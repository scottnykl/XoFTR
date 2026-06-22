"""Run XoFTR cross-modal matching on live AftrBurner image streams.

Connects to two AftrBurner ImGui_Stream_Source endpoints (one visible, one
thermal), pairs incoming frames, and runs XoFTR matching in real time.

Usage:
  python run_XoFTR_on_AftrImgPairs.py \
      --vis_port 12676 \
      --tir_port 12698 \
      --ckpt weights/weights_xoftr_640.ckpt \
      --show

  # Single stream with TypeA message differentiation (both on same port):
  python run_XoFTR_on_AftrImgPairs.py \
      --vis_port 12676 \
      --single_stream \
      --ckpt weights/weights_xoftr_640.ckpt \
      --show

Requires the xoftr conda environment (PyTorch, kornia, etc.).
"""

import asyncio
import argparse
import sys
import threading
import time
import cv2
import numpy as np
from pathlib import Path
from collections import deque

from aftr_tcp_listener import (
    AftrTcpListener, GCamImage, ComponentLayout, OriginCorner, MSG_ID,
)


def load_xoftr(ckpt_path, match_threshold=0.3, fine_threshold=0.1):
    from src.xoftr import XoFTR
    from src.config.default import get_cfg_defaults
    from src.utils.data_io import DataIOWrapper, lower_config

    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    config["xoftr"]["match_coarse"]["thr"] = match_threshold
    config["xoftr"]["fine"]["thr"] = fine_threshold

    matcher = XoFTR(config=config["xoftr"])
    matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt_path)
    return matcher


def gcam_to_bgr(img: GCamImage) -> np.ndarray:
    """Convert a GCamImage to BGR numpy array for OpenCV/XoFTR."""
    pixels = img.pixels
    if img.comp_layout == ComponentLayout.RGB:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    elif img.comp_layout == ComponentLayout.RGBA:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    elif img.comp_layout == ComponentLayout.BGRA:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    elif img.comp_layout == ComponentLayout.GRAY:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    if img.origin_corner == OriginCorner.LOWER_LEFT:
        pixels = cv2.flip(pixels, 0)
    return pixels


def draw_matches(img0, img1, mkpts0, mkpts1, mconf, max_kpts=500):
    """Draw matching keypoints side-by-side on a single image."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:] = img1

    n = len(mkpts0)
    if n == 0:
        cv2.putText(canvas, "No matches", (w0 // 2, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return canvas

    indices = np.arange(n)
    if n > max_kpts:
        indices = np.argsort(mconf)[-max_kpts:]

    for idx in indices:
        pt0 = tuple(mkpts0[idx].astype(int))
        pt1 = (int(mkpts1[idx][0]) + w0, int(mkpts1[idx][1]))
        c = mconf[idx]
        color = (0, int(255 * c), int(255 * (1 - c)))
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 3, color, -1, cv2.LINE_AA)

    cv2.putText(canvas, "{} matches".format(n), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return canvas


class XoFTR_AftrBridge:
    """Bridges AftrBurner image streams to XoFTR matching."""

    def __init__(self, matcher, K0=None, K1=None, dist0=None, dist1=None,
                 show=False):
        self.matcher = matcher
        self.K0 = K0
        self.K1 = K1
        self.dist0 = dist0
        self.dist1 = dist1
        self.show = show

        self._vis_queue = deque(maxlen=2)
        self._tir_queue = deque(maxlen=2)
        self._lock = threading.Lock()
        self._match_count = 0
        self._last_result = None

    def on_vis_image(self, img: GCamImage):
        bgr = gcam_to_bgr(img)
        with self._lock:
            self._vis_queue.append((img.frame_idx, bgr))
        self._try_match()

    def on_tir_image(self, img: GCamImage):
        bgr = gcam_to_bgr(img)
        with self._lock:
            self._tir_queue.append((img.frame_idx, bgr))
        self._try_match()

    def on_single_stream_image(self, img: GCamImage):
        """For single-stream mode: NetMsgSend_GCam_Image = visible,
        TypeA_NetMsgSend_GCam_Image = thermal."""
        bgr = gcam_to_bgr(img)
        with self._lock:
            self._vis_queue.append((img.frame_idx, bgr))
        self._try_match()

    def on_single_stream_image_typeA(self, img: GCamImage):
        bgr = gcam_to_bgr(img)
        with self._lock:
            self._tir_queue.append((img.frame_idx, bgr))
        self._try_match()

    def _try_match(self):
        with self._lock:
            if not self._vis_queue or not self._tir_queue:
                return
            _, vis_bgr = self._vis_queue.popleft()
            _, tir_bgr = self._tir_queue.popleft()

        t0 = time.perf_counter()
        result = self.matcher.from_cv_imgs(
            vis_bgr, tir_bgr,
            K0=self.K0, K1=self.K1,
            dist0=self.dist0, dist1=self.dist1,
        )
        dt = time.perf_counter() - t0

        self._match_count += 1
        n_matches = len(result["mkpts0"])
        print("Match #{}: {} keypoints in {:.1f}ms".format(
            self._match_count, n_matches, dt * 1000))

        self._last_result = result

        if self.show:
            canvas = draw_matches(
                vis_bgr, tir_bgr,
                result["mkpts0"], result["mkpts1"], result["mconf"],
            )
            cv2.imshow("XoFTR Matches", canvas)
            cv2.waitKey(1)


async def run_dual_sink(bridge, host, vis_port, tir_port):
    """Connect to two separate Source endpoints (visible + thermal)."""
    vis_listener = AftrTcpListener()
    tir_listener = AftrTcpListener()

    vis_listener.on_image(bridge.on_vis_image)
    tir_listener.on_image(bridge.on_tir_image)

    vis_task = asyncio.create_task(
        vis_listener.run_sink(host, vis_port))
    tir_task = asyncio.create_task(
        tir_listener.run_sink(host, tir_port))

    print("Connecting to visible stream on port {} and thermal stream on port {}...".format(
        vis_port, tir_port))
    await asyncio.gather(vis_task, tir_task)


async def run_single_sink(bridge, host, port):
    """Connect to one Source, differentiate by message type."""
    listener = AftrTcpListener()

    # Override dispatch to separate NetMsgSend_GCam_Image vs TypeA
    original_dispatch = listener._dispatch

    def custom_dispatch(msg_id, payload):
        from aftr_tcp_listener import parse_gcam_image
        if msg_id == MSG_ID["NetMsgSend_GCam_Image"]:
            img = parse_gcam_image(payload)
            bridge.on_single_stream_image(img)
        elif msg_id == MSG_ID["TypeA_NetMsgSend_GCam_Image"]:
            img = parse_gcam_image(payload)
            bridge.on_single_stream_image_typeA(img)
        else:
            listener._msg_count += 1
            for cb in listener._raw_callbacks:
                cb(msg_id, payload)

    listener._dispatch = custom_dispatch

    print("Connecting to single stream on port {}...".format(port))
    await listener.run_sink(host, port)


def main():
    parser = argparse.ArgumentParser(
        description="Run XoFTR matching on live AftrBurner image streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="AftrBurner Source host (default: 127.0.0.1)")
    parser.add_argument("--vis_port", type=int, default=12676,
                        help="TCP port for visible image stream (default: 12676)")
    parser.add_argument("--tir_port", type=int, default=12698,
                        help="TCP port for thermal image stream (default: 12698)")
    parser.add_argument("--single_stream", action="store_true",
                        help="Both image types on one port (NetMsgSend_GCam_Image=vis, "
                             "TypeA=tir). Only --vis_port is used.")
    parser.add_argument("--ckpt", default="weights/weights_xoftr_640.ckpt",
                        help="XoFTR checkpoint path (default: weights/weights_xoftr_640.ckpt)")
    parser.add_argument("--match_threshold", type=float, default=0.3,
                        help="Coarse matching threshold (default: 0.3)")
    parser.add_argument("--fine_threshold", type=float, default=0.1,
                        help="Fine matching threshold (default: 0.1)")
    parser.add_argument("--show", action="store_true",
                        help="Display match visualization in an OpenCV window")

    # Optional calibration (if not provided, no undistortion is applied)
    parser.add_argument("--calib", default=None,
                        help="Path to calibration YAML (same format as create_scene_npz.py)")

    args = parser.parse_args()

    # Load calibration if provided
    K0, K1, dist0, dist1 = None, None, None, None
    if args.calib:
        from create_scene_npz import load_calibration
        calib = load_calibration(args.calib)
        K0 = calib["visible"]["K"].astype(np.float32)
        K1 = calib["thermal"]["K"].astype(np.float32)
        dist0 = calib["visible"]["dist"]
        dist1 = calib["thermal"]["dist"]
        print("Loaded calibration from {}".format(args.calib))

    print("Loading XoFTR from {}...".format(args.ckpt))
    matcher = load_xoftr(args.ckpt, args.match_threshold, args.fine_threshold)
    print("XoFTR loaded.")

    bridge = XoFTR_AftrBridge(
        matcher, K0=K0, K1=K1, dist0=dist0, dist1=dist1, show=args.show,
    )

    if args.show:
        cv2.namedWindow("XoFTR Matches", cv2.WINDOW_NORMAL)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        if args.single_stream:
            asyncio.run(run_single_sink(bridge, args.host, args.vis_port))
        else:
            asyncio.run(run_dual_sink(bridge, args.host, args.vis_port, args.tir_port))
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
