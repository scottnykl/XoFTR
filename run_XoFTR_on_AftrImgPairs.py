"""Run XoFTR cross-modal matching on live AftrBurner image streams.

Primary mode — receive NetMsgSend_ImagePair (paired visible+thermal in one msg):

  python run_XoFTR_on_AftrImgPairs.py \
      --port 12676 \
      --ckpt weights/weights_xoftr_640.ckpt \
      --show

Dual-sink mode — two separate Source endpoints (one visible, one thermal):

  python run_XoFTR_on_AftrImgPairs.py \
      --mode dual \
      --vis_port 12676 \
      --tir_port 12698 \
      --ckpt weights/weights_xoftr_640.ckpt \
      --show

Single-stream mode — TypeA message differentiation (both on same port):

  python run_XoFTR_on_AftrImgPairs.py \
      --mode single_stream \
      --port 12676 \
      --ckpt weights/weights_xoftr_640.ckpt \
      --show

Requires the xoftr conda environment (PyTorch, kornia, etc.).
"""

import asyncio
import argparse
import signal
import sys
import time
import cv2
import numpy as np
from collections import deque

from aftr_tcp_listener import (
    AftrTcpListener, GCamImage, ImagePair, ComponentLayout, OriginCorner,
    MSG_ID, parse_gcam_image, _run_async, build_inference_result,
)


def estimate_relative_pose(mkpts0, mkpts1, K0, K1, thresh=0.5, conf=0.99999):
    """Estimate relative pose from matched keypoints and intrinsics.

    Returns (yaw_deg, pitch_deg, roll_deg, t_user, n_inliers) in user coords
    (X=forward, Y=left, Z=up), or None if estimation fails.
    """
    from src.utils.metrics import estimate_pose

    ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh, conf)
    if ret is None:
        return None

    R_cv, t_cv, inliers = ret

    # OpenCV camera coords: X=right, Y=down, Z=forward
    # User coords:          X=forward, Y=left, Z=up
    C = np.array([[0, 0, 1],
                  [-1, 0, 0],
                  [0, -1, 0]], dtype=np.float64)

    R_user = C @ R_cv @ C.T
    t_user = C @ t_cv

    # Extract YPR from R = Rx(roll) * Ry(pitch) * Rz(yaw)
    pitch = np.degrees(np.arcsin(np.clip(R_user[0, 2], -1.0, 1.0)))
    yaw = np.degrees(np.arctan2(-R_user[0, 1], R_user[0, 0]))
    roll = np.degrees(np.arctan2(-R_user[1, 2], R_user[2, 2]))

    return yaw, pitch, roll, t_user, int(np.sum(inliers))


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
                 show=False, step=False, top_k=None, listener=None):
        self.matcher = matcher
        self.K0 = K0
        self.K1 = K1
        self.dist0 = dist0
        self.dist1 = dist1
        self.show = show
        self.step = step
        self.top_k = top_k
        self.listener = listener
        self._shutting_down = False

        self._vis_queue = deque(maxlen=2)
        self._tir_queue = deque(maxlen=2)
        self._match_count = 0
        self._last_result = None

    # -- ImagePair mode (primary) ------------------------------------------

    def on_image_pair(self, pair: ImagePair):
        """Process a NetMsgSend_ImagePair — imgA is visible, imgB is thermal."""
        vis_bgr = gcam_to_bgr(pair.imgA)
        tir_bgr = gcam_to_bgr(pair.imgB)
        self._run_match(vis_bgr, tir_bgr,
                        frame_a=pair.imgA.frame_idx, frame_b=pair.imgB.frame_idx,
                        time_utc_a=pair.imgA.timestamp_utc,
                        orig_w_a=pair.imgA.width, orig_h_a=pair.imgA.height,
                        orig_w_b=pair.imgB.width, orig_h_b=pair.imgB.height)

    # -- Dual-sink mode ----------------------------------------------------

    def on_vis_image(self, img: GCamImage):
        self._vis_queue.append(gcam_to_bgr(img))
        self._try_queued_match()

    def on_tir_image(self, img: GCamImage):
        self._tir_queue.append(gcam_to_bgr(img))
        self._try_queued_match()

    # -- Single-stream mode ------------------------------------------------

    def on_single_stream_vis(self, img: GCamImage):
        self._vis_queue.append(gcam_to_bgr(img))
        self._try_queued_match()

    def on_single_stream_tir(self, img: GCamImage):
        self._tir_queue.append(gcam_to_bgr(img))
        self._try_queued_match()

    # -- Internals ---------------------------------------------------------

    def _send_inference_result(self, result, frame_idx, time_utc,
                               orig_w_a, orig_h_a, orig_w_b, orig_h_b):
        """Normalize keypoints to [0,1] and send back to AftrBurner."""
        if self.listener is None:
            print("  [InferenceResult] skipped: listener is None", flush=True)
            return
        if self.listener.writer is None:
            print("  [InferenceResult] skipped: listener.writer is None", flush=True)
            return
        if frame_idx is None or time_utc is None:
            print("  [InferenceResult] skipped: frame_idx={} time_utc={}".format(
                frame_idx, time_utc), flush=True)
            return

        mkpts0 = result["mkpts0"]
        mkpts1 = result["mkpts1"]
        mconf = result["mconf"]
        n = len(mkpts0)
        if n == 0:
            return

        order = np.argsort(mconf)[::-1]

        matches = []
        for i in order:
            ua = float(mkpts0[i, 0]) / orig_w_a
            va = float(mkpts0[i, 1]) / orig_h_a
            ub = float(mkpts1[i, 0]) / orig_w_b
            vb = float(mkpts1[i, 1]) / orig_h_b
            matches.append((ua, va, ub, vb, float(mconf[i])))

        msg_bytes = build_inference_result(frame_idx, time_utc, matches)
        self.listener.writer.write(msg_bytes)
        print("  Sent {} matches back to AftrBurner (frame {})".format(
            len(matches), frame_idx), flush=True)

    def _try_queued_match(self):
        if not self._vis_queue or not self._tir_queue:
            return
        vis_bgr = self._vis_queue.popleft()
        tir_bgr = self._tir_queue.popleft()
        self._run_match(vis_bgr, tir_bgr)

    def _run_match(self, vis_bgr, tir_bgr, frame_a=None, frame_b=None,
                   time_utc_a=None, orig_w_a=None, orig_h_a=None,
                   orig_w_b=None, orig_h_b=None):
        t0 = time.perf_counter()
        result = self.matcher.from_cv_imgs(
            vis_bgr, tir_bgr,
            K0=self.K0, K1=self.K1,
            dist0=self.dist0, dist1=self.dist1,
        )
        dt = time.perf_counter() - t0

        if self.top_k is not None and len(result["mkpts0"]) > self.top_k:
            top_idx = np.argsort(result["mconf"])[-self.top_k:]
            result["mkpts0"] = result["mkpts0"][top_idx]
            result["mkpts1"] = result["mkpts1"][top_idx]
            result["mconf"] = result["mconf"][top_idx]
            result["matches"] = result["matches"][top_idx]

        self._match_count += 1
        n_matches = len(result["mkpts0"])
        frame_info = ""
        if frame_a is not None:
            frame_info = " [frameA={} frameB={}]".format(frame_a, frame_b)
        print("Match #{}: {} keypoints in {:.1f}ms{}".format(
            self._match_count, n_matches, dt * 1000, frame_info), flush=True)

        self._last_result = result

        self._send_inference_result(
            result, frame_a, time_utc_a,
            orig_w_a, orig_h_a, orig_w_b, orig_h_b)

        pose_lines = []
        if self.K0 is not None and self.K1 is not None:
            pose_K0 = result.get("new_K0", self.K0).astype(np.float64)
            pose_K1 = result.get("new_K1", self.K1).astype(np.float64)
            pose = estimate_relative_pose(
                result["mkpts0"], result["mkpts1"], pose_K0, pose_K1)
            if pose is not None:
                yaw, pitch, roll, t, n_inl = pose
                print("  YPR: ({:.2f} {:.2f} {:.2f})  T: ({:.3f} {:.3f} {:.3f})  inliers: {}".format(
                    yaw, pitch, roll, t[0], t[1], t[2], n_inl), flush=True)
                pose_lines.append("YPR: ({:.2f} {:.2f} {:.2f})".format(yaw, pitch, roll))
                pose_lines.append("T: ({:.3f} {:.3f} {:.3f})".format(t[0], t[1], t[2]))
                pose_lines.append("inliers: {}".format(n_inl))
            else:
                print("  Pose estimation failed (< 5 matches or degenerate)", flush=True)

        if self.show:
            canvas = draw_matches(
                vis_bgr, tir_bgr,
                result["mkpts0"], result["mkpts1"], result["mconf"],
            )
            if pose_lines:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.6
                thickness = 2
                margin = 10
                line_h = 28
                max_w = max(cv2.getTextSize(l, font, scale, thickness)[0][0]
                           for l in pose_lines)
                ch, cw = canvas.shape[:2]
                x0 = cw - max_w - margin
                y0 = ch - margin - line_h * len(pose_lines)
                cv2.rectangle(canvas,
                              (x0 - 6, y0 - 20),
                              (cw - margin + 6, ch - margin + 6),
                              (0, 0, 0), -1)
                for i, line in enumerate(pose_lines):
                    cv2.putText(canvas, line,
                                (x0, y0 + i * line_h),
                                font, scale, (0, 255, 255), thickness, cv2.LINE_AA)
            cv2.imshow("XoFTR Matches", canvas)
            if self.step:
                print("Press any key in the image window for next pair...", flush=True)
                while cv2.waitKey(100) == -1:
                    if self._shutting_down:
                        return
            else:
                cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Async entry points
# ---------------------------------------------------------------------------

async def run_pair_sink(bridge, host, port):
    """Connect to a Source streaming NetMsgSend_ImagePair messages."""
    listener = AftrTcpListener()
    listener.on_image_pair(bridge.on_image_pair)
    print("Connecting to ImagePair stream on {}:{}...".format(host, port), flush=True)
    await listener.run_sink(host, port)


async def run_dual_sink(bridge, host, vis_port, tir_port):
    """Connect to two separate Source endpoints (visible + thermal)."""
    vis_listener = AftrTcpListener()
    tir_listener = AftrTcpListener()

    vis_listener.on_image(bridge.on_vis_image)
    tir_listener.on_image(bridge.on_tir_image)

    print("Connecting to visible on port {} and thermal on port {}...".format(
        vis_port, tir_port), flush=True)
    await asyncio.gather(
        vis_listener.run_sink(host, vis_port),
        tir_listener.run_sink(host, tir_port),
    )


async def run_single_sink(bridge, host, port):
    """Connect to one Source, differentiate by message type."""
    listener = AftrTcpListener()

    def custom_dispatch(msg_id, payload):
        if msg_id == MSG_ID["NetMsgSend_GCam_Image"]:
            bridge.on_single_stream_vis(parse_gcam_image(payload))
        elif msg_id == MSG_ID["TypeA_NetMsgSend_GCam_Image"]:
            bridge.on_single_stream_tir(parse_gcam_image(payload))
        else:
            listener._msg_count += 1

    listener._dispatch = custom_dispatch

    print("Connecting to single stream on port {}...".format(port), flush=True)
    await listener.run_sink(host, port)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run XoFTR matching on live AftrBurner image streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["pair", "dual", "single_stream"],
                        default="pair",
                        help="pair: NetMsgSend_ImagePair on one port (default). "
                             "dual: two separate Source ports. "
                             "single_stream: TypeA differentiation on one port.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="AftrBurner Source host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12676,
                        help="TCP port for pair/single_stream mode (default: 12676)")
    parser.add_argument("--vis_port", type=int, default=12676,
                        help="TCP port for visible stream in dual mode (default: 12676)")
    parser.add_argument("--tir_port", type=int, default=12698,
                        help="TCP port for thermal stream in dual mode (default: 12698)")
    parser.add_argument("--ckpt", default="weights/weights_xoftr_640.ckpt",
                        help="XoFTR checkpoint path (default: weights/weights_xoftr_640.ckpt)")
    parser.add_argument("--match_threshold", type=float, default=0.3,
                        help="Coarse matching threshold (default: 0.3)")
    parser.add_argument("--fine_threshold", type=float, default=0.1,
                        help="Fine matching threshold (default: 0.1)")
    parser.add_argument("--show", action="store_true",
                        help="Display match visualization in an OpenCV window")
    parser.add_argument("--step", action="store_true",
                        help="Step-through mode: process one pair at a time, "
                             "wait for keypress before next (implies --show)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Use and display only the top K matches by confidence")
    parser.add_argument("--calib", default=None,
                        help="Path to calibration YAML (same format as create_scene_npz.py)")

    args = parser.parse_args()

    if args.step:
        args.show = True

    # Load calibration if provided
    K0, K1, dist0, dist1 = None, None, None, None
    if args.calib:
        from create_scene_npz import load_calibration
        calib = load_calibration(args.calib)
        K0 = calib["visible"]["K"].astype(np.float32)
        K1 = calib["thermal"]["K"].astype(np.float32)
        dist0 = calib["visible"]["dist"]
        dist1 = calib["thermal"]["dist"]
        print("Loaded calibration from {}".format(args.calib), flush=True)

    print("Loading XoFTR from {}...".format(args.ckpt), flush=True)
    matcher = load_xoftr(args.ckpt, args.match_threshold, args.fine_threshold)
    print("XoFTR loaded.", flush=True)

    bridge = XoFTR_AftrBridge(
        matcher, K0=K0, K1=K1, dist0=dist0, dist1=dist1,
        show=args.show, step=args.step, top_k=args.top_k,
    )

    if args.show:
        cv2.namedWindow("XoFTR Matches", cv2.WINDOW_NORMAL)

    # Build the async coroutine for the selected mode
    # We reuse _run_async from aftr_tcp_listener for proper Ctrl+C on Windows
    class _Args:
        pass
    run_args = _Args()
    run_args.mode = "sink"
    run_args.host = args.host
    run_args.subscribe_idx = -1

    if args.mode == "pair":
        run_args.port = args.port
        listener = AftrTcpListener()
        bridge.listener = listener
        listener.on_image_pair(bridge.on_image_pair)
    elif args.mode == "dual":
        # For dual mode we build a custom listener and override run
        listener = None
    elif args.mode == "single_stream":
        run_args.port = args.port
        listener = AftrTcpListener()
        bridge.listener = listener
        def custom_dispatch(msg_id, payload):
            if msg_id == MSG_ID["NetMsgSend_GCam_Image"]:
                bridge.on_single_stream_vis(parse_gcam_image(payload))
            elif msg_id == MSG_ID["TypeA_NetMsgSend_GCam_Image"]:
                bridge.on_single_stream_tir(parse_gcam_image(payload))
        listener._dispatch = custom_dispatch

    # Run with proper Ctrl+C handling
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if args.mode == "dual":
        main_coro = run_dual_sink(bridge, args.host, args.vis_port, args.tir_port)
    elif args.mode == "pair":
        main_coro = listener.run_sink(args.host, args.port, step=True)
    else:
        main_coro = listener.run_sink(args.host, args.port, step=True)

    main_task = loop.create_task(main_coro)

    async def _keepalive():
        while True:
            await asyncio.sleep(0.2)

    keepalive_task = loop.create_task(_keepalive())

    def _shutdown(signum=None, frame=None):
        print("\nCtrl+C received, shutting down...", flush=True)
        bridge._shutting_down = True
        main_task.cancel()
        keepalive_task.cancel()

    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        keepalive_task.cancel()
        try:
            loop.run_until_complete(keepalive_task)
        except asyncio.CancelledError:
            pass
        loop.close()
        cv2.destroyAllWindows()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
