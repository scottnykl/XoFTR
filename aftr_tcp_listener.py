"""AftrBurner TCP Listener — receives NetMsg messages from an AftrBurner engine.

Can operate in two modes:

  SERVER mode (default): Python listens on a port. The AftrBurner engine
  connects to Python via NetMessengerClient. Use this when the engine is
  configured to send data to Python's IP:PORT.

    python aftr_tcp_listener.py --mode server --port 12690

  SINK mode: Python connects to an AftrBurner ImGui_Stream_Source, subscribes,
  and receives a live stream of images. Use this to tap into an existing
  Source that is already running.

    python aftr_tcp_listener.py --mode sink --host 127.0.0.1 --port 12676

Received images are stored as numpy arrays and can be accessed via callbacks.
Run standalone to see incoming messages printed to the console.
"""

import asyncio
import struct
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
import argparse
import signal
import sys
import logging

log = logging.getLogger("aftr_tcp")


# ---------------------------------------------------------------------------
# AftrBurner NetMsg wire protocol constants
# ---------------------------------------------------------------------------

HEADER_SIZE = 8  # 4 bytes msg_id + 4 bytes payload_length, both big-endian
HEADER_STRUCT = struct.Struct(">II")  # two big-endian uint32


def fnv1a_32(name: str) -> int:
    """Compute FNV-1a 32-bit hash of a class name string, matching AftrBurner's implementation."""
    h = 2166136261  # FNV offset basis
    for c in name.encode("ascii"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFFFFFFFFFF  # 64-bit intermediate
    return h & 0xFFFFFFFF


# Pre-compute message IDs for known NetMsg types
MSG_ID = {
    "NetMsgGeneric": 1,
    "NetMsgSend_GCam_Image": fnv1a_32("NetMsgSend_GCam_Image"),
    "TypeA_NetMsgSend_GCam_Image": fnv1a_32("TypeA_NetMsgSend_GCam_Image"),
    "NetMsg_Subscribe_to_Stream_Source": fnv1a_32("NetMsg_Subscribe_to_Stream_Source"),
    "NetMsg_SessionStreamMode_to_LiveStream": fnv1a_32("NetMsg_SessionStreamMode_to_LiveStream"),
    "NetMsg_SessionStreamMode_to_UponRequest": fnv1a_32("NetMsg_SessionStreamMode_to_UponRequest"),
    "NetMsg_Request_Next_SensorDatum": fnv1a_32("NetMsg_Request_Next_SensorDatum"),
    "NetMsgSendString": fnv1a_32("NetMsgSendString"),
    "NetMsgSendDCM3x3": fnv1a_32("NetMsgSendDCM3x3"),
    "NetMsgSend_ImagePair": fnv1a_32("NetMsgSend_ImagePair"),
    "NetMsg_2D_to_2D_InferenceResult": fnv1a_32("NetMsg_2D_to_2D_InferenceResult"),
}

# Reverse lookup: ID -> name
MSG_NAME = {v: k for k, v in MSG_ID.items()}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ComponentLayout(IntEnum):
    GRAY = 0
    RGB = 1
    BGR = 2
    RGBA = 3
    BGRA = 4


class OriginCorner(IntEnum):
    UPPER_LEFT = 0
    LOWER_LEFT = 1


@dataclass
class GCamImage:
    width: int
    height: int
    num_components: int
    bytes_per_component: int
    is_interleaved: bool
    comp_layout: ComponentLayout
    origin_corner: OriginCorner
    pixels: np.ndarray
    timestamp_utc: str
    frame_idx: int


@dataclass
class ImagePair:
    imgA: GCamImage
    imgB: GCamImage


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

class NetMsgReader:
    """Reads fields from a NetMsg payload buffer in big-endian order."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read_int32(self) -> int:
        val = struct.unpack_from(">i", self._data, self._pos)[0]
        self._pos += 4
        return val

    def read_uint32(self) -> int:
        val = struct.unpack_from(">I", self._data, self._pos)[0]
        self._pos += 4
        return val

    def read_bool(self) -> bool:
        val = self._data[self._pos]
        self._pos += 1
        return val != 0

    def read_uchar(self) -> int:
        val = self._data[self._pos]
        self._pos += 1
        return val

    def read_float(self) -> float:
        val = struct.unpack_from(">f", self._data, self._pos)[0]
        self._pos += 4
        return val

    def read_double(self) -> float:
        val = struct.unpack_from(">d", self._data, self._pos)[0]
        self._pos += 8
        return val

    def read_bytes(self, n: int) -> bytes:
        val = self._data[self._pos : self._pos + n]
        self._pos += n
        return val

    def read_cstring(self) -> str:
        end = self._data.index(b"\x00", self._pos)
        val = self._data[self._pos : end].decode("utf-8")
        self._pos = end + 1
        return val

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos


class NetMsgWriter:
    """Builds a NetMsg payload buffer in big-endian order."""

    def __init__(self):
        self._buf = bytearray()

    def write_int32(self, val: int):
        self._buf.extend(struct.pack(">i", val))

    def write_uint32(self, val: int):
        self._buf.extend(struct.pack(">I", val))

    def write_bool(self, val: bool):
        self._buf.append(1 if val else 0)

    def write_uchar(self, val: int):
        self._buf.append(val & 0xFF)

    def write_cstring(self, val: str):
        self._buf.extend(val.encode("utf-8"))
        self._buf.append(0)

    def write_float(self, val: float):
        self._buf.extend(struct.pack(">f", val))

    def write_bytes(self, data: bytes):
        self._buf.extend(data)

    def to_bytes(self) -> bytes:
        return bytes(self._buf)


def build_netmsg(msg_name: str, payload: bytes = b"") -> bytes:
    """Build a complete NetMsg frame: 8-byte header + payload."""
    msg_id = MSG_ID.get(msg_name)
    if msg_id is None:
        msg_id = fnv1a_32(msg_name)
    header = HEADER_STRUCT.pack(msg_id, len(payload))
    return header + payload


def build_inference_result(frame_idx: int, time_utc: str, matches) -> bytes:
    """Build a NetMsg_2D_to_2D_InferenceResult frame.

    Args:
        frame_idx: Frame index relayed from the received image pair.
        time_utc: UTC time string relayed from the received image pair.
        matches: Iterable of (ua, va, ub, vb, conf) tuples sorted by
                 descending confidence.
    """
    w = NetMsgWriter()
    w.write_uint32(frame_idx)
    w.write_cstring(time_utc)
    w.write_uint32(len(matches))
    for ua, va, ub, vb, conf in matches:
        w.write_float(ua)
        w.write_float(va)
        w.write_float(ub)
        w.write_float(vb)
        w.write_float(conf)
    return build_netmsg("NetMsg_2D_to_2D_InferenceResult", w.to_bytes())


# ---------------------------------------------------------------------------
# Message parsers
# ---------------------------------------------------------------------------

def _parse_gcam_from_reader(r: NetMsgReader) -> GCamImage:
    """Parse one GCamImage (image + timestamp + frameIdx) from a reader."""
    w = r.read_int32()
    h = r.read_int32()
    num_comp = r.read_int32()
    bpc = r.read_int32()
    is_interleaved = r.read_bool()
    comp_layout = ComponentLayout(r.read_uchar())
    origin_corner = OriginCorner(r.read_uchar())

    pixel_size = w * h * num_comp * bpc
    pixel_bytes = r.read_bytes(pixel_size)

    timestamp = r.read_cstring()
    frame_idx = r.read_uint32()

    if bpc == 1:
        dtype = np.uint8
    elif bpc == 2:
        dtype = np.uint16
    elif bpc == 4:
        dtype = np.float32
    else:
        dtype = np.uint8

    pixels = np.frombuffer(pixel_bytes, dtype=dtype)
    if is_interleaved and num_comp > 1:
        pixels = pixels.reshape((h, w, num_comp))
    elif num_comp == 1:
        pixels = pixels.reshape((h, w))
    else:
        pixels = pixels.reshape((num_comp, h, w))

    return GCamImage(
        width=w, height=h, num_components=num_comp,
        bytes_per_component=bpc, is_interleaved=is_interleaved,
        comp_layout=comp_layout, origin_corner=origin_corner,
        pixels=pixels, timestamp_utc=timestamp, frame_idx=frame_idx,
    )


def parse_gcam_image(data: bytes) -> GCamImage:
    return _parse_gcam_from_reader(NetMsgReader(data))


def parse_image_pair(data: bytes) -> ImagePair:
    r = NetMsgReader(data)
    imgA = _parse_gcam_from_reader(r)
    imgB = _parse_gcam_from_reader(r)
    return ImagePair(imgA=imgA, imgB=imgB)


def parse_send_string(data: bytes) -> str:
    r = NetMsgReader(data)
    return r.read_cstring()


def parse_dcm3x3(data: bytes) -> np.ndarray:
    r = NetMsgReader(data)
    dcm = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for col in range(3):
            dcm[row, col] = r.read_double()
    return dcm


# ---------------------------------------------------------------------------
# TCP stream reading
# ---------------------------------------------------------------------------

async def read_exactly(reader: asyncio.StreamReader, n: int) -> bytes:
    """Read exactly n bytes from the stream, raising on disconnect."""
    data = await reader.readexactly(n)
    return data


async def read_one_netmsg(reader: asyncio.StreamReader) -> Tuple[int, bytes]:
    """Read one NetMsg from the TCP stream. Returns (msg_id, payload_bytes)."""
    header = await read_exactly(reader, HEADER_SIZE)
    msg_id, payload_len = HEADER_STRUCT.unpack(header)
    if payload_len > 0:
        payload = await read_exactly(reader, payload_len)
    else:
        payload = b""
    return msg_id, payload


# ---------------------------------------------------------------------------
# AftrBurner TCP Listener
# ---------------------------------------------------------------------------

class AftrTcpListener:
    """Receives AftrBurner NetMsg messages over TCP.

    Register callbacks with `on_image`, `on_message`, etc. before calling
    `run_server()` or `run_sink()`.
    """

    def __init__(self):
        self._image_callbacks: List[Callable[[GCamImage], None]] = []
        self._pair_callbacks: List[Callable[[ImagePair], None]] = []
        self._raw_callbacks: List[Callable[[int, bytes], None]] = []
        self._msg_count = 0
        self._image_count = 0
        self.writer: Optional[asyncio.StreamWriter] = None

    def on_image(self, callback: Callable[[GCamImage], None]):
        self._image_callbacks.append(callback)

    def on_image_pair(self, callback: Callable[[ImagePair], None]):
        self._pair_callbacks.append(callback)

    def on_raw_message(self, callback: Callable[[int, bytes], None]):
        self._raw_callbacks.append(callback)

    def _dispatch(self, msg_id: int, payload: bytes):
        self._msg_count += 1

        if msg_id in (MSG_ID["NetMsgSend_GCam_Image"], MSG_ID["TypeA_NetMsgSend_GCam_Image"]):
            img = parse_gcam_image(payload)
            self._image_count += 1
            for cb in self._image_callbacks:
                cb(img)
        elif msg_id == MSG_ID["NetMsgSend_ImagePair"]:
            pair = parse_image_pair(payload)
            self._image_count += 2
            for cb in self._pair_callbacks:
                cb(pair)
        else:
            for cb in self._raw_callbacks:
                cb(msg_id, payload)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        print(f"Connection from {peer}")
        try:
            while True:
                msg_id, payload = await read_one_netmsg(reader)
                self._dispatch(msg_id, payload)
        except (asyncio.IncompleteReadError, ConnectionError):
            print(f"Connection closed by {peer}")
        finally:
            writer.close()

    async def run_server(self, host: str = "0.0.0.0", port: int = 12690):
        """Listen for incoming TCP connections from the AftrBurner engine."""
        server = await asyncio.start_server(self._handle_connection, host, port)
        addrs = [s.getsockname() for s in server.sockets]
        print(f"AftrBurner TCP listener serving on {addrs}")
        async with server:
            await server.serve_forever()

    async def run_sink(self, host: str = "127.0.0.1", port: int = 12676,
                       subscribe_idx: int = -1, step: bool = False,
                       drain: bool = False):
        """Connect to an AftrBurner ImGui_Stream_Source and receive live images.

        If step=True, uses UponRequest mode — sends NetMsg_Request_Next_SensorDatum
        before each read, so the dispatch callback controls the pace.

        If drain=True, after reading a message, drains any additional complete
        messages already buffered and only dispatches the latest one.  Useful when
        the source streams faster than the consumer can process.
        """
        print(f"Connecting to AftrBurner Source at {host}:{port}...", flush=True)
        reader, writer = await asyncio.open_connection(host, port)
        self.writer = writer
        print(f"Connected to {host}:{port}.", flush=True)

        # Send NetMsg_Subscribe_to_Stream_Source (payload: int32 idx)
        w = NetMsgWriter()
        w.write_int32(subscribe_idx)
        sub_bytes = build_netmsg("NetMsg_Subscribe_to_Stream_Source", w.to_bytes())
        writer.write(sub_bytes)
        await writer.drain()
        log.debug("Sent Subscribe (%d bytes): %s", len(sub_bytes), sub_bytes.hex())
        print(f"Sent subscription (idx={subscribe_idx}).", flush=True)

        if step:
            upon_req_bytes = build_netmsg("NetMsg_SessionStreamMode_to_UponRequest")
            writer.write(upon_req_bytes)
            await writer.drain()
            print("Sent UponRequest stream mode. Step-through active.", flush=True)
        else:
            live_bytes = build_netmsg("NetMsg_SessionStreamMode_to_LiveStream")
            writer.write(live_bytes)
            await writer.drain()
            log.debug("Sent LiveStream (%d bytes): %s", len(live_bytes), live_bytes.hex())
            print("Sent live stream request. Waiting for data...", flush=True)

        req_next_bytes = build_netmsg("NetMsg_Request_Next_SensorDatum")

        if drain:
            await self._run_sink_drain(reader, writer, step, req_next_bytes)
        else:
            await self._run_sink_loop(reader, writer, step, req_next_bytes)

    async def _run_sink_loop(self, reader, writer, step, req_next_bytes):
        try:
            while True:
                if step:
                    writer.write(req_next_bytes)
                    await writer.drain()
                msg_id, payload = await read_one_netmsg(reader)
                log.debug("Recv msg_id=0x%08X len=%d", msg_id, len(payload))
                self._dispatch(msg_id, payload)
        except asyncio.CancelledError:
            print("Sink task cancelled.", flush=True)
        except (asyncio.IncompleteReadError, ConnectionError) as e:
            print(f"Connection to source closed: {e}", flush=True)
        finally:
            writer.close()

    async def _run_sink_drain(self, reader, writer, step, req_next_bytes):
        """Drain mode: a background task reads messages continuously and
        overwrites a 'latest' slot.  The main loop picks up whatever is
        newest after each dispatch, skipping stale frames."""
        latest = [None]          # type: List[Optional[Tuple[int, bytes]]]
        ready = asyncio.Event()
        reader_done = [False]

        async def _reader():
            try:
                while True:
                    msg = await read_one_netmsg(reader)
                    latest[0] = msg
                    ready.set()
            except (asyncio.CancelledError, asyncio.IncompleteReadError,
                    ConnectionError):
                reader_done[0] = True
                ready.set()

        reader_task = asyncio.ensure_future(_reader())

        try:
            while True:
                await ready.wait()
                if reader_done[0]:
                    break
                ready.clear()
                # Yield briefly so the reader task can consume any remaining
                # buffered messages and overwrite latest with the newest one.
                await asyncio.sleep(0.05)
                msg_id, payload = latest[0]
                log.debug("Recv msg_id=0x%08X len=%d", msg_id, len(payload))
                self._dispatch(msg_id, payload)
        except asyncio.CancelledError:
            print("Sink task cancelled.", flush=True)
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
            writer.close()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _make_image_handler(show: bool):
    cv2_window = None

    def handler(img: GCamImage):
        nonlocal cv2_window
        print(f"  Image: {img.width}x{img.height} {img.comp_layout.name} "
              f"frame={img.frame_idx} ts={img.timestamp_utc} "
              f"pixels={img.pixels.shape} dtype={img.pixels.dtype}", flush=True)

        if not show:
            return

        import cv2
        display = img.pixels
        if img.comp_layout == ComponentLayout.RGB:
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        elif img.comp_layout == ComponentLayout.RGBA:
            display = cv2.cvtColor(display, cv2.COLOR_RGBA2BGRA)
        elif img.comp_layout == ComponentLayout.BGRA:
            display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
        if img.origin_corner == OriginCorner.LOWER_LEFT:
            display = cv2.flip(display, 0)

        if cv2_window is None:
            cv2_window = "AftrBurner Stream"
            cv2.namedWindow(cv2_window, cv2.WINDOW_NORMAL)

        cv2.imshow(cv2_window, display)
        cv2.waitKey(1)

    return handler


def _make_pair_handler(show: bool):
    cv2_window = None

    def handler(pair: ImagePair):
        nonlocal cv2_window
        a, b = pair.imgA, pair.imgB
        print(f"  ImagePair: A={a.width}x{a.height} {a.comp_layout.name} frame={a.frame_idx} "
              f"| B={b.width}x{b.height} {b.comp_layout.name} frame={b.frame_idx}", flush=True)

        if not show:
            return

        import cv2

        def to_display(img):
            d = img.pixels
            if img.comp_layout == ComponentLayout.RGB:
                d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)
            elif img.comp_layout == ComponentLayout.RGBA:
                d = cv2.cvtColor(d, cv2.COLOR_RGBA2BGRA)
            elif img.comp_layout == ComponentLayout.BGRA:
                d = cv2.cvtColor(d, cv2.COLOR_BGRA2BGR)
            elif img.comp_layout == ComponentLayout.GRAY:
                d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
            if img.origin_corner == OriginCorner.LOWER_LEFT:
                d = cv2.flip(d, 0)
            return d

        dispA = to_display(a)
        dispB = to_display(b)

        hA, wA = dispA.shape[:2]
        hB, wB = dispB.shape[:2]
        h = max(hA, hB)
        if hA != h:
            dispA = cv2.resize(dispA, (int(wA * h / hA), h))
        if hB != h:
            dispB = cv2.resize(dispB, (int(wB * h / hB), h))

        canvas = np.hstack([dispA, dispB])
        cv2.putText(canvas, "A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas, "B", (dispA.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cv2_window is None:
            cv2_window = "AftrBurner ImagePair"
            cv2.namedWindow(cv2_window, cv2.WINDOW_NORMAL)

        cv2.imshow(cv2_window, canvas)
        cv2.waitKey(1)

    return handler


def _default_raw_handler(msg_id: int, payload: bytes):
    name = MSG_NAME.get(msg_id, f"Unknown(0x{msg_id:08X})")
    print(f"  NetMsg: {name} (id={msg_id}) payload={len(payload)} bytes", flush=True)

    if msg_id == MSG_ID.get("NetMsgSendString"):
        print(f"    String: '{parse_send_string(payload)}'")
    elif msg_id == MSG_ID.get("NetMsgSendDCM3x3"):
        print(f"    DCM:\n{parse_dcm3x3(payload)}")


def main():
    parser = argparse.ArgumentParser(
        description="AftrBurner TCP Listener — receive NetMsg over TCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["server", "sink"], default="server",
                        help="server: listen for connections. sink: connect to a Source.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="server: bind address. sink: source address. (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=12690,
                        help="TCP port (default: 12690)")
    parser.add_argument("--subscribe-idx", type=int, default=-1,
                        help="Subscription index for sink mode (default: -1)")
    parser.add_argument("--show", action="store_true",
                        help="Display received images in an OpenCV window")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (hex dumps, per-message logs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    listener = AftrTcpListener()
    listener.on_image(_make_image_handler(args.show))
    listener.on_image_pair(_make_pair_handler(args.show))
    listener.on_raw_message(_default_raw_handler)

    _run_async(listener, args)


def _run_async(listener, args):
    """Run the asyncio event loop with proper Ctrl+C handling on Windows."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if args.mode == "server":
        main_coro = listener.run_server(args.host, args.port)
    else:
        main_coro = listener.run_sink(args.host, args.port, args.subscribe_idx)

    main_task = loop.create_task(main_coro)

    # On Windows, asyncio doesn't wake up to handle KeyboardInterrupt while
    # blocked on I/O. This periodic no-op task forces the loop to wake up
    # so Ctrl+C is noticed.
    async def _keepalive():
        while True:
            await asyncio.sleep(0.2)

    keepalive_task = loop.create_task(_keepalive())

    def _shutdown(signum=None, frame=None):
        print("\nCtrl+C received, shutting down...", flush=True)
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
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
