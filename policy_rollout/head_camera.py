"""Head-camera sources for policy rollout.

The MXT head tokenizer consumes one RGB frame laid out as a stereo pair: two
views side by side, ``H x 2W`` (480 x 1280 at 480p). HTD's own G1 has a stereo
USB camera that delivers that pair directly, mounted upside down. Other robots
have a different camera, so rollout separates three choices that the original
code fixed together:

* **source** - where frames come from:
    ``stereo_usb``  a V4L2 stereo camera found by name (HTD's ``3D USB Camera``)
    ``opencv``      any V4L2/OpenCV device (name, /dev path or index), mono or stereo
    ``realsense``   an Intel RealSense colour stream on this PC (pyrealsense2)
    ``zmq``         a RealSense served over ZMQ REQ/REP, e.g. the G1 Jetson camera
                    server (reply part 0 is a JPEG BGR colour frame)
    ``static``      a fixed image file, or a grey test frame (sim tests, no hardware)
* **layout** - how one source frame becomes the ``H x 2W`` pair. It must match
  how the training images were made:
    ``stereo``      the frame already is a side-by-side pair; resize to 2W x H
    ``duplicate``   mono frame, resized to W x H and placed in both halves
                    (htd-dataset-converter ``--main-layout duplicate``)
    ``stretch``     mono frame resized to 2W x H
                    (htd-dataset-converter ``--main-layout stretch``)
* **rotate** - degrees clockwise applied to the raw source frame before the
  layout (0, 90, 180 or 270). HTD's G1 stereo camera needs 180.

``HeadCamera.read()`` returns a BGR ``uint8`` array of shape ``(H, 2W, 3)``, or
``None`` when no new frame is available. Optional dependencies (pyrealsense2,
pyzmq) are imported only by the source that needs them.
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

SOURCES = ('stereo_usb', 'opencv', 'realsense', 'zmq', 'static')
LAYOUTS = ('stereo', 'duplicate', 'stretch')
ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
DEFAULT_STEREO_USB_NAME = '3D USB Camera'
DEFAULT_ZMQ_ADDRESS = 'tcp://192.168.123.164:5556'


def compose_head_frame(frame, layout, view_size, rotate=0):
    """Rotate a raw source frame and lay it out as a side-by-side pair.

    ``view_size`` is (H, W) of ONE view; the result is ``(H, 2W, C)``.
    """
    if layout not in LAYOUTS:
        raise ValueError(f'unknown head camera layout {layout!r}; choose from {LAYOUTS}')
    if rotate not in ROTATIONS:
        raise ValueError(f'head camera rotation must be one of {sorted(ROTATIONS)}, got {rotate!r}')
    if ROTATIONS[rotate] is not None:
        frame = cv2.rotate(frame, ROTATIONS[rotate])
    h, w = view_size
    if layout == 'duplicate':
        half = frame if frame.shape[:2] == (h, w) else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return np.concatenate([half, half], axis=1)
    if frame.shape[:2] == (h, 2 * w):
        return frame
    # 'stretch' matches the converter (INTER_AREA); 'stereo' keeps upstream rollout's default interpolation.
    interpolation = cv2.INTER_AREA if layout == 'stretch' else cv2.INTER_LINEAR
    return cv2.resize(frame, (2 * w, h), interpolation=interpolation)


class _OpenCVSource:
    """A V4L2/OpenCV capture device (stereo_usb is this with a name lookup)."""

    def __init__(self, device, width, height, fps, mjpg=True):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open head camera device {device!r}')
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class _RealSenseSource:
    """RealSense colour stream on this PC."""

    def __init__(self, width, height, fps, serial=None):
        import pyrealsense2 as rs  # optional dependency

        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy() if color else None

    def close(self):
        self.pipeline.stop()


class _ZmqSource:
    """REQ/REP client for a camera server whose reply part 0 is a JPEG BGR frame.

    An empty reply means the server has no fresh frame; ``read`` then returns
    None. A timed-out REQ socket cannot send again, so it is rebuilt.
    """

    def __init__(self, address, timeout_ms=1000):
        import zmq  # optional dependency

        self._zmq = zmq
        self.address = address
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self._connect()

    def _connect(self):
        zmq = self._zmq
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self.address)

    def read(self):
        try:
            self.socket.send(b'x')
            parts = self.socket.recv_multipart()
        except self._zmq.Again:
            self.socket.close()
            self._connect()
            return None
        if not parts or not parts[0]:
            return None
        return cv2.imdecode(np.frombuffer(parts[0], np.uint8), cv2.IMREAD_COLOR)

    def close(self):
        self.socket.close()


class _StaticSource:
    """The same frame on every read (no hardware)."""

    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame.copy()

    def close(self):
        pass


def _static_frame(device, width, height):
    """``device`` as an image file resized to width x height, else mid-grey with a white 20-px border."""
    frame = cv2.imread(device) if isinstance(device, str) and os.path.isfile(device) else None
    if frame is not None:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    if device is not None:
        print(f'Warning: cannot read head camera image {device!r}; using a grey frame')
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    frame[20:-20, 20:-20] = 128
    return frame


class HeadCamera:
    """A head-camera source plus the rotation and layout that match training."""

    def __init__(self, source, layout, view_size, rotate=0, stale_warn_s=1.0):
        if rotate not in ROTATIONS:
            raise ValueError(f'head camera rotation must be one of {sorted(ROTATIONS)}, got {rotate!r}')
        if layout not in LAYOUTS:
            raise ValueError(f'unknown head camera layout {layout!r}; choose from {LAYOUTS}')
        self.source = source
        self.layout = layout
        self.view_size = tuple(view_size)
        self.rotate = rotate
        self.stale_warn_s = stale_warn_s
        self.last_frame_time = None
        self._last_warn = 0.0

    def read(self):
        frame = self.source.read()
        now = time.time()
        if frame is None:
            since = now - (self.last_frame_time or now)
            if since > self.stale_warn_s and now - self._last_warn > 5.0:
                print(f'Warning: no new head camera frame for {since:.1f}s')
                self._last_warn = now
            return None
        self.last_frame_time = now
        return compose_head_frame(frame, self.layout, self.view_size, self.rotate)

    def close(self):
        self.source.close()


def add_head_camera_args(parser):
    """Register the head-camera options on an argparse parser."""
    group = parser.add_argument_group('head camera')
    group.add_argument('--head_camera_source', choices=SOURCES, default='stereo_usb',
                       help='where head frames come from (default: HTD stereo USB camera)')
    group.add_argument('--head_camera_device', type=str, default=None,
                       help='stereo_usb/opencv: device name substring, /dev path or index; '
                            f'realsense: serial number; zmq: address (default {DEFAULT_ZMQ_ADDRESS}); '
                            'static: image file')
    group.add_argument('--head_camera_layout', choices=LAYOUTS, default=None,
                       help='how a source frame becomes the stereo pair; must match training data '
                            '(default: stereo for stereo_usb, duplicate otherwise)')
    group.add_argument('--head_camera_rotate', type=int, choices=sorted(ROTATIONS), default=None,
                       help='clockwise rotation of the raw frame (default: 180 for stereo_usb on G1, else 0)')
    group.add_argument('--head_camera_source_res', type=str, default=None, metavar='WxH',
                       help='requested source resolution for opencv/realsense, e.g. 640x480 '
                            '(default: one view for mono layouts, the full pair for stereo)')


def resolve_head_camera_args(args, embodiment=''):
    """Fill source-dependent defaults in place; returns args."""
    if args.head_camera_layout is None:
        args.head_camera_layout = 'stereo' if args.head_camera_source == 'stereo_usb' else 'duplicate'
    if args.head_camera_rotate is None:
        args.head_camera_rotate = 180 if (args.head_camera_source == 'stereo_usb' and 'g1' in embodiment) else 0
    return args


def _parse_res(text):
    try:
        w, h = (int(v) for v in text.lower().split('x'))
    except ValueError:
        raise ValueError(f'--head_camera_source_res must look like 640x480, got {text!r}') from None
    return w, h


def make_head_camera(args, view_size, fps, embodiment='', device_lookup=None):
    """Build the HeadCamera selected by the ``add_head_camera_args`` options.

    ``view_size`` is (H, W) of one view. ``device_lookup(name)`` resolves a V4L2
    device name to a path (rollout passes data_collect's v4l2 helper).
    """
    resolve_head_camera_args(args, embodiment)
    h, w = view_size
    if args.head_camera_source_res:
        src_w, src_h = _parse_res(args.head_camera_source_res)
    elif args.head_camera_layout == 'stereo':
        src_w, src_h = 2 * w, h
    else:
        src_w, src_h = w, h

    source_name = args.head_camera_source
    device = args.head_camera_device
    if source_name in ('stereo_usb', 'opencv'):
        if source_name == 'stereo_usb' and device is None:
            device = DEFAULT_STEREO_USB_NAME
        if device is None:
            device = 0
        elif isinstance(device, str) and device.isdigit():
            device = int(device)
        elif isinstance(device, str) and not device.startswith('/dev/') and device_lookup is not None:
            path = device_lookup(device)
            if path is None:
                raise RuntimeError(f'no video device named {device!r}')
            device = path
        print(f'Head camera: {source_name} device {device!r}')
        source = _OpenCVSource(device, src_w, src_h, fps)
    elif source_name == 'realsense':
        print(f'Head camera: RealSense {device or "(first found)"} {src_w}x{src_h}@{fps}')
        source = _RealSenseSource(src_w, src_h, fps, serial=device)
    elif source_name == 'zmq':
        address = device or DEFAULT_ZMQ_ADDRESS
        print(f'Head camera: ZMQ server {address}')
        source = _ZmqSource(address)
    elif source_name == 'static':
        print(f'Head camera: static frame {src_w}x{src_h}')
        source = _StaticSource(_static_frame(device, src_w, src_h))
    else:
        raise ValueError(f'unknown head camera source {source_name!r}')
    print(f'Head camera layout={args.head_camera_layout} rotate={args.head_camera_rotate} '
          f'-> {h}x{2 * w} pair')
    return HeadCamera(source, args.head_camera_layout, view_size, args.head_camera_rotate)
