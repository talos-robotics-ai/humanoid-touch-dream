"""Tests for head_camera.py. Run: python test_head_camera.py  (or pytest).

No camera is needed: layouts are checked against the converter's reference
composition, and the ZMQ source against an in-process fake server.
"""
import argparse
import threading

import cv2
import numpy as np

import head_camera as hc

VIEW = (480, 640)


def _frame(h, w, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def converter_main_image(rgb, layout):
    """htd-dataset-converter htd_convert/convert.py main_image(), verbatim logic."""
    if layout == 'stretch':
        return cv2.resize(rgb, (1280, 480), interpolation=cv2.INTER_AREA)
    half = rgb if rgb.shape[:2] == (480, 640) else cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)
    return np.concatenate([half, half], axis=1)


def test_mono_layouts_match_converter():
    for h, w in ((480, 640), (720, 1280), (480, 848)):
        f = _frame(h, w)
        for layout in ('duplicate', 'stretch'):
            got = hc.compose_head_frame(f, layout, VIEW)
            assert got.shape == (480, 1280, 3)
            assert np.array_equal(got, converter_main_image(f, layout)), (layout, h, w)


def test_stereo_layout_matches_upstream_rollout():
    f = _frame(480, 1280)
    expected = cv2.resize(cv2.rotate(f, cv2.ROTATE_180), (1280, 480))  # upstream head_camera_stream_thread
    assert np.array_equal(hc.compose_head_frame(f, 'stereo', VIEW, rotate=180), expected)
    f = _frame(720, 2560)
    expected = cv2.resize(cv2.rotate(f, cv2.ROTATE_180), (1280, 480))
    assert np.array_equal(hc.compose_head_frame(f, 'stereo', VIEW, rotate=180), expected)


def test_rotation_before_layout():
    f = _frame(640, 480)  # portrait-mounted camera
    got = hc.compose_head_frame(f, 'duplicate', VIEW, rotate=90)
    assert np.array_equal(got, converter_main_image(cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE), 'duplicate'))


def test_invalid_options_rejected():
    for kwargs in ({'layout': 'side'}, {'layout': 'duplicate', 'rotate': 45}):
        try:
            hc.compose_head_frame(_frame(480, 640), view_size=VIEW, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f'accepted {kwargs}')


def _args(*argv):
    p = argparse.ArgumentParser()
    hc.add_head_camera_args(p)
    return p.parse_args(list(argv))


def test_defaults_keep_upstream_behaviour():
    a = hc.resolve_head_camera_args(_args(), 'g1_inspire_wb')
    assert (a.head_camera_source, a.head_camera_layout, a.head_camera_rotate) == ('stereo_usb', 'stereo', 180)
    a = hc.resolve_head_camera_args(_args('--head_camera_source', 'zmq'), 'g1_inspire_wb')
    assert (a.head_camera_layout, a.head_camera_rotate) == ('duplicate', 0)
    a = hc.resolve_head_camera_args(_args('--head_camera_source', 'realsense', '--head_camera_layout', 'stretch',
                                          '--head_camera_rotate', '180'), 'g1_inspire_wb')
    assert (a.head_camera_layout, a.head_camera_rotate) == ('stretch', 180)


def test_static_source_has_view_size():
    a = hc.resolve_head_camera_args(_args('--head_camera_source', 'static'), 'g1_inspire_wb')
    assert (a.head_camera_layout, a.head_camera_rotate) == ('duplicate', 0)
    assert hc.make_head_camera(a, VIEW, 30).read().shape == (480, 1280, 3)


class _FakeSource:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return self.frames.pop(0)

    def close(self):
        pass


def test_head_camera_skips_missing_frames():
    f = _frame(480, 640)
    cam = hc.HeadCamera(_FakeSource([None, f]), 'duplicate', VIEW)
    assert cam.read() is None
    out = cam.read()
    assert out.shape == (480, 1280, 3) and np.array_equal(out[:, :640], out[:, 640:])


def test_zmq_source_against_fake_server():
    try:
        import zmq
    except ImportError:
        print('  (skipped: pyzmq not installed)')
        return
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    port = rep.bind_to_random_port('tcp://127.0.0.1')
    bgr = _frame(480, 640, seed=3)
    ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    # Replies like the G1 camera server: one empty "not ready" reply, then [rgb, ir, depth, meta].
    replies = [[b''], [jpg.tobytes(), b'ir', b'depth', b'{}']]

    def serve():
        for parts in replies:
            rep.recv()
            rep.send_multipart(parts)
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    src = hc._ZmqSource(f'tcp://127.0.0.1:{port}', timeout_ms=2000)
    cam = hc.HeadCamera(src, 'duplicate', VIEW)
    assert cam.read() is None                       # empty reply -> no frame
    out = cam.read()
    decoded = cv2.imdecode(np.frombuffer(jpg.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(out, converter_main_image(decoded, 'duplicate'))
    t.join(1)
    rep.close()
    # A server that stops answering: read() times out, returns None and reconnects.
    src.timeout_ms = 200
    src.socket.close()
    src._connect()
    assert src.read() is None
    src.close()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'{len(tests)} tests passed')
