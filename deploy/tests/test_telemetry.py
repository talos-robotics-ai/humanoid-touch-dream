import numpy as np, pytest, cv2
from deploy.telemetry import camera_message, chunk_message, is_ready

def traj(H=60):
    return {'body_pose': np.zeros((H, 6)), 'left_eef_to_body': np.ones((H, 6)), 'right_eef_to_body': np.zeros((H, 6)),
            'left_eef_link': np.zeros((H, 15)), 'right_eef_link': np.zeros((H, 15)), 'velocity_commands': np.full((H, 3), 0.25)}

def test_chunk_message_flattens_row_major():
    m = chunk_message(traj(), t_obs=90, interval=45, inference_ms=6.6, stamp=1.0)
    assert (m.horizon, m.interval, m.t_obs) == (60, 45, 90)
    assert len(m.left_eef_link) == 900 and m.velocity_commands[:3] == [0.25, 0.25, 0.25]
    assert np.array(m.left_eef_to_body).reshape(60, 6).tolist() == np.ones((60, 6)).tolist()

def test_chunk_message_rejects_bad_shapes():
    bad = traj(); bad['velocity_commands'] = np.zeros((60, 2))
    with pytest.raises(ValueError):
        chunk_message(bad, 0, 45, 1.0, 0.0)
    with pytest.raises(ValueError):
        chunk_message({k: v for k, v in traj().items() if k != 'body_pose'}, 0, 45, 1.0, 0.0)

def test_camera_message_downscales_and_encodes_jpeg():
    rgb = np.zeros((480, 1280, 3), dtype=np.uint8); rgb[..., 0] = 200
    m = camera_message(rgb, stamp=2.0)
    assert (m.width, m.height) == (640, 240)
    img = cv2.imdecode(np.frombuffer(bytes(m.jpeg), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 640, 3) and img[0, 0, 2] > 150           # red survived (BGR order)

def test_is_ready_names_the_first_missing_condition():
    assert is_ready(False, (0.1, 0.1, 0.1), 5.0, 750.0) == (False, "warming up")
    assert is_ready(True, (2.0, 0.1, 0.1), 5.0, 750.0) == (False, "no robot state")
    assert is_ready(True, (0.1, 2.0, 0.1), 5.0, 750.0) == (False, "no hand state")
    assert is_ready(True, (0.1, 0.1, 2.0), 5.0, 750.0) == (False, "no camera frame")
    assert is_ready(True, (0.1, 0.1, 0.1), 900.0, 750.0) == (False, "inference slow")
    assert is_ready(True, (0.1, 0.1, 0.1), None, 750.0) == (True, "ready")
