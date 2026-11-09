import math, os, sys
import numpy as np, pytest
from deploy.g1_spec import G1_SPEC
from deploy.msgs import PredictedChunk
from deploy.plot_frames import SERIES, measured_point, predicted_series

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "robot_control"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "policy_rollout"))
from teleop_dds_msgs import EEFState, G1RobotState  # noqa: E402
import g1_command  # noqa: E402

def robot_state(**over):
    z6 = [0.0] * 6
    f = dict(timestamp=0.0, torso_pose=[0, 0, 0.78, 0, 0, 0], left_end_effector_pose=z6, right_end_effector_pose=z6,
             pelvis_pose=z6, left_foot_pose=z6, right_foot_pose=z6,
             left_end_effector_to_torso_pose=[0.3, 0.2, -0.1, 0.1, 0.2, 0.3], right_end_effector_to_torso_pose=z6,
             joint_positions=[0.0] * 29, joint_velocities=[0.0] * 29, velocity_commands=[0.2, 0.0, -0.1],
             imu_quaternion=[1, 0, 0, 0], imu_accelerometer=[0, 0, 9.8], imu_gyroscope=[0, 0, 0], imu_rpy=z6[:3])
    return G1RobotState(**{**f, **over})

def chunk(H=60, **over):
    f = dict(timestamp=0.0, t_obs=0, horizon=H, interval=45, inference_ms=5.0,
             body_pose=[0, 0, 0.03, 0, 0, 0] * H, left_eef_to_body=[0.3, 0.2, -0.1, 0.1, 0.2, 0.3] * H,
             right_eef_to_body=[0.0] * (6 * H), left_eef_link=[0.1, 0, 0] * 5 * H, right_eef_link=[0.0] * (15 * H),
             velocity_commands=[0.2, 0.0, -0.1] * H)
    return PredictedChunk(**{**f, **over})

def test_measured_point_keys_and_values():
    tips = [0.08, 0, 0] * 5 + [0.06, 0, 0] * 5     # right 5 tips at 8 cm, left at 6 cm
    eef = EEFState(timestamp=0.0, joint_positions=[0.0] * 12, finger_forces=[0.0] * 12, hand_link_poses=tips)
    p = measured_point(robot_state(), eef, G1_SPEC.plot)
    assert set(p) == set(SERIES)
    assert (p["left_hand_x"], p["torso_height"], p["vx"], p["wz"]) == (0.3, 0.78, 0.2, -0.1)
    assert p["left_fingers_mm"] == pytest.approx(60.0) and p["right_fingers_mm"] == pytest.approx(80.0)

def test_prediction_is_drawn_as_the_command_rollout_sends():
    rs = robot_state()
    s = predicted_series(chunk(), {"left": rs.left_end_effector_to_torso_pose[3:6], "right": rs.right_end_effector_to_torso_pose[3:6]}, G1_SPEC.plot)
    pos, _ = g1_command.wrist_target([0.3, 0.2, -0.1, 0.1, 0.2, 0.3], "left")
    assert (s["left_hand_x"][0], s["left_hand_y"][0], s["left_hand_z"][0]) == pytest.approx(tuple(pos))
    assert s["torso_height"][0] == pytest.approx(0.03 + g1_command.TORSO_HEIGHT_OFFSET)
    assert s["left_fingers_mm"][0] == pytest.approx(100.0) and len(s["vx"]) == 60 and s["wz"][5] == -0.1

def test_rotation_error_is_zero_for_the_measured_rotation():
    # a chunk whose rotation equals the measured one, once the IK remap is undone
    rs = robot_state()
    R = g1_command.rpy_to_rot_mat([0.1, 0.2, 0.3]) @ g1_command.UNIFIED_TO_UNITREE_EEF_ROT_REMAP["left"].T
    rpy = g1_command.rot_mat_to_rpy(R)
    c = chunk(left_eef_to_body=[0.3, 0.2, -0.1, *rpy] * 60)
    s = predicted_series(c, {"left": [0.1, 0.2, 0.3], "right": [0, 0, 0]}, G1_SPEC.plot)
    assert s["left_rot_deg"][0] == pytest.approx(0.0, abs=1e-6)

def test_bad_chunk_rejected():
    assert predicted_series(chunk(velocity_commands=[0.0] * 10), {"left": [0, 0, 0], "right": [0, 0, 0]}, G1_SPEC.plot) is None
    assert predicted_series(chunk(body_pose=[math.nan] * 360), {"left": [0, 0, 0], "right": [0, 0, 0]}, G1_SPEC.plot) is None