"""HTD g1_inspire_wb action -> the whole-body teleop targets policy_rollout publishes.

One action step (``body_pose``, ``left/right_eef_to_body``, ``left/right_eef_link``,
``velocity_commands``) becomes the ``TeleopDataWB`` fields the robot-side controller
consumes: 4x4 wrist targets in the arm-IK frame, 75-value hand keypoints, torso
attitude and height, and the velocity command. Shared by ``policy_rollout.py`` and
the offline replay tools, so a replayed command stream reaches the controller exactly
as rollout would send it. No DDS, robot or torch dependencies.
"""
import numpy as np

# Wrist target = HTD's eef_to_body + this offset (m), the inverse of the converter's
# "minus (0.15, 0, 0.45)", followed by the rotation remap below.
EEF_POSITION_OFFSET = np.array([0.15, 0.0, 0.45])
# torso_height sent = actions/body[2] + this (m).
TORSO_HEIGHT_OFFSET = 0.75

UNIFIED_TO_UNITREE_EEF_ROT_REMAP = {
    'left': np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
    'right': np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
}

# Inspire hand base frame <-> HTD's unified hand frame, per hand type.
INSPIRE_TO_UNIFIED_EEF = {
    'f1': np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64),
    'dftp': np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64),
}

# Fingertip rows of the 25-keypoint hand skeleton: thumb, index, middle, ring, pinky.
FINGERTIP_KEYPOINTS = (4, 9, 14, 19, 24)


def rpy_to_rot_mat(rpy):
    """ZYX roll/pitch/yaw -> rotation matrix, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    x, y, z = rpy
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    return np.array([
        [cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ])


def rot_mat_to_rpy(R):
    """Rotation matrix -> ZYX roll/pitch/yaw, stable near the pitch singularity."""
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy < 1e-6:
        return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(sy, cy), 0.0])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(sy, cy), np.arctan2(R[1, 0], R[0, 0])])


def eef_link_to_hand_keypoints(eef_link_15d, inspire_to_unified):
    """15 fingertip coordinates (5 x xyz, unified frame) -> 75-value keypoint array
    (25 x xyz, Inspire frame) with only the fingertip rows set."""
    hand = np.zeros((25, 3))
    hand[list(FINGERTIP_KEYPOINTS)] = np.asarray(eef_link_15d, dtype=np.float64).reshape(5, 3)
    return (inspire_to_unified[:3, :3].T @ hand.T).T.flatten()


def wrist_target(eef_to_body, side):
    """HTD eef_to_body pose (xyz + rpy) -> (position, rpy) of the arm-IK wrist target."""
    eef_to_body = np.asarray(eef_to_body, dtype=np.float64)
    pos = eef_to_body[:3] + EEF_POSITION_OFFSET
    rot = rpy_to_rot_mat(eef_to_body[3:6]) @ UNIFIED_TO_UNITREE_EEF_ROT_REMAP[side]
    return pos, rot_mat_to_rpy(rot)


def pose_matrix(pos, rpy):
    """4x4 homogeneous transform from a position and ZYX rpy."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot_mat(rpy)
    T[:3, 3] = pos
    return T


def teleop_targets(cmd, hand_type='dftp', send_velocity=True):
    """One HTD action step -> the TeleopDataWB fields (without the timestamp).

    ``cmd`` maps body_pose, left/right_eef_to_body, left/right_eef_link and, when
    ``send_velocity``, velocity_commands to per-step vectors. Returns a dict with
    left/right_arm_pose (16, row-major 4x4), left/right_hand_pose (75), torso_rpy (3),
    torso_height and velocity_commands ([] when not sent).
    """
    inspire_to_unified = INSPIRE_TO_UNIFIED_EEF[hand_type]
    body = np.asarray(cmd['body_pose'], dtype=np.float64)
    out = {}
    for side in ('left', 'right'):
        pos, rpy = wrist_target(cmd[f'{side}_eef_to_body'], side)
        out[f'{side}_arm_pose'] = pose_matrix(pos, rpy).flatten()
        out[f'{side}_hand_pose'] = eef_link_to_hand_keypoints(cmd[f'{side}_eef_link'], inspire_to_unified)
    out['torso_rpy'] = body[3:6].copy()
    out['torso_height'] = float(body[2] + TORSO_HEIGHT_OFFSET)
    vel = cmd.get('velocity_commands') if send_velocity else None
    out['velocity_commands'] = [] if vel is None else np.asarray(vel, dtype=np.float64)
    return out
