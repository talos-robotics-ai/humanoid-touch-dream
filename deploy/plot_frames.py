"""Measured state and predicted chunk as the UI plots them, in the frame of the command rollout sends."""
import math
import os
import sys

import numpy as np

_POLICY_ROLLOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policy_rollout")
if _POLICY_ROLLOUT not in sys.path:
    sys.path.append(_POLICY_ROLLOUT)
from g1_command import UNIFIED_TO_UNITREE_EEF_ROT_REMAP, rpy_to_rot_mat  # noqa: E402

SERIES = ("left_hand_x", "left_hand_y", "left_hand_z", "right_hand_x", "right_hand_y", "right_hand_z",
          "left_rot_deg", "right_rot_deg", "left_fingers_mm", "right_fingers_mm", "torso_height", "vx", "wz")

# PredictedChunk field -> values per step.
_CHUNK_WIDTHS = (("body_pose", 6), ("left_eef_to_body", 6), ("right_eef_to_body", 6),
                 ("left_eef_link", 15), ("right_eef_link", 15), ("velocity_commands", 3))


def _vec(seq, n):
    """First n values as floats; NaN where the sequence is short."""
    a = np.full(n, math.nan)
    if seq is not None:
        v = np.asarray(seq, dtype=np.float64).reshape(-1)[:n]
        a[:v.size] = v
    return a


def _fingers_mm(tips):
    """Mean distance of the fingertips (..., 5, 3) from the hand origin, mm."""
    return np.mean(np.linalg.norm(tips, axis=-1), axis=-1) * 1000.0


def measured_point(robot_state, eef_state, spec) -> dict:
    """One measured sample of every series; `eef_state` may be None (fingers NaN)."""
    left = _vec(robot_state.left_end_effector_to_torso_pose, 3)
    right = _vec(robot_state.right_end_effector_to_torso_pose, 3)
    torso = _vec(robot_state.torso_pose, 3)
    vel = _vec(robot_state.velocity_commands, 3)
    tips = _vec(eef_state.hand_link_poses if eef_state is not None else None, 30).reshape(2, 5, 3)
    p = {f"{side}_hand_{ax}": float(v[i]) for side, v in (("left", left), ("right", right))
         for i, ax in enumerate("xyz")}
    p.update(left_rot_deg=0.0, right_rot_deg=0.0,
             right_fingers_mm=float(_fingers_mm(tips[0])), left_fingers_mm=float(_fingers_mm(tips[1])),
             torso_height=float(torso[2]), vx=float(vel[0]), wz=float(vel[2]))
    return p


def _rotations(rpy):
    """(H, 3) ZYX rpy -> (H, 3, 3), with g1_command's convention."""
    return np.moveaxis(rpy_to_rot_mat((rpy[:, 0], rpy[:, 1], rpy[:, 2])), -1, 0)


def _angle_deg(R):
    """Rotation angle of each (H, 3, 3) matrix, deg; atan2 keeps it exact near zero."""
    cos = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    skew = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=-1)
    return np.degrees(np.arctan2(np.linalg.norm(skew, axis=-1) / 2.0, cos))


def predicted_series(chunk, measured_rpy: dict, spec) -> dict | None:
    """Every series over the chunk's horizon, or None for a malformed chunk."""
    H = int(chunk.horizon)
    if H <= 0:
        return None
    a = {}
    for key, width in _CHUNK_WIDTHS:
        v = np.asarray(getattr(chunk, key), dtype=np.float64).reshape(-1)
        if v.size != H * width or not np.all(np.isfinite(v)):
            return None
        a[key] = v.reshape(H, width)
    out = {}
    for side in ("left", "right"):
        pose = a[f"{side}_eef_to_body"]
        pos = pose[:, 0:3] + np.asarray(spec.eef_offset, dtype=np.float64)
        for i, ax in enumerate("xyz"):
            out[f"{side}_hand_{ax}"] = pos[:, i].tolist()
        R_pred = _rotations(pose[:, 3:6]) @ UNIFIED_TO_UNITREE_EEF_ROT_REMAP[side]
        R_meas = rpy_to_rot_mat(_vec(measured_rpy.get(side), 3))
        out[f"{side}_rot_deg"] = _angle_deg(R_meas.T @ R_pred).tolist()
        out[f"{side}_fingers_mm"] = _fingers_mm(a[f"{side}_eef_link"].reshape(H, 5, 3)).tolist()
    out["torso_height"] = (a["body_pose"][:, 2] + spec.torso_height_offset).tolist()
    out["vx"] = a["velocity_commands"][:, 0].tolist()
    out["wz"] = a["velocity_commands"][:, 2].tolist()
    return out
