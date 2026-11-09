"""What one TeleopDataWB message from policy_rollout asks of the walker.

policy_rollout publishes HTD's whole-body action as ``TeleopDataWB`` on
``rt/teleop/synchronized_data`` (policy_rollout.py ``_execute_g1_command``):
the wrist targets in the arm-IK frame, the 75-value hand keypoints, the torso
attitude, ``torso_height = actions/body[2] + 0.75`` and, under
``--send_velocity_commands True``, ``velocity_commands = [vx, vy, wz]``. This
module turns that into the three walker commands the teleop's backend sends in
``--lower jetson`` (teleop ``g1_backend.py`` ``_apply`` and ``_tick_walk``):

* **walk** ``(vx, wz)``, clipped to the walker's own command box. The walker
  has no lateral axis; the training data's vy is zero throughout.
* **torso** ``(height, roll, pitch, yaw)``: the height is travel about the
  teleop's stand height, ``torso_height - G1_STAND_HEIGHT`` (the HTD
  converter wrote ``actions/body[2] = 0.78 + travel - 0.75``, so this undoes
  it exactly). The attitude follows ``TORSO_ATTITUDE_MODES``; the default,
  ``yaw``, is what the teleop sent during the recording (README.md, "Torso
  command").

Nothing here touches DDS, the link or the robot: the controller feeds it and
the unit tests drive it directly.
"""
from dataclasses import dataclass

import numpy as np

#: How the HTD torso attitude becomes the walker's (roll, pitch, yaw), named
#: after the teleop's PSI_TORSO_ATTITUDE values it reproduces:
#: ``yaw`` -- roll and pitch commanded zero, the yaw passed on (the teleop
#: under PSI_TORSO_ATTITUDE=yaw, the recording's setting);
#: ``hold`` -- all three zero (PSI_TORSO_ATTITUDE=hold);
#: ``follow`` -- all three passed on (PSI_TORSO_ATTITUDE=1).
TORSO_ATTITUDE_MODES = ("yaw", "hold", "follow")

ARM_POSE_LEN = 16
HAND_POSE_LEN = 75


@dataclass(frozen=True)
class PolicyFrame:
    """One validated TeleopDataWB message."""
    left_pose: np.ndarray            # 4x4, arm-IK (pelvis) frame
    right_pose: np.ndarray
    left_hand: np.ndarray            # 75 = 25 keypoints x 3
    right_hand: np.ndarray
    torso_rpy: np.ndarray            # 3
    torso_height: float              # m, HTD's absolute torso height
    velocity: np.ndarray             # [vx, vy, wz], or None when not sent
    stamp: float                     # the publisher's time.time()


def parse_teleop_data(msg):
    """A :class:`PolicyFrame` from a TeleopDataWB sample, or None for one
    with a wrong field length or a non-finite value -- the same lengths HTD's
    g1_inspire_wb_controller accepts, plus finiteness, since these values go
    to an IK solve and a walking policy."""
    lp, rp = np.asarray(msg.left_arm_pose, float), np.asarray(msg.right_arm_pose, float)
    lh, rh = np.asarray(msg.left_hand_pose, float), np.asarray(msg.right_hand_pose, float)
    rpy = np.asarray(msg.torso_rpy, float)
    vel = np.asarray(msg.velocity_commands, float)
    if (lp.size != ARM_POSE_LEN or rp.size != ARM_POSE_LEN or lh.size != HAND_POSE_LEN
            or rh.size != HAND_POSE_LEN or rpy.size != 3 or vel.size not in (0, 3)):
        return None
    values = np.concatenate([lp, rp, lh, rh, rpy, vel, [msg.torso_height]])
    if not np.all(np.isfinite(values)):
        return None
    return PolicyFrame(lp.reshape(4, 4), rp.reshape(4, 4), lh, rh, rpy,
                       float(msg.torso_height), vel if vel.size == 3 else None,
                       float(msg.timestamp))


def walk_command(frame, vx_range, wz_range):
    """``(vx, wz)`` for the walker, clipped to its box; zero when the
    policy sends no velocity."""
    if frame.velocity is None:
        return 0.0, 0.0
    vx = float(np.clip(frame.velocity[0], *vx_range))
    wz = float(np.clip(frame.velocity[2], *wz_range))
    return vx, wz


def torso_command(frame, stand_height, mode="yaw"):
    """``(height, roll, pitch, yaw)`` for the walker: metres about the
    stand height, radians. The walker clamps each axis to its own box."""
    if mode not in TORSO_ATTITUDE_MODES:
        raise ValueError("torso attitude mode must be one of %s, got %r"
                         % (TORSO_ATTITUDE_MODES, mode))
    height = frame.torso_height - stand_height
    roll, pitch, yaw = (float(v) for v in frame.torso_rpy)
    if mode == "yaw":
        roll = pitch = 0.0
    elif mode == "hold":
        roll = pitch = yaw = 0.0
    return float(height), roll, pitch, yaw


class WalkGate:
    """The teleop backend's walk gate (``_tick_walk`` / ``_rearm_walk_gate``)
    with the policy's velocity in place of the sticks: after anything that
    re-arms it -- the handover, a session boundary, the robot standing
    itself on a silent link -- the walk stays zero until the policy has
    been seen asking for rest, and only then is the walker told to resume.

    "Rest" is ``|vx| <= rest_vx`` and ``|wz| <= rest_wz``: the policy's
    velocity is a regression output, not a centred stick, so it is never
    exactly zero."""

    def __init__(self, rest_vx=0.05, rest_wz=0.05):
        self.rest_vx = float(rest_vx)
        self.rest_wz = float(rest_wz)
        self.pending = True

    def rearm(self):
        self.pending = True

    def step(self, vx, wz, allowed):
        """``(vx, wz, release)``: the command to send this tick, and whether
        the gate opened on it (the caller then calls the walker's
        ``resume()``)."""
        if not allowed:
            return 0.0, 0.0, False
        if self.pending:
            if abs(vx) > self.rest_vx or abs(wz) > self.rest_wz:
                return 0.0, 0.0, False
            self.pending = False
            return vx, wz, True
        return vx, wz, False
