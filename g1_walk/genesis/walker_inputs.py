#!/usr/bin/env python3
"""Turn an HTD command stream into the per-tick commands a lower-body controller receives.

    python g1_walk/genesis/walker_inputs.py --commands EP_commands_SCHEDULE.npz \
        --source policy|recorded --out EP_<source>_inputs.npz

The input is a ``policy_rollout/replay_eval.py --save_commands`` stream (30 Hz).
Each step goes through ``policy_rollout/g1_command.teleop_targets`` -- the
TeleopDataWB policy_rollout publishes -- and then through the same code the two
controllers run on it, at their 50 Hz control rate:

* **g1_walk** (``controller.py``): HTD's arm IK once per policy frame, the arm
  shaper (engage glide, 4 rad/s cap, jump guard), the walk gate and clip, the
  torso mapping (``--torso_attitude yaw``) and its engage glide. Written as
  ``walk_req`` (vx, wz), ``torso_req`` (height about the stand height, roll,
  pitch, yaw) and ``arm_wire`` (14 joint targets): what controller.py hands
  the walker's link, and ``hand_q01``, the Inspire commands its hand loop
  sends (HTD's DexPilot retargeting, registers / 1000, 1 = open; left then
  right, register order). The walker's own command stage (box clamps, slews) is
  applied by the simulator driver, which has the walker's run config.
* **HTD's WBC** (``robot_control/g1_inspire_wb_controller.py``): the frame's
  values as that controller passes them on -- ``vel_raw`` (vx, vy, wz),
  ``torso_raw`` (absolute height, roll, pitch, yaw) -- and ``arm_ik``, the raw
  IK solution it commands the arms with.

Before the first finite policy step (the rollout sends nothing before its
first chunk) the arms are at the teleop's home pose and every command is
neutral; ``session`` marks the ticks a policy frame drives. The IK is warm
started from its own previous answer, standing in for the measured arm.
Runs where the controller runs (needs the teleop pin, pinocchio and casadi).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
sys.path.insert(0, MOD)
sys.path.insert(0, os.path.join(os.path.dirname(MOD), "policy_rollout"))
os.environ.setdefault("PSI_DDS_IFACE", "lo")   # the teleop's modules read it at import
import paths  # noqa: E402
from commands import WalkGate, parse_teleop_data, torso_command, walk_command  # noqa: E402
from shaping import ArmShaper, TorsoEngage  # noqa: E402

CONTROL_HZ = 50.0
# The walker's deployed walk box (g1_lower: the deploy class's vx_range / wz_range);
# controller.py clips to it before the walker clamps again.
G1_WALK_VX_RANGE = (-0.4, 0.8)
G1_WALK_WZ_RANGE = (-1.0, 1.0)


class _Msg:
    """A TeleopDataWB-shaped record for ``parse_teleop_data``."""

    def __init__(self, targets, timestamp):
        self.left_arm_pose = targets["left_arm_pose"]
        self.right_arm_pose = targets["right_arm_pose"]
        self.left_hand_pose = targets["left_hand_pose"]
        self.right_hand_pose = targets["right_hand_pose"]
        self.torso_rpy = targets["torso_rpy"]
        self.torso_height = targets["torso_height"]
        self.velocity_commands = targets["velocity_commands"]
        self.timestamp = timestamp


def load_stream(path, source):
    """``({rollout command name: (T, D)}, fps)`` for the chosen source."""
    z = np.load(path)
    prefix = "cmd_" if source == "policy" else "gt_"
    rename = {"left_eef_to_body_pose": "left_eef_to_body", "right_eef_to_body_pose": "right_eef_to_body",
              "left_eef_link_pose": "left_eef_link", "right_eef_link_pose": "right_eef_link"}
    keys = ("body_pose", "left_eef_to_body_pose", "right_eef_to_body_pose",
            "left_eef_link_pose", "right_eef_link_pose", "velocity_commands")
    return {rename.get(k, k): z[prefix + k].astype(np.float64) for k in keys}, float(z["fps"])


def build_controller_parts(dt):
    """The IK, arm shaper, torso engage and walk gate, built as controller.py builds them."""
    paths.add_teleop()
    import pinocchio as pin
    from teleop.robot_control.drive_safety import JointJumpGuard, arm_jump_thresholds
    from teleop.robot_control.g1_backend import ARM_HOLD_MAX, G1_STAND_HEIGHT, JUMP_CREEP_RAD
    from teleop.utils.psi_env import psi_float
    from controller import load_home_pose
    cwd = os.getcwd()
    paths.enter_htd_robot_control()        # robot_state and the IK import HTD's modules by bare name
    from robot_arm_ik_g1 import G1_29_ArmIK
    from robot_state import G1_URDFS
    from hands import REGISTER_MAX, HtdInspireModel, to_registers
    ik = G1_29_ArmIK(urdf_path="./assets/g1/" + G1_URDFS[0], urdf_dir="./assets/g1/")
    hand_model = HtdInspireModel()

    def hands(left75, right75):
        """InspireHands.command's registers, as fractions (1 = open): left 6 then right 6,
        Inspire register order; None while either hand's keypoints are all zero."""
        q = hand_model.retarget(left75, right75)
        if q is None:
            return None
        return np.array(to_registers(q[0]) + to_registers(q[1]), float) / REGISTER_MAX
    os.chdir(cwd)
    model = ik.reduced_robot.model
    data = model.createData()
    zeros = np.zeros(model.nv)

    def gravity(q):
        return np.asarray(pin.rnea(model, data, np.asarray(q, float), zeros, zeros)).copy()
    guard = JointJumpGuard(arm_jump_thresholds(psi_float("PSI_G1_ARM_JUMP_SCALE", "1.0")),
                           "g1_walk", "arms", max_hold_steps=ARM_HOLD_MAX)
    engage_s = psi_float("PSI_ENGAGE_S", "1.0")
    arms = ArmShaper(guard, gravity, JUMP_CREEP_RAD, dt, 4.0, engage_s,
                     psi_float("PSI_ENGAGE_VEL", "1.0"), psi_float("PSI_ENGAGE_MAX_S", "4.0"))
    return ik, hands, arms, TorsoEngage(engage_s), WalkGate(), load_home_pose(), G1_STAND_HEIGHT


def walker_inputs(stream, fps, settle_s):
    from g1_command import teleop_targets
    dt = 1.0 / CONTROL_HZ
    ik, hands, arms, torso, gate, home, stand_height = build_controller_parts(dt)
    steps = len(stream["body_pose"])
    n_settle = int(round(settle_s * CONTROL_HZ))
    n = n_settle + int(np.ceil(steps / fps * CONTROL_HZ))
    out = {k: np.zeros((n, d)) for k, d in (("walk_req", 2), ("torso_req", 4), ("arm_wire", 14),
                                             ("vel_raw", 3), ("torso_raw", 4), ("arm_ik", 14))}
    out["session"] = np.zeros(n, bool)
    out["hand_q01"] = np.ones((n, 12))         # the hands boot open
    out["torso_raw"][:, 0] = 0.75              # HTD's controller's neutral height
    out["arm_wire"][:] = home
    out["arm_ik"][:] = home
    q_wire, q_ik, frame, solved_i, session = home.copy(), home.copy(), None, -1, False
    hand = np.ones(12)
    for k in range(n):
        now = k * dt
        i = int(np.floor((k - n_settle) * dt * fps + 1e-9)) if k >= n_settle else -1
        if 0 <= i < steps and i != solved_i:
            cmd = {key: v[i] for key, v in stream.items()}
            if all(np.all(np.isfinite(v)) for v in cmd.values()):
                f = parse_teleop_data(_Msg(teleop_targets(cmd, "dftp", send_velocity=True), now))
                if f is not None:
                    q_prev = q_ik
                    q_ik, _ = ik.solve_ik(f.left_pose, f.right_pose, q_ik, (q_ik - q_prev) / dt)
                    q_ik = np.asarray(q_ik, float)
                    frame, solved_i = f, i
                    h = hands(f.left_hand, f.right_hand)
                    if h is not None:              # else the hands keep their last command
                        hand = h
        if frame is not None and not session:      # controller._start_session
            arms.begin(q_wire)
            torso.begin((0.0, 0.0, 0.0, 0.0), now)
            gate.rearm()
            session = True
        if session:
            shaped = arms.shape(q_ik, now)
            if shaped is not None:
                q_wire = shaped[0]
            vx, wz = walk_command(frame, G1_WALK_VX_RANGE, G1_WALK_WZ_RANGE)
            vx, wz, _ = gate.step(vx, wz, allowed=True)
            out["walk_req"][k] = (vx, wz)
            out["torso_req"][k] = torso.shape(torso_command(frame, stand_height, "yaw"), now)
            out["vel_raw"][k] = frame.velocity if frame.velocity is not None else 0.0
            out["torso_raw"][k] = (frame.torso_height, *frame.torso_rpy)
            out["arm_ik"][k] = q_ik
            out["session"][k] = True
            out["hand_q01"][k] = hand
        out["arm_wire"][k] = q_wire
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commands", required=True, help="replay_eval --save_commands npz")
    ap.add_argument("--source", choices=("policy", "recorded"), default="policy")
    ap.add_argument("--settle_s", type=float, default=3.0,
                    help="seconds of standing, arms at home, before the stream starts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    commands = os.path.abspath(args.commands)
    out_path = os.path.abspath(args.out)
    stream, fps = load_stream(commands, args.source)
    out = walker_inputs(stream, fps, args.settle_s)
    np.savez_compressed(out_path, control_hz=CONTROL_HZ, settle_s=args.settle_s, source=args.source,
                        commands=commands, episode=os.path.basename(commands).split("_commands_")[0], **out)
    s = out["session"]
    print("[walker_inputs] %s %s: %d ticks (%.1f s), session from %.1f s; walk |vx| max %.2f, "
          "|wz| max %.2f -> %s" % (os.path.basename(commands), args.source, len(s), len(s) / CONTROL_HZ,
                                   np.argmax(s) / CONTROL_HZ, np.abs(out["walk_req"][:, 0]).max(),
                                   np.abs(out["walk_req"][:, 1]).max(), out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
