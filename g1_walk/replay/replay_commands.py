#!/usr/bin/env python3
"""Replay a recorded HTD command stream through the controller and log what the robot does.

Run by sim_replay.sh once the sim host, the walker and the controller are up.

The input is an ``<episode>_commands_<schedule>.npz`` written by
``policy_rollout/replay_eval.py --save_commands``: ``cmd_*`` is what the policy's
rollout schedule sends at each 30 Hz step for a recorded episode (NaN before its
first chunk), ``gt_*`` the operator's recorded commands for the same episode.
``--source policy`` replays the first, ``--source recorded`` the second.

Each step is turned into TeleopDataWB by ``policy_rollout/g1_command.teleop_targets``
-- the conversion policy_rollout itself uses -- and published from a process with
policy_rollout's DDS configuration, with velocity commands on (rollout's
``--send_velocity_commands True``). Meanwhile the robot is logged: rt/lowstate
(29 joint positions and the IMU attitude), rt/odommodestate (base position and
attitude), and the controller's rt/robot/state and rt/robot/eef_state.
Writes everything to --out (npz).
"""
import argparse
import os
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.dirname(HERE)
sys.path.insert(0, MOD)
sys.path.insert(0, os.path.join(MOD, "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(MOD), "policy_rollout"))
import paths  # noqa: E402
from sim_drive import BAND_RELEASE_PORT, SETTLE_S, Recorder, policy_rollout_dds  # noqa: E402

COMMAND_KEYS = ("body_pose", "left_eef_to_body_pose", "right_eef_to_body_pose",
                "left_eef_link_pose", "right_eef_link_pose", "velocity_commands")


def load_stream(path, source):
    """{rollout command name: (T, D)} for the chosen source, and the step rate."""
    z = np.load(path)
    prefix = "cmd_" if source == "policy" else "gt_"
    # replay_eval names the heads after the model; g1_command takes rollout's names.
    rename = {"left_eef_to_body_pose": "left_eef_to_body", "right_eef_to_body_pose": "right_eef_to_body",
              "left_eef_link_pose": "left_eef_link", "right_eef_link_pose": "right_eef_link"}
    stream = {rename.get(k, k): z[prefix + k].astype(np.float64) for k in COMMAND_KEYS}
    return stream, float(z["fps"])


def policy_side(conn, stream_path, source):
    """policy_rollout's part, in a process of its own: the stream out, the controller's topics in."""
    policy_rollout_dds("lo")
    paths.add_teleop()
    paths.enter_htd_robot_control()
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from g1_command import teleop_targets
    from teleop_dds_msgs import EEFState, G1RobotState, TeleopDataWB

    stream, fps = load_stream(stream_path, source)
    steps = len(stream["body_pose"])
    rec = Recorder(period=0.0)
    dds = DomainParticipant(0)
    writer = DataWriter(dds, Topic(dds, "rt/teleop/synchronized_data", TeleopDataWB))
    readers = {"robot_state": DataReader(dds, Topic(dds, "rt/robot/state", G1RobotState)),
               "eef_state": DataReader(dds, Topic(dds, "rt/robot/eef_state", EEFState))}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            for s in readers["robot_state"].take(N=64):
                rec.add("robot_state", list(s.torso_pose) + list(s.left_end_effector_to_torso_pose)
                        + list(s.right_end_effector_to_torso_pose) + list(s.joint_positions))
            for s in readers["eef_state"].take(N=64):
                rec.add("eef_state", list(s.joint_positions) + list(s.finger_forces))
            time.sleep(0.005)
    threading.Thread(target=poll, daemon=True).start()

    print("[replay] settling %.0f s on the walker alone" % SETTLE_S, flush=True)
    time.sleep(SETTLE_S)
    print("[replay] %s stream: %d steps at %.0f Hz (%.1f s)" % (source, steps, fps, steps / fps), flush=True)
    period, t_start = 1.0 / fps, time.time()
    sent = 0
    for i in range(steps):
        cmd = {k: v[i] for k, v in stream.items()}
        if all(np.all(np.isfinite(v)) for v in cmd.values()):   # rollout sends nothing before its first chunk
            t = teleop_targets(cmd, "dftp", send_velocity=True)
            writer.write(TeleopDataWB(
                timestamp=time.time(), left_arm_pose=t["left_arm_pose"].tolist(),
                right_arm_pose=t["right_arm_pose"].tolist(), left_hand_pose=t["left_hand_pose"].tolist(),
                right_hand_pose=t["right_hand_pose"].tolist(), torso_rpy=t["torso_rpy"].tolist(),
                torso_height=t["torso_height"], velocity_commands=t["velocity_commands"].tolist()))
            sent += 1
        delay = t_start + (i + 1) * period - time.time()
        if delay > 0:
            time.sleep(delay)
    time.sleep(1.0)
    stop.set()
    print("[replay] sent %d messages in %.1f s" % (sent, time.time() - t_start), flush=True)
    conn.send((t_start, rec.series))


def series_arrays(series):
    out = {}
    for name, rows in series.items():
        if rows:
            out[f"{name}_t"] = np.array([t for t, _ in rows])
            out[name] = np.stack([r for _, r in rows])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commands", required=True, help="replay_eval --save_commands npz")
    ap.add_argument("--source", choices=("policy", "recorded"), default="policy")
    ap.add_argument("--out", required=True, help="output npz")
    args = ap.parse_args(argv)
    import multiprocessing as mp
    paths.add_teleop()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    rec = Recorder(period=0.01)
    ChannelFactoryInitialize(0, "lo")
    lowstate = ChannelSubscriber("rt/lowstate", LowState_)
    odom = ChannelSubscriber("rt/odommodestate", SportModeState_)
    lowstate.Init(lambda m: rec.add("lowstate", [m.motor_state[i].q for i in range(29)]
                                    + list(m.imu_state.rpy)), 10)
    odom.Init(lambda m: rec.add("odom", list(m.position) + list(m.imu_state.rpy)), 10)

    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"{}", ("127.0.0.1", BAND_RELEASE_PORT))
    print("[replay] sim pelvis band release requested", flush=True)
    ctx = mp.get_context("spawn")
    mine, theirs = ctx.Pipe()
    child = ctx.Process(target=policy_side, args=(theirs, args.commands, args.source), name="policy-side")
    child.start()
    t_start, series = mine.recv()
    child.join(timeout=10.0)
    rec.series.update(series)
    stream, fps = load_stream(args.commands, args.source)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, t_start=t_start, fps=fps, source=args.source,
                        commands=os.path.abspath(args.commands),
                        **{f"stream_{k}": v for k, v in stream.items()}, **series_arrays(rec.series))
    print("[replay] wrote %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
