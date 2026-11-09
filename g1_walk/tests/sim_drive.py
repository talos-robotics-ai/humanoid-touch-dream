#!/usr/bin/env python3
"""Drive the sim end to end in policy_rollout's place, and score it.

Run by sim_e2e.sh once the sim host, the walker and the controller are up.
It plays a scripted choreography as TeleopDataWB (what policy_rollout
publishes), from a process whose DDS is configured as policy_rollout configures
it for ``--dds_interface lo``, and listens to the robot meanwhile:

    settle   band released, nothing published         (the walker alone)
    hold     the arms at home, hands open                3 s
    arms     both arms and fingers move, torso yaw       8 s
    walk     arms at home, vx 0.3 m/s                    8 s
    turn     wz 0.5 rad/s                                6 s
    silence  nothing published, mid-turn                 3 s  (the session ends, the walk stops)
    rest     zero velocity                               3 s  (a new session)
    reset    a ResetCommand, then nothing                5 s  (arms glide home)

The arm targets are HTD's arm-IK forward kinematics of a joint trajectory
around the teleop's home pose, i.e. exactly what policy_rollout sends for a
policy that commands those joints; the fingertip keypoints are the Inspire
hand's own fingertips at the commanded finger angles divided by DexPilot's
scaling, which is how the HTD converter synthesised them for training.

Pass criteria (each reported with its number): the robot stays up (pelvis
height, tilt), the commanded arms follow the trajectory and the measured arms
follow the command, the emulated fingers close and open as commanded (read
back through eef_state), the walk and the turn move the base, exactly one
rt/lowcmd writer (the walker) and no rt/arm_sdk writer, rt/robot/state,
eef_state and the touch topics published, the session ends on silence.
Writes summary.json to --log; exit status 1 when a criterion fails.
"""
import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402

PUBLISH_HZ = 30.0          # policy_rollout's default --control_freq
BAND_RELEASE_PORT = 14651  # the G1 sim host's target-viz port; any datagram starts the session
SETTLE_S = 14.0            # the band ramps off over PSI_SIM_HOLD_RELEASE_S (10 s)
PHASES = (("hold", 3.0), ("arms", 8.0), ("walk", 8.0), ("turn", 6.0), ("silence", 3.0),
          ("rest", 3.0), ("reset", 5.0))
DEXPILOT_SCALING = 1.20    # robot_control/assets/inspire_hand/inspire_hand_dftp.yml
HAND_OPEN = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])        # retargeting order
HAND_CLOSED = np.array([0.5, 0.25, 1.0, 1.0, 1.0, 1.0])
TIP_ROWS = (4, 9, 14, 19, 24)
ARM_AMPL = np.array([-0.25, 0.0, 0.0, 0.3, 0.2, 0.0, 0.0])  # shoulder pitch, elbow, wrist roll
ARM_PERIOD_S = 4.0


class Recorder:
    """Latest-sample buffers, appended at most every ``period`` seconds."""

    def __init__(self, period=0.01):
        self.period = period
        self.series = {}
        self._last = {}
        self.lock = threading.Lock()

    def add(self, name, row):
        now = time.time()
        with self.lock:
            if now - self._last.get(name, 0.0) < self.period:
                return
            self._last[name] = now
            self.series.setdefault(name, []).append((now, np.asarray(row, float)))

    def window(self, name, t0, t1):
        with self.lock:
            rows = [(t, r) for t, r in self.series.get(name, ()) if t0 <= t <= t1]
        if not rows:
            return np.zeros(0), np.zeros((0, 0))
        return np.array([t for t, _ in rows]), np.stack([r for _, r in rows])


def arm_trajectory(home, t):
    """14 arm joints at ``t`` seconds into the arm phase: both arms swing
    the same way, mirrored on the wrist roll."""
    s = np.sin(2 * np.pi * t / ARM_PERIOD_S)
    left = home[:7] + s * ARM_AMPL
    right = home[7:] + s * ARM_AMPL * np.array([1, 1, 1, 1, -1, 1, 1])
    return np.concatenate([left, right])


class Choreography:
    def __init__(self, home):
        import pinocchio as pin
        from robot_arm_ik_g1 import G1_29_ArmIK
        from hands import HtdInspireModel
        self.pin = pin
        self.home = home
        ik = G1_29_ArmIK(urdf_path="./assets/g1/g1_body29_hand14_wrist4010.urdf",
                         urdf_dir="./assets/g1/")
        self.model = ik.reduced_robot.model
        self.data = self.model.createData()
        self.frames = (self.model.getFrameId("L_ee"), self.model.getFrameId("R_ee"))
        self.hands = HtdInspireModel()

    def wrists(self, q):
        self.pin.framesForwardKinematics(self.model, self.data, q)
        return [self.data.oMf[f].homogeneous.copy() for f in self.frames]

    def keypoints(self, side, q_hand):
        kp = np.zeros((25, 3))
        kp[list(TIP_ROWS)] = self.hands.fingertips(side, q_hand) / DEXPILOT_SCALING
        return kp.ravel()

    def at(self, phase, t):
        """(arm joints, hand closure 0..1, velocity, torso yaw)."""
        if phase == "arms":
            closure = 0.5 - 0.5 * np.cos(2 * np.pi * t / ARM_PERIOD_S)
            return arm_trajectory(self.home, t), closure, (0.0, 0.0, 0.0), \
                0.3 * np.sin(np.pi * t / 8.0)
        vel = {"walk": (0.3, 0.0, 0.0), "turn": (0.0, 0.0, 0.5)}.get(phase, (0.0, 0.0, 0.0))
        return self.home.copy(), 0.0, vel, 0.0

    def message(self, phase, t, TeleopDataWB):
        q, closure, vel, yaw = self.at(phase, t)
        lw, rw = self.wrists(q)
        q_hand = HAND_OPEN + closure * (HAND_CLOSED - HAND_OPEN)
        return TeleopDataWB(
            timestamp=time.time(), left_arm_pose=lw.ravel().tolist(),
            right_arm_pose=rw.ravel().tolist(),
            left_hand_pose=self.keypoints("left", q_hand).tolist(),
            right_hand_pose=self.keypoints("right", q_hand).tolist(),
            torso_rpy=[0.0, 0.0, float(yaw)], torso_height=0.03 + 0.75,
            velocity_commands=list(vel))


def count_writers(participant, topic):
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    reader = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    seen = {}
    t_end = time.time() + 2.0
    while time.time() < t_end:
        for s in reader.take(N=256):
            seen[s.key] = s.topic_name
        time.sleep(0.05)
    return sum(1 for name in seen.values() if name == topic)


def policy_rollout_dds(interface):
    """The DDS configuration policy_rollout makes for ``--dds_interface``
    (policy_rollout.py ``_setup_dds_interface``), so this side of the test
    discovers the controller the way the real rollout does."""
    import tempfile
    xml = ('<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS xmlns="https://cdds.io/config">'
           '<Domain><General><Interfaces><NetworkInterface name="%s"/></Interfaces>'
           '</General></Domain></CycloneDDS>' % interface)
    fd, path = tempfile.mkstemp(suffix=".xml", prefix="cyclonedds_")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    os.environ["CYCLONEDDS_URI"] = "file://" + path


def policy_side(conn):
    """policy_rollout's part, in a process of its own (its DDS configuration
    differs from the Unitree SDK's): the choreography out, the controller's
    topics in. Sends (phase bounds, recorded series)."""
    policy_rollout_dds("lo")
    paths.add_teleop()
    paths.enter_htd_robot_control()
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from inspire_sdkpy import inspire_dds
    from teleop_dds_msgs import EEFState, G1RobotState, ResetCommand, TeleopDataWB
    from controller import load_home_pose

    chor = Choreography(load_home_pose())
    rec = Recorder(period=0.0)          # every message: their rates are scored
    dds = DomainParticipant(0)
    writer = DataWriter(dds, Topic(dds, "rt/teleop/synchronized_data", TeleopDataWB))
    reset_writer = DataWriter(dds, Topic(dds, "rt/robot/reset_command", ResetCommand))
    readers = {
        "robot_state": DataReader(dds, Topic(dds, "rt/robot/state", G1RobotState)),
        "eef_state": DataReader(dds, Topic(dds, "rt/robot/eef_state", EEFState)),
        "touch_l": DataReader(dds, Topic(dds, "rt/inspire_hand/touch/l",
                                         inspire_dds.inspire_hand_touch)),
    }
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            for name, r in readers.items():
                for s in r.take(N=64):
                    if name == "robot_state":
                        rec.add(name, list(s.velocity_commands))
                    elif name == "eef_state":
                        rec.add(name, list(s.joint_positions))
                    else:
                        rec.add(name, [0.0])
            time.sleep(0.005)
    threading.Thread(target=poll, daemon=True).start()

    print("[drive] settling %.0f s on the walker alone" % SETTLE_S, flush=True)
    time.sleep(SETTLE_S)
    bounds = {}
    for phase, dur in PHASES:
        t0 = time.time()
        bounds[phase] = (t0, t0 + dur)
        print("[drive] %s (%.0f s)" % (phase, dur), flush=True)
        if phase == "reset":
            reset_writer.write(ResetCommand(timestamp=time.time(), reset_requested=True))
        while time.time() < t0 + dur:
            if phase not in ("silence", "reset"):
                writer.write(chor.message(phase, time.time() - t0, TeleopDataWB))
            time.sleep(1.0 / PUBLISH_HZ)
    stop.set()
    conn.send((bounds, rec.series))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True, help="directory for summary.json")
    args = ap.parse_args(argv)
    import multiprocessing as mp
    paths.add_teleop()

    from cyclonedds.domain import DomainParticipant
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from controller import load_home_pose

    rec = Recorder()
    ChannelFactoryInitialize(0, "lo")
    subs = [
        ChannelSubscriber("rt/lowstate", LowState_),
        ChannelSubscriber("rt/lowcmd", LowCmd_),
        ChannelSubscriber("rt/odommodestate", SportModeState_),
    ]
    subs[0].Init(lambda m: rec.add("state", [m.motor_state[i].q for i in range(15, 29)]
                                   + list(m.imu_state.rpy)), 10)
    subs[1].Init(lambda m: rec.add("cmd", [m.motor_cmd[i].q for i in range(15, 29)]), 10)
    subs[2].Init(lambda m: rec.add("odom", list(m.position) + list(m.imu_state.rpy)), 10)

    t_start = time.time()
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"{}", ("127.0.0.1", BAND_RELEASE_PORT))
    print("[drive] sim pelvis band release requested", flush=True)
    mine, theirs = mp.get_context("spawn").Pipe()
    child = mp.get_context("spawn").Process(target=policy_side, args=(theirs,), name="policy-side")
    child.start()
    bounds, series = mine.recv()
    child.join(timeout=10.0)
    rec.series.update(series)
    dds = DomainParticipant(0)
    lowcmd_writers = count_writers(dds, "rt/lowcmd")
    arm_sdk_writers = count_writers(dds, "rt/arm_sdk")
    summary = score(rec, bounds, load_home_pose(), t_start, lowcmd_writers, arm_sdk_writers)
    os.makedirs(args.log, exist_ok=True)
    with open(os.path.join(args.log, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary["checks"].items():
        print("[drive] %-28s %s  %s" % (k, "PASS" if v["pass"] else "FAIL", v["value"]))
    return 0 if all(v["pass"] for v in summary["checks"].values()) else 1


def score(rec, bounds, home, t_start, lowcmd_writers, arm_sdk_writers):
    checks = {}

    def check(name, value, ok):
        checks[name] = {"value": value, "pass": bool(ok)}

    t_end = bounds["reset"][1]
    t, odom = rec.window("odom", t_start + 2.0, t_end)
    check("pelvis_height_min_m", round(float(odom[:, 2].min()), 3) if len(t) else None,
          len(t) and odom[:, 2].min() > 0.6)
    tilt = np.abs(odom[:, 3:5]).max() if len(t) else np.inf
    check("tilt_max_rad", round(float(tilt), 3), tilt < 0.35)
    check("stood_s", round(float(t[-1] - t[0]), 1) if len(t) else 0.0,
          len(t) and t[-1] - t[0] > 0.9 * (t_end - t_start - 2.0))

    a0, a1 = bounds["arms"]
    tc, cmd = rec.window("cmd", a0 + 2.0, a1)
    want = np.stack([arm_trajectory(home, ti - a0) for ti in tc])
    err_cmd = np.abs(cmd - want).max(axis=1)
    check("arm_cmd_vs_target_p95_rad", round(float(np.percentile(err_cmd, 95)), 4),
          np.percentile(err_cmd, 95) < 0.2)
    ts, st = rec.window("state", a0 + 2.0, a1)
    cmd_at = np.stack([np.interp(ts, tc, cmd[:, j]) for j in range(14)], axis=1)
    err_meas = np.abs(st[:, :14] - cmd_at).max(axis=1)
    check("arm_meas_vs_cmd_p95_rad", round(float(np.percentile(err_meas, 95)), 4),
          np.percentile(err_meas, 95) < 0.1)
    check("arm_cmd_swing_rad", round(float(np.ptp(cmd[:, 0])), 3), np.ptp(cmd[:, 0]) > 0.3)
    # eef_state joints are right hand first, retargeting order: index is column 2
    te, ej = rec.window("eef_state", a0, a1)
    swing = float(np.ptp(ej[:, 2])) if len(te) else 0.0
    check("finger_measured_swing_rad", round(swing, 3), swing > 0.5)

    w0, w1 = bounds["walk"]
    tw, ow = rec.window("odom", w0, w1 + 1.0)
    dist = float(np.linalg.norm(ow[-1, :2] - ow[0, :2])) if len(tw) else 0.0
    check("walk_displacement_m", round(dist, 3), dist > 0.5)
    u0, u1 = bounds["turn"]
    tu, ou = rec.window("odom", u0, u1 + 1.0)
    dyaw = float(np.abs(np.unwrap(ou[:, 5])[-1] - np.unwrap(ou[:, 5])[0])) if len(tu) else 0.0
    check("turn_yaw_change_rad", round(dyaw, 3), dyaw > 1.0)
    tr, vr = rec.window("robot_state", w0, w1)
    check("walker_applied_vx_max", round(float(vr[:, 0].max()), 3) if len(tr) else None,
          len(tr) and vr[:, 0].max() > 0.15)
    s0, s1 = bounds["silence"]
    ts, vs = rec.window("robot_state", s0 + 1.5, s1)
    check("walk_zero_on_silence", round(float(np.abs(vs[:, [0, 2]]).max()), 3) if len(ts) else None,
          len(ts) and np.abs(vs[:, [0, 2]]).max() < 0.05)
    r0, r1 = bounds["reset"]
    tr2, cr = rec.window("cmd", r1 - 1.0, r1)
    home_err = float(np.abs(cr - home).max()) if len(tr2) else np.inf
    check("arms_home_after_reset_rad", round(home_err, 4), home_err < 0.05)

    check("lowcmd_writers", lowcmd_writers, lowcmd_writers == 1)
    check("arm_sdk_writers", arm_sdk_writers, arm_sdk_writers == 0)
    # At least policy_rollout's own 30 Hz control rate: one fresh state per
    # policy step. The controller asks for 50 Hz; its IK solve holds the
    # interpreter for about 20 ms per policy frame, as in HTD's controller.
    h0 = bounds["hold"][0]
    for name in ("robot_state", "eef_state", "touch_l"):
        hz = len(rec.window(name, h0, t_end)[0]) / (t_end - h0)
        check("%s_hz" % name, round(hz, 1), hz > PUBLISH_HZ)
    return {"phases": {k: [round(a - t_start, 2), round(b - t_start, 2)] for k, (a, b) in bounds.items()},
            "checks": checks}


if __name__ == "__main__":
    sys.exit(main())
