#!/usr/bin/env python3
"""HTD's policy on the g1_walk walker: the process that replaces
robot_control/g1_inspire_wb_controller.py when the legs are the teleop's
walking policy.

    python g1_walk/controller.py --iface lo                    # the MuJoCo sim
    python g1_walk/controller.py --iface <robot NIC>           # the robot

It needs a running walker (``g1_walk/run_walker.sh``) and, for a session,
policy_rollout publishing on the same DDS interface. It reads:

* ``rt/teleop/synchronized_data`` (TeleopDataWB) and ``rt/robot/reset_command``
  from policy_rollout, unchanged;
* rt/lowstate, through the teleop's ``G1_29_JetsonBody``;
* the Inspire hands, through the teleop's Modbus driver.

It publishes ``rt/robot/state``, ``rt/robot/eef_state`` and
``rt/inspire_hand/touch/{l,r}`` as HTD's controller does, and sends the arm,
walk and torso commands to the walker over its UDP link. It never publishes
rt/lowcmd or rt/arm_sdk: the walker process is the only one commanding the
motors.

Lifecycle, the teleop backend's under ``--lower jetson``:

1. connect to the walker; glide the arms from where they are to the spec's
   home pose while the walker runs its handover (gain ramp, glide, parity
   check, blend); nothing else is commanded until the walker is live;
2. a session starts on the first fresh policy frame once the arms are home:
   the walker is engaged, the link watchdog armed, the arms eased onto the
   IK solution and the torso onto its command;
3. per tick: IK, engage glide, speed cap, jump guard, arms to the walker;
   walk command through the walk gate; torso command; hands retargeted;
4. a session ends -- arms held, walk to zero, the walker stands -- when the
   policy falls silent for ``--policy_timeout`` s, on ``s`` (stop) at the
   keyboard, on a latch (Unitree remote A or L2+B, a walker or link fault);
   a reset from policy_rollout also glides the arms home and opens the hands;
5. ``q``, Ctrl-C or SIGTERM: the walker stands and the link is left (detach):
   the walker keeps the robot balancing until it is stopped in its own
   terminal.

Keyboard (Enter after each): ``s`` stop and hold, ``g`` accept the policy
again, ``q`` quit.
"""
import argparse
import logging
import os
import signal
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
from commands import (TORSO_ATTITUDE_MODES, WalkGate, parse_teleop_data,  # noqa: E402
                      torso_command, walk_command)
from shaping import ArmShaper, Glide, TorsoEngage  # noqa: E402

TOPIC_TELEOP = "rt/teleop/synchronized_data"
TOPIC_RESET = "rt/robot/reset_command"
TOPIC_STATE = "rt/robot/state"
TOPIC_EEF = "rt/robot/eef_state"
TOPIC_TOUCH = ("rt/inspire_hand/touch/l", "rt/inspire_hand/touch/r")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iface", default=os.environ.get("PSI_DDS_IFACE"),
                    help="DDS interface: lo for the sim, the robot NIC on the robot "
                         "(default PSI_DDS_IFACE)")
    ap.add_argument("--walker_host", default=os.environ.get("PSI_G1_JETSON_HOST", "127.0.0.1"),
                    help="host the walker runs on (default PSI_G1_JETSON_HOST or 127.0.0.1)")
    ap.add_argument("--walker_port", type=int,
                    default=int(os.environ.get("PSI_G1_JETSON_PORT", "") or 5591),
                    help="the walker's UDP link port (default PSI_G1_JETSON_PORT or 5591)")
    ap.add_argument("--g1_type", type=int, default=0, choices=(0, 1),
                    help="G1 wrist variant, as HTD's controller: 0 = wrist4010, 1 = wrist5010")
    ap.add_argument("--control_freq", type=float, default=50.0,
                    help="arm / walk / torso / hand command rate, Hz (HTD's default 50)")
    ap.add_argument("--publish_freq", type=float, default=50.0,
                    help="rt/robot/state, eef_state and touch rate, Hz (HTD's default 50)")
    ap.add_argument("--policy_timeout", type=float, default=0.5,
                    help="seconds without a policy frame after which the session ends: "
                         "arms held, walk to zero")
    ap.add_argument("--torso_attitude", default="yaw", choices=TORSO_ATTITUDE_MODES,
                    help="which torso attitude axes reach the walker; yaw is what the "
                         "teleop sent during the recording (see README.md)")
    ap.add_argument("--arm_max_vel", type=float, default=4.0,
                    help="rad/s any arm joint may be commanded to move")
    ap.add_argument("--allow_low_power", action="store_true",
                    help="drive the robot even on battery or a power-saving profile (not recommended)")
    ap.add_argument("--hands", default="modbus", choices=("modbus", "none"),
                    help="none runs without the Inspire hands (no eef_state, no touch)")
    return ap.parse_args(argv)


def load_home_pose():
    """The spec's ``[pose] home``: the stand pose the teleop parks the arms
    at between sessions (``ArmPipeline.home``)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(paths.G1_ARM_SPEC, "rb") as f:
        home = np.asarray(tomllib.load(f)["pose"]["home"], dtype=float)
    if home.shape != (14,):
        raise ValueError("%s: [pose] home must have 14 entries" % paths.G1_ARM_SPEC)
    return home


class Controller:
    def __init__(self, args):
        # Imported here: the modules below need paths set up and, for the
        # teleop's, PSI_DDS_IFACE already in the environment.
        import pinocchio as pin
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from robot_arm_ik_g1 import G1_29_ArmIK
        from teleop.robot_control.drive_safety import (EStopLatch, JointJumpGuard,
                                                       arm_jump_thresholds)
        from teleop.robot_control.g1_backend import (ARM_HOLD_MAX, G1_STAND_HEIGHT,
                                                     JUMP_CREEP_RAD, LAUNCH_HOME_RAD_S)
        from teleop.utils.logger import logger
        from teleop.utils.psi_env import psi_float
        from teleop_dds_msgs import EEFState, G1RobotState, ResetCommand, TeleopDataWB
        from robot_state import G1_URDFS, HtdStateEstimate
        from walker import WalkerLink

        self.args = args
        self.log = logger
        self.dt = 1.0 / float(args.control_freq)
        self.stand_height = G1_STAND_HEIGHT
        self.launch_home_rad_s = LAUNCH_HOME_RAD_S
        # The teleop backend's knobs, read under the same names and defaults.
        self.rehome_s = psi_float("PSI_REHOME_S", "2.0")
        engage_s = psi_float("PSI_ENGAGE_S", "1.0")
        self.home = load_home_pose()

        self._lock = threading.RLock()
        self._estop = EStopLatch()
        self.quit = threading.Event()
        self.session = False
        self.ready = False              # the walker's handover completed
        self.operator_hold = False      # 's' at the keyboard
        self.resetting = False
        self._fresh_after = 0.0         # frames received before this are stale
        # The latest policy frame, under its own lock: the control tick holds
        # self._lock through an IK solve, which must not stall the reader.
        self._frame_lock = threading.Lock()
        self._frame = None
        self._frame_t = None
        self._solved = None             # (frame, IK solution) of the last solve
        self._reset_pending = threading.Event()
        self._open_hands = threading.Event()
        self._home_glide = None
        self._arms_home = False
        self._last_arm_cmd = None
        self._said = set()

        # Everything slow to build comes before the walker link: once linked,
        # this process must answer the walker every 0.25 s or it stands itself.
        urdf_dir = "./assets/g1/"
        self.ik = G1_29_ArmIK(urdf_path=urdf_dir + G1_URDFS[args.g1_type], urdf_dir=urdf_dir)
        model = self.ik.reduced_robot.model
        data = model.createData()
        zeros = np.zeros(model.nv)
        self.gravity = lambda q: np.asarray(pin.rnea(model, data, np.asarray(q, float),
                                                     zeros, zeros)).copy()
        guard = JointJumpGuard(
            arm_jump_thresholds(psi_float("PSI_G1_ARM_JUMP_SCALE", "1.0")), "g1_walk", "arms",
            max_hold_steps=ARM_HOLD_MAX,
            on_expire=lambda label: logger.critical(
                "[g1_walk] the %s have refused every command for %d frames" % (label, ARM_HOLD_MAX)))
        self.arms = ArmShaper(guard, self.gravity, JUMP_CREEP_RAD, self.dt, args.arm_max_vel,
                              engage_s, psi_float("PSI_ENGAGE_VEL", "1.0"),
                              psi_float("PSI_ENGAGE_MAX_S", "4.0"))
        self.torso = TorsoEngage(engage_s)
        self.gate = WalkGate()
        self.hands = None
        if args.hands == "modbus":
            from hands import InspireHands
            self.hands = InspireHands()

        self.walker = WalkerLink(args.walker_host, args.walker_port, self.estop)
        self.body = self.walker.body
        self.walker.strategy.on_remote_stand = self._on_remote_stand
        self.estimate = HtdStateEstimate(args.g1_type, self.body.get_current_motor_q)

        # HTD's topics, on a participant created after the body's
        # ChannelFactoryInitialize so both share its DDS interface.
        self.dds = DomainParticipant(0)
        self._msg_types = (TeleopDataWB, ResetCommand)
        self.reader_teleop = DataReader(self.dds, Topic(self.dds, TOPIC_TELEOP, TeleopDataWB))
        self.reader_reset = DataReader(self.dds, Topic(self.dds, TOPIC_RESET, ResetCommand))
        self.writer_state = DataWriter(self.dds, Topic(self.dds, TOPIC_STATE, G1RobotState))
        if self.hands is not None:
            from inspire_sdkpy import inspire_dds
            self.writer_eef = DataWriter(self.dds, Topic(self.dds, TOPIC_EEF, EEFState))
            self.writer_touch = [DataWriter(self.dds, Topic(self.dds, t, inspire_dds.inspire_hand_touch))
                                 for t in TOPIC_TOUCH]

    # -- latches ---------------------------------------------------------------
    def estop(self, reason="operator"):
        """The body's e-stop hook (A on the remote, damping, the walker's own
        latch adopted): ``G1Backend.estop`` -- latch, freeze the arms and
        waist on the walker, stand it. Latched until this process restarts."""
        if not self._estop.trip(reason):
            return False
        try:
            self.body.latch_estop(reason)
        except Exception:
            self.log.exception("[g1_walk] freezing the walker's body failed")
        try:
            self.walker.strategy.stand()
        except Exception:
            self.log.exception("[g1_walk] standing the walker after the e-stop failed")
        # No lock: this runs on the body's state thread or the link's receive
        # thread, which a control tick waiting on the walker may need. The
        # tick sees the latch through fault() and ends the session itself.
        self.gate.rearm()
        self._home_glide = None
        return True

    def fault(self):
        """Why no command may go out now, or None."""
        estop = self._estop.fault()
        if estop is None and self.body._estop_reason is not None:
            estop = "estop_%s" % self.body._estop_reason
        return estop or self.walker.fault() or (
            None if self.body.fault is None else "link_%s" % self.body.fault)

    def _on_remote_stand(self):
        """The walker stood itself on a silent link: the policy must be seen
        at rest again before it walks."""
        self.gate.rearm()

    # -- inputs ------------------------------------------------------------------
    def _dds_in_loop(self):
        while not self.quit.is_set():
            try:
                self._take_inputs()
            except Exception:
                self._say_once("dds_in", "[g1_walk] reading policy_rollout's topics failed "
                               "(logged once; the session ends on the silence)", exc=True)
            time.sleep(0.002)

    def _take_inputs(self):
        teleop_type, reset_type = self._msg_types
        for sample in self.reader_teleop.take(N=16):
            if not isinstance(sample, teleop_type):
                continue            # a writer's dispose or unregister
            frame = parse_teleop_data(sample)
            if frame is None:
                self._say_once("bad_frame", "[g1_walk] a TeleopDataWB message with a wrong "
                               "length or a non-finite value was dropped")
                continue
            with self._frame_lock:
                self._frame, self._frame_t = frame, time.monotonic()
        for sample in self.reader_reset.take(N=4):
            if isinstance(sample, reset_type) and sample.reset_requested:
                self._reset_pending.set()

    def _fresh_frame(self, now):
        with self._frame_lock:
            frame, t = self._frame, self._frame_t
        if frame is None or t <= self._fresh_after or now - t > self.args.policy_timeout:
            return None
        return frame

    def _keyboard_loop(self):
        for line in sys.stdin:
            key = line.strip().lower()
            if key == "s":
                with self._lock:
                    self.operator_hold = True
                    if self.session:
                        self._end_session("stopped at the keyboard", glide_home=False)
                self.log.info("[g1_walk] stopped: the walker stands, the arms hold; g resumes")
            elif key == "g":
                with self._lock:
                    self.operator_hold = False
                    self._fresh_after = time.monotonic()
                self.log.info("[g1_walk] accepting the policy again from its next frame")
            elif key == "q":
                self.quit.set()
                return

    # -- the control loop ----------------------------------------------------------
    def _control_loop(self):
        next_t = time.monotonic()
        while not self.quit.is_set():
            try:
                with self._lock:
                    self._tick(time.monotonic())
            except Exception:
                self.log.exception("[g1_walk] control tick failed")
            next_t += self.dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    def _tick(self, now):
        fault = self.fault()
        if self._reset_pending.is_set():
            self._reset_pending.clear()
            self._start_reset()
        if self.session:
            frame = self._fresh_frame(now)
            if fault is not None:
                self._end_session("latched: %s" % fault, glide_home=False)
            elif frame is None:
                self._end_session("no policy frame for %.2f s" % self.args.policy_timeout,
                                  glide_home=False)
            else:
                self._session_tick(frame, now)
                return
        self._tick_between(now)
        self.walker.strategy.set_walk_command(0.0, 0.0)
        if self._may_start(fault, now):
            self._start_session(now)

    def _session_tick(self, frame, now):
        # One IK solve per policy frame (policy_rollout publishes at 30 Hz, a
        # solve takes about 20 ms); every tick shapes and sends the latest.
        if self._solved is None or self._solved[0] is not frame:
            q_meas = self.body.get_current_dual_arm_q()
            dq_meas = self.body.get_current_dual_arm_dq()
            q_sol, _ = self.ik.solve_ik(frame.left_pose, frame.right_pose, q_meas, dq_meas)
            self._solved = (frame, q_sol)
        out = self.arms.shape(self._solved[1], now)
        if out is not None:
            self.walker.send_arms(*out)
            self._last_arm_cmd = out[0]
        s = self.walker.strategy
        if frame.velocity is not None and abs(frame.velocity[1]) > 0.05:
            self._say_once("vy", "[g1_walk] the policy asks for vy %.2f m/s; the walker has "
                           "no lateral axis and it is dropped" % frame.velocity[1])
        vx, wz = walk_command(frame, s.vx_range, s.wz_range)
        vx, wz, release = self.gate.step(vx, wz, allowed=self.walker.live)
        if release:
            s.resume()
            self.log.info("[g1_walk] the policy asked for rest -- it walks the robot now")
        s.set_walk_command(vx, wz)
        target = torso_command(frame, self.stand_height, self.args.torso_attitude)
        h, r, p, y = self.torso.shape(target, now)
        s.set_torso_command(height=h, roll=r, pitch=p, yaw=y)

    def _may_start(self, fault, now):
        if (not self.ready or fault is not None or self.operator_hold or self.resetting
                or self._home_glide is not None or not self.walker.live):
            return False
        return self._fresh_frame(now) is not None

    def _start_session(self, now):
        """``G1Backend.on_session_start`` + ``_engage_lower`` + the
        pipeline's session boundary."""
        if not self.walker.engage():
            self.operator_hold = True
            self.log.critical("[g1_walk] session refused; s/g at the keyboard tries again")
            return
        self.body.arm_link_watchdog()
        q_from = self._last_arm_cmd if self._last_arm_cmd is not None \
            else self.body.get_current_dual_arm_q()
        self.arms.begin(q_from)
        self.torso.begin(self.walker.applied_torso(), now)
        self.gate.rearm()
        self._open_hands.clear()
        self.session = True
        self.log.info("[g1_walk] session started: the policy drives the arms, walk and torso")

    def _end_session(self, why, glide_home):
        """``G1Backend.on_session_end`` and ``reset_session``: the walker
        stands (walk and torso slewed back to nominal), the walk gate
        re-arms, the link watchdog is disarmed, the walker's body re-anchors
        its arm command; the arms hold, or glide home."""
        self.session = False
        s = self.walker.strategy
        s.set_walk_command(0.0, 0.0)
        s.stand()
        self.gate.rearm()
        self.body.disarm_link_watchdog()
        self.body.reset(hold_pose=False)
        if glide_home:
            self._start_home_glide()
        self.log.warning("[g1_walk] session ended (%s); the walker stands, the arms %s"
                         % (why, "glide home" if glide_home else "hold"))

    def _start_reset(self):
        """policy_rollout's reset: HTD's controller opens the hands, returns
        the arms and torso to default and waits for new frames."""
        self.log.info("[g1_walk] reset requested: arms home, hands open")
        if self.session:
            self._end_session("reset requested", glide_home=True)
        elif self._estop.fault() is None:
            self.walker.strategy.stand()
            self._start_home_glide()
        self._open_hands.set()
        self.resetting = True

    # -- between sessions: the arms' glide home ----------------------------------------
    def _start_home_glide(self, launch=False):
        """``G1Backend._start_arm_home``: from the last command on the wire (the
        measured arms at launch) to home, smoothstep, no faster than the
        launch speed bound."""
        if self.fault() is not None:
            return
        q0 = self._last_arm_cmd
        if launch or q0 is None:
            q0 = self.body.get_current_dual_arm_q()
        travel = float(np.abs(self.home - q0).max())
        dur = max(self.rehome_s, travel / self.launch_home_rad_s)
        self._home_glide = Glide(q0, dur, time.monotonic())
        self._arms_home = False
        self.log.info("[g1_walk] arms gliding home over %.1f s" % dur)

    def _tick_between(self, now):
        """``G1Backend._tick_arm_home``: one tick of the glide, then the hold
        at home; nothing while the arms hold elsewhere or a latch froze them."""
        if self.fault() is not None:
            self._home_glide = None
            return
        if self._home_glide is not None:
            q, done = self._home_glide.at(self.home, now)
            if done:
                self._home_glide = None
                self._arms_home = True
                if self.resetting:
                    self.resetting = False
                    self._fresh_after = now
                    self.log.info("[g1_walk] reset complete, waiting for new policy frames")
        elif self._arms_home:
            q = self.home
        else:
            return
        self.walker.send_arms(q, self.gravity(q))
        self._last_arm_cmd = np.asarray(q, float).copy()

    # -- hands and state ------------------------------------------------------------------
    def _hands_loop(self):
        period = 1.0 / float(self.args.control_freq)
        while not self.quit.is_set():
            t0 = time.monotonic()
            try:
                if self._open_hands.is_set():
                    self.hands.open()
                elif self.session:
                    frame = self._fresh_frame(t0)
                    if frame is not None:
                        self.hands.command(frame.left_hand, frame.right_hand)
            except Exception:
                self.log.exception("[g1_walk] hand command failed")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _publish_state_loop(self):
        """rt/robot/state, as HTD's ``_robot_state_pub_loop``."""
        from teleop.robot_control.robot_body import LowStateStale
        from robot_state import robot_state_message
        period = 1.0 / float(self.args.publish_freq)
        while not self.quit.is_set():
            t0 = time.monotonic()
            try:
                self.writer_state.write(robot_state_message(
                    self.estimate, self.body, self.walker.strategy.walk_command))
            except LowStateStale as e:
                self._say_once("stale", "[g1_walk] robot state not published: %s" % e)
            except Exception:
                self._say_once("state", "[g1_walk] publishing rt/robot/state failed "
                               "(logged once)", exc=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _publish_hands_loop(self):
        """rt/robot/eef_state, as HTD's ``_hand_state_pub_loop``, and the
        touch topics its DFTP hand driver publishes."""
        from hands import touch_message
        from robot_state import eef_state_message
        period = 1.0 / float(self.args.publish_freq)
        while not self.quit.is_set():
            t0 = time.monotonic()
            try:
                joints, forces, tips, touch_l, touch_r = self.hands.read()
                self.writer_eef.write(eef_state_message(joints, forces, tips))
                self.writer_touch[0].write(touch_message(touch_l))
                self.writer_touch[1].write(touch_message(touch_r))
            except Exception:
                self._say_once("hand_state", "[g1_walk] publishing the hand state failed "
                               "(logged once)", exc=True)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _say_once(self, key, line, exc=False):
        if key not in self._said:
            self._said.add(key)
            (self.log.exception if exc else self.log.warning)(line)

    # -- the process ------------------------------------------------------------------------
    def run(self):
        loops = [self._dds_in_loop, self._control_loop, self._publish_state_loop]
        if self.hands is not None:
            loops += [self._hands_loop, self._publish_hands_loop]
        if sys.stdin is not None and sys.stdin.isatty():
            loops.append(self._keyboard_loop)
        with self._lock:
            self._start_home_glide(launch=True)
        for fn in loops:
            threading.Thread(target=fn, name=fn.__name__.strip("_"), daemon=True).start()
        if not self.walker.handover():
            self.quit.set()
            return 1
        with self._lock:
            self.ready = True
            self.gate.rearm()
        s = self.walker.strategy
        self.log.info("[g1_walk] the walker's command box: vx %s m/s, wz %s rad/s, torso "
                      "height travel %s m" % (s.vx_range, s.wz_range, s.h_range))
        print("READY: g1_walk controller: the walker is live; sessions start on the "
              "policy's frames (s stop, g go, q quit)", flush=True)
        self.quit.wait()
        return 0

    def shutdown(self):
        """Stand the walker and detach from it (the teleop's shutdown under
        --lower jetson): the walker keeps the robot balancing."""
        self.quit.set()
        with self._lock:
            if self.session:
                self._end_session("shutting down", glide_home=False)
            else:
                self.walker.strategy.set_walk_command(0.0, 0.0)
                self.walker.strategy.stand()
        # One link heartbeat carries the stand before the link closes.
        time.sleep(0.1)
        self.walker.detach()


def configure_logging():
    """One line per record. The teleop's logger has a handler of its own,
    and the teleop's Inspire driver logs on the root logger, which then
    configures itself and would print every teleop record a second time."""
    from teleop.utils.logger import logger as teleop_logger
    teleop_logger.propagate = False
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def main(argv=None):
    args = parse_args(argv)
    if not args.iface:
        print("FAILED: no DDS interface: pass --iface (lo for the sim)", file=sys.stderr)
        return 2
    from power import power_problems
    problems = power_problems()
    if problems:
        # The sim on loopback is allowed (its timing checks will say so); the robot is not.
        if args.iface != "lo" and not args.allow_low_power:
            print("FAILED: this machine is not set up for real-time control: %s "
                  "(--allow_low_power overrides)" % "; ".join(problems), file=sys.stderr)
            return 2
        print("WARNING: not set up for real-time control: %s" % "; ".join(problems),
              file=sys.stderr, flush=True)
    os.environ["PSI_DDS_IFACE"] = args.iface
    paths.add_teleop()
    paths.enter_htd_robot_control()
    configure_logging()
    ctrl = Controller(args)

    def _on_signal(signum, frame):
        ctrl.quit.set()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    try:
        code = ctrl.run()
    finally:
        ctrl.shutdown()
    # The walker is stood and detached: nothing the robot relies on is left in this
    # process. Stop its worker processes (multiprocessing's own exit hook would, but it
    # runs during interpreter teardown), then skip that teardown, where native threads
    # of the DDS and Modbus libraries still alive can abort the process ("terminate
    # called without an active exception").
    import multiprocessing
    for child in multiprocessing.active_children():
        child.terminate()
    for child in multiprocessing.active_children():
        child.join(timeout=2.0)
        if child.is_alive():
            child.kill()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)


if __name__ == "__main__":
    sys.exit(main())
