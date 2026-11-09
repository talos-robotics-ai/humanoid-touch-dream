"""g1_walk/controller.py's supervisor tick on a fake walker link: what reaches the walker per state."""
import os
import sys
import threading
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "g1_walk"))
import controller  # noqa: E402
from commands import PolicyFrame, WalkGate  # noqa: E402
from deploy.msgs import PolicyStatus  # noqa: E402
from deploy.supervisor import State  # noqa: E402


class Stale(RuntimeError):
    pass


class Strategy:
    vx_range, wz_range = (-0.3, 0.5), (-0.8, 0.8)
    walk_command = (0.0, 0.0)

    def __init__(self, calls):
        self.calls, self.phase = calls, "live"

    def set_walk_command(self, vx, wz): self.calls.append(("walk", round(vx, 3), round(wz, 3)))
    def set_torso_command(self, **kw): self.calls.append(("torso",))
    def stand(self): self.calls.append(("stand",))
    def resume(self): self.calls.append(("resume",))


class Body:
    def __init__(self, calls):
        self.calls, self.fault, self._estop_reason, self.age = calls, None, None, 0.01
        self.remote_controller = types.SimpleNamespace(lx=0.0, ly=0.0, rx=0.0, ry=0.0, button=[0] * 16)

    def lowstate_age(self): return self.age

    def get_imu_data(self):
        if self.age > 0.5:
            raise Stale()
        return types.SimpleNamespace(gyroscope=[0.0, 0.0, 0.0])

    def get_current_motor_dq(self): return np.zeros(29)
    def get_current_dual_arm_q(self): return np.zeros(14)
    def get_current_dual_arm_dq(self): return np.zeros(14)

    def arm_link_watchdog(self):
        self.calls.append(("arm_wd",))
        if self._estop_reason is None:
            self.fault = None

    def disarm_link_watchdog(self): self.calls.append(("disarm_wd",))
    def reset(self, hold_pose=True): self.calls.append(("body_reset", hold_pose))


class Walker:
    def __init__(self, calls):
        self.calls, self.strategy, self.body, self.lower_fault = calls, Strategy(calls), Body(calls), None
        self.engages = True

    @property
    def live(self): return self.strategy.phase == "live"

    def engage(self):
        self.calls.append(("engage",))
        return self.engages

    def fault(self): return self.lower_fault
    def send_arms(self, q, tau): self.calls.append(("arms",))
    def applied_torso(self): return (0.0, 0.0, 0.0, 0.0)


FRAME = PolicyFrame(np.eye(4), np.eye(4), np.zeros(75), np.zeros(75), np.zeros(3), 0.78,
                    np.zeros(3), 0.0)


class Rig:
    def __init__(self, monkeypatch, *argv):
        self.t = 100.0
        monkeypatch.setattr(controller, "time", types.SimpleNamespace(
            monotonic=lambda: self.t, time=time.time, perf_counter=time.perf_counter))
        self.calls = []
        c = controller.Controller.__new__(controller.Controller)
        c.args = controller.parse_args(["--iface", "lo", "--supervisor", "on", *argv])
        c.log = types.SimpleNamespace(**{k: (lambda *a, **kw: None) for k in
                                         ("info", "warning", "critical", "exception")})
        c.dt, c.stand_height, c.home = 0.02, 0.75, np.full(14, 0.1)
        c.rehome_s, c.launch_home_rad_s = 2.0, 0.5
        c._lock, c._frame_lock = threading.RLock(), threading.Lock()
        c._estop = types.SimpleNamespace(fault=lambda: None)
        c.quit = threading.Event()
        c.session = c.operator_hold = c.resetting = False
        c.ready = True
        c._fresh_after, c._frame, c._frame_t, c._solved = 0.0, None, None, None
        c._reset_pending, c._open_hands = threading.Event(), threading.Event()
        c._home_glide, c._arms_home, c._last_arm_cmd = None, True, np.full(14, 0.1)
        c._said, c._stop_request = set(), None
        c.gravity = lambda q: np.zeros(14)
        c.arms = types.SimpleNamespace(begin=lambda q: None, shape=lambda q, now: (np.zeros(14), np.zeros(14)))
        c.torso = types.SimpleNamespace(begin=lambda a, now: None, shape=lambda target, now: (0.0, 0.0, 0.0, 0.0))
        c.ik = types.SimpleNamespace(solve_ik=lambda *a: (np.zeros(14), None))
        c.gate = WalkGate()
        c.walker = Walker(self.calls)
        c.body = c.walker.body
        c._init_supervisor(Stale)
        self.status = []                # SupervisorStatus messages written
        c.writer_sup = types.SimpleNamespace(write=self.status.append)
        self.c = c

    def tick(self, n=1, frames=True, dt=0.02):
        c = self.c
        for _ in range(n):
            self.t += dt
            if frames:
                c._frame, c._frame_t = FRAME, self.t - 0.001
            c._torso_pose = (np.array([0.0, 0.0, 0.78, 0.0, 0.0, 0.0]), self.t)
            c._policy_status = PolicyStatus(time.time(), True, True, 20.0, 1, 0, 0.0, 0.0, 0.0,
                                            1.0, 2.0, 45, 60, "test", "")
            self.calls.clear()
            c._tick(self.t)
        return list(self.calls)

    @property
    def state(self): return self.c.sup.state

    def to_ready(self):
        self.tick(110)                  # BOOT -> STABILIZING, then the 2 s standing hold
        assert self.state is State.READY

    def to_running(self):
        self.to_ready()
        self.c._ui_cmds.append("start")
        self.tick(10)                   # past the start-frame margin
        assert self.state is State.RUNNING and self.c.session


@pytest.fixture
def rig(monkeypatch):
    return Rig(monkeypatch)


def test_start_waits_for_a_frame_newer_than_the_start(rig):
    rig.to_ready()
    rig.c._ui_cmds.append("start")
    calls = rig.tick(frames=False)
    assert rig.state is State.RUNNING and not rig.c.session
    assert ("engage",) not in calls and ("walk", 0.0, 0.0) in calls
    rig.tick(7)                         # inside the start-frame margin
    assert not rig.c.session
    calls = rig.tick()
    assert rig.c.session and ("engage",) in calls and ("arm_wd",) in calls


def test_start_ignores_frames_for_the_margin(rig):
    rig.to_ready()
    rig.c._ui_cmds.append("start")
    rig.tick(frames=False)
    rig.tick(2)                         # frame 0.04 s after the start edge
    assert not rig.c.session
    rig.tick(8)                         # frame 0.20 s after
    assert rig.c.session


def test_start_without_a_new_frame_holds_after_the_policy_timeout(rig):
    rig.to_ready()
    rig.c._ui_cmds.append("start")
    rig.tick(frames=False)
    rig.tick(20, frames=False)          # 0.4 s: still waiting
    assert rig.state is State.RUNNING
    rig.tick(15, frames=False)
    assert rig.state is State.HOLD and not rig.c.session


def test_freeze_never_stands_the_walker(rig):
    rig.to_running()
    rig.c._kb_cmds.append("stop")
    calls = rig.tick()
    assert rig.state is State.HOLD and ("stand",) not in calls and ("disarm_wd",) in calls
    calls = rig.tick()
    assert ("arms",) not in calls and ("stand",) not in calls     # arms and torso hold


def test_sticks_take_over_from_running_without_a_stand(monkeypatch):
    rig = Rig(monkeypatch, "--manual_walk", "on")
    rig.to_running()
    rig.c.body.remote_controller.ly = 0.8
    calls = rig.tick()
    assert rig.state is State.MANUAL and ("stand",) not in calls and ("resume",) in calls
    walks = [c for c in rig.tick() if c[0] == "walk"]
    assert walks and walks[-1][1] > 0.0


def test_estop_commands_nothing(rig):
    rig.to_running()
    rig.c.walker.lower_fault = "lower_tilt"
    calls = rig.tick()
    assert rig.state is State.ESTOP and ("stand",) not in calls
    assert rig.tick(5) == []


def test_jump_guard_expiry_stops_the_policy(rig):
    rig.to_running()
    rig.c._stop_request = "the arms refused every command for 25 frames"
    rig.tick()
    assert rig.state is State.HOLD and rig.c._stop_request is None


def test_rearm_lifts_a_link_fault(rig):
    rig.to_running()
    rig.c.body.fault = "jetson_silent"
    rig.tick()
    assert rig.state is State.HOLD
    rig.c._kb_cmds.append("rearm")
    calls = rig.tick()
    assert rig.c.body.fault is None and ("arm_wd",) in calls and ("stand",) in calls
    rig.tick(110)
    assert rig.state is State.READY


def test_a_silent_walker_holds_and_recovers(rig):
    rig.to_running()
    rig.c.walker.lower_fault = "lower_jetson_silent_1.2s"
    rig.tick()
    assert rig.state is State.HOLD
    rig.c.walker.lower_fault = None     # the walker's status returns
    rig.c._kb_cmds.append("rearm")
    rig.tick(110)
    assert rig.state is State.READY


def test_reset_under_a_link_fault_lifts_it_and_completes(rig):
    rig.to_running()
    rig.c.body.fault = "jetson_silent"
    rig.tick()
    assert rig.state is State.HOLD
    rig.c._reset_pending.set()          # rollout's reset
    calls = rig.tick()
    assert rig.state is State.STABILIZING and rig.c.body.fault is None and ("arm_wd",) in calls
    assert rig.c._home_glide is not None and rig.c._open_hands.is_set()
    rig.tick(110)
    assert not rig.c.resetting and rig.state is State.READY


def test_reset_whose_glide_is_refused_does_not_stay_resetting(rig):
    rig.to_running()
    rig.c.walker.lower_fault = "lower_jetson_silent_1.2s"    # not lifted by the watchdog: still silent
    rig.tick()
    rig.c._kb_cmds.append("reset")
    rig.tick()
    assert rig.state is State.STABILIZING and rig.c._home_glide is None and not rig.c.resetting
    rig.c.walker.lower_fault = None
    rig.tick(110)
    assert rig.state is State.READY


def test_refused_engage_goes_to_hold(rig):
    rig.to_ready()
    rig.c.walker.engages = False
    rig.c._ui_cmds.append("start")
    rig.tick(9)
    assert not rig.c.session
    rig.tick()
    assert rig.state is State.HOLD


def test_rearm_restarts_the_standing_hold(rig):
    rig.to_running()
    rig.tick(100)                       # 2 s of standing already held in RUNNING
    rig.c._kb_cmds.append("stop")
    rig.tick()
    assert rig.state is State.HOLD
    rig.c._kb_cmds.append("rearm")
    rig.tick()
    assert rig.state is State.STABILIZING
    rig.tick(94)                        # 1.9 s after the re-arm, standing good throughout
    assert rig.state is State.STABILIZING
    rig.tick(16)                        # 2.22 s
    assert rig.state is State.READY


def test_reset_restarts_the_standing_hold(rig):
    rig.to_ready()
    rig.tick(100)
    rig.c.rehome_s = 0.5                # a glide home shorter than the hold
    rig.c._kb_cmds.append("reset")
    rig.tick()
    assert rig.state is State.STABILIZING
    rig.tick(94)                        # glide done at 0.5 s; 1.9 s of the hold
    assert rig.state is State.STABILIZING and not rig.c.resetting
    rig.tick(16)
    assert rig.state is State.READY


def test_a_state_change_writes_the_status_at_once(rig):
    rig.to_ready()
    assert [m.state for m in rig.status] == ["STABILIZING", "READY"]    # one write per transition
    rig.tick(5)
    assert len(rig.status) == 2                                          # none without a transition
    rig.c._ui_cmds.append("start")
    rig.tick()
    assert rig.status[-1].state == "RUNNING" and len(rig.status) == 3


def test_the_status_carries_the_recent_log_lines(rig):
    rig.to_ready()
    log = list(rig.status[-1].recent_log)
    assert log == ["[supervisor] BOOT -> STABILIZING: walker live",
                   "[supervisor] STABILIZING -> READY: standing 2.0 s, policy ready, frames fresh",
                   "HTDPOLICY READY TO DEPLOY"]
    for _ in range(15):
        rig.c._ui_cmds.append("reset")
        rig.tick(2)
    assert len(rig.c._recent_log) == 12 and rig.c._recent_log[-1].endswith(": reset")


def test_no_motor_state_yet_fails_the_standing_check_safe(rig):
    def no_lowstate():
        raise AttributeError("'NoneType' object has no attribute 'motor_state'")
    rig.c.body.get_current_motor_dq = no_lowstate
    rig.tick(120)
    assert rig.state is State.STABILIZING
