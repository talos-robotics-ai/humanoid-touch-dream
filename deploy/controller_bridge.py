"""What g1_walk/controller.py needs between its own data and the supervisor."""
import math
import threading
import time
from collections import deque

import numpy as np

from deploy.g1_spec import StandingSpec
from deploy.msgs import UI_COMMANDS, SupervisorStatus, UiCommand
from deploy.remote import RemoteEvents, RemoteSnapshot
from deploy.standing import StandingReport, StandingSample
from deploy.supervisor import Decision, Supervisor


class TickTimer:
    """Tick durations of the last `window_s`; added from the control thread, read from another."""

    def __init__(self, window_s: float = 1.0):
        self.window_s = float(window_s)
        self._samples: deque = deque()
        self._lock = threading.Lock()

    def add(self, seconds: float, now: float) -> None:
        with self._lock:
            self._samples.append((now, seconds * 1000.0))
            while self._samples and now - self._samples[0][0] > self.window_s:
                self._samples.popleft()

    def stats(self, now: float) -> tuple[float, float]:
        with self._lock:
            ms = sorted(v for t, v in self._samples if now - t <= self.window_s)
        if not ms:
            return 0.0, 0.0
        return ms[len(ms) // 2], ms[-1]


class UiCommandFilter:
    def __init__(self, max_age_s: float):
        self.max_age_s = float(max_age_s)
        self._seen: dict = {}       # nonce -> command timestamp

    def accept(self, cmd: UiCommand, now_wall: float) -> str | None:
        # A nonce is forgotten once its command is too old to pass the age check anyway.
        self._seen = {n: t for n, t in self._seen.items() if now_wall - t <= self.max_age_s}
        if cmd.command not in UI_COMMANDS or not 0.0 <= now_wall - cmd.timestamp <= self.max_age_s:
            return None
        if cmd.nonce in self._seen:
            return None
        self._seen[cmd.nonce] = float(cmd.timestamp)
        return cmd.command


def standing_sample(walker_live: bool, torso_pose, gyro, leg_dq, lowstate_age_s: float,
                    spec: StandingSpec) -> StandingSample:
    """A missing input makes the sample fail the check rather than raise."""
    nan, inf = math.nan, math.inf
    if torso_pose is None:
        walker_live, height, roll, pitch = False, nan, nan, nan
    else:
        height, roll, pitch = (float(torso_pose[i]) for i in (2, 3, 4))
    if gyro is None:
        gyro_norm, lowstate_age_s = nan, inf
    else:
        gyro_norm = float(np.linalg.norm(np.asarray(gyro, float)))
    if leg_dq is None:
        leg_max = inf
    else:
        leg_max = float(np.abs(np.asarray(leg_dq, float)[list(spec.leg_joints)]).max())
    return StandingSample(bool(walker_live), height, roll, pitch, gyro_norm, leg_max,
                          float(lowstate_age_s))


def status_message(sup: Supervisor, decision: Decision, standing: StandingReport,
                   events: RemoteEvents, snap: RemoteSnapshot, policy_ready: bool,
                   frame_fresh: bool, walker_live: bool, fault: str | None, walk_enabled: bool,
                   manual_walk_enabled: bool, tick_stats: tuple[float, float], last_log: str,
                   recent_log, now_wall: float, now_mono: float | None = None) -> SupervisorStatus:
    """`recent_log`: the supervisor's last log lines, oldest first.
    `now_mono` is the supervisor's clock (time.monotonic) for the time in state."""
    if now_mono is None:
        now_mono = time.monotonic()
    p50, mx = tick_stats
    return SupervisorStatus(
        timestamp=float(now_wall),
        state=sup.state.value,
        since_s=max(0.0, float(now_mono - sup.since)),
        reason=decision.reason,
        walker_live=bool(walker_live),
        standing=bool(standing.standing),
        standing_for_s=float(standing.held_s),
        policy_ready=bool(policy_ready),
        frames_fresh=bool(frame_fresh),
        walk_enabled=bool(walk_enabled),
        manual_walk_enabled=bool(manual_walk_enabled),
        fault=fault or "",
        gating=list(decision.gating),
        stick_lx=float(snap.lx), stick_ly=float(snap.ly),
        stick_rx=float(snap.rx), stick_ry=float(snap.ry),
        buttons=list(events.pressed),
        manual_vx=float(events.manual_vx), manual_wz=float(events.manual_wz),
        tick_p50_ms=float(p50), tick_max_ms=float(mx),
        log_line=last_log,
        recent_log=list(recent_log),
    )
