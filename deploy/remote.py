"""Turns the Unitree remote's sticks and buttons into supervisor events."""
import math
from dataclasses import dataclass

from deploy.g1_spec import RemoteSpec

# Button names by bit index of the wireless-remote key word (same order as the teleop KeyMap).
BUTTON_NAMES = ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
                "A", "B", "X", "Y", "up", "right", "down", "left")


@dataclass(frozen=True)
class RemoteSnapshot:
    lx: float
    ly: float
    rx: float
    ry: float
    buttons: frozenset
    stale: bool


@dataclass(frozen=True)
class RemoteEvents:
    start: bool
    stop: bool
    reset: bool
    rearm: bool
    deflected: bool
    manual_vx: float
    manual_wz: float
    pressed: tuple


def _axis(v) -> float:
    v = float(v)
    return v if math.isfinite(v) else 0.0


def snapshot(rc, stale: bool) -> RemoteSnapshot:
    """Copy `rc` (a teleop RemoteController); a stale link reads as centred, no buttons."""
    if stale:
        return RemoteSnapshot(0.0, 0.0, 0.0, 0.0, frozenset(), True)
    buttons = frozenset(i for i, v in enumerate(rc.button) if v)
    return RemoteSnapshot(_axis(rc.lx), _axis(rc.ly), _axis(rc.rx), _axis(rc.ry), buttons, False)


def stick_to_walk(ly: float, rx: float, spec: RemoteSpec) -> tuple[float, float]:
    """Stick deflection to (vx, wz); non-finite input commands no motion."""
    if not (math.isfinite(ly) and math.isfinite(rx)):
        return 0.0, 0.0

    def scaled(v):
        if abs(v) <= spec.deadband:
            return 0.0
        s = (abs(v) - spec.deadband) / (1.0 - spec.deadband)
        return math.copysign(min(s, 1.0), v)

    f = scaled(ly)
    vx = f * (spec.vx_forward_max if f > 0 else spec.vx_backward_max)
    return vx, -scaled(rx) * spec.wz_max


class RemoteInput:
    def __init__(self, spec: RemoteSpec):
        self.spec = spec
        self._down_since: dict[int, float] = {}
        self._fired: set = set()     # buttons whose event already fired in this press
        self._rearm_fired = False

    def update(self, snap: RemoteSnapshot, now: float) -> RemoteEvents:
        s = self.spec
        for b in list(self._down_since):
            if b not in snap.buttons:
                del self._down_since[b]
                self._fired.discard(b)
        for b in snap.buttons:
            self._down_since.setdefault(b, now)

        def fires(b):
            t = self._down_since.get(b)
            if t is None or b in self._fired or now - t < s.debounce_s:
                return False
            self._fired.add(b)
            return True

        start, stop, reset = fires(s.button_start), fires(s.button_stop), fires(s.button_reset)
        mod = s.button_rearm_modifier
        if mod in self._down_since:
            start = False
        both = mod in self._down_since and s.button_start in self._down_since
        rearm = False
        if both:
            since = max(self._down_since[mod], self._down_since[s.button_start])
            if not self._rearm_fired and now - since >= s.rearm_hold_s:
                rearm = self._rearm_fired = True
        else:
            self._rearm_fired = False

        deflected = abs(snap.ly) > s.deadband or abs(snap.rx) > s.deadband
        vx, wz = stick_to_walk(snap.ly, snap.rx, s) if deflected else (0.0, 0.0)
        pressed = tuple(BUTTON_NAMES[i] for i in sorted(snap.buttons))
        return RemoteEvents(start, stop, reset, rearm, deflected, vx, wz, pressed)
