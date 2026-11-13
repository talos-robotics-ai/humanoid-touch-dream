"""Deployment supervisor: one state-machine step per control tick."""
from dataclasses import dataclass
from enum import Enum

from deploy.g1_spec import SupervisorSpec
from deploy.remote import RemoteEvents
from deploy.standing import StandingReport


class State(str, Enum):
    BOOT = "BOOT"
    STABILIZING = "STABILIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    MANUAL = "MANUAL"
    ESTOP = "ESTOP"


def classify_fault(reason: str | None) -> str | None:
    """None, "estop" (latched) or "link" (recoverable). Unknown strings fail closed to "link".

    A silent walker (`lower_jetson_silent_<age>s`, derived from the link's age) clears when its
    status returns, so it is a link fault; an `estop_*` is latched in the body whatever its reason."""
    if reason is None:
        return None
    if reason.startswith("estop_") or reason in ("damp", "damped"):
        return "estop"
    if reason.startswith("lower_") and "silent" not in reason:
        return "estop"
    return "link"


@dataclass(frozen=True)
class Inputs:
    walker_live: bool
    fault: str | None
    standing: StandingReport
    policy_ready: bool
    frame_fresh: bool
    remote: RemoteEvents
    ui: tuple[str, ...]
    keyboard: tuple[str, ...]
    lowstate_stale: bool
    # What RUNNING watches besides the frames: the policy's inputs are fresh (its robot state,
    # hand state and head camera, as its status reports them) and the robot's tilt
    # (max |roll|, |pitch|; NaN when unknown, which fails closed).
    observations_ok: bool
    tilt: float


@dataclass(frozen=True)
class Decision:
    state: State
    changed: bool
    actions: tuple[str, ...]
    log: tuple[str, ...]
    gating: tuple[str, ...]
    reason: str


@dataclass
class _Ctx:
    i: Inputs
    fault_class: str | None
    start: bool
    stop: bool
    rearm: bool
    reset: bool
    released_s: float  # seconds the sticks have been centred (MANUAL only)
    spec: SupervisorSpec

    @property
    def ok(self) -> bool:
        return (self.i.standing.standing and self.i.policy_ready and self.i.frame_fresh
                and self.fault_class is None)


def _lost(c: _Ctx) -> str:
    if c.fault_class is not None:
        return f"fault: {c.i.fault}"
    if not c.i.standing.standing:
        return "standing lost: " + ",".join(c.i.standing.failing) if c.i.standing.failing else "standing lost"
    if not c.i.policy_ready:
        return "policy not ready"
    return "policy frames stopped"


def _running_exit(c: _Ctx) -> str | None:
    if c.stop:
        return "stop"
    if not c.i.frame_fresh:
        return "policy frames stopped"
    if c.fault_class == "link":
        return f"link fault: {c.i.fault}"
    if c.i.lowstate_stale:
        return "low-state stale"
    if not c.i.tilt <= c.spec.running_tilt_max:       # `not x <= limit` so NaN fails closed
        return "tilt %.2f rad > %.2f" % (c.i.tilt, c.spec.running_tilt_max)
    if not c.i.observations_ok:
        return "policy observations stale"
    return None


def _manual_exit(c: _Ctx) -> str | None:
    if c.released_s >= c.spec.manual_release_s:
        return "sticks released for %.1f s" % c.spec.manual_release_s
    if c.i.lowstate_stale:
        return "low-state stale"
    if c.fault_class == "link":
        return f"link fault: {c.i.fault}"
    return None


# (from states, to, guard -> reason or None, entry actions); first match wins.
# The ESTOP row is handled separately in step() because it applies to every state but ESTOP.
def _manual_entry(c: _Ctx) -> str | None:
    # Not while an exit condition holds, or HOLD <-> MANUAL would alternate every tick.
    if c.i.remote.deflected and not c.i.lowstate_stale and c.fault_class != "link":
        return "sticks deflected"
    return None


_TABLE = (
    ({State.BOOT}, State.STABILIZING, lambda c: "walker live" if c.i.walker_live else None, lambda f: ()),
    ({State.STABILIZING, State.READY, State.HOLD, State.RUNNING}, State.MANUAL, _manual_entry,
     lambda f: ("freeze", "engage_manual") if f is State.RUNNING else ("engage_manual",)),
    ({State.STABILIZING}, State.READY,
     lambda c: "standing %.1f s, policy ready, frames fresh" % c.i.standing.held_s if c.ok else None,
     lambda f: ("log_ready",)),
    # Stop first: a stop beats a start (or a reset) in the same tick, and the start that follows
    # it is ignored in HOLD, which only a deliberate re-arm leaves.
    ({State.READY}, State.HOLD, lambda c: "stop" if c.stop else None, lambda f: ("freeze",)),
    # Reset before the lost row: a reset in the tick a condition drops must still reset.
    ({State.READY}, State.STABILIZING, lambda c: "reset" if c.reset else None, lambda f: ("reset",)),
    ({State.READY}, State.STABILIZING, lambda c: None if c.ok else _lost(c), lambda f: ()),
    ({State.READY}, State.RUNNING, lambda c: "start" if c.start else None, lambda f: ("start_session",)),
    ({State.RUNNING}, State.HOLD, _running_exit, lambda f: ("freeze",)),
    ({State.MANUAL}, State.HOLD, _manual_exit, lambda f: ("freeze",)),
    ({State.HOLD}, State.STABILIZING, lambda c: "re-arm" if c.rearm else None, lambda f: ("stand",)),
    ({State.MANUAL}, State.STABILIZING,
     lambda c: "re-arm" if c.rearm and not c.i.remote.deflected else None, lambda f: ("freeze", "stand")),
    ({State.STABILIZING, State.HOLD}, State.STABILIZING,
     lambda c: "reset" if c.reset else None, lambda f: ("reset",)),
    ({State.RUNNING}, State.STABILIZING, lambda c: "reset" if c.reset else None, lambda f: ("freeze", "reset")),
)


class Supervisor:
    def __init__(self, spec: SupervisorSpec, now: float):
        self.spec = spec
        self.state = State.BOOT
        self.since = now
        self.last_reason = "boot"
        self._centred_since: float | None = None
        self._ignored_logged = float("-inf")

    def step(self, inputs: Inputs, now: float) -> Decision:
        prev = self.state
        if prev is State.MANUAL:
            if inputs.remote.deflected:
                self._centred_since = None
            elif self._centred_since is None:
                self._centred_since = now
        released = 0.0 if self._centred_since is None else now - self._centred_since
        i = inputs

        def flag(name: str, remote_event: bool) -> bool:
            return remote_event or name in i.ui or name in i.keyboard

        r = i.remote
        c = _Ctx(i, classify_fault(i.fault), flag("start", r.start), flag("stop", r.stop),
                 flag("rearm", r.rearm), flag("reset", r.reset), released, self.spec)

        target, reason, entry = None, "", ()
        if prev is not State.ESTOP and c.fault_class == "estop":
            target, reason, entry = State.ESTOP, f"estop: {i.fault}", ("freeze",)
        else:
            for froms, to, guard, actions in _TABLE:
                if prev in froms and (why := guard(c)):
                    target, reason, entry = to, why, actions(prev)
                    break

        log: tuple[str, ...] = ()
        if c.start and "start_session" not in entry and now - self._ignored_logged >= 1.0:
            self._ignored_logged = now
            log = (f"[supervisor] start ignored in {prev.value}",)
        if target is None:
            actions = ("manual_walk",) if prev is State.MANUAL else ()
            reason = self.last_reason
        else:
            actions = entry
            line = f"[supervisor] {prev.value} -> {target.value}: {reason}"
            log += (line, "HTDPOLICY READY TO DEPLOY") if target is State.READY else (line,)
            self.last_reason = reason
            if target is not prev:
                self.state, self.since = target, now
                self._centred_since = None

        return Decision(self.state, self.state is not prev, actions, log,
                        self._gating(i), reason)

    def _gating(self, i: Inputs) -> tuple[str, ...]:
        s = self.state
        if s is State.BOOT:
            return ("walker not live",)
        if s is State.STABILIZING:
            out = []
            if not i.walker_live:
                out.append("walker not live")
            if i.fault is not None:
                out.append(f"fault: {i.fault}")
            if not i.standing.standing:
                out.append("standing: " + ",".join(i.standing.failing) if i.standing.failing
                           else "standing %.1f / %.1f s" % (i.standing.held_s, self.spec.standing_hold_s))
            if not i.policy_ready:
                out.append("policy not ready")
            if not i.frame_fresh:
                out.append("no policy frames")
            return tuple(out)
        return {State.READY: ("press Start",), State.RUNNING: (),
                State.HOLD: ("re-arm: L1+Start 1 s, or Reset",),
                State.MANUAL: ("release the sticks",), State.ESTOP: ("restart the processes",)}[s]
