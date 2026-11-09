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
    """None, "estop" (latched) or "link" (recoverable). Unknown strings fail closed to "link"."""
    if reason is None:
        return None
    if reason.startswith(("estop_", "lower_")) or reason in ("damp", "damped"):
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
    state: State
    fault_class: str | None
    start: bool
    stop: bool
    rearm: bool
    reset: bool
    released_s: float  # seconds the sticks have been centred (MANUAL only)
    release_needed: float

    @property
    def ok(self) -> bool:
        return self.i.standing.standing and self.i.policy_ready and self.i.frame_fresh


def _lost(c: _Ctx) -> str:
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
    return None


def _manual_exit(c: _Ctx) -> str | None:
    if c.released_s >= c.release_needed:
        return "sticks released for %.1f s" % c.release_needed
    if c.i.lowstate_stale:
        return "low-state stale"
    if c.fault_class == "link":
        return f"link fault: {c.i.fault}"
    return None


S, R, RU, H, M, ST, B = (State.STABILIZING, State.READY, State.RUNNING, State.HOLD,
                         State.MANUAL, State.ESTOP, State.BOOT)

# (from states, to, guard -> reason or None, entry actions); first match wins.
# The ESTOP row is handled separately in step() because it applies to every state but ESTOP.
_TABLE = (
    ({B}, S, lambda c: "walker live" if c.i.walker_live else None, lambda f: ()),
    ({S, R, H, RU}, M, lambda c: "sticks deflected" if c.i.remote.deflected else None,
     lambda f: ("freeze", "engage_manual") if f is RU else ("engage_manual",)),
    ({S}, R, lambda c: "standing %.1f s, policy ready, frames fresh" % c.i.standing.held_s if c.ok else None,
     lambda f: ("log_ready",)),
    ({R}, S, lambda c: None if c.ok else _lost(c), lambda f: ()),
    ({R}, RU, lambda c: "start" if c.start else None, lambda f: ("start_session",)),
    ({RU}, H, _running_exit, lambda f: ("freeze",)),
    ({M}, H, _manual_exit, lambda f: ("freeze",)),
    ({H}, S, lambda c: "re-arm" if c.rearm else None, lambda f: ("stand",)),
    ({M}, S, lambda c: "re-arm" if c.rearm and not c.i.remote.deflected else None, lambda f: ("freeze", "stand")),
    ({S, R, H}, S, lambda c: "reset" if c.reset else None, lambda f: ("reset",)),
    ({RU}, S, lambda c: "reset" if c.reset else None, lambda f: ("freeze", "reset")),
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
        flag = lambda name: bool(getattr(i.remote, name, False)) or name in i.ui or name in i.keyboard
        c = _Ctx(i, prev, classify_fault(i.fault), flag("start"), flag("stop"), flag("rearm"),
                 flag("reset"), released, self.spec.manual_release_s)

        target, reason, entry = None, "", ()
        if prev is not State.ESTOP and c.fault_class == "estop":
            target, reason, entry = State.ESTOP, f"estop: {i.fault}", ("freeze",)
        else:
            for froms, to, guard, actions in _TABLE:
                if prev in froms and (why := guard(c)):
                    target, reason, entry = to, why, actions(prev)
                    break

        log: tuple[str, ...] = ()
        if target is None:
            actions = ("manual_walk",) if prev is State.MANUAL else ()
            if c.start and prev is not State.READY and now - self._ignored_logged >= 1.0:
                self._ignored_logged = now
                log = (f"[supervisor] start ignored in {prev.value}",)
            reason = self.last_reason
        else:
            actions = entry
            line = f"[supervisor] {prev.value} -> {target.value}: {reason}"
            log = (line, "HTDPOLICY READY TO DEPLOY") if target is State.READY else (line,)
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
