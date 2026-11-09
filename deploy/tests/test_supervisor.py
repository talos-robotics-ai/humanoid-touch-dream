from deploy.g1_spec import G1_SPEC
from deploy.remote import RemoteEvents
from deploy.standing import StandingReport
from deploy.supervisor import Inputs, State, Supervisor, classify_fault

STANDING = StandingReport(standing=True, held_s=2.0, failing=())
NOT_STANDING = StandingReport(standing=False, held_s=0.4, failing=("gyro",))
QUIET = RemoteEvents(start=False, stop=False, reset=False, rearm=False, deflected=False, manual_vx=0.0, manual_wz=0.0, pressed=())

def inputs(**over):
    base = dict(walker_live=True, fault=None, standing=STANDING, policy_ready=True, frame_fresh=True,
                remote=QUIET, ui=(), keyboard=(), lowstate_stale=False)
    return Inputs(**{**base, **over})

def remote(**over):
    return RemoteEvents(**{**QUIET.__dict__, **over})

def to_ready(sup, t=0.0):
    sup.step(inputs(), t)                      # BOOT -> STABILIZING
    return sup.step(inputs(), t + 0.02)        # STABILIZING -> READY

def test_boot_to_ready_logs_the_exact_line_once():
    sup = Supervisor(G1_SPEC.supervisor, 0.0)
    d = sup.step(inputs(standing=NOT_STANDING), 0.0)
    assert d.state is State.STABILIZING and "standing: gyro" in d.gating
    d = sup.step(inputs(), 0.1)
    assert d.state is State.READY and d.log[-1] == "HTDPOLICY READY TO DEPLOY" and "log_ready" in d.actions
    assert sup.step(inputs(), 0.2).log == ()

def test_ready_falls_back_when_a_condition_is_lost():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    d = sup.step(inputs(policy_ready=False), 1.0)
    assert d.state is State.STABILIZING and "policy not ready" in d.gating and "policy" in d.reason

def test_start_runs_only_from_ready():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    d = sup.step(inputs(remote=remote(start=True)), 1.0)
    assert d.state is State.RUNNING and d.actions == ("start_session",)

def test_start_outside_ready_ignored():
    sup = Supervisor(G1_SPEC.supervisor, 0.0)
    sup.step(inputs(standing=NOT_STANDING), 0.0)
    d = sup.step(inputs(standing=NOT_STANDING, ui=("start",)), 0.1)
    assert d.state is State.STABILIZING and d.log == ("[supervisor] start ignored in STABILIZING",)

def test_start_ignored_log_is_rate_limited_to_once_per_second():
    sup = Supervisor(G1_SPEC.supervisor, 0.0)
    sup.step(inputs(standing=NOT_STANDING), 0.0)
    first = sup.step(inputs(standing=NOT_STANDING, ui=("start",)), 0.1)
    again = sup.step(inputs(standing=NOT_STANDING, ui=("start",)), 0.6)
    later = sup.step(inputs(standing=NOT_STANDING, ui=("start",)), 1.2)
    assert first.log and again.log == () and later.log == first.log

def test_stop_freezes_and_needs_rearm():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    sup.step(inputs(ui=("start",)), 1.0)
    d = sup.step(inputs(remote=remote(stop=True)), 2.0)
    assert d.state is State.HOLD and d.actions == ("freeze",) and "stand" not in d.actions
    d = sup.step(inputs(keyboard=("rearm",)), 3.0)
    assert d.state is State.STABILIZING and d.actions == ("stand",)
    assert sup.step(inputs(), 3.1).state is State.READY

def test_no_auto_resume_from_hold():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    sup.step(inputs(ui=("start",)), 1.0)
    d = sup.step(inputs(frame_fresh=False), 2.0)
    assert d.state is State.HOLD and "frame" in d.reason
    assert sup.step(inputs(frame_fresh=True), 2.5).state is State.HOLD

def test_sticks_take_over_from_running_and_release_to_hold():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    sup.step(inputs(ui=("start",)), 1.0)
    d = sup.step(inputs(remote=remote(deflected=True, manual_vx=0.2)), 2.0)
    assert d.state is State.MANUAL and d.actions == ("freeze", "engage_manual")
    assert sup.step(inputs(remote=remote(deflected=True, manual_vx=0.2)), 2.1).actions == ("manual_walk",)
    assert sup.step(inputs(), 2.5).state is State.MANUAL            # sticks centred: release timer starts
    assert sup.step(inputs(), 3.4).state is State.MANUAL            # centred for 0.9 s
    d = sup.step(inputs(), 3.6)                                      # centred for 1.1 s
    assert d.state is State.HOLD and d.actions == ("freeze",)

def test_manual_from_ready_without_freeze():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    d = sup.step(inputs(remote=remote(deflected=True)), 1.0)
    assert d.state is State.MANUAL and d.actions == ("engage_manual",)

def test_manual_exits_on_stale_state():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    sup.step(inputs(remote=remote(deflected=True)), 1.0)
    d = sup.step(inputs(remote=remote(deflected=True), lowstate_stale=True), 1.1)
    assert d.state is State.HOLD and d.actions == ("freeze",) and "stale" in d.reason

def test_rearm_from_manual_requires_centred_sticks():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    sup.step(inputs(remote=remote(deflected=True)), 1.0)
    assert sup.step(inputs(remote=remote(deflected=True, rearm=True)), 1.1).state is State.MANUAL
    d = sup.step(inputs(remote=remote(rearm=True)), 1.2)
    assert d.state is State.STABILIZING and d.actions == ("freeze", "stand")

def test_reset_everywhere_it_is_allowed():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    d = sup.step(inputs(remote=remote(reset=True)), 1.0)
    assert d.state is State.STABILIZING and d.actions == ("reset",)
    sup.step(inputs(), 1.1); sup.step(inputs(ui=("start",)), 1.2)
    d = sup.step(inputs(ui=("reset",)), 2.0)
    assert d.state is State.STABILIZING and d.actions == ("freeze", "reset")

def test_fault_classes():
    assert classify_fault(None) is None
    assert classify_fault("estop_operator") == "estop" and classify_fault("lower_tilt") == "estop"
    assert classify_fault("link_jetson_silent") == "link" and classify_fault("link_watchdog") == "link"

def test_link_fault_holds_estop_latches():
    sup = Supervisor(G1_SPEC.supervisor, 0.0); to_ready(sup)
    sup.step(inputs(ui=("start",)), 1.0)
    assert sup.step(inputs(fault="link_watchdog"), 2.0).state is State.HOLD
    sup.step(inputs(keyboard=("rearm",)), 3.0)
    d = sup.step(inputs(fault="estop_operator"), 4.0)
    assert d.state is State.ESTOP and d.gating == ("restart the processes",)
    assert sup.step(inputs(keyboard=("rearm",)), 5.0).state is State.ESTOP

def test_gating_texts_in_stabilizing():
    sup = Supervisor(G1_SPEC.supervisor, 0.0)
    d = sup.step(inputs(walker_live=False, standing=StandingReport(False, 1.3, ()), policy_ready=False, frame_fresh=False), 0.0)
    assert d.state is State.BOOT
    d = sup.step(inputs(standing=StandingReport(False, 1.3, ()), policy_ready=False, frame_fresh=False), 0.1)
    assert d.gating == ("standing 1.3 / 2.0 s", "policy not ready", "no policy frames")
