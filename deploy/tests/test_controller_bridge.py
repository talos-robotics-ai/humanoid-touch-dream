import math
import numpy as np
from deploy.controller_bridge import TickTimer, UiCommandFilter, standing_sample, status_message
from deploy.g1_spec import G1_SPEC
from deploy.msgs import SupervisorStatus, UiCommand
from deploy.remote import RemoteEvents, RemoteSnapshot
from deploy.standing import StandingReport
from deploy.supervisor import Inputs, State, Supervisor

def test_tick_timer_window_and_percentiles():
    t = TickTimer(window_s=1.0)
    assert t.stats(0.0) == (0.0, 0.0)
    for i, ms in enumerate([1, 2, 3, 4, 50]):
        t.add(ms / 1000.0, 0.1 * i)
    p50, mx = t.stats(0.5)
    assert p50 == 3.0 and mx == 50.0
    assert t.stats(1.35) == (50.0, 50.0)          # only the sample at 0.4 s is inside the 1 s window

def test_ui_commands_drop_stale_and_repeated():
    f = UiCommandFilter(max_age_s=0.5)
    assert f.accept(UiCommand(timestamp=100.0, nonce=1, command="start", source="ui"), 100.1) == "start"
    assert f.accept(UiCommand(timestamp=100.0, nonce=1, command="start", source="ui"), 100.2) is None   # repeated nonce
    assert f.accept(UiCommand(timestamp=100.0, nonce=2, command="stop", source="ui"), 100.9) is None    # 0.9 s old
    assert f.accept(UiCommand(timestamp=101.0, nonce=3, command="dance", source="ui"), 101.0) is None   # unknown
    assert f.accept(UiCommand(timestamp=101.0, nonce=4, command="rearm", source="cli"), 101.1) == "rearm"

def test_standing_sample_from_controller_data():
    spec = G1_SPEC.standing
    dq = np.zeros(29); dq[3] = 0.4; dq[20] = 3.0          # knee moving a little, an arm joint a lot
    s = standing_sample(True, np.array([0, 0, 0.79, 0.01, -0.02, 1.0]), np.array([0.1, 0.2, 0.0]), dq, 0.02, spec)
    assert (s.walker_live, s.torso_height, s.roll, s.pitch) == (True, 0.79, 0.01, -0.02)
    assert abs(s.gyro_norm - math.hypot(0.1, 0.2)) < 1e-12 and s.leg_dq_max == 0.4 and s.lowstate_age_s == 0.02

def test_standing_sample_with_missing_data_fails_safe():
    s = standing_sample(True, None, None, None, 0.0, G1_SPEC.standing)
    assert s.walker_live is False and s.lowstate_age_s == math.inf

def test_status_message_carries_state_flags_remote_and_timing():
    sup = Supervisor(G1_SPEC.supervisor, 10.0)
    standing = StandingReport(standing=False, held_s=1.3, failing=())
    events = RemoteEvents(start=False, stop=False, reset=False, rearm=False, deflected=True,
                          manual_vx=0.12, manual_wz=-0.2, pressed=("L1", "start"))
    snap = RemoteSnapshot(0.0, 0.5, -0.4, 0.0, frozenset({1, 2}), False)
    d = sup.step(Inputs(True, None, standing, False, True, events, (), (), False), 10.5)
    m = status_message(sup, d, standing, events, snap, False, True, True, None, True, False,
                       (2.5, 7.0), "[supervisor] BOOT -> STABILIZING: walker live", 1000.0, now_mono=11.0)
    assert isinstance(m, SupervisorStatus)
    assert (m.state, m.since_s, m.reason, m.timestamp) == (State.STABILIZING.value, 0.5, "walker live", 1000.0)
    assert (m.walker_live, m.standing, m.standing_for_s, m.policy_ready, m.frames_fresh) == (True, False, 1.3, False, True)
    assert (m.walk_enabled, m.manual_walk_enabled, m.fault) == (True, False, "")
    assert list(m.gating) == list(d.gating) and "policy not ready" in m.gating
    assert (m.stick_lx, m.stick_ly, m.stick_rx, m.stick_ry) == (0.0, 0.5, -0.4, 0.0)
    assert list(m.buttons) == ["L1", "start"] and (m.manual_vx, m.manual_wz) == (0.12, -0.2)
    assert (m.tick_p50_ms, m.tick_max_ms) == (2.5, 7.0) and m.log_line.startswith("[supervisor] BOOT")

def test_ui_command_nonces_expire_with_their_commands():
    f = UiCommandFilter(max_age_s=0.5)
    for i in range(100):
        assert f.accept(UiCommand(timestamp=100.0 + i, nonce=i, command="stop", source="ui"), 100.0 + i) == "stop"
    assert len(f._seen) <= 1                                  # only nonces still young enough to replay
    assert f.accept(UiCommand(timestamp=199.0, nonce=99, command="stop", source="ui"), 199.3) is None   # still remembered
    assert f.accept(UiCommand(timestamp=199.0, nonce=99, command="stop", source="ui"), 199.6) is None   # expired, and too old
