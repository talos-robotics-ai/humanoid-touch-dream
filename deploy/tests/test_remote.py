import os, sys
import pytest
from deploy.g1_spec import G1_SPEC
from deploy.remote import RemoteInput, RemoteSnapshot, snapshot, stick_to_walk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "g1_walk"))
import paths; paths.add_teleop()  # noqa: E402
from teleop.robot_control.remote_controller import KeyMap, RemoteController, pack_wireless_remote  # noqa: E402

SPEC = G1_SPEC.remote

def snap(now_buttons=(), **sticks):
    rc = RemoteController()
    rc.set(pack_wireless_remote(buttons=now_buttons, **sticks))
    return snapshot(rc, stale=False)

def test_snapshot_decodes_teleop_block():
    s = snap((KeyMap.start, KeyMap.L1), ly=0.5, rx=-0.25)
    assert s.buttons == frozenset({KeyMap.start, KeyMap.L1}) and (s.ly, s.rx) == (0.5, -0.25)

def test_stale_snapshot_reads_as_centred():
    rc = RemoteController(); rc.set(pack_wireless_remote(ly=1.0, buttons=(KeyMap.start,)))
    s = snapshot(rc, stale=True)
    assert s.ly == 0.0 and s.buttons == frozenset() and s.stale

def test_button_press_is_debounced_edge():
    r = RemoteInput(SPEC)
    assert not r.update(snap((KeyMap.start,)), 0.00).start      # just pressed
    assert not r.update(snap((KeyMap.start,)), 0.03).start      # bounce window
    assert r.update(snap((KeyMap.start,)), 0.06).start          # held 50 ms: one event
    assert not r.update(snap((KeyMap.start,)), 0.50).start      # still held: no repeat
    r.update(snap(()), 0.60)
    assert r.update(snap((KeyMap.start,)), 0.70).start is False and r.update(snap((KeyMap.start,)), 0.76).start

def test_stop_and_reset_buttons():
    r = RemoteInput(SPEC)
    r.update(snap((KeyMap.X,)), 0.0); assert r.update(snap((KeyMap.X,)), 0.06).stop
    r.update(snap(()), 0.1)
    r.update(snap((KeyMap.Y,)), 0.2); assert r.update(snap((KeyMap.Y,)), 0.26).reset

def test_rearm_needs_l1_and_start_held_one_second():
    r = RemoteInput(SPEC)
    both = (KeyMap.L1, KeyMap.start)
    assert not r.update(snap(both), 0.0).rearm
    e = r.update(snap(both), 0.9)
    assert not e.rearm and not e.start                            # start suppressed while L1 is down
    assert r.update(snap(both), 1.0).rearm
    assert not r.update(snap(both), 1.5).rearm                    # once

def test_stick_mapping():
    assert stick_to_walk(0.0, 0.0, SPEC) == (0.0, 0.0)
    assert stick_to_walk(0.3, 0.0, SPEC) == (0.0, 0.0)                     # deadband edge
    assert stick_to_walk(1.0, 0.0, SPEC) == pytest.approx((0.30, 0.0))
    assert stick_to_walk(-1.0, 0.0, SPEC) == pytest.approx((-0.20, 0.0))
    assert stick_to_walk(0.65, 0.0, SPEC)[0] == pytest.approx(0.15)
    assert stick_to_walk(0.0, 1.0, SPEC) == pytest.approx((0.0, -0.50))    # stick right = clockwise
    assert stick_to_walk(0.0, -1.0, SPEC) == pytest.approx((0.0, 0.50))

def test_deflection_and_manual_command():
    r = RemoteInput(SPEC)
    e = r.update(snap(ly=0.8), 0.0)
    assert e.deflected and e.manual_vx == pytest.approx(stick_to_walk(0.8, 0.0, SPEC)[0])
    e = r.update(snap(ly=0.1), 0.1)
    assert not e.deflected and (e.manual_vx, e.manual_wz) == (0.0, 0.0)

def test_pressed_names_for_display():
    r = RemoteInput(SPEC)
    assert r.update(snap((KeyMap.A, KeyMap.L2, KeyMap.B)), 0.0).pressed == ("L2", "A", "B")

def test_nan_axis_fails_closed():
    rc = RemoteController(); rc.set(pack_wireless_remote(ly=0.5))
    rc.ly = float("nan")
    s = snapshot(rc, stale=False)
    assert s.ly == 0.0
    assert stick_to_walk(float("nan"), 0.5, SPEC) == (0.0, 0.0)
    e = RemoteInput(SPEC).update(snap(ly=float("nan")), 0.0)
    assert e.deflected is False and (e.manual_vx, e.manual_wz) == (0.0, 0.0)

def stale():
    return RemoteSnapshot(0.0, 0.0, 0.0, 0.0, frozenset(), True)

def test_held_start_does_not_refire_after_stale_tick():
    r = RemoteInput(SPEC)
    r.update(snap((KeyMap.start,)), 0.0)
    assert r.update(snap((KeyMap.start,)), 0.06).start
    r.update(stale(), 0.10)
    assert not any(r.update(snap((KeyMap.start,)), t).start for t in (0.15, 0.20, 0.50))
    r.update(snap(()), 0.6)
    r.update(snap((KeyMap.start,)), 0.7)
    assert r.update(snap((KeyMap.start,)), 0.76).start

def test_held_rearm_does_not_refire_after_stale_tick():
    r = RemoteInput(SPEC)
    both = (KeyMap.L1, KeyMap.start)
    r.update(snap(both), 0.0)
    assert r.update(snap(both), 1.0).rearm
    r.update(stale(), 1.1)
    assert not any(r.update(snap(both), t).rearm for t in (1.2, 2.3, 3.0))

def test_stale_period_cannot_complete_rearm_hold():
    r = RemoteInput(SPEC)
    both = (KeyMap.L1, KeyMap.start)
    r.update(snap(both), 0.0)
    r.update(stale(), 0.5)
    assert not any(r.update(snap(both), t).rearm for t in (1.2, 2.5, 3.0))

def test_start_stays_suppressed_when_l1_released_first():
    r = RemoteInput(SPEC)
    r.update(snap((KeyMap.L1, KeyMap.start)), 0.0)
    assert not r.update(snap((KeyMap.L1, KeyMap.start)), 0.06).start
    assert not r.update(snap((KeyMap.start,)), 0.10).start
    assert not r.update(snap((KeyMap.start,)), 0.30).start

def test_rearm_refires_only_after_release():
    r = RemoteInput(SPEC)
    both = (KeyMap.L1, KeyMap.start)
    r.update(snap(both), 0.0)
    assert r.update(snap(both), 1.0).rearm
    assert not r.update(snap((KeyMap.start,)), 1.1).rearm
    r.update(snap(both), 1.2)
    assert not r.update(snap(both), 2.0).rearm
    assert r.update(snap(both), 2.3).rearm

def test_stick_mapping_clips_beyond_unit():
    assert stick_to_walk(5.0, 0.0, SPEC) == pytest.approx((0.30, 0.0))
    assert stick_to_walk(-5.0, 0.0, SPEC) == pytest.approx((-0.20, 0.0))
    assert stick_to_walk(0.0, 5.0, SPEC) == pytest.approx((0.0, -0.50))
    assert stick_to_walk(0.0, -5.0, SPEC) == pytest.approx((0.0, 0.50))

def test_hand_built_nan_snapshot_is_not_deflected():
    e = RemoteInput(SPEC).update(RemoteSnapshot(0.0, float("nan"), 0.9, 0.0, frozenset(), False), 0.0)
    assert e.deflected is False and (e.manual_vx, e.manual_wz) == (0.0, 0.0)
