# Deployment Supervisor, Joystick Takeover and Monitoring UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A supervisor state machine around the `g1_walk` controller that stabilizes the robot, declares `HTDPOLICY READY TO DEPLOY`, starts and stops the HTD policy from the Unitree remote or a browser UI, hands the legs to the joystick on demand, and shows the robot and the policy's planned actions live without adding latency to the control loops.

**Architecture:** A new `deploy/` package holds pure, unit-tested logic (state machine, standing check, remote decoding, telemetry message builders, plot frame conversion) plus DDS message types. `g1_walk/controller.py` gains a `--supervisor` mode that feeds those modules once per tick and publishes `SupervisorStatus`; `policy_rollout/policy_rollout.py` gains a `--supervisor` mode that auto-starts, publishes `PolicyStatus`, `PredictedChunk` and `CameraFrame` from non-control threads, and restarts its chunk when the supervisor enters RUNNING. `deploy/ui/server.py` is a separate aiohttp process that only subscribes and serves a self-contained page.

**Tech Stack:** Python 3.10, cyclonedds (IdlStruct dataclasses, best-effort QoS), numpy, OpenCV (JPEG), aiohttp 3.14 (already in the `htd_rollout` env), plain HTML/JS/CSS with canvas plots, pytest 9.

**Spec:** `docs/superpowers/specs/2026-11-09-deploy-supervisor-ui-design.md`

## Global Constraints

- Never modify anything under `g1_walk/teleop-cbf-perception/` (pinned submodule).
- Do not modify `g1_walk/paths.py`, `policy_rollout/g1_command.py`, `g1_walk/genesis/`, `g1_walk/replay/`, `policy_rollout/chunk_schedule.py`, `policy_rollout/replay_eval.py`. `policy_rollout/head_camera.py` may only gain the additive `static` source (agreed with the other session).
- No new work on the controller's 50 Hz control thread beyond reading already-decoded fields, scalar comparisons, the supervisor step and one non-blocking queue read; nothing new on rollout's 30 Hz control thread.
- All new DDS writers and readers use `Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1), Policy.Durability.Volatile)`.
- Remote buttons A and L2+B stay the walker's; the supervisor never acts on them except by observing the resulting latch.
- Exact log line on entering READY: `HTDPOLICY READY TO DEPLOY`.
- Defaults from the spec (all in `deploy/g1_spec.py`): stand height 0.78 m, height tolerance 0.06 m, tilt 0.10 rad, gyro 0.25 rad/s, leg joint velocity 0.5 rad/s, low-state age 0.1 s, standing hold 2.0 s; stick deadband 0.3, manual vx forward 0.30 m/s, backward 0.20 m/s, wz 0.50 rad/s, manual release 1.0 s, re-arm hold 1.0 s, button debounce 0.05 s; UI command max age 0.5 s; policy timeout 0.5 s (existing `--policy_timeout`).
- `--supervisor` defaults to **off** in `controller.py` (the other session's sim scripts rely on today's behaviour) and to **False** in `policy_rollout.py`; the deploy docs pass both.
- Run every test with `~/.venvs/htd_rollout/bin/python -m pytest` from the repository root (`deploy/tests/`), unless the step says otherwise. The teleop submodule is importable through `g1_walk/paths.py`'s `add_teleop()`.
- Commit after every task with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` as the last line; commit only the files the task names (the index holds another session's staged changes: always `git commit --only -- <paths>`).

## Review Focus

1. A `UiCommand` replayed or duplicated (same nonce, or older than 0.5 s) must be ignored. Test: Task 5 (`test_controller_bridge.py::test_ui_commands_drop_stale_and_repeated`).
2. A frozen remote snapshot (low-state stale) while in MANUAL must zero the walk and leave MANUAL, never keep the last stick value. Test: Task 3 (`test_remote.py::test_stale_snapshot_reads_as_centred`) and Task 4 (`test_supervisor.py::test_manual_exits_on_stale_state`).
3. Policy frames that resume after a gap in RUNNING must not restart the policy by themselves: HOLD stays HOLD. Test: Task 4 (`test_supervisor.py::test_no_auto_resume_from_hold`).
4. Start pressed outside READY (STABILIZING, HOLD, MANUAL) is ignored and logged once. Test: Task 4 (`test_supervisor.py::test_start_outside_ready_ignored`).
5. A `PredictedChunk` with a wrong length or a non-finite value must not crash the UI server or produce a plot. Test: Task 7 (`test_plot_frames.py::test_bad_chunk_rejected`).

---

## File structure

| Path | Responsibility |
| --- | --- |
| `deploy/__init__.py` | empty |
| `deploy/msgs.py` | DDS message types and topic names shared by controller, rollout and UI |
| `deploy/dds.py` | best-effort QoS, interface configuration (CYCLONEDDS_URI), reader/writer helpers |
| `deploy/g1_spec.py` | every G1 number and button code; the robot-agnostic seam |
| `deploy/standing.py` | `StandingCheck` |
| `deploy/remote.py` | remote snapshot to events and manual walk command |
| `deploy/supervisor.py` | the state machine and fault classification |
| `deploy/controller_bridge.py` | small pure helpers the controller uses: tick timer, UI command filter, standing sample builder, status message builder |
| `deploy/telemetry.py` | rollout-side message builders: predicted chunk, camera frame, policy status, loop timer |
| `deploy/plot_frames.py` | measured state and predicted chunk into one plotting frame |
| `deploy/ui/server.py`, `deploy/ui/cmd.py`, `deploy/ui/static/{index.html,app.js,style.css}` | the UI |
| `deploy/tests/test_*.py`, `deploy/tests/sim_supervisor.sh`, `deploy/tests/ws_probe.py` | tests |
| `g1_walk/controller.py` | `--supervisor` mode |
| `policy_rollout/policy_rollout.py` | `--supervisor` mode |
| `policy_rollout/requirements-rollout.txt` | `aiohttp` |
| `docs/deployment.md`, `docs/README.md` | the deploy session and the UI |

---

### Task 1: Messages, DDS helpers and the G1 spec

**Files:**
- Create: `deploy/__init__.py`, `deploy/msgs.py`, `deploy/dds.py`, `deploy/g1_spec.py`
- Test: `deploy/tests/test_msgs.py`, `deploy/tests/test_g1_spec.py`

**Interfaces:**
- Produces, `deploy/msgs.py`: topic constants `TOPIC_SUPERVISOR = "rt/htd/supervisor"`, `TOPIC_POLICY_STATUS = "rt/htd/policy_status"`, `TOPIC_PREDICTED_CHUNK = "rt/htd/predicted_chunk"`, `TOPIC_CAMERA = "rt/htd/camera"`, `TOPIC_UI_COMMAND = "rt/htd/ui_command"`; IdlStruct dataclasses (style of `robot_control/teleop_dds_msgs.py`):
  - `SupervisorStatus(timestamp: float, state: str, since_s: float, reason: str, walker_live: bool, standing: bool, standing_for_s: float, policy_ready: bool, frames_fresh: bool, walk_enabled: bool, manual_walk_enabled: bool, fault: str, gating: sequence[str], stick_lx: float, stick_ly: float, stick_rx: float, stick_ry: float, buttons: sequence[str], manual_vx: float, manual_wz: float, tick_p50_ms: float, tick_max_ms: float, log_line: str)`
  - `PolicyStatus(timestamp: float, ready: bool, publishing: bool, inference_ms: float, chunks: int32, overrun_steps: int32, robot_state_age_s: float, eef_state_age_s: float, camera_age_s: float, loop_p50_ms: float, loop_max_ms: float, interval: int32, horizon: int32, checkpoint: str, detail: str)`
  - `PredictedChunk(timestamp: float, t_obs: int32, horizon: int32, interval: int32, inference_ms: float, body_pose: sequence[float], left_eef_to_body: sequence[float], right_eef_to_body: sequence[float], left_eef_link: sequence[float], right_eef_link: sequence[float], velocity_commands: sequence[float])` — each sequence row-major `horizon x D` with D = 6, 6, 6, 15, 15, 3.
  - `CameraFrame(timestamp: float, width: int32, height: int32, jpeg: sequence[uint8])`
  - `UiCommand(timestamp: float, nonce: int32, command: str, source: str)` — `command` in `("start", "stop", "rearm", "reset")`; module constant `UI_COMMANDS = ("start", "stop", "rearm", "reset")`.
- Produces, `deploy/dds.py`: `telemetry_qos() -> Qos` (BestEffort, KeepLast(1), Volatile); `configure_interface(iface: str | None) -> str | None` (writes the CycloneDDS XML `policy_rollout.py:_setup_dds_interface` writes, sets `CYCLONEDDS_URI` unless already set, returns the file path or None); `writer(participant, topic_name, msg_type)` and `reader(participant, topic_name, msg_type)` returning cyclonedds `DataWriter`/`DataReader` built with `Topic(participant, topic_name, msg_type)` and `telemetry_qos()`.
- Produces, `deploy/g1_spec.py`: frozen dataclasses with the spec defaults: `StandingSpec(stand_height=0.78, height_tol=0.06, tilt_max=0.10, gyro_max=0.25, leg_dq_max=0.5, lowstate_max_age=0.1, hold_s=2.0, leg_joints=tuple(range(12)))`; `RemoteSpec(button_start=2, button_stop=10, button_reset=11, button_rearm_modifier=1, deadband=0.3, vx_forward_max=0.30, vx_backward_max=0.20, wz_max=0.50, release_s=1.0, rearm_hold_s=1.0, debounce_s=0.05)` (bits are the teleop `KeyMap`: start=2, L1=1, X=10, Y=11); `SupervisorSpec(ui_command_max_age=0.5, policy_status_max_age=1.0, standing_hold_s=2.0, manual_release_s=1.0)`; `PlotSpec(eef_offset=(0.15, 0.0, 0.45), torso_height_offset=0.75)` imported from `policy_rollout/g1_command.py` (`EEF_POSITION_OFFSET`, `TORSO_HEIGHT_OFFSET`) via a `sys.path` insert of `policy_rollout/`; `G1_SPEC = G1DeploySpec(standing=..., remote=..., supervisor=..., plot=...)`.

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_msgs.py
import math
from deploy import msgs

def test_topics_are_distinct_and_namespaced():
    topics = [msgs.TOPIC_SUPERVISOR, msgs.TOPIC_POLICY_STATUS, msgs.TOPIC_PREDICTED_CHUNK,
              msgs.TOPIC_CAMERA, msgs.TOPIC_UI_COMMAND]
    assert len(set(topics)) == 5 and all(t.startswith("rt/htd/") for t in topics)

def test_messages_round_trip_through_cdr():
    chunk = msgs.PredictedChunk(timestamp=1.5, t_obs=30, horizon=2, interval=45, inference_ms=6.6,
                                body_pose=[0.0] * 12, left_eef_to_body=[0.1] * 12, right_eef_to_body=[0.2] * 12,
                                left_eef_link=[0.3] * 30, right_eef_link=[0.4] * 30, velocity_commands=[0.5] * 6)
    assert msgs.PredictedChunk.deserialize(chunk.serialize()) == chunk
    cmd = msgs.UiCommand(timestamp=2.0, nonce=7, command="start", source="ui")
    assert msgs.UiCommand.deserialize(cmd.serialize()).command == "start"
    frame = msgs.CameraFrame(timestamp=3.0, width=4, height=2, jpeg=list(b"\xff\xd8\x00\xff"))
    assert bytes(msgs.CameraFrame.deserialize(frame.serialize()).jpeg) == b"\xff\xd8\x00\xff"

def test_ui_commands_constant():
    assert msgs.UI_COMMANDS == ("start", "stop", "rearm", "reset")
```

```python
# deploy/tests/test_g1_spec.py
import os, sys
from deploy.g1_spec import G1_SPEC

def test_button_bits_match_the_teleop_keymap():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "g1_walk"))
    import paths; paths.add_teleop()
    from teleop.robot_control.remote_controller import KeyMap
    r = G1_SPEC.remote
    assert (r.button_start, r.button_stop, r.button_reset, r.button_rearm_modifier) == (KeyMap.start, KeyMap.X, KeyMap.Y, KeyMap.L1)

def test_plot_offsets_come_from_g1_command():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "policy_rollout"))
    import g1_command
    assert tuple(G1_SPEC.plot.eef_offset) == tuple(g1_command.EEF_POSITION_OFFSET)
    assert G1_SPEC.plot.torso_height_offset == g1_command.TORSO_HEIGHT_OFFSET

def test_standing_defaults():
    s = G1_SPEC.standing
    assert (s.stand_height, s.height_tol, s.tilt_max, s.gyro_max, s.leg_dq_max, s.lowstate_max_age, s.hold_s) == (0.78, 0.06, 0.10, 0.25, 0.5, 0.1, 2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_msgs.py deploy/tests/test_g1_spec.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy'`

- [ ] **Step 3: Implement `deploy/__init__.py` (empty), `deploy/msgs.py`, `deploy/dds.py`, `deploy/g1_spec.py` as specified in Interfaces**

`configure_interface` copies the XML from `policy_rollout.py:_setup_dds_interface` verbatim (temp file via `tempfile.mkstemp(suffix='.xml', prefix='cyclonedds_')`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_msgs.py deploy/tests/test_g1_spec.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add deploy/__init__.py deploy/msgs.py deploy/dds.py deploy/g1_spec.py deploy/tests/test_msgs.py deploy/tests/test_g1_spec.py
git commit --only -m "deploy: DDS messages, QoS helpers and the G1 deploy spec" -- deploy
```

---

### Task 2: Standing check

**Files:**
- Create: `deploy/standing.py`
- Test: `deploy/tests/test_standing.py`

**Interfaces:**
- Consumes: `StandingSpec` from Task 1.
- Produces: `StandingSample(walker_live: bool, torso_height: float, roll: float, pitch: float, gyro_norm: float, leg_dq_max: float, lowstate_age_s: float)` (frozen dataclass); `StandingReport(standing: bool, held_s: float, failing: tuple[str, ...])` with check names `"walker"`, `"height"`, `"tilt"`, `"gyro"`, `"legs"`, `"stale"`; `class StandingCheck: __init__(self, spec: StandingSpec)`, `update(self, sample: StandingSample, now: float) -> StandingReport`, `reset(self) -> None`. `standing` is true only when `failing == ()` and the checks have passed continuously for `spec.hold_s`; `held_s` is the time passed since the checks began passing (0.0 while failing).

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_standing.py
from deploy.g1_spec import G1_SPEC
from deploy.standing import StandingCheck, StandingSample

GOOD = dict(walker_live=True, torso_height=0.78, roll=0.0, pitch=0.02, gyro_norm=0.05, leg_dq_max=0.1, lowstate_age_s=0.02)

def test_needs_hold_time():
    c = StandingCheck(G1_SPEC.standing)
    r = c.update(StandingSample(**GOOD), 0.0)
    assert (r.standing, r.failing) == (False, ()) and r.held_s == 0.0
    r = c.update(StandingSample(**GOOD), 1.9)
    assert not r.standing and abs(r.held_s - 1.9) < 1e-9
    assert c.update(StandingSample(**GOOD), 2.0).standing

def test_each_threshold_names_its_check():
    c = StandingCheck(G1_SPEC.standing)
    bad = {"walker": dict(walker_live=False), "height": dict(torso_height=0.70), "tilt": dict(pitch=0.2),
           "gyro": dict(gyro_norm=0.3), "legs": dict(leg_dq_max=0.8), "stale": dict(lowstate_age_s=0.5)}
    for name, over in bad.items():
        r = c.update(StandingSample(**{**GOOD, **over}), 0.0)
        assert r.failing == (name,), name
    r = c.update(StandingSample(**{**GOOD, "roll": 0.2, "gyro_norm": 0.9}), 0.0)
    assert r.failing == ("tilt", "gyro")

def test_failure_restarts_the_hold():
    c = StandingCheck(G1_SPEC.standing)
    c.update(StandingSample(**GOOD), 0.0)
    c.update(StandingSample(**GOOD), 1.5)
    c.update(StandingSample(**{**GOOD, "gyro_norm": 0.5}), 1.6)
    r = c.update(StandingSample(**GOOD), 1.7)
    assert not r.standing and r.held_s == 0.0
    assert c.update(StandingSample(**GOOD), 3.7).standing

def test_height_band_is_symmetric_about_stand_height():
    c = StandingCheck(G1_SPEC.standing)
    assert c.update(StandingSample(**{**GOOD, "torso_height": 0.78 + 0.059}), 0.0).failing == ()
    c.reset()
    assert c.update(StandingSample(**{**GOOD, "torso_height": 0.78 - 0.061}), 0.0).failing == ("height",)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_standing.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.standing'`

- [ ] **Step 3: Implement `deploy/standing.py`**

Checks in the order `walker, height, tilt, gyro, legs, stale`; `tilt` fails when `abs(roll) > tilt_max or abs(pitch) > tilt_max`; `held_s` is `now - pass_since` where `pass_since` is set on the first passing update after a failure.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_standing.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git commit --only -m "deploy: standing check" -- deploy/standing.py deploy/tests/test_standing.py
```

---

### Task 3: Remote input

**Files:**
- Create: `deploy/remote.py`
- Test: `deploy/tests/test_remote.py`

**Interfaces:**
- Consumes: `RemoteSpec` from Task 1; the teleop's `RemoteController` and `pack_wireless_remote` (tests only).
- Produces: `RemoteSnapshot(lx: float, ly: float, rx: float, ry: float, buttons: frozenset[int], stale: bool)`; `snapshot(rc, stale: bool) -> RemoteSnapshot` where `rc` has `lx, ly, rx, ry, button` like the teleop's `RemoteController` (a stale snapshot has zero sticks and no buttons regardless of `rc`); `RemoteEvents(start: bool, stop: bool, reset: bool, rearm: bool, deflected: bool, manual_vx: float, manual_wz: float, pressed: tuple[str, ...])`; `stick_to_walk(ly: float, rx: float, spec: RemoteSpec) -> tuple[float, float]`; `class RemoteInput: __init__(self, spec: RemoteSpec)`, `update(self, snap: RemoteSnapshot, now: float) -> RemoteEvents`. Button names in `pressed`: `"start"`, `"X"`, `"Y"`, `"L1"`, plus `"A"`, `"B"`, `"L2"`, `"R1"`, `"R2"`, `"select"` for display.

Semantics: a button event fires once when the button has been down for `debounce_s` after being up (edge + debounce); `rearm` fires once when `L1` and `start` have both been down for `rearm_hold_s` (and the `start` event is suppressed while `L1` is down); `deflected` is `abs(ly) > deadband or abs(rx) > deadband`; `manual_vx, manual_wz` are `stick_to_walk(ly, rx)` and both 0.0 when not deflected. `stick_to_walk`: forward `ly > 0` maps `(ly - deadband) / (1 - deadband)` to `[0, vx_forward_max]`, backward to `[0, -vx_backward_max]`; `wz = -(rx sign-scaled) * wz_max` (right stick right = clockwise = negative yaw rate, the teleop's convention `wz = -rx * ...`); results clipped to the maxima.

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_remote.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_remote.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.remote'`

- [ ] **Step 3: Implement `deploy/remote.py`**

Keep per-button `down_since` times; `pressed` is ordered by bit index (`R1, L1, start, select, R2, L2, F1, F2, A, B, X, Y, up, right, down, left`) filtered to the pressed ones. `deploy/remote.py` must not import the teleop (names are a local table keyed by bit index).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_remote.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git commit --only -m "deploy: remote buttons, debounce, hold-to-rearm and stick mapping" -- deploy/remote.py deploy/tests/test_remote.py
```

---

### Task 4: The supervisor state machine

**Files:**
- Create: `deploy/supervisor.py`
- Test: `deploy/tests/test_supervisor.py`

**Interfaces:**
- Consumes: `StandingReport` (Task 2), `RemoteEvents` (Task 3), `SupervisorSpec` (Task 1).
- Produces:
  - `class State(str, Enum)`: `BOOT, STABILIZING, READY, RUNNING, HOLD, MANUAL, ESTOP`.
  - `classify_fault(reason: str | None) -> str | None`: `None` for `None`; `"estop"` when the reason starts with `estop_` or `lower_` or equals `damp`/`damped`; `"link"` otherwise (`link_*`, `jetson_silent`, anything else).
  - `Inputs(walker_live: bool, fault: str | None, standing: StandingReport, policy_ready: bool, frame_fresh: bool, remote: RemoteEvents, ui: tuple[str, ...], keyboard: tuple[str, ...], lowstate_stale: bool)` — `ui` and `keyboard` hold command names from `UI_COMMANDS` seen this tick.
  - `Decision(state: State, changed: bool, actions: tuple[str, ...], log: tuple[str, ...], gating: tuple[str, ...], reason: str)`.
  - Actions vocabulary (the controller executes them in order): `"start_session"`, `"freeze"` (walk 0, keep torso and arm commands, disarm watchdog, re-arm walk gate), `"stand"` (walker `stand()`: torso to nominal), `"engage_manual"` (walker engage, arm watchdog, resume), `"manual_walk"` (send `remote.manual_vx/manual_wz`), `"reset"` (today's `_start_reset`), `"log_ready"`.
  - `class Supervisor: __init__(self, spec: SupervisorSpec, now: float)`, attributes `state`, `since`, `last_reason`; `step(self, inputs: Inputs, now: float) -> Decision`.

Transition table (from the spec; `start`/`stop`/`rearm`/`reset` are true when the remote event, a UI command or a keyboard command of that name arrived this tick):

| From | To | Condition (first match wins) | Actions on entry |
| --- | --- | --- | --- |
| any but ESTOP | ESTOP | `classify_fault(fault) == "estop"` | `("freeze",)` |
| BOOT | STABILIZING | `walker_live` | `()` |
| STABILIZING, READY, HOLD, RUNNING | MANUAL | `remote.deflected` | RUNNING: `("freeze", "engage_manual")`; others: `("engage_manual",)` |
| STABILIZING | READY | `standing.standing and policy_ready and frame_fresh` | `("log_ready",)` |
| READY | STABILIZING | not (`standing.standing and policy_ready and frame_fresh`) | `()`; reason names what was lost |
| READY | RUNNING | `start` | `("start_session",)` |
| RUNNING | HOLD | `stop`, or `not frame_fresh`, or `classify_fault(fault) == "link"` | `("freeze",)` |
| MANUAL | HOLD | `not remote.deflected` for `release_s` (clock in the supervisor), or `lowstate_stale`, or `classify_fault(fault) == "link"` | `("freeze",)` |
| HOLD | STABILIZING | `rearm` | `("stand",)` |
| MANUAL | STABILIZING | `rearm and not remote.deflected` | `("freeze", "stand")` |
| STABILIZING, READY, HOLD | STABILIZING | `reset` | `("reset",)` (stays/returns to STABILIZING; `changed` true only if the state changed) |
| RUNNING | STABILIZING | `reset` | `("freeze", "reset")` |

Per-tick actions when no transition happens: RUNNING → `()` (the controller runs its session tick); MANUAL → `("manual_walk",)`; others `()`. `start` in any state but READY: no transition, one log line `"[supervisor] start ignored in <STATE>"` (at most once per second). The READY entry log is two lines: `"[supervisor] STABILIZING -> READY: standing 2.0 s, policy ready, frames fresh"` and `"HTDPOLICY READY TO DEPLOY"`. Every other transition logs one `"[supervisor] <FROM> -> <TO>: <reason>"`. `gating` texts: STABILIZING: for each missing condition one of `"walker not live"`, `"standing: <failing joined by ,>"` or `"standing %.1f / %.1f s"`, `"policy not ready"`, `"no policy frames"`; READY: `("press Start",)`; RUNNING: `()`; HOLD: `("re-arm: L1+Start 1 s, or Reset",)`; MANUAL: `("release the sticks",)`; ESTOP: `("restart the processes",)`.

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_supervisor.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_supervisor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.supervisor'`

- [ ] **Step 3: Implement `deploy/supervisor.py`**

One `step` that computes the merged command flags, evaluates the table in the listed order (one transition per step), applies entry actions, and builds `gating` from the state after the step. The MANUAL release timer uses `spec.manual_release_s`, the gating text `spec.standing_hold_s`. `gating` for BOOT is `("walker not live",)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests -q`
Expected: all tests in `deploy/tests` pass (`26 passed`)

- [ ] **Step 5: Commit**

```bash
git commit --only -m "deploy: supervisor state machine" -- deploy/supervisor.py deploy/tests/test_supervisor.py
```

---

### Task 5: Controller integration (`--supervisor`)

**Files:**
- Create: `deploy/controller_bridge.py`
- Modify: `g1_walk/controller.py` (argparse at 68-96; `Controller.__init__` 113-205; `_take_inputs` 251-266; `_keyboard_loop` 274-291; `_tick` 308-327; `_end_session` 377-393; `_hands_loop` 445-459; `_publish_state_loop` 460-476; `run` 501-524; `main` 549-593)
- Test: `deploy/tests/test_controller_bridge.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces, `deploy/controller_bridge.py`:
  - `class TickTimer: __init__(self, window_s: float = 1.0)`, `add(self, seconds: float, now: float) -> None`, `stats(self, now: float) -> tuple[float, float]` returning `(p50_ms, max_ms)` over the samples of the last `window_s` (both 0.0 when empty).
  - `class UiCommandFilter: __init__(self, max_age_s: float)`, `accept(self, cmd: UiCommand, now_wall: float) -> str | None` returning the command name when `cmd.command in UI_COMMANDS`, `now_wall - cmd.timestamp <= max_age_s` and the nonce was not seen before; else `None`.
  - `standing_sample(walker_live: bool, torso_pose, gyro, leg_dq, lowstate_age_s: float, spec: StandingSpec) -> StandingSample` with `torso_pose` the 6-vector `(x, y, z, roll, pitch, yaw)` from the estimator, `gyro` a 3-vector, `leg_dq` the 29 joint velocities (uses `spec.leg_joints`); any `None` input yields a sample that fails (`walker_live=False` for None torso; `lowstate_age_s=inf` for None gyro).
  - `status_message(sup: Supervisor, decision: Decision, standing: StandingReport, events: RemoteEvents, snap: RemoteSnapshot, policy_ready: bool, frame_fresh: bool, walker_live: bool, fault: str | None, walk_enabled: bool, manual_walk_enabled: bool, tick_stats: tuple[float, float], last_log: str, now_wall: float) -> SupervisorStatus`.

Controller behaviour under `--supervisor on`:
- New arguments: `--supervisor {on,off}` default `off`; `--walk {on,off}` default `on`; `--manual_walk {on,off}` default `off`; `--log_file PATH` default `None` (with supervisor on and None: `<repo>/logs/deploy_<YYYYmmdd_HHMMSS>.log`; `configure_logging` adds a `FileHandler` with the same format).
- `sys.path.insert(0, os.path.dirname(HERE))` next to the existing insert so `import deploy...` works.
- `__init__`: `self.sup = Supervisor(G1_SPEC.supervisor, time.monotonic())`, `self.standing_check = StandingCheck(G1_SPEC.standing)`, `self.remote_in = RemoteInput(G1_SPEC.remote)`, `self.ui_filter = UiCommandFilter(G1_SPEC.supervisor.ui_command_max_age)`, `self.ticks = TickTimer()`, `self._ui_cmds: list[str]` and `self._kb_cmds: list[str]` under `self._frame_lock`, `self._torso_pose = None` (set by `_publish_state_loop` from the state it computes, under `self._frame_lock`), `self._policy_status = None` (latest `PolicyStatus`, set in `_take_inputs`), `self._last_log = ""`, readers `reader_ui` (`UiCommand`) and `reader_policy` (`PolicyStatus`), writer `writer_sup` (`SupervisorStatus`), all through `deploy.dds.reader/writer` on `self.dds`.
- `_take_inputs`: also drains `reader_ui` through `ui_filter.accept(cmd, time.time())` into `_ui_cmds`, and `reader_policy` into `_policy_status`.
- `_keyboard_loop`: in supervisor mode `s` → `_kb_cmds.append("stop")`, `g` → `"rearm"`, `r` → `"reset"`, `q` quits; off mode unchanged.
- `_tick` dispatches to `_supervisor_tick(now)` when `args.supervisor == "on"`, else today's body unchanged.
- `_supervisor_tick(now)`: measure duration with `time.perf_counter()` into `self.ticks`. Build: `fault = self.fault()`; `stale = self.body.lowstate_age() > G1_SPEC.standing.lowstate_max_age`; `snap = remote.snapshot(self.body.remote_controller, stale)`; `events = self.remote_in.update(snap, now)`; `sample = standing_sample(self.walker.live, torso_pose, gyro_or_None, self.body.get_current_motor_dq(), self.body.lowstate_age(), spec)` where `gyro_or_None` is `self.body.get_imu_data().gyroscope` or `None` on `LowStateStale`; `report = self.standing_check.update(sample, now)`; `policy_ready = status is not None and status.ready and time.time() - status.timestamp <= spec.policy_status_max_age`; `frame = self._fresh_frame(now)`; `ui, kb = drain lists`; `decision = self.sup.step(Inputs(...), now)`. Log every `decision.log` line (`self.log.info`, and keep the last one in `self._last_log`). Execute actions:
  - `start_session` → `self._fresh_after = now; self._start_session(now)` but **only when a fresh frame arrives after `_fresh_after`**: until then RUNNING ticks call `_tick_between` and send walk 0 (a pending flag `self._awaiting_fresh = True`; cleared by the first fresh frame, which then calls `_start_session`). If no fresh frame arrives within `policy_timeout`, the supervisor sees `frame_fresh=False` next tick and goes to HOLD.
  - `freeze` → `self._end_session_freeze()`: new method = today's `_end_session(why, glide_home=False)` **without** `s.stand()`: `session=False; s.set_walk_command(0,0); gate.rearm(); body.disarm_link_watchdog(); body.reset(hold_pose=False)`; log line taken from the decision.
  - `stand` → `self.walker.strategy.stand()`.
  - `engage_manual` → `self.walker.engage()`, `self.body.arm_link_watchdog()`, `self.walker.strategy.resume()`.
  - `manual_walk` → `s.set_walk_command(vx, wz)` with `(events.manual_vx, events.manual_wz)` if `args.manual_walk == "on"` else `(0.0, 0.0)`.
  - `reset` → today's `_start_reset()` body minus its `_end_session` call (the `freeze` action already ran when needed).
  - `log_ready` → nothing extra (the line is in `decision.log`).
  - In RUNNING with a session: today's `_session_tick(frame, now)` but with the walk gate forced to zero when `args.walk == "off"`. In every other state: `_tick_between(now)` as today; MANUAL skips the `set_walk_command(0, 0)` that follows it.
- `_publish_state_loop`: store the computed `s["torso_pose"]` into `self._torso_pose`; every 5th iteration write `status_message(...)` to `writer_sup` (10 Hz).
- `_hands_loop`: sends hand commands only while `self.sup.state is State.RUNNING` (supervisor mode); `_open_hands` behaviour unchanged.
- `run`: unchanged `READY:` print; under supervisor the line reads `READY: g1_walk controller (supervisor): the walker is live; waiting for the robot to stand and the policy to be ready`.

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_controller_bridge.py
import math
import numpy as np
from deploy.controller_bridge import TickTimer, UiCommandFilter, standing_sample
from deploy.g1_spec import G1_SPEC
from deploy.msgs import UiCommand

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_controller_bridge.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.controller_bridge'`

- [ ] **Step 3: Implement `deploy/controller_bridge.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_controller_bridge.py -q`
Expected: `4 passed`

- [ ] **Step 5: Integrate into `g1_walk/controller.py` as specified above**

- [ ] **Step 6: Verify the controller still imports and parses both modes, and that the existing sim end-to-end test is unaffected**

Run: `~/.venvs/htd_rollout/bin/python g1_walk/controller.py --help | grep -E 'supervisor|manual_walk|log_file|--walk'`
Expected: the four options listed.
Run: `~/.venvs/htd_rollout/bin/python -c "import ast,sys; ast.parse(open('g1_walk/controller.py').read())"`
Expected: no output.
Run (on mains power, performance profile, nothing else of the stack running): `bash g1_walk/tests/sim_e2e.sh --python ~/.venvs/htd_rollout/bin/python`
Expected: last line `drive PASS, silence PASS, controller shutdown PASS, teardown PASS` (the default `--supervisor off` keeps today's behaviour).

- [ ] **Step 7: Commit**

```bash
git commit --only -m "g1_walk controller: --supervisor mode (state machine, remote, UI commands, status)" -- deploy/controller_bridge.py deploy/tests/test_controller_bridge.py g1_walk/controller.py
```

(`g1_walk/controller.py` is untracked peer work; committing it here is agreed with the other session before this task runs. If it is not, leave it unstaged and commit only the `deploy/` files; the final whole-branch commit picks it up.)

---

### Task 6: Rollout integration (`--supervisor`)

**Files:**
- Create: `deploy/telemetry.py`
- Modify: `policy_rollout/policy_rollout.py` (`__init__` 150-202; `head_camera_stream_thread` 318-340; `_robot_state_subscription_loop`/`_eef_state_subscription_loop` keep as is; `warmup_inference` 810-827; `inference_worker_thread` 891-903; `on_press_key` 1324-1344; `run` 1350-1397; argparse at 1907-1940)
- Modify (additive only): `policy_rollout/head_camera.py` (`SOURCES`, `add_head_camera_args` help, `make_head_camera` dispatch), `policy_rollout/test_head_camera.py` (one new test)
- Test: `deploy/tests/test_telemetry.py`

**Interfaces:**
- Consumes: `msgs` (Task 1).
- Produces, `deploy/telemetry.py`:
  - `chunk_message(traj: dict, t_obs: int, interval: int, inference_ms: float, stamp: float) -> PredictedChunk` where `traj` has keys `body_pose (H,6)`, `left_eef_to_body (H,6)`, `right_eef_to_body (H,6)`, `left_eef_link (H,15)`, `right_eef_link (H,15)`, `velocity_commands (H,3)` (numpy); raises `ValueError` on a missing key or wrong width.
  - `camera_message(rgb: np.ndarray, stamp: float, max_width: int = 640, quality: int = 70) -> CameraFrame`: downscale to `max_width` keeping aspect, RGB→BGR, `cv2.imencode('.jpg', ..., [cv2.IMWRITE_JPEG_QUALITY, quality])`.
  - `class LoopTimer`: same API as `TickTimer` (import and re-export `TickTimer` from `controller_bridge` as `LoopTimer`).
  - `policy_status_message(ready: bool, publishing: bool, inference_ms: float, chunks: int, overrun_steps: int, ages: tuple[float, float, float], loop_stats: tuple[float, float], interval: int, horizon: int, checkpoint: str, detail: str, stamp: float) -> PolicyStatus`.
  - `is_ready(warmed_up: bool, ages: tuple[float, float, float], last_inference_ms: float | None, budget_ms: float) -> tuple[bool, str]`: ready when warmed up, every age ≤ 1.0 s and (`last_inference_ms is None` or `≤ budget_ms`); the string names the first missing condition (`"warming up"`, `"no robot state"`, `"no hand state"`, `"no camera frame"`, `"inference slow"`) or `"ready"`.

Rollout behaviour under `--supervisor True`:
- `sys.path.insert(0, <repo root>)` next to the htd_policy insert in `mxt_loading.py`'s style, in `policy_rollout.py`.
- `self._warmed_up = True` at the end of `warmup_inference`; `self.head_frame_t = time.monotonic()` set in `head_camera_stream_thread` after `self.head_color_frame = ...`; `self._inference_ms = collections.deque(maxlen=10)` appended in `inference_worker_thread` with `1000 * seconds`.
- `run()`: when supervisor, set `self.on_rollout = True` before the loop (auto start; keys still work) and start three daemon threads: `_telemetry_status_thread` (5 Hz: `PolicyStatus` via `is_ready(...)` with `budget_ms = 0.5 * 1000 * interval / control_freq`), `_telemetry_camera_thread` (5 Hz: `camera_message(self.head_color_frame, time.time())`), `_supervisor_listener_thread` (reads `SupervisorStatus`; on an edge where the previous state was not `"RUNNING"` and the new one is, sets `self._restart_chunk.set()` (a `threading.Event`)).
- Control loop: at the top of each iteration, `if self._restart_chunk.is_set(): self._restart_chunk.clear(); self.scheduler.reset(); self.rollout_counter = 0` (two attribute writes and a flag check: negligible).
- `inference_worker_thread`: after a successful `_run_inference`, when supervisor, `self.writer_chunk.write(chunk_message(traj, t_obs, self.infer_interval, 1000 * seconds, time.time()))` inside its own try/except that prints once.
- Writers/readers through `deploy.dds.writer/reader` on `self.dds_participant`, created in `__init__` after `init_dds_publisher` when supervisor.
- New argparse: `--supervisor` (`str2bool`, default False, help `run under the g1_walk supervisor: start inference automatically, publish status, predicted chunks and camera frames, restart the chunk when the supervisor enters RUNNING`).
- `head_camera.py`: `SOURCES` gains `'static'`; `class _StaticSource(frame)` with `read()` returning a copy of the frame and `close()` doing nothing; `make_head_camera` builds it from `--head_camera_device` when that names a readable image file (`cv2.imread`, resized to `src_w x src_h`), else a mid-grey `src_h x src_w x 3` frame with a white 20-px border. Layout and rotation apply as for any source. `test_head_camera.py` gains `test_static_source_has_view_size`: `--head_camera_source static` with `resolve_head_camera_args` gives layout `duplicate`, and `make_head_camera(args, (480, 640), 30).read().shape == (480, 1280, 3)`.

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_telemetry.py
import numpy as np, pytest, cv2
from deploy.telemetry import camera_message, chunk_message, is_ready

def traj(H=60):
    return {'body_pose': np.zeros((H, 6)), 'left_eef_to_body': np.ones((H, 6)), 'right_eef_to_body': np.zeros((H, 6)),
            'left_eef_link': np.zeros((H, 15)), 'right_eef_link': np.zeros((H, 15)), 'velocity_commands': np.full((H, 3), 0.25)}

def test_chunk_message_flattens_row_major():
    m = chunk_message(traj(), t_obs=90, interval=45, inference_ms=6.6, stamp=1.0)
    assert (m.horizon, m.interval, m.t_obs) == (60, 45, 90)
    assert len(m.left_eef_link) == 900 and m.velocity_commands[:3] == [0.25, 0.25, 0.25]
    assert np.array(m.left_eef_to_body).reshape(60, 6).tolist() == np.ones((60, 6)).tolist()

def test_chunk_message_rejects_bad_shapes():
    bad = traj(); bad['velocity_commands'] = np.zeros((60, 2))
    with pytest.raises(ValueError):
        chunk_message(bad, 0, 45, 1.0, 0.0)
    with pytest.raises(ValueError):
        chunk_message({k: v for k, v in traj().items() if k != 'body_pose'}, 0, 45, 1.0, 0.0)

def test_camera_message_downscales_and_encodes_jpeg():
    rgb = np.zeros((480, 1280, 3), dtype=np.uint8); rgb[..., 0] = 200
    m = camera_message(rgb, stamp=2.0)
    assert (m.width, m.height) == (640, 240)
    img = cv2.imdecode(np.frombuffer(bytes(m.jpeg), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 640, 3) and img[0, 0, 2] > 150           # red survived (BGR order)

def test_is_ready_names_the_first_missing_condition():
    assert is_ready(False, (0.1, 0.1, 0.1), 5.0, 750.0) == (False, "warming up")
    assert is_ready(True, (2.0, 0.1, 0.1), 5.0, 750.0) == (False, "no robot state")
    assert is_ready(True, (0.1, 2.0, 0.1), 5.0, 750.0) == (False, "no hand state")
    assert is_ready(True, (0.1, 0.1, 2.0), 5.0, 750.0) == (False, "no camera frame")
    assert is_ready(True, (0.1, 0.1, 0.1), 900.0, 750.0) == (False, "inference slow")
    assert is_ready(True, (0.1, 0.1, 0.1), None, 750.0) == (True, "ready")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_telemetry.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'deploy.telemetry'`

- [ ] **Step 3: Implement `deploy/telemetry.py`**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_telemetry.py -q`
Expected: `4 passed`

- [ ] **Step 5: Integrate into `policy_rollout/policy_rollout.py` as specified**

- [ ] **Step 6: Verify the existing rollout tests and argument parsing**

Run: `cd policy_rollout && ~/.venvs/htd_rollout/bin/python -m pytest test_chunk_schedule.py test_head_camera.py test_replay_eval.py -q && ~/.venvs/htd_rollout/bin/python policy_rollout.py --help | grep -c supervisor`
Expected: tests pass (`test_head_camera.py` now has 8); `1`.

- [ ] **Step 7: Commit**

```bash
git commit --only -m "policy_rollout: --supervisor mode (auto start, status, predicted chunk, camera, chunk restart); static head-camera source" -- deploy/telemetry.py deploy/tests/test_telemetry.py policy_rollout/policy_rollout.py policy_rollout/head_camera.py policy_rollout/test_head_camera.py
```

(Same note as Task 5 about `policy_rollout.py` being untracked peer work.)

---

### Task 7: The UI server, the command-line tool and the page

**Files:**
- Create: `deploy/plot_frames.py`, `deploy/ui/__init__.py`, `deploy/ui/server.py`, `deploy/ui/cmd.py`, `deploy/ui/static/index.html`, `deploy/ui/static/app.js`, `deploy/ui/static/style.css`
- Modify: `policy_rollout/requirements-rollout.txt` (add `aiohttp>=3.9`)
- Test: `deploy/tests/test_plot_frames.py`, `deploy/tests/test_ui_server.py`

**Interfaces:**
- Consumes: `msgs`, `dds` (Task 1), `PlotSpec` (Task 1), `g1_command` (read-only).
- Produces, `deploy/plot_frames.py`:
  - `SERIES = ("left_hand_x", "left_hand_y", "left_hand_z", "right_hand_x", "right_hand_y", "right_hand_z", "left_rot_deg", "right_rot_deg", "left_fingers_mm", "right_fingers_mm", "torso_height", "vx", "wz")`.
  - `measured_point(robot_state, eef_state, spec: PlotSpec) -> dict[str, float]`: from a `G1RobotState` and an `EEFState` (or `None`): hands from `left/right_end_effector_to_torso_pose[0:3]` as measured; `*_rot_deg` = 0.0 (the measured rotation is the reference); fingers = mean distance of the five fingertips from the origin in `hand_link_poses` (right 15 then left 15, as `robot_state.py` writes it) in mm; `torso_height = torso_pose[2]`; `vx, wz = velocity_commands[0], velocity_commands[2]`.
  - `predicted_series(chunk: PredictedChunk, measured_rpy: dict[str, tuple], spec: PlotSpec) -> dict[str, list[float]] | None`: `None` when any sequence has the wrong length for `horizon` or a non-finite value; hands = `eef_to_body[:, 0:3] + spec.eef_offset`; `*_rot_deg` = angle (deg) between each predicted step's rotation (`rpy_to_rot_mat(eef_to_body[k, 3:6]) @ UNIFIED_TO_UNITREE_EEF_ROT_REMAP[side]`, `g1_command.wrist_target`) and the measured `left/right_end_effector_to_torso_pose[3:6]` rotation; fingers from `left/right_eef_link` reshaped (H, 5, 3); `torso_height = body_pose[:, 2] + spec.torso_height_offset`; `vx, wz`.
- Produces, `deploy/ui/server.py`: `class Telemetry` (DDS thread: readers for `rt/robot/state`, `rt/robot/eef_state`, the five `rt/htd/*` topics; `snapshot() -> dict` with keys `supervisor`, `policy`, `measured`, `predicted`, `ages` (seconds since each topic's last sample), `camera_seq`; `camera() -> bytes | None`; `send_command(name: str, source: str = "ui") -> int` (nonce)); `make_app(telemetry) -> aiohttp.web.Application` with routes `GET /` (index.html), `GET /static/{name}`, `GET /ws` (sends `{"type": "snapshot", ...}` JSON every `1/rate` s and a binary frame when `camera_seq` changed; accepts `{"cmd": "start"|"stop"|"rearm"|"reset"}` and answers `{"type": "ack", "cmd": ..., "nonce": n}`), `POST /cmd` (`{"cmd": ...}`, same effect); `main()` with `--iface`, `--port 8765`, `--rate 20`.
- Produces, `deploy/ui/cmd.py`: `python deploy/ui/cmd.py [--iface IFACE] start|stop|rearm|reset` publishing one `UiCommand` with `source="cli"`, printing the nonce, exit 0.
- Page (`static/`): sections per the spec: state strip with the seven states, gating list, Start (disabled unless `state == "READY"`), Stop, Re-arm and Reset (`confirm()` dialogs), `walk`/`manual walk` indicators; robot panel (torso height, roll/pitch from `measured`, applied walk, sticks as two small crosshairs, pressed buttons, `tick_p50_ms`/`tick_max_ms`, `loop_p50_ms`/`loop_max_ms`, `inference_ms`, every topic age in grey turning red past 1 s); camera `<img>` from the binary frames (`Blob` → `URL.createObjectURL`); six canvas plots over one axis −6…+2 s (left hand xyz, right hand xyz, rotation deg, fingers mm, torso height, vx/wz) with ring buffers of measured points (timestamped on arrival) and the latest prediction drawn from `t_pred_received + k/30 s` with a vertical mark at `interval/30` s; log pane (last 200 `log_line` values, de-duplicated by consecutive repeats, newest at the bottom, text filter). All colours via CSS variables, `prefers-color-scheme` honoured. No external assets.

- [ ] **Step 1: Write the failing tests**

```python
# deploy/tests/test_plot_frames.py
import math, os, sys
import numpy as np, pytest
from deploy.g1_spec import G1_SPEC
from deploy.msgs import PredictedChunk
from deploy.plot_frames import SERIES, measured_point, predicted_series

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "robot_control"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "policy_rollout"))
from teleop_dds_msgs import EEFState, G1RobotState  # noqa: E402
import g1_command  # noqa: E402

def robot_state(**over):
    z6 = [0.0] * 6
    f = dict(timestamp=0.0, torso_pose=[0, 0, 0.78, 0, 0, 0], left_end_effector_pose=z6, right_end_effector_pose=z6,
             pelvis_pose=z6, left_foot_pose=z6, right_foot_pose=z6,
             left_end_effector_to_torso_pose=[0.3, 0.2, -0.1, 0.1, 0.2, 0.3], right_end_effector_to_torso_pose=z6,
             joint_positions=[0.0] * 29, joint_velocities=[0.0] * 29, velocity_commands=[0.2, 0.0, -0.1],
             imu_quaternion=[1, 0, 0, 0], imu_accelerometer=[0, 0, 9.8], imu_gyroscope=[0, 0, 0], imu_rpy=z6[:3])
    return G1RobotState(**{**f, **over})

def chunk(H=60, **over):
    f = dict(timestamp=0.0, t_obs=0, horizon=H, interval=45, inference_ms=5.0,
             body_pose=[0, 0, 0.03, 0, 0, 0] * H, left_eef_to_body=[0.3, 0.2, -0.1, 0.1, 0.2, 0.3] * H,
             right_eef_to_body=[0.0] * (6 * H), left_eef_link=[0.1, 0, 0] * 5 * H, right_eef_link=[0.0] * (15 * H),
             velocity_commands=[0.2, 0.0, -0.1] * H)
    return PredictedChunk(**{**f, **over})

def test_measured_point_keys_and_values():
    tips = [0.08, 0, 0] * 5 + [0.06, 0, 0] * 5     # right 5 tips at 8 cm, left at 6 cm
    eef = EEFState(timestamp=0.0, joint_positions=[0.0] * 12, finger_forces=[0.0] * 12, hand_link_poses=tips)
    p = measured_point(robot_state(), eef, G1_SPEC.plot)
    assert set(p) == set(SERIES)
    assert (p["left_hand_x"], p["torso_height"], p["vx"], p["wz"]) == (0.3, 0.78, 0.2, -0.1)
    assert p["left_fingers_mm"] == pytest.approx(60.0) and p["right_fingers_mm"] == pytest.approx(80.0)

def test_prediction_is_drawn_as_the_command_rollout_sends():
    rs = robot_state()
    s = predicted_series(chunk(), {"left": rs.left_end_effector_to_torso_pose[3:6], "right": rs.right_end_effector_to_torso_pose[3:6]}, G1_SPEC.plot)
    pos, _ = g1_command.wrist_target([0.3, 0.2, -0.1, 0.1, 0.2, 0.3], "left")
    assert (s["left_hand_x"][0], s["left_hand_y"][0], s["left_hand_z"][0]) == pytest.approx(tuple(pos))
    assert s["torso_height"][0] == pytest.approx(0.03 + g1_command.TORSO_HEIGHT_OFFSET)
    assert s["left_fingers_mm"][0] == pytest.approx(100.0) and len(s["vx"]) == 60 and s["wz"][5] == -0.1

def test_rotation_error_is_zero_for_the_measured_rotation():
    # a chunk whose rotation equals the measured one, once the IK remap is undone
    rs = robot_state()
    R = g1_command.rpy_to_rot_mat([0.1, 0.2, 0.3]) @ g1_command.UNIFIED_TO_UNITREE_EEF_ROT_REMAP["left"].T
    rpy = g1_command.rot_mat_to_rpy(R)
    c = chunk(left_eef_to_body=[0.3, 0.2, -0.1, *rpy] * 60)
    s = predicted_series(c, {"left": [0.1, 0.2, 0.3], "right": [0, 0, 0]}, G1_SPEC.plot)
    assert s["left_rot_deg"][0] == pytest.approx(0.0, abs=1e-6)

def test_bad_chunk_rejected():
    assert predicted_series(chunk(velocity_commands=[0.0] * 10), {"left": [0, 0, 0], "right": [0, 0, 0]}, G1_SPEC.plot) is None
    assert predicted_series(chunk(body_pose=[math.nan] * 360), {"left": [0, 0, 0], "right": [0, 0, 0]}, G1_SPEC.plot) is None
```

```python
# deploy/tests/test_ui_server.py
import asyncio, json
from aiohttp import WSMsgType
from aiohttp.test_utils import TestClient, TestServer
from deploy.ui.server import make_app

class FakeTelemetry:
    def __init__(self):
        self.sent = []
    def snapshot(self):
        return {"supervisor": {"state": "READY", "gating": ["press Start"], "log_line": "HTDPOLICY READY TO DEPLOY"},
                "policy": {"ready": True}, "measured": {"torso_height": 0.78}, "predicted": None,
                "ages": {"supervisor": 0.05}, "camera_seq": 1}
    def camera(self):
        return b"\xff\xd8jpeg"
    def send_command(self, name, source="ui"):
        self.sent.append((name, source)); return len(self.sent)

async def _scenario():
    fake = FakeTelemetry()
    async with TestClient(TestServer(make_app(fake, rate_hz=50.0))) as client:
        r = await client.get("/")
        assert r.status == 200 and "HTD" in await r.text()
        async with client.ws_connect("/ws") as ws:
            seen_text = seen_binary = False
            for _ in range(6):
                m = await ws.receive()
                if m.type == WSMsgType.TEXT:
                    d = json.loads(m.data)
                    assert d["type"] == "snapshot" and d["supervisor"]["state"] == "READY"
                    seen_text = True
                elif m.type == WSMsgType.BINARY:
                    assert m.data == b"\xff\xd8jpeg"; seen_binary = True
                if seen_text and seen_binary:
                    break
            assert seen_text and seen_binary
            await ws.send_json({"cmd": "start"})
            while True:
                m = await ws.receive_json()
                if m.get("type") == "ack":
                    break
            assert m["cmd"] == "start" and fake.sent[-1] == ("start", "ui")
            await ws.send_json({"cmd": "dance"})
            while True:
                m = await ws.receive_json()
                if m.get("type") == "error":
                    break
            assert fake.sent[-1] == ("start", "ui")
        r = await client.post("/cmd", json={"cmd": "stop"})
        assert r.status == 200 and fake.sent[-1] == ("stop", "ui")

def test_page_snapshot_camera_and_commands():
    asyncio.run(_scenario())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests/test_plot_frames.py deploy/tests/test_ui_server.py -q`
Expected: FAIL with `ModuleNotFoundError` for `deploy.plot_frames` / `deploy.ui.server`

- [ ] **Step 3: Implement `deploy/plot_frames.py`, `deploy/ui/server.py`, `deploy/ui/cmd.py`, the static page; add `aiohttp>=3.9` to `policy_rollout/requirements-rollout.txt` under a comment `# deploy/ui/server.py (the monitoring UI)`**

`Telemetry` runs one thread polling each reader with `take(N=4)` every 5 ms and keeps the latest sample and its arrival time; `snapshot()` converts with `plot_frames` under a lock. The WebSocket handler loop: `await asyncio.sleep(1/rate)`, send JSON, send binary if the camera sequence changed; a concurrent receive task handles commands. `make_app(telemetry, rate_hz=20.0, static_dir=<deploy/ui/static>)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.venvs/htd_rollout/bin/python -m pytest deploy/tests -q`
Expected: all pass.

- [ ] **Step 5: Check the page renders with fake data**

Run: `~/.venvs/htd_rollout/bin/python deploy/ui/server.py --demo --port 8765 &` (add `--demo`: a `FakeTelemetry`-like source that animates a READY→RUNNING cycle and a synthetic chunk, so the page can be checked without the robot), open `http://127.0.0.1:8765` in a browser, confirm: state strip lights up, six plots draw a measured trace flowing into a dashed prediction, buttons send and the log pane shows the ack; then stop the server.
Expected: no console errors (check with the browser's developer tools), the page works at 1280 px wide and on a phone-width window without horizontal scrolling.

- [ ] **Step 6: Commit**

```bash
git commit --only -m "deploy UI: DDS-to-WebSocket server, command tool, self-contained page with state-into-prediction plots" -- deploy/plot_frames.py deploy/ui deploy/tests/test_plot_frames.py deploy/tests/test_ui_server.py policy_rollout/requirements-rollout.txt
```

---

### Task 8: Sim smoke test and the latency check

**Files:**
- Create: `deploy/tests/sim_supervisor.sh`, `deploy/tests/ws_probe.py`
- Depends on: Task 6 (`--head_camera_source static`).

**Interfaces:**
- `deploy/tests/ws_probe.py --port 8765 --seconds S --out probe.json`: connects to `/ws`, records every snapshot's `supervisor.state`, `supervisor.tick_p50_ms`, `supervisor.tick_max_ms`, `policy.loop_max_ms`, `supervisor.log_line`, writes them as JSON, exit 0.
- `deploy/tests/sim_supervisor.sh --model_path RUN_DIR [--python PY]`: brings up the sim host, the walker and the controller **with** `--supervisor on --walk on --manual_walk off --log_file LOG/deploy.log` (own bring-up, reusing `stop_pid` and `wait_for` by sourcing `g1_walk/tests/sim_stack.sh` without calling `stack_start`; same env as there), then rollout with `--supervisor True --head_camera_source static --dds_interface lo --send_velocity_commands True --inference_interval 45 --model_path RUN_DIR` plus the run's YAMLs, then the UI server. Script of checks, each printed `PASS`/`FAIL`:
  1. `HTDPOLICY READY TO DEPLOY` appears in `controller.log` within 90 s.
  2. 20 s with no WebSocket client, then `ws_probe.py` for 20 s: from the controller's status, `tick_max_ms` < 10 in both windows and `|tick_p50(with) - tick_p50(without)|` < 1 ms; rollout `loop_max_ms` < 16 in both.
  3. `cmd.py start` → `[supervisor] READY -> RUNNING` in the log within 2 s; 10 s later `cmd.py stop` → `RUNNING -> HOLD`; `cmd.py rearm` → `HOLD -> STABILIZING` then `STABILIZING -> READY` within 10 s; `cmd.py reset` → `arms gliding home` in the log.
  4. Teardown clean (every process stopped; `rt/lowcmd` writers gone).
  Exit 0 only if all pass. Logs under `deploy/tests/logs/<time>/` (gitignored: add `deploy/tests/logs/` to the root `.gitignore`).

- [ ] **Step 1: Write `ws_probe.py` and `sim_supervisor.sh`**

- [ ] **Step 2: Run it (mains power, performance profile, nothing else of the stack running)**

Run: `bash deploy/tests/sim_supervisor.sh --model_path ~/htd_models/rack_white_becker_HTD --python ~/.venvs/htd_rollout/bin/python`
Expected: four `PASS` lines and exit 0. If check 1 fails on the standing thresholds in the sim (the sim walker's stance height), tune `StandingSpec` within ±0.02 of the defaults, note the change in the spec document, and re-run.

- [ ] **Step 3: Record the latency numbers**

Append the measured `tick_p50/max` with and without the UI, and rollout's `loop_max_ms`, to `docs/deployment.md` (Task 9 writes the section; keep the numbers in `deploy/tests/logs/<time>/summary.json` meanwhile).

- [ ] **Step 4: Commit**

```bash
git commit --only -m "deploy: MuJoCo smoke test of the supervisor, UI and latency budget" -- deploy/tests/sim_supervisor.sh deploy/tests/ws_probe.py .gitignore
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/deployment.md` (rewrite), `docs/README.md` (pipeline table row, environments), `README.md` (one line in the fork table: Deployment entry point `deploy/`)

Content of `docs/deployment.md`: the four-terminal session for MuJoCo and for the robot (walker, controller `--supervisor on`, rollout `--supervisor True --inference_interval 45 --head_camera_source realsense --head_camera_layout duplicate`, UI server), the state table and transitions (copied from the spec, shortened), the button map, the `HTDPOLICY READY TO DEPLOY` line, what stop/hold means (freeze in place), re-arm vs reset, manual walking and the gantry check of the stick axes before `--manual_walk on`, the UI sections and the plot's frames, the latency numbers from Task 8, troubleshooting (each gating text and what clears it). Mention that `g1_walk/README.md` keeps the architecture and the sim details.

- [ ] **Step 1: Write the docs**
- [ ] **Step 2: Run the link checker used earlier (relative links and anchors resolve)**
- [ ] **Step 3: Commit**

```bash
git commit --only -m "docs: deployment with the supervisor, buttons, joystick and UI" -- docs/deployment.md docs/README.md README.md
```

---

## Self-review notes

- Spec coverage: states and transitions (Task 4), standing (Task 2), policy ready (Task 6 `is_ready`), log line (Task 4), buttons and sticks (Task 3, Task 5), fresh chunk on start (Task 6 listener + Task 5 `_fresh_after`), freeze on stop (Task 5 `freeze`), re-arm/reset (Tasks 4, 5), UI with plots, camera, log, buttons (Task 7), latency budget and verification (Tasks 5, 6, 8), docs (Task 9), robot-agnostic seam (`g1_spec.py`, Task 1). Not covered on purpose: items under "Later" in the spec.
- Names used across tasks: `State`, `Inputs`, `Decision`, `Supervisor.step`, `StandingCheck.update`, `StandingSample`, `StandingReport`, `RemoteInput.update`, `RemoteSnapshot`, `RemoteEvents`, `snapshot`, `stick_to_walk`, `TickTimer`, `UiCommandFilter.accept`, `standing_sample`, `status_message`, `chunk_message`, `camera_message`, `is_ready`, `policy_status_message`, `measured_point`, `predicted_series`, `SERIES`, `make_app`, `Telemetry`, topic constants and message classes from Task 1.
