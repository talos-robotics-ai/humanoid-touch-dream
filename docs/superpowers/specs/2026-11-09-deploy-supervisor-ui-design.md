# Deployment supervisor, joystick takeover and monitoring UI

Design, 2026-11-09. Status: decisions A and B taken by the user; awaiting approval of the whole.

## Intent

Deploying a trained HTD policy on the G1 today means starting the walker, the
`g1_walk` controller and `policy_rollout.py` in three terminals and pressing keys in two
of them; the controller starts driving the robot on the first policy frame it sees. The
operator has no confirmation that the robot is stable or that the policy is actually
ready, no single button to start or stop, no way to move the robot by hand without
killing processes, and nothing that shows what the policy is about to do.

This design adds a supervisor state machine around the existing stack, a Unitree-remote
and UI button interface to it, joystick takeover of the walking policy, and a browser UI
that shows the state machine, the robot and the policy's planned actions, without adding
latency to the control loops.

Everything runs on the operator's laptop (RTX 5070): the walker, the controller, rollout,
the UI server and the browser. No computation on the robot's Jetson. First in MuJoCo
through the existing `g1_walk` sim stack, then on the robot.

## Goals

1. A supervisor that (a) stabilizes the robot on the `g1_walk` walking policy, (b)
   confirms the robot is standing correctly, (c) confirms the HTD transformer is loaded,
   warmed up and producing frames, and only then (d) writes the line
   `HTDPOLICY READY TO DEPLOY` to the log.
2. One **start** button that lets the policy drive the robot once ready, and one **stop**
   button that stops the policy while the robot keeps standing: arms and torso hold their
   last targets, hands keep their grip, walk goes to zero, the walker keeps balancing.
   The same holds when the policy finishes or its frames stop: nothing is switched off.
3. Joystick access to the walking policy at any time, for moving the robot by hand.
4. A UI to monitor the robot while the state machine runs: the state machine, the robot
   state, the camera, the log, start/stop buttons, and a time plot per command group in
   which the measured state over the last seconds flows into the policy's predicted
   2-second action chunk.
5. The telemetry path to the UI adds no measurable latency to the control loops.

## Non-goals

- Changing the walking policy or anything under `g1_walk/teleop-cbf-perception` (pinned
  submodule, never modified).
- Collision checking of the arm targets, workspace boxes, perception-based safety. The
  existing guards stay (jump guard, speed cap, walker tilt abort, link watchdog); new
  checks are listed under "Later" and are not part of this build.
- Task-success detection. HTD has no notion of a finished task; "the policy finishes"
  means the operator stops it or its frames stop.
- Remote (off-laptop) UI access. The browser runs on the laptop. The design does not
  prevent a tablet on the same network later.

## Anti-goals

- Any new work on the controller's 50 Hz control thread or rollout's 30 Hz control
  thread beyond reading already-decoded fields and a few scalar comparisons. If the UI
  slows the robot, the design has failed.
- A UI that can start the robot from a state the supervisor did not declare ready.
- A stop that releases the arms or the legs. Stop means hold.
- Hidden automatic resumes: nothing goes from HOLD or MANUAL back to RUNNING without
  the operator's explicit action.

## Constraints

- Three processes already exist and keep their roles: the walker (pinned teleop code,
  only writer of `rt/lowcmd`), `g1_walk/controller.py` (only process commanding the
  walker over its UDP link), `policy_rollout/policy_rollout.py` (inference, publishes
  `TeleopDataWB`). DDS (CycloneDDS, domain 0) is the bus between them.
- The Unitree remote is read by the walker's body for A (freeze and stand, latched) and
  L2+B (damp). Those two stay as they are. The controller's body object already decodes
  the same remote block (`body.remote_controller`: `lx, ly, rx, ry, button[16]`, key map
  in the teleop's `remote_controller.py`); the sticks and the other buttons are unused
  today.
- The laptop may be on the robot's network with no internet: the UI page is
  self-contained (no CDN, no external fonts or scripts).
- Python 3.10 in the `htd_rollout` environment layered on the teleop environment; new
  dependencies are kept to one small pure-Python web package.
- The other session plans a robot-agnostic split of `g1_walk`. Robot specifics introduced
  here (topics, button codes, limits, frame offsets) live in one place so they can move.

## Architecture

```text
 laptop                                                       robot (or MuJoCo sim host)
 ┌─────────────────────────────┐
 │ policy_rollout.py           │  TeleopDataWB (30 Hz, existing) ─────┐
 │  inference thread ──► PredictedChunk (per inference, new)         │
 │  status thread    ──► PolicyStatus (5 Hz, new)                    │
 │  camera thread    ──► CameraFrame JPEG (≤5 Hz, new)               ▼
 └─────────────────────────────┘                     ┌──────────────────────────────┐
                                                     │ g1_walk/controller.py         │  UDP link   ┌──────────┐  rt/lowcmd  ┌─────────┐
 ┌─────────────────────────────┐   UiCommand (new)   │  + deploy.supervisor FSM      ├────────────►│ walker   ├────────────►│ G1 / sim│
 │ deploy/ui/server.py          ├────────────────────►│  + deploy.remote (sticks,     │             │ (pinned) │◄────────────┤         │
 │  DDS readers ─► WebSocket    │◄────────────────────┤    buttons) from the body's   │◄────────────┤          │  rt/lowstate│         │
 │  (≈10 Hz) ─► browser         │  SupervisorStatus   │    rt/lowstate decode         │   status    └──────────┘  (+remote)  └─────────┘
 └─────────────────────────────┘  (10 Hz, new) +      │  publishes rt/robot/state,    │
                                   existing topics    │  eef_state, touch (existing)  │   Modbus ──► Inspire hands
                                                     └──────────────────────────────┘
```

Rollout runs inference continuously once it has observations (new `--auto_start`,
default on when `--supervisor` is set) and publishes frames; it does not decide when
the robot moves. The controller's supervisor decides: frames are **received** in every
state and **applied** only in RUNNING. In READY and HOLD the policy therefore runs in
shadow: the transformer is exercised on live observations, its chunks reach the UI plot,
and nothing moves.

The UI server is a separate process that only subscribes. It can be killed or frozen at
any time without any effect on the robot.

### New package: `deploy/`

| Module | Content |
| --- | --- |
| `deploy/msgs.py` | the new DDS message types (cyclonedds `IdlStruct` dataclasses, same style as `robot_control/teleop_dds_msgs.py`) and topic names |
| `deploy/supervisor.py` | the state machine: pure Python, no I/O, no threads; fed events and a clock, returns the actions to take. Fully unit-tested |
| `deploy/standing.py` | the "standing correctly" check: pure function of (walker live, torso pose, IMU, joint velocities, time) |
| `deploy/remote.py` | remote snapshot (`lx, ly, rx, ry, button[]`) to events: button presses with debounce and hold timers, stick deflection to manual walk command |
| `deploy/g1_spec.py` | every G1-specific number in one file: button map, stick mapping and limits, standing thresholds, frame offsets for the plot, topic names (the seam for the robot-agnostic split) |
| `deploy/ui/server.py` | DDS to WebSocket bridge and static file server |
| `deploy/ui/static/` | one HTML page, one JS file, one CSS file; plots drawn on canvas; no external assets |
| `deploy/tests/` | unit tests for supervisor, standing, remote, frame conversion; a sim smoke test script |

The controller imports `deploy.supervisor`, `deploy.standing`, `deploy.remote`,
`deploy.g1_spec`, `deploy.msgs`. Rollout imports `deploy.msgs` only. The repository root
goes on `sys.path` the way `g1_walk/paths.py` already does for other directories.

## The supervisor

### States

| State | Meaning | Robot behaviour (what the controller sends) |
| --- | --- | --- |
| `BOOT` | process started, walker link not yet live | nothing (today's launch glide of the arms to home runs as before) |
| `STABILIZING` | walker handover done; waiting for the robot to be still and upright | arms at home (hold), walk 0, torso nominal |
| `READY` | standing confirmed and policy ready; **log line `HTDPOLICY READY TO DEPLOY`** on entry | arms hold, walk 0, torso nominal; policy frames received, not applied |
| `RUNNING` | the policy drives the robot | today's session tick: IK, shaping, arms, walk through the gate, torso, hands |
| `HOLD` | stopped by the operator, by frame silence, by a fault, or by the policy process ending | **freeze in place**: arms and torso stay at the last **sent** targets (walk command set to zero, no walker `stand()`), hands hold their last command. Nothing moves on a stop |
| `MANUAL` | the operator drives the legs with the sticks | arms and torso hold as in HOLD; walk from the sticks; hands hold |
| `ESTOP` | the walker's own latch tripped (remote A, L2+B, walker fault, link fault) | nothing is commanded (as today: the walker froze the arms and stands, or damped). Exit only by restarting the processes, as today |

### Transitions

| From | To | On |
| --- | --- | --- |
| BOOT | STABILIZING | walker handover completed (`walker.live`) |
| STABILIZING | READY | `standing` has been true for `standing_hold_s` (default 2.0 s) **and** `policy_ready` **and** a fresh policy frame arrived within `policy_timeout` **and** no fault is asserted |
| READY | STABILIZING | standing lost, or policy frames stop, or policy status not ready, or a fault appears; the log says which |
| READY | RUNNING | **start**: remote `Start` button, or UI start command |
| RUNNING | HOLD | **stop**: remote `X` button, or UI stop; or no fresh frame for `policy_timeout` (0.5 s, as today); or the policy's status says it exited; or the arm jump guard has refused commands for `ARM_HOLD_MAX` frames |
| RUNNING | MANUAL | any stick deflection beyond the deadband, while the low-state is fresh and no link fault is asserted (otherwise the next row applies) |
| READY, HOLD, STABILIZING | MANUAL | any stick deflection beyond the deadband, while the low-state is fresh and no link fault is asserted |
| MANUAL | HOLD | sticks back inside the deadband for `manual_release_s` (1.0 s), or the robot's low-state older than 0.1 s (the remote snapshot is then unknown: walk goes to zero first). The policy does **not** resume |
| HOLD | STABILIZING | **re-arm**: remote `L1 + Start` held for 1 s, or UI "Re-arm" (confirmed). The torso is slewed back to nominal (the walker's `stand()`), the arms and hands stay where they are; READY is reached again only through the standing and readiness checks, and the policy restarts from a fresh chunk |
| HOLD, READY, STABILIZING, RUNNING | STABILIZING | **reset**: remote `Y`, UI "Reset" (confirmed), or rollout's `ResetCommand` (its key 1): today's reset, arms glide home, hands open, torso nominal. In RUNNING it stops first (HOLD) and then resets |
| any | ESTOP | the walker's own latches: `estop_*` (remote A, damping), `lower_*` (composer trip, tilt abort). Link faults (`link_*`, a silent walker) go to HOLD instead and are cleared by re-arm, as today's session start clears them |

Guards and details:

- Start is accepted in READY only. A start pressed elsewhere is logged and ignored.
- Stop is accepted everywhere it means something (RUNNING, MANUAL goes to HOLD when
  sticks are centred). Stop never does anything the walker's A button does: it does not
  freeze the walker's own arm reference or stand through the e-stop path. It ends the
  session like today's `s` key (`_end_session(glide_home=False)`) **except** that the
  walker's `stand()` is not called: the walk command goes to zero and the torso command
  stays (decision A). The link watchdog is disarmed and the walk gate re-armed as today.
- "Policy frames stop" during RUNNING goes to HOLD (today: session ends with arms held,
  walk zero). This is the "policy finishes" case: stability is maintained, nothing is
  deactivated.
- Entering RUNNING from READY uses today's session start (`_start_session`): walker
  engage, arm engage glide from the current arm command onto the policy's solution,
  torso engage glide, walk gate re-armed (the walk starts only after the policy has
  been seen asking for rest, as today).
- The robot executes only chunks planned after start (my call): rollout subscribes to
  `SupervisorStatus`; on the edge into RUNNING it resets its chunk scheduler and infers
  at once, and the controller ignores frames whose `chunk_stamp` (new field of
  `TeleopDataWB`'s companion `PredictedChunk`, carried as the frame's `timestamp`
  origin) precedes the start instant. Until that first fresh frame arrives (about one
  inference, 10-50 ms) RUNNING holds like READY.
- The walking command can be disabled for a whole deployment (`--walk off`, default
  on): in RUNNING the walk gate then sends zero regardless of the policy. The UI shows
  the setting; it cannot be changed while RUNNING.

### "Standing correctly" (`deploy/standing.py`)

All of the following, evaluated every control tick from data the controller already
has, true continuously for `standing_hold_s` = 2.0 s:

| Check | Source | Threshold (default, `g1_spec.py`) |
| --- | --- | --- |
| walker live | `walker.live` (phase `live`) | required |
| torso height | `estimate` torso pose z (feet frame) | within ±0.06 m of the stand height 0.78 m |
| torso tilt | torso pose roll, pitch | each within ±0.10 rad |
| body at rest | IMU gyroscope norm | < 0.25 rad/s |
| legs at rest | joint velocities of the 12 leg joints | max < 0.5 rad/s |
| state fresh | the body's low-state age | < 0.1 s |

The check is also evaluated in READY and HOLD; losing it in READY returns to
STABILIZING. In RUNNING it is not a stop condition (the robot walks and bends by
design); the walker's own tilt abort (1.0 rad) remains the fall guard.

### "Policy ready" (`PolicyStatus` from rollout)

Rollout publishes `PolicyStatus` at 5 Hz from a background thread: `ready` is true when
the checkpoint is loaded, the warm-up and CUDA-graph capture are done, a robot state,
hand state and camera frame have each arrived within the last second, and the last
inference finished within the chunk budget. Also carried: inference time (ms, median of
the last 10), chunk count, overrun steps, and `publishing` (frames are being sent).
`ready` must be true and the status itself fresh (< 1 s) for the supervisor's
`policy_ready`.

### The log

The controller logs every transition as
`[supervisor] <FROM> -> <TO>: <reason>`. On entering READY it logs exactly
`HTDPOLICY READY TO DEPLOY` on its own line (plus the usual timestamp prefix), and the
same line is repeated in the UI log pane. `--log_file PATH` (new) adds a file handler so
the deployment has a record; default `logs/deploy_<YYYYmmdd_HHMMSS>.log` under the
repository, created only when `--supervisor` is on.

### Decisions taken by the user (2026-11-09)

**A. Torso after a stop: freeze in place.** A stop sets the walk command to zero and
leaves the torso and arm commands exactly where they were; the walker keeps tracking
them and balancing. Today's `stand()` slew back to nominal is **not** called on stop.
A robot bent over a table stays bent. Consequence: the standing check (nominal height)
cannot pass in HOLD; re-arm brings the torso back to nominal first.

**B. Re-arm leaves arms and hands; Reset is a separate action.** Re-arm (HOLD to
STABILIZING) slews the torso to nominal and re-checks standing and readiness, with the
arms and hands untouched, so the policy can be restarted while the robot holds an
object. Reset (remote `Y`, UI button, or rollout's `ResetCommand`) is today's reset:
arms glide home, hands open, torso nominal. With `--auto_start` nothing resets before
the first run: the arms are at home from the launch glide, the hands in whatever state
they were; press Reset if they should be opened first.

## Buttons and sticks (`deploy/remote.py`, `deploy/g1_spec.py`)

Read on the controller's existing low-state path (`body.remote_controller`, already
updated at the rt/lowstate rate); one snapshot per control tick, a few comparisons.

| Input | Action | Where accepted |
| --- | --- | --- |
| `Start` | start the policy | READY |
| `X` | stop (HOLD) | RUNNING |
| `L1 + Start` held 1.0 s | re-arm | HOLD, MANUAL (sticks centred) |
| `Y` | reset: arms home, hands open, torso nominal | HOLD, READY, STABILIZING; RUNNING (stops first) |
| left stick y (`ly`) | forward speed | any state except ESTOP: enters MANUAL |
| right stick x (`rx`) | turn rate | same |
| `A`, `L2 + B` | the walker's own freeze / damp | unchanged (walker); the supervisor goes to ESTOP when it sees the latch |

Buttons are edge-triggered with a 50 ms debounce; the hold timer for re-arm is in the
supervisor's clock, not a thread.

Entering MANUAL engages the walker and arms the link watchdog exactly as a session
start does (my call): the operator driving the legs is a session without a policy. A
stale low-state anywhere is read as centred sticks and no buttons.

Stick mapping (my call): deadband 0.3 on each axis (the teleop's deadband), then
proportional from the deadband edge to full deflection, scaled to
`manual_vx_max` = 0.30 m/s forward, 0.20 m/s backward, `manual_wz_max` = 0.50 rad/s,
all clipped to the walker's command box. Slower than the teleop's 0.55 m/s bang-bang
mapping on purpose: this is for repositioning by hand, not for walking fast. The walker
slews and clamps as it does for every command. The walk gate is bypassed in MANUAL
(the operator is the one asking for motion).

The pinned teleop notes that the remote's `ly` byte offset was never verified against
the Unitree struct, and MuJoCo cannot exercise the remote. Manual walking therefore
ships behind `--manual_walk` (default off): with it off, stick deflection still enters
MANUAL (policy stopped, arms hold) but sends zero walk, and the UI shows the live stick
values and buttons so the axes can be checked on the gantry. Turning it on is the first
step of the robot session, after that check (my call).

The UI mirrors `start`, `stop`, `re-arm` and `reset` (the last two ask for confirmation in
the page), and so does the command-line tool `deploy/ui/cmd.py`, which publishes the
same `UiCommand`. UI commands carry a timestamp and a nonce; the controller accepts a command
only if it is younger than 0.5 s and the nonce is new. Physical inputs always win: a
stick deflection overrides a UI start in the same tick, and the walker's A and L2+B
remain latched regardless of the UI.

## Rollout changes (`policy_rollout/policy_rollout.py`)

- `--supervisor` (default off, for upstream compatibility; `g1_walk/README.md` will tell
  users to pass it): enables `--auto_start` behaviour (inference and frame publishing
  start as soon as observations are available, no key presses), the three new
  publishers below, and `PolicyStatus`.
- `PredictedChunk`: published from the **inference worker thread** right after
  `_run_inference` returns, before the result is queued to the control thread. Fields:
  `t_obs` (control step index), `stamp`, `horizon` (60), `interval` (45),
  `inference_ms`, and one flat float sequence per command group:
  `body_pose` (60x6), `left_eef_to_body` (60x6), `right_eef_to_body` (60x6),
  `left_eef_link` (60x15), `right_eef_link` (60x15), `velocity_commands` (60x3), in the
  action frame the policy outputs (before `g1_command` offsets). About 12 KB per
  inference, one DDS write per 1.5 s. Zero work on the control thread.
- `CameraFrame`: JPEG bytes of the policy's head frame at most 5 Hz, from a new
  low-priority thread that encodes `head_color_frame` (quality 70, 640x240 downscale).
  For the `zmq` source the JPEG the server sent is forwarded without re-encoding (not
  used now: no Jetson computation, the RealSense is on the laptop, so this path encodes).
- `PolicyStatus` at 5 Hz from the same status thread.
- The existing keys 1/2/3 keep working for upstream-style use without `--supervisor`.
- A synthetic head-camera source for sim stack tests (`--head_camera_source static`,
  a fixed image or a sim render): left to the builder to place where the existing
  camera sources live, coordinated with the other session that owns `head_camera.py`.

## Controller changes (`g1_walk/controller.py`)

- `--supervisor` (default on when the flag is implemented; the current "start on the
  first frame" behaviour stays available as `--supervisor off` for the replay tools that
  drive the controller directly).
- The supervisor object owns the state; the existing booleans (`session`,
  `operator_hold`, `ready`, `resetting`) become derived from it or are removed. The
  existing `_tick` branches map onto the states: `_session_tick` in RUNNING,
  `_tick_between` in STABILIZING/READY/HOLD/MANUAL, plus the manual walk command in
  MANUAL.
- Per tick, in this order: read `fault()`; snapshot the remote; take UI commands from the
  DDS reader (already on the input thread); update `standing`; update `policy_ready`
  from the latest `PolicyStatus`; feed the supervisor the event set; apply its decision.
- `SupervisorStatus` is published at 10 Hz from the existing state-publish thread (one
  extra small write), not from the control thread. Fields: state, since (s), reason of
  the last transition, the flags (`walker_live`, `standing`, `standing_for_s`,
  `policy_ready`, `frames_fresh`, `walk_enabled`), the remote snapshot (sticks, pressed
  buttons), the manual walk command, the last log line, and the controller's control-tick
  timing (p50 and max of the last second) so the UI can show the loop is healthy.
- Keyboard `s`/`g`/`q` remain: `s` = stop, `g` = re-arm, `q` = quit; `r` = reset.
- Hands: in HOLD and MANUAL the hand loop stops sending (the Inspire hands hold their
  last position); only Reset opens them.

## The UI (`deploy/ui/`)

One page, one screen, dark or light following the browser. Sections, top to bottom:

1. **State strip**: the state machine as a row of boxes BOOT → STABILIZING → READY →
   RUNNING with HOLD, MANUAL, ESTOP beside; the current state lit; under it the
   conditions that gate the next transition, each with a check mark or the live value
   (`standing 1.3 / 2.0 s`, `policy ready: warming up`, `frames: 30 Hz`). Buttons:
   **Start** (enabled only in READY), **Stop**, **Re-arm** and **Reset** (both with
   confirmation). `walk: on/off` and `manual walk: on/off` indicators.
2. **Robot**: torso height and tilt, walk command applied, remote sticks and pressed
   buttons, control-tick timing from both processes, inference time, the age of every
   input the UI shows (so a frozen value is visibly stale).
3. **Camera**: the latest `CameraFrame`.
4. **What happens next**: six time plots sharing one time axis from −6 s to +2 s.
   Left of 0: the measured state (solid). Right of 0: the predicted chunk (dashed,
   another colour), redrawn when a new `PredictedChunk` arrives and shifted left as time
   passes until the next one. The predicted step at `interval` (1.5 s) is marked: that is
   where rollout will switch to the next chunk. Groups and units:

   | Plot | Measured (from `rt/robot/state`, `rt/robot/eef_state`) | Predicted (from `PredictedChunk`) |
   | --- | --- | --- |
   | left hand position, 3 lines | `left_end_effector_to_torso_pose[0:3]` as measured | `left_eef_to_body[:, 0:3]` + the command offset (0.15, 0, 0.45), i.e. the wrist target rollout would send |
   | right hand position | same, right | same, right |
   | hand rotation, 2 lines (left, right) | angle between measured and the chunk's first rotation (deg) | rotation error of each predicted step to the current measured (deg) |
   | fingers, 2 lines | mean fingertip distance from the hand origin, from `hand_link_poses` (mm) | same from `left/right_eef_link` |
   | torso height | `torso_pose[2]` (m) | `body_pose[:, 2]` + 0.75 |
   | walk, 2 lines | the applied `velocity_commands` vx and wz | `velocity_commands[:, 0]`, `[:, 2]` |

   The offsets are the ones `policy_rollout/g1_command.py` applies when it turns an
   action into a command (`wrist_target`: action + (0.15, 0, 0.45); torso: body[2] + 0.75),
   so the prediction is drawn as the command the robot will receive, in the frame the
   measured state is reported in. `test_plot_frames.py` checks this against `g1_command`.
   Measured history is kept in the browser (a ring buffer per line); the server does not
   store history.
5. **Log**: the last 200 `[supervisor]` lines and the `HTDPOLICY READY TO DEPLOY` line,
   newest at the bottom, with a filter box.

The server (`deploy/ui/server.py --iface IFACE --port 8765`) subscribes to the topics
with best-effort, keep-last-1 QoS, coalesces to one JSON message per 100 ms (camera
frames as a separate binary message), and serves the static page. It never publishes
anything except `UiCommand` when a button is pressed. It is started as a fourth terminal
command, and `g1_walk/README.md` gets the four-terminal sequence for the sim and the
robot.

## Latency budget and how it is verified

What the control threads do that they did not do before:

| Thread | Added work per tick |
| --- | --- |
| controller control thread (50 Hz) | read ~6 fields of an already-decoded remote; the standing check (a dozen float comparisons); the supervisor step (table lookups); one `ui_commands.get_nowait()` |
| rollout control thread (30 Hz) | nothing |
| rollout inference thread (per 1.5 s) | one DDS write of ~12 KB |
| new low-rate threads | status writes at 5–10 Hz; JPEG encode at ≤5 Hz (releases the GIL) |

All new publishers use best-effort QoS with history depth 1, so a slow or absent
subscriber never blocks a writer. The UI process is separate and runs at `nice 10`.

Verification (part of the build, in MuJoCo): run the full stack for 60 s with and
without the UI connected, and compare the controller's control-tick p50/max and
rollout's loop overrun count, both already measured in the processes and now carried in
`SupervisorStatus`/`PolicyStatus`. Acceptance: no difference beyond noise (max tick
time stays under half the period, 10 ms for the controller, 16 ms for rollout). The
numbers go into `g1_walk/README.md`.

## Testing

- `deploy/tests/test_supervisor.py`: table-driven transitions, every guard, the hold
  timer, the log line emitted exactly once per READY entry, no path from HOLD/MANUAL to
  RUNNING without re-arm.
- `deploy/tests/test_standing.py`: each threshold, the 2 s hold, hysteresis on loss.
- `deploy/tests/test_remote.py`: `pack_wireless_remote` from the pinned teleop builds
  low-state blocks; debounce, hold-to-re-arm, deadband, stick scaling, precedence over
  UI commands, UI command freshness and nonce.
- `deploy/tests/test_plot_frames.py`: measured-to-command-frame conversion equals
  `g1_command` for a known pose.
- `deploy/tests/sim_supervisor.sh`: the MuJoCo stack plus rollout with a synthetic camera
  and the UI server; drives the state machine through `UiCommand` messages (a small CLI,
  `deploy/ui/cmd.py start|stop|rearm|reset`, publishes them; it is also useful at the robot)
  and asserts the transitions and the latency acceptance above. The sim host publishes
  no remote block, so the remote path is covered by the unit tests on fabricated
  low-state blocks and by the first session on the robot, on the gantry.

## Rollout of the work

1. `deploy/` package with the pure modules and their unit tests.
2. Controller integration behind `--supervisor`; sim run without the UI.
3. Rollout publishers and `--supervisor`; sim run; the log line appears.
4. UI server and page; the latency comparison.
5. Docs: `g1_walk/README.md` session sequence, `docs/deployment.md` update.

## Left to the builder

- Exact message field names and the WebSocket JSON shape.
- The canvas plotting code and page layout (no external libraries; the design fixes the
  content, not the look).
- Where the synthetic camera source lives (coordinate with the other session).
- Thread and queue details in the controller, as long as the latency budget holds.
- Tuning of thresholds after the first sim runs, within the documented defaults.

## Later (explicitly out of this build)

Wrist workspace box and keep-out zones, collision checks on the IK targets (the pinned
teleop's `armkit/safety_filter.py` is importable), stale-observation rejection in
rollout, a dead-man switch while RUNNING, the robot-agnostic split of `g1_walk`.
