# Deployment

A trained policy runs on the G1 through the teleop stack's walking policy. A supervisor
in the controller decides when the policy may drive the robot: it confirms that the
robot stands and that the policy is ready, logs `HTDPOLICY READY TO DEPLOY`, and waits
for the operator's start. Stop, re-arm, reset and joystick walking come from the Unitree
remote, a browser UI or the command line. This page is the operator's guide.
[`g1_walk/README.md`](../g1_walk/README.md) keeps the architecture, the simulator
details and the safety notes; the design is in
[the spec](superpowers/specs/2026-11-09-deploy-supervisor-ui-design.md).

## What runs where

Everything runs on the operator laptop. The robot's own computer (the Jetson) runs
nothing of this; the laptop talks to the robot over the cable.

| Process | Command | Role |
| --- | --- | --- |
| walker | `g1_walk/run_walker.sh` | the teleop's walking policy; the only process that writes `rt/lowcmd` to the motors |
| controller | `g1_walk/controller.py` | arm IK (inverse kinematics, solved in a child process), walk, torso and hand commands to the walker; the supervisor state machine; reads the remote |
| rollout | `policy_rollout/policy_rollout.py` | HTD inference at 30 Hz; publishes the policy's frames, its status, the predicted chunk and the camera image |
| UI | `deploy/ui/server.py` | subscribes to all of the above and serves the monitoring page; can be stopped at any time without effect on the robot |

The processes talk over DDS (Data Distribution Service, CycloneDDS, domain 0) on one
network interface: `lo` in MuJoCo, the robot's NIC (network interface) on the robot.
The head camera is a RealSense plugged into the laptop
(`--head_camera_source realsense --head_camera_layout duplicate`); in MuJoCo rollout
uses a fixed test image (`--head_camera_source static`).

Use one controller only. `robot_control/`'s controller and `g1_walk/controller.py` both
command the robot; never run both. The training data was recorded on the teleop's
walking policy, so `g1_walk/` is the one to use.

## Before a session

1. **Power.** Laptop on mains, platform profile `performance`
   (`powerprofilesctl set performance`). On the robot the controller refuses to start on
   battery or a power-saving profile unless `--allow_low_power` is passed; in MuJoCo it
   only warns, and the stack then misses its timing.

2. **Environment.** One `htd_rollout` environment runs all four processes. Either a new
   conda environment, or a thin venv on top of the teleop environment, which already has
   CUDA PyTorch:

   ```bash
   bash policy_rollout/setup_rollout_env.sh
   bash policy_rollout/setup_rollout_env.sh --base-python ~/anaconda3/envs/teleop/bin/python
   bash g1_walk/setup.sh --python ~/.venvs/htd_rollout/bin/python
   ```

   `g1_walk/setup.sh` fetches the pinned teleop code and SDKs and checks the
   environment. Below, `PY` is that environment's python. Clear a ROS `PYTHONPATH` in
   every terminal (`export PYTHONPATH=`): it breaks the stack's imports.

3. **Checkpoint.** Copy the run directory, or at least the checkpoint
   (`policy_last.ckpt` or the one you chose), `all_configs.json` and
   `dataset_stats.pkl`. Verify it on the laptop
   ([evaluation.md](evaluation.md#2-inference-check-on-the-deployment-machine)):

   ```bash
   cd policy_rollout && $PY check_inference.py --model_path RUN_DIR
   ```

   Rollout needs the run's model YAMLs, the pair recorded in its `all_configs.json`
   (`embodiment_config_path`, `trunk_config_path`; the walk profile by default, see
   [training.md](training.md#environment)).

4. **Gantry.** The robot stays on its gantry until the walker has taken the legs over,
   and for [the manual-walk check](#manual-walking-check-on-the-gantry) before the first
   `--manual_walk on`.

## The state machine

The controller logs every transition as `[supervisor] FROM -> TO: reason`. On entering
READY it logs `HTDPOLICY READY TO DEPLOY` on a line of its own. With the supervisor on
(the default) the log also goes to `logs/deploy_<YYYYmmdd_HHMMSS>.log` under the
repository; `--log_file PATH` chooses another file. From the MuJoCo check:

```text
[2026-11-09 13:46:15,053] [INFO] [supervisor] BOOT -> STABILIZING: walker live
[2026-11-09 13:46:18,853] [INFO] [supervisor] STABILIZING -> READY: standing 3.8 s, policy ready, frames fresh
[2026-11-09 13:46:18,853] [INFO] HTDPOLICY READY TO DEPLOY
[2026-11-09 13:47:02,935] [INFO] [supervisor] start from cli
[2026-11-09 13:47:02,953] [INFO] [supervisor] READY -> RUNNING: start
[2026-11-09 13:47:12,969] [INFO] [supervisor] RUNNING -> HOLD: stop
```

| State | What the robot does | How it is left |
| --- | --- | --- |
| BOOT | the walker hands the legs over; the arms glide to their home pose; nothing else is commanded | automatically, once the walker is live (the controller prints `READY:`) |
| STABILIZING | arms at home, or held where they were after a re-arm; walk zero; torso nominal. Policy frames are received, not applied | to READY once the robot stands and the policy is ready (below); sticks: MANUAL; Reset |
| READY | as STABILIZING. The policy runs in shadow: its predictions reach the UI, nothing moves | Start: RUNNING. Stop: HOLD (a stop beats a start in the same tick, and a later start is ignored until a re-arm). A lost check: back to STABILIZING (the log says which); sticks: MANUAL; Reset |
| RUNNING | the policy drives the arms, hands, torso and walk | Stop, 0.5 s without a policy frame, a link fault, the arm jump guard, the robot's low-state older than 0.1 s, torso tilt above 0.35 rad, or stale policy observations (below): HOLD. Sticks: MANUAL. Reset: stops, then resets |
| HOLD | freeze in place: walk zero, torso and arms hold their last sent targets, the hands keep their grip, the walker keeps balancing | Re-arm or Reset: STABILIZING; sticks: MANUAL |
| MANUAL | arms, torso and hands hold as in HOLD; the walk comes from the sticks | sticks centred for 1.0 s, or the robot's low-state older than 0.1 s, or a link fault: HOLD. Re-arm with the sticks centred: STABILIZING |
| ESTOP | nothing is commanded; the walker froze the arms and stands (remote A, its own abort) or damped the motors (L2+B) | restart the processes |

Nothing returns to RUNNING from HOLD or MANUAL without a re-arm, a fresh READY and a
new start.

**Standing check.** All of these, without interruption, for 2.0 s:

| Check | Limit |
| --- | --- |
| walker live | handover done |
| torso height | within ±0.06 m of 0.78 m |
| torso roll and pitch | each within ±0.10 rad |
| body at rest | gyroscope norm < 0.25 rad/s |
| legs at rest | every leg joint velocity < 0.5 rad/s |
| low-state fresh | newer than 0.1 s |

The check does not apply in RUNNING, where the robot walks and bends by design. RUNNING
has its own, looser checks, each going to HOLD:

| Check in RUNNING | Limit |
| --- | --- |
| low-state fresh | newer than 0.1 s (the remote arrives in it, so a stale low-state also means the controller cannot read the sticks or X) |
| torso roll and pitch | each within ±0.35 rad (teleop episodes peak at about 0.22 rad on the real robot); unknown counts as failed |
| policy observations | rollout's robot state, hand state and head camera each newer than 0.3 s, as its status reports them, and the status newer than 0.5 s; a frozen camera must not drive the robot open loop. A slow inference alone does not stop the robot |

The walker's own tilt abort (1.0 rad) remains the fall guard behind these.

**Controller failures.** A control tick that raises, or no tick completing for 0.1 s
(a hang), sets the walk to zero at once and stops the policy: HOLD on the next tick that
runs (with `--supervisor off`: as the `s` key). After 5 failed ticks in a row, or a
stall of 0.5 s, the walker is also told to stand. These come from the controller itself:
the walker link's heartbeat would otherwise keep resending the last walk command. The
log says `control tick failed (N in a row)` or `no control tick for <s> s`, once a
second.

**Policy ready.** Rollout reports ready when the checkpoint is loaded and warmed up, a
robot state, a hand state and a camera frame have each arrived within the last second,
and the last inference fit the chunk budget (half the inference interval). The status
itself must be younger than 1 s, and a policy frame must have arrived within 0.5 s.
Frames do not count while the arms glide home. Rollout runs no inference and publishes
no frame until a robot state, a hand state and a real head frame have each arrived (it
logs `observations present: inference and publishing start`).

**Start.** The robot executes only chunks planned after the start: rollout restarts its
chunk when it sees RUNNING, and the controller ignores frames older than the start plus
0.15 s. Until the first fresh frame arrives the robot holds as in READY; if none
arrives, HOLD follows after about 0.65 s (the 0.5 s frame timeout, counted from the
0.15 s margin). The walk starts only once the policy has asked for rest
(|vx| and |wz| ≤ 0.05), as before. `--walk off` replaces the policy's walk with zero for
the whole run.

**Stop** freezes in place. A robot bent over a table stays bent, and the standing check
then cannot pass.

**Re-arm** (HOLD to STABILIZING) slews the torso back to nominal, leaves the arms and
hands where they are, clears a link fault and restarts the 2 s standing hold. The policy
can be restarted while the robot holds an object.

**Reset** is separate: the arms glide home, the hands open, the torso goes to nominal.
In RUNNING it stops first.

**HOLD or ESTOP.** A silent walker or a link blip goes to HOLD from RUNNING and MANUAL,
and back to STABILIZING from READY; re-arm (or reset) clears it. The walker's own
latches go to ESTOP: remote A, L2+B, a walker fault or its tilt abort. They hold until
the processes restart (at least the walker and the controller).

## Buttons and sticks

The controller reads the Unitree remote from the robot's low-state, once per control
tick. Buttons fire on the press, after a 50 ms debounce.

| Input | Action | Accepted in |
| --- | --- | --- |
| Start | start the policy | READY; elsewhere logged as `start ignored in <STATE>` |
| X | stop: freeze in place | RUNNING; READY (goes to HOLD, so a start right after it is ignored) |
| L1 + Start, held 1.0 s | re-arm | HOLD. From MANUAL: centre the sticks, wait for HOLD (1.0 s), then L1 + Start |
| Y | reset: arms home, hands open, torso nominal | STABILIZING, READY, HOLD; RUNNING (stops first) |
| left stick forward / back | walk speed vx: up to 0.30 m/s forward, 0.20 m/s back | every state except BOOT and ESTOP, and not while the low-state is stale or a link fault is asserted; enters MANUAL |
| right stick left / right | yaw rate: up to 0.50 rad/s; stick right turns clockwise | as above |
| A | the walker's freeze and stand | always: ESTOP, restart needed |
| L2 + B | the walker damps every motor | always: ESTOP, restart needed |

- **Re-arm: press L1 first, then Start.** A Start pressed more than 50 ms before L1 is
  read as a start.
- **Sticks.** Deadband 0.3 of full deflection, then proportional up to the maximum. Any
  deflection beyond the deadband enters MANUAL: the policy stops and the arms hold. Not
  while the low-state is stale (the sticks then read as centred) or a link fault is
  asserted (RUNNING then goes to HOLD). The left stick's sideways axis and the right
  stick's forward axis are not used.
- **Manual walking is off by default.** Without `--manual_walk on` MANUAL sends walk
  zero. Turn it on only after the
  [check on the gantry](#manual-walking-check-on-the-gantry).
- X does nothing in MANUAL: centre the sticks, and HOLD follows after 1.0 s.
- After a low-state dropout a held button must be released and pressed again (X
  excepted).
- MuJoCo publishes no remote: in the sim use the UI or `deploy/ui/cmd.py`.

The same commands, without the remote:

| Where | Start | Stop | Re-arm | Reset |
| --- | --- | --- | --- | --- |
| UI | Start (enabled in READY only) | Stop | Re-arm (asks to confirm) | Reset (asks to confirm) |
| command line | `cmd.py start` | `cmd.py stop` | `cmd.py rearm` | `cmd.py reset` |
| controller terminal (Enter after each) | | `s` | `g` | `r` |

`cmd.py` is `$PY deploy/ui/cmd.py --iface <iface> <command>`; it waits up to 2 s for
the command topic's reader and exits with 1 and a message if none matched (the
controller or the UI not running on that interface). `q` in the controller
terminal quits: the walker stands and keeps balancing. A UI or command-line command
older than 0.5 s is dropped, and the remote wins over them: a stick deflection in the
same tick as a UI start enters MANUAL.

Rollout's keys `1`, `2`, `3` are disabled with `--supervisor True`: it starts no keyboard
listener and prints `supervisor mode: keyboard keys 1/2/3 disabled; use the remote, the UI
or deploy/ui/cmd.py`. Without the supervisor they act as before
(see [Troubleshooting](#troubleshooting)).

## Running in MuJoCo

Five terminals, each from the repository root, in this order. The first three are
[g1_walk's sim sequence](../g1_walk/README.md#running-in-mujoco-on-one-machine) with the
supervisor on. That README's commands predate the supervisor: with today's default its
controller command runs the supervisor too, so use the sequences on this page (the
replay tools pass `--supervisor off`).

```bash
# 1. the teleop's sim host: G1 with Inspire hands, walker stance, pelvis band
cd g1_walk/teleop-cbf-perception/teleop
MUJOCO_GL=egl PSI_SIM_HOLD_RELEASE=session $PY sim/sim_host.py --iface lo --lower jetson --inspire
# wait for "Publishing rt/lowstate"

# 2. the walker (prints READY)
bash g1_walk/run_walker.sh --iface lo --python $PY

# 3. the controller (prints READY: once the walker is live)
PSI_INSPIRE_LEFT_IP=127.0.0.1 PSI_INSPIRE_RIGHT_IP=127.0.0.2 \
    $PY g1_walk/controller.py --iface lo --supervisor on

# 4. release the sim's pelvis band, then start rollout
$PY -c "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'{}', ('127.0.0.1', 14651))"
cd policy_rollout && OMP_NUM_THREADS=1 $PY policy_rollout.py --dds_interface lo \
    --supervisor True --inference_interval 45 --send_velocity_commands True \
    --head_camera_source static \
    --model_path RUN_DIR --policy_class MXT --embodiment g1_inspire_wb --inspire_hand_type dftp \
    --use_wrist_camera False --device cuda:0 --control_freq 30 \
    --embodiment_config_path ../htd_policy/configs/htd/dftp_walk_latent16.yaml \
    --trunk_config_path ../htd_policy/configs/htd/trunk_walk.yaml

# 5. the UI, then open http://127.0.0.1:8765
$PY deploy/ui/server.py --iface lo --port 8765
```

Without `--send_velocity_commands True` the walk stays zero. A few seconds after rollout
has warmed up the controller logs `HTDPOLICY READY TO DEPLOY`; then press Start in the
UI, or `$PY deploy/ui/cmd.py --iface lo start`.

**One-command check.** `deploy/tests/sim_supervisor.sh` brings up the same five
processes, waits for the ready line, measures the control-loop timing with and without
a UI client, drives start, stop, re-arm and reset through `cmd.py`, and checks that
everything stops cleanly:

```bash
bash deploy/tests/sim_supervisor.sh --model_path RUN_DIR --python $PY
```

It prints PASS or FAIL for each of the four checks, exits 0 only if all pass, and writes
the logs and `summary.json` to `deploy/tests/logs/<time>/`. Nothing else may be running
the sim stack or using port 8765.

The replay and test tools that drive the controller directly
(`g1_walk/tests/sim_e2e.sh`, `g1_walk/replay/sim_replay.sh`) pass `--supervisor off`:
a session then starts on the first policy frame, without the state machine.

## Running on the robot

The walker runs on the laptop and reaches the robot over the cable. The steps follow
[g1_walk's robot sequence](../g1_walk/README.md#running-on-the-robot-from-this-laptop),
whose commands predate the supervisor; with today's default the controller runs it, so
use the commands below.

1. Robot on its gantry, powered; the laptop on the robot's network (192.168.123.x);
   the RealSense on the laptop.
2. Release Unitree's own controller (debug mode). The robot goes limp:

   ```bash
   cd g1_walk/teleop-cbf-perception/teleop && $PY tools/enter_dev_mode.py --iface <NIC>
   ```

3. Four terminals:

   ```bash
   # 1. the walker
   bash g1_walk/run_walker.sh --iface <NIC> --python $PY

   # 2. the controller; the Inspire hands' Modbus addresses (HTD's rig: .210 left, .211 right)
   PSI_INSPIRE_LEFT_IP=192.168.123.210 PSI_INSPIRE_RIGHT_IP=192.168.123.211 \
       $PY g1_walk/controller.py --iface <NIC> --supervisor on

   # 3. rollout
   cd policy_rollout && OMP_NUM_THREADS=1 $PY policy_rollout.py --dds_interface <NIC> \
       --supervisor True --inference_interval 45 --send_velocity_commands True \
       --head_camera_source realsense --head_camera_layout duplicate \
       --model_path RUN_DIR --policy_class MXT --embodiment g1_inspire_wb --inspire_hand_type dftp \
       --use_wrist_camera False --device cuda:0 --control_freq 30 \
       --embodiment_config_path ../htd_policy/configs/htd/dftp_walk_latent16.yaml \
       --trunk_config_path ../htd_policy/configs/htd/trunk_walk.yaml

   # 4. the UI, then open http://127.0.0.1:8765
   $PY deploy/ui/server.py --iface <NIC> --port 8765
   ```

4. The controller's handover (3 s gain ramp, 2 s glide, parity check, 5 s blend) puts
   the legs on the walker. Lower the gantry once the controller prints `READY:`.
5. Wait for `HTDPOLICY READY TO DEPLOY`, check the camera image and the plots, then
   press Start.

Differences from MuJoCo: the controller's real-time power check is enforced; the remote
works; the hands are the real Inspire hands, which hold their last position in HOLD and
MANUAL and open only on Reset; the camera is the RealSense. Other camera sources are in
[policy_rollout/README.md](../policy_rollout/README.md#head-camera), and the rollout
options (`--crossfade_steps`, `--async_inference`, `--checkpoint`, `--dream_touch`) in
[policy_rollout/README.md](../policy_rollout/README.md#run-htd-or-mxt).

To stop: press Stop (X or the UI), then Reset if the arms should go home and the hands
open, then quit rollout (Ctrl-C), the controller (`q`: the walker stands and keeps
balancing), the walker, with the robot on the gantry or sitting first
([g1_walk's safety notes](../g1_walk/README.md#stopping-and-safety)). The UI can be
stopped at any time.

### Manual walking: check on the gantry

The remote's stick offsets in the low-state were never verified against the Unitree
layout, and MuJoCo cannot exercise them. Before the first `--manual_walk on`, on the
gantry, with the controller at its default `--manual_walk off`:

1. Push the left stick forward. The UI's left-stick dot moves up, the state goes to
   MANUAL, and the Robot panel's `manual walk` row shows a positive vx, up to 0.30 at
   full deflection. Pulled back: negative, down to -0.20.
2. Push the right stick right. The dot moves right, `manual walk` shows a negative wz
   (clockwise), down to -0.50.
3. Release the sticks: HOLD after 1.0 s.
4. In HOLD, press the buttons one at a time and check the name the UI shows under
   `buttons`. **Never press A**: the walker reads it as an e-stop, the supervisor
   latches ESTOP, and the walker and the controller must be restarted, handover
   included. **Never press L2 and B together**: the walker damps every motor.
5. Press Y last: it fires a reset (arms home, hands open) and leaves HOLD for
   STABILIZING, so no re-arm is needed after it.

If all of it matches, quit the controller (`q`) and start it again with
`--manual_walk on`. Otherwise leave manual walking off: deflecting a stick still stops
the policy and holds the robot.

## The UI

```bash
$PY deploy/ui/server.py --iface <iface> --port 8765    # then open http://127.0.0.1:8765
$PY deploy/ui/server.py --demo                          # the page with fake data, no robot
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--iface` | none | DDS network interface (`lo` in MuJoCo) |
| `--port` | 8765 | HTTP and WebSocket port |
| `--host` | 127.0.0.1 | address to listen on; the page is meant for the laptop's own browser |
| `--rate` | 20 | snapshots per second sent to each browser |
| `--demo` | off | animated fake telemetry, no DDS |

The server only subscribes (best-effort, keep-last-1) and runs at `nice 10`. It
publishes one command message per button press and nothing else. The page loads nothing
from the internet, so it works on the robot's network.

Sections, top to bottom:

1. **State.** The states as a row of boxes, the current one lit, the time in it, the
   last transition's reason and any fault. Below, the conditions (walker live, standing
   for N s, policy ready with rollout's reason, policy frames) and the gating text of
   the next transition (see [Troubleshooting](#troubleshooting)). Buttons Start, Stop,
   Re-arm, Reset, and `walk` and `manual walk`: read-only indicators of the
   controller's `--walk` and `--manual_walk` flags.
2. **Robot.** Torso height, roll and pitch; the walk command applied; the manual walk
   command from the sticks; the controller's tick time, its arm IK solve time and
   rollout's loop time (median and maximum over the last second); inference time, chunk count and overruns; the two
   sticks and the pressed buttons; the age of every input.
3. **Camera.** The policy's head image, at most 5 frames per second, with its age.
4. **What happens next.** Six plots on one time axis, -6 s to +2 s. Left of 0, the
   measured state (solid). Right of 0, the latest predicted chunk (dashed, a lighter
   tint of its measured line's colour, over a tinted background), shifted left as time passes until
   the next chunk arrives. The vertical mark is the chunk's step 45 (1.5 s after it was
   planned), where rollout switches to the next chunk.
5. **Log.** The supervisor's recent lines: each status carries its last 12, so a line is
   not lost between two statuses and the lines from before the page was opened show
   too. Plus the page's own lines (connected, command sent); the last 200, newest at the
   bottom, with a filter box. The log file is the complete record.

The prediction is drawn as the command the robot will receive, in the frame the
measured state is reported in:

| Plot | Measured | Predicted |
| --- | --- | --- |
| left hand, right hand (x, y, z, m) | wrist position in the torso frame | the action's wrist position + (0.15, 0, 0.45), as rollout turns it into a command |
| hand rotation error (deg) | 0 by definition | angle between each predicted wrist rotation and the measured one |
| fingers (mm) | mean fingertip distance from the hand origin | the same from the predicted finger links |
| torso height (m) | torso z | the action's body z + 0.75 |
| walk (m/s, rad/s) | the applied vx and yaw rate | the predicted vx and yaw rate |

**Ages.** An input older than 1 s is shown red; the predicted chunk after 3 s (one
arrives every 1.5 s, so red means two were missed). A stale supervisor status greys the
state section and disables Start. `ui DDS poll` is red when the server's DDS thread has
stopped or not polled within 1 s. If the page receives no snapshot for 1 s (server,
browser link or DDS gone), it marks the state and every age stale, disables Start and
shows `no data for N s` in the state line until snapshots resume.

## Troubleshooting

The UI's gating list and the controller log say what holds a state.

| Gating text | Meaning | What clears it |
| --- | --- | --- |
| `walker not live` | the walker's handover is not done, or the walker is not running | start the walker before the controller; wait for the controller's `READY:` |
| `fault: <reason>` | a fault is asserted (below) | a link fault: Reset, or it clears when the walker answers again; an e-stop goes to ESTOP |
| `standing: height,tilt,...` | these checks fail now (`walker`, `height`, `tilt`, `gyro`, `legs`, `stale`) | let the robot settle on its feet; after a stop, re-arm brings the torso to nominal height. `stale`: the robot's low-state is not arriving |
| `standing 1.3 / 2.0 s` | all checks pass, the 2 s hold is counting | wait |
| `policy not ready` | rollout's status is not ready, or older than 1 s; the conditions line gives rollout's reason | see the next table |
| `no policy frames` | no frame from rollout within 0.5 s, or the arms are gliding home | wait for the glide; check rollout runs with `--supervisor True` on the same interface and has logged `observations present` |
| `press Start` | READY | start |
| `re-arm: L1+Start 1 s, or Reset` | HOLD | re-arm or reset |
| `release the sticks` | MANUAL | centre the sticks for 1.0 s |
| `restart the processes` | ESTOP | stop rollout, the controller and the walker, then start again |

Rollout's reasons for not ready:

| Reason | What clears it |
| --- | --- |
| `warming up` | wait for the checkpoint load, warm-up and CUDA graph capture |
| `no robot state` | the controller's `rt/robot/state` is not arriving: controller running, same interface |
| `no hand state` | `rt/robot/eef_state` is not arriving: the hands' Modbus link. With `--hands none` it never arrives, so READY is never reached |
| `no camera frame` | the head camera: RealSense connected, or `static` in MuJoCo |
| `inference slow` | the last inference took more than half the inference interval (750 ms at 45 steps): power and profile, other GPU load |

Why the robot went to HOLD (the log line's reason):

| Reason | Cause |
| --- | --- |
| `stop` | X, the UI, `cmd.py stop`, `s`, from RUNNING or READY; also the arm jump guard refusing every command for 50 frames, the walker refusing to engage, or a controller failure (`control tick failed`, `no control tick for <s> s`), each logged as `[supervisor] stop: ...` first |
| `policy frames stopped` | no frame for 0.5 s: rollout ended or crashed, or no fresh frame came after a start |
| `link fault: <reason>` | the walker fell silent or the state link dropped |
| `sticks released for 1.0 s` | MANUAL ended |
| `low-state stale` | RUNNING or MANUAL with the robot's low-state older than 0.1 s |
| `tilt <x> rad > 0.35` | RUNNING with the torso rolled or pitched beyond 0.35 rad (`nan`: the torso pose is not arriving) |
| `policy observations stale` | RUNNING with rollout's robot state, hand state or head camera older than 0.3 s, or its status older than 0.5 s |

ESTOP reasons: `estop_<reason>` (remote A, L2+B damping, or a latch the walker reported)
and `lower_<reason>` (the walker's composer trip or tilt abort). A silent walker,
`lower_jetson_silent_<age>s`, is a link fault and goes to HOLD instead.

**Rollout's keys.** With `--supervisor True` rollout starts no keyboard listener: `1`,
`2`, `3` do nothing, and typing digits in any window is safe. Without the supervisor it
reads them through a desktop-wide listener: a `1` typed anywhere, for example in an IP
address, sends a reset (the hands open, the arms glide home), `3` pauses publishing and
`2` resumes.

## Verified numbers

MuJoCo on the operator laptop (RTX 5070 Laptop GPU), `sim_supervisor.sh`, run
`20261109_134551`, all four checks PASS. Times in ms.

| Measured | No UI client | UI connected | Acceptance |
| --- | --- | --- | --- |
| controller tick, READY, median | 0.26 | 0.26 | within 1 ms of each other |
| controller tick, READY, max | 1.46 | 2.36 | < 10 in both, within 2 ms of each other |
| rollout loop, READY, max | 2.49 | 2.24 | < 16 |
| inference, median | 10.2 | 9.8 | |
| controller tick, RUNNING, median | 25.46 | 25.94 | within 2 ms of each other |
| rollout loop, RUNNING, max | 4.1 | 3.7 | |

The RUNNING rows above predate the IK worker: the tick then carried the arm IK solve,
about 25 ms, as before the supervisor. The solve now runs in a child process of the
controller (`g1_walk/ik_worker.py`, OpenMP on one thread): the tick hands it the newest
policy frame and sends the latest solution, and never waits for a solve. A session's
first tick sends no arm command (no solution yet); a worker that owes a solution for
longer than `--policy_timeout` logs `IK worker stalled: arms hold` and the arms keep
their last command while walk and torso continue. Offline, without the sim
(`g1_walk/tests/bench_ik.py --tick`, 50 Hz, a frame every 1/30 s): tick median 0.10 ms
and max 0.37 ms against 12.5 and 18.9 ms with the solve inline; solve 2.8 ms. The
Robot panel's `arm IK solve` row shows the live solve time.

| Transition | Time |
| --- | --- |
| controller's `READY:` to `HTDPOLICY READY TO DEPLOY` | 3.7 s |
| re-arm to READY | 2.4 s from launching `cmd.py` (2.0 s standing hold) |
| reset in READY to READY again | 2.4 s from launching `cmd.py` |
| start to RUNNING, stop to HOLD, re-arm to STABILIZING | under 20 ms after the controller receives the command; 0.4 s from launching `cmd.py` |
| teardown of all five processes | 2.0 s |

## Tests

No robot or camera is needed:

```bash
PYTHONPATH= $PY -m pytest deploy/tests -q                                  # 101 tests: supervisor, standing, remote, plots, UI server
PYTHONPATH= $PY -m unittest discover -s g1_walk/tests -p 'test_*.py'       # g1_walk command shaping, walk gate, hands, IK worker
bash deploy/tests/sim_supervisor.sh --model_path RUN_DIR --python $PY      # the whole stack in MuJoCo, about 2 min
```

The remote's buttons and sticks are covered by the unit tests on fabricated low-state
blocks and by the gantry check above.
