# g1_walk: HTD's policy on the teleop's G1 walking policy

The HTD data this repository trains on was recorded with the teleop stack
[teleop-cbf-perception](https://github.com/talos-robotics-ai/teleop-cbf-perception)
driving the Unitree G1's legs with its **g1_walk** walking policy (run
`g1_teleop_anyadapter`, checkpoint 9999) under `--lower jetson`, not with HTD's own
whole-body controller (`robot_control/g1_wholebody.py`). This module runs a trained HTD
policy on that same walker:

* the walker is the teleop's own code, pinned as a git submodule at the commit the data
  was recorded with, and run unchanged (its network on this machine's GPU);
* `controller.py` takes the place of `robot_control/g1_inspire_wb_controller.py`: it
  turns policy_rollout's messages into the walker's arm, walk and torso commands and
  the Inspire hands' registers, and publishes the robot state policy_rollout reads.

The directory is named after the walking policy it integrates, the name the teleop gives
it (`third_party/g1_walk`, the `talos-robotics-ai/g1_walk_cbf` checkpoint repo).

## Architecture

```
 policy_rollout.py (unchanged)                         this machine (GPU)
   TeleopDataWB  rt/teleop/synchronized_data  ─┐
   ResetCommand  rt/robot/reset_command       ─┤
   ◄─ rt/robot/state, rt/robot/eef_state       │
   ◄─ rt/inspire_hand/touch/{l,r}              │
                                               ▼
 g1_walk/controller.py ──────────────────────────────────────────────────────────┐
   HTD G1_29_ArmIK ─► engage glide ─► speed cap ─► teleop JointJumpGuard ─┐       │
   velocity_commands ─► (vx, wz) clipped ─► walk gate ────────────────────┤       │
   torso_height − 0.78, torso yaw ─► torso engage glide ──────────────────┤       │
                                   teleop G1_29_JetsonBody / RemoteLower ◄┘       │
   HTD DexPilot ─► registers ─► teleop Inspire_Modbus_Controller ──► Modbus TCP ──┼─► Inspire hands
   HTD state estimator ◄─ rt/lowstate (JetsonBody, no publisher)                  │
 ─────────────────────────────────────────────────────────────────────────────────┘
          │ UDP link :5591 (teleop jetson_link.py: ops + arm/walk/torso hot state)
          ▼
 g1_walk/run_walker.sh ─► teleop g1_jetson_walker.py (pinned, unchanged)
   body process: G1_29_BodyController ─► rt/lowcmd, all 29 motors (the ONLY writer)
   policy process: talos_walk composer, network on CUDA, link server
   reads the Unitree remote itself: A freezes arms and stands, L2+B damps
          │ rt/lowcmd / rt/lowstate (DDS)
          ▼
   MuJoCo sim host (--iface lo)   or   the G1 (--iface <robot NIC>)
```

The arm IK solves in a child process of the controller (`ik_worker.py`): the 50 Hz
tick hands it the newest policy frame and sends the latest solution, never waiting.

| Process | Publishes | Started by |
|---|---|---|
| walker (2 processes) | `rt/lowcmd` | `bash g1_walk/run_walker.sh` |
| controller | `rt/robot/state`, `rt/robot/eef_state`, `rt/inspire_hand/touch/{l,r}`; never `rt/lowcmd` or `rt/arm_sdk` | `python g1_walk/controller.py` |
| policy_rollout | `rt/teleop/synchronized_data`, `rt/robot/reset_command` | `python policy_rollout/policy_rollout.py` |
| sim host (sim only) | `rt/lowstate`, `rt/odommodestate`; Modbus servers 127.0.0.1/.2:6000 | the submodule's `sim/sim_host.py` |

## Setup (once)

From the repository root, with the Python environment that will run everything: the
teleop conda env at the pinned commit (`g1_walk/teleop-cbf-perception/environment.yml`)
or a venv layered on it, such as the one `policy_rollout/setup_rollout_env.sh
--base-python` makes. Git 2.36 or newer.

```bash
bash g1_walk/setup.sh --python /path/to/python
```

It fetches the teleop submodule at the pin and the one nested submodule the walker
imports (`third_party/g1_walk`, the checkpoint; also `third_party/unitree_sdk2_python`
when the environment has no `unitree_sdk2py`), clones the Inspire SDK at its pin
(`pins.env`), installs `requirements.txt`, the SDK and `robot_control/dex-retargeting`,
then runs `check_pin.py` and `check_env.py`. About 1.3 GB on disk. policy_rollout needs
the Inspire SDK in its environment for the touch topics: run both from this one
environment.

## Running in MuJoCo on one machine

Five terminals, each from the repository root, in this order. `PY` is the environment's
python. The full procedure, with the supervisor's states, buttons and UI, is in
[docs/deployment.md](../docs/deployment.md).

```bash
# 1. the teleop's sim host: G1 with Inspire hands, walker stance, pelvis band
cd g1_walk/teleop-cbf-perception/teleop
MUJOCO_GL=egl PSI_SIM_HOLD_RELEASE=session $PY sim/sim_host.py --iface lo --lower jetson --inspire
# (add --spectator for a window; wait for "Publishing rt/lowstate")

# 2. the walker (prints READY; holds the measured pose until the handover)
bash g1_walk/run_walker.sh --iface lo --python $PY

# 3. the controller: runs the walker's handover, then waits for the policy
PSI_INSPIRE_LEFT_IP=127.0.0.1 PSI_INSPIRE_RIGHT_IP=127.0.0.2 $PY g1_walk/controller.py --iface lo \
    --supervisor on

# 4. release the sim's pelvis band (the host lets it go over 10 s on the first datagram
#    to its target-viz port), then start the policy
$PY -c "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'{}', ('127.0.0.1', 14651))"
cd policy_rollout && $PY policy_rollout.py --dds_interface lo --send_velocity_commands True \
    --supervisor True --inference_interval 45 \
    --model_path <run dir> ...   # the rest as in policy_rollout/README.md

# 5. the monitoring UI, then open http://127.0.0.1:8765
$PY deploy/ui/server.py --iface lo --port 8765
```

Without `--send_velocity_commands True` the walk stays at zero. Rollout's keyboard keys
(`1`, `2`, `3`) are disabled in supervisor mode: start, stop, re-arm and reset come from
the remote, the UI or `deploy/ui/cmd.py`. Stop in reverse order
(below). The sim's hands report zero touch; their angles follow the commands.

## Running on the robot from this laptop

The walker runs on the laptop and publishes `rt/lowcmd` to the robot over the cable; the
robot's PC2 runs nothing of this module.

1. Robot on its gantry, powered, the laptop on its network (192.168.123.x).
2. Release Unitree's own controller (debug mode). The robot goes limp:
   `cd g1_walk/teleop-cbf-perception/teleop && $PY tools/enter_dev_mode.py --iface <NIC>`
3. The Inspire hands' Modbus addresses: the teleop driver's `PSI_INSPIRE_LEFT_IP`,
   `PSI_INSPIRE_RIGHT_IP`, `PSI_INSPIRE_PORT` (defaults 127.0.0.1 / 127.0.0.2 / 6000,
   the teleop's SSH tunnel; HTD's rig uses 192.168.123.210 left / .211 right).
4. ```bash
   bash g1_walk/run_walker.sh --iface <NIC> --python $PY
   PSI_INSPIRE_LEFT_IP=<left> PSI_INSPIRE_RIGHT_IP=<right> $PY g1_walk/controller.py --iface <NIC> \
       --supervisor on
   (cd policy_rollout && $PY policy_rollout.py --dds_interface <NIC> --send_velocity_commands True \
       --supervisor True --inference_interval 45 ...)
   $PY deploy/ui/server.py --iface <NIC> --port 8765     # the UI: http://127.0.0.1:8765
   ```
   The controller's handover (3 s gain ramp, 2 s glide, parity check, 5 s blend) puts
   the legs on the walker; lower the gantry once it prints READY. Rollout's keyboard
   keys are disabled in supervisor mode; the full procedure is in
   [docs/deployment.md](../docs/deployment.md#running-on-the-robot).

## What the controller does

* **Start** (teleop `G1Backend._start_policy`): the arms glide from where they are to
  the spec's home pose (`assets/g1/arm_spec.toml [pose] home`, at most 0.6 rad/s) while
  the walker runs its handover. Nothing reaches the arms from the policy before the
  walker reports `live`.
* **Session** (teleop `on_session_start` + `_engage_lower`): the first fresh policy
  frame once the arms are home engages the walker, arms the link watchdog, and eases
  the arms (teleop engage glide, `PSI_ENGAGE_S` / `PSI_ENGAGE_VEL`) and the torso onto
  the policy's commands.
* **Per tick** (50 Hz): HTD's `G1_29_ArmIK` once per policy frame, solved in the
  controller's IK process (`ik_worker.py`; a tick never waits for it, and sends no arm
  command until a session's first solution arrives), then the engage glide, a 4 rad/s joint speed cap (`--arm_max_vel`), the teleop's jump guard, and the
  command with its gravity torque goes to the walker through `G1_29_JetsonBody` (as
  `_safely_set_arms`) and to the walker's arm reference. Walk: `(vx, wz)` from
  `velocity_commands`, clipped to the walker's box; vy is dropped (the walker has no
  lateral axis; the data's vy is zero). After every session start, the walk stays zero
  until the policy asks for rest (|vx|, |wz| ≤ 0.05), the teleop's walk gate with the
  policy in place of the sticks. Torso: below.
* **Session end** — the walker stands (walk and torso slew to nominal), the arms hold:
  no policy frame for `--policy_timeout` (0.5 s); `s` at the keyboard (`g` resumes); a
  latch (remote A or L2+B, a walker or link fault). A reset from policy_rollout also
  glides the arms home and opens the hands, as HTD's controller does, then waits for
  new frames. The walker itself does the same when this process falls silent for
  0.25 s.
* **Quit** (`q`, Ctrl-C, SIGTERM): the walker stands and the link is closed without
  stopping it (detach). The walker keeps the robot balancing; restarting the controller
  takes the link over again.
* **State**: `rt/robot/state` from HTD's own estimator code on the walker's rt/lowstate,
  `rt/robot/eef_state` from HTD's hand-state code on the Modbus readings, and the touch
  topics from the Modbus driver's tactile sweep, at 50 Hz.

### Torso command

The recording's `actions/body` torso attitude is the `action` tail
`torso_roll/pitch/yaw`, which the teleop's lean master writes from the arm pipeline's
`result.torso_rpy` (`teleop/robot_control/joint_command.py:167-168`,
`teleop/lean_master.py:785-788`). The backend sends that same attitude to the walker
(`teleop/robot_control/g1_backend.py:2465-2467`: `set_torso_command(height=new_h -
G1_STAND_HEIGHT, roll, pitch, yaw)` from `_torso_command`, `:2017-2056`, which returns
`res.torso_rpy` outside the session-start engage glide, a re-home and a pause), through
`RemoteStrategy.set_torso_command` to the walker, which clamps it to its box (yaw ±0.6
rad) and feeds it to the network (`teleop/robot_control/g1_lower.py:725-726`). Under
`PSI_TORSO_ATTITUDE=yaw` the pipeline commands roll and pitch zero, the height at the
stand and the yaw of the operator's head (`teleop/robot_control/arm_pipeline.py:286-292`,
`:2255-2273`). The data fits that setting: height travel exactly 0
(`actions/body` z constant 0.03), roll and pitch about 0, yaw in [−0.30, 0.58] rad; the
operator laptop's teleop TUI profile (`teleop/tui_config.toml`) has `torso_yaw_only = true`, which the TUI turns into `PSI_TORSO_ATTITUDE=yaw` (`teleop/tui/config.py:281`).

So `--torso_attitude yaw` (the default) sends roll = pitch = 0 and the policy's yaw;
`hold` sends all three zero, `follow` passes all three. The height is
`torso_height − 0.78` in every mode, which inverts the converter exactly (0 for the
trained constant). The teleop's session-start torso glide is reproduced.

### Hands

The teleop's `Inspire_Modbus_Controller` drives the hands, the driver that commanded
them during the recording (same slew limit, force and speed registers, touch sweep). It
takes 0..1000 registers, which HTD's DexPilot chain produces (`Inspire_Controller`'s
retargeting and `joint_angles_to_hardware`, reused through a subclass without its DDS
endpoints). HTD's own DDS hand bridge (`hand_drivers/inspire_dftp.py`) would need the
Inspire SDK's Modbus stack in a second process and is not what the data was recorded
with.

## Provenance and the pin

`g1_walk/teleop-cbf-perception` is a submodule at `TELEOP_COMMIT` in `pins.env`
(60785385, the "60785385-dirty" of the recording); its own gitlinks fix
`third_party/g1_walk` (6a3e9505) and `third_party/unitree_sdk2_python` (cf26ccef).

```bash
python g1_walk/check_pin.py                                   # also run by run_walker.sh
python g1_walk/check_pin.py --against ~/path/to/teleop-cbf-perception
```

The first verifies the checkout, the index's gitlink and the nested commit, and that
the walker's deploy set (every tracked file under `teleop/` and `assets/` plus
`third_party/g1_walk`, what the teleop's `tools/g1_jetson_walker.sh` pushes to PC2) has
no local change. `--against` lists every deploy-set file whose bytes differ in another
checkout.

**Not in the pin.** The teleop working tree the data was recorded from was dirty, and
the operator laptop's teleop tree still carries uncommitted changes to 15 deploy-set
files, among them `g1_jetson_walker.py` (SIGTERM damps all 29 motors and holds the
damping 3 s before exiting), `g1_jetson_remote.py` (`RemoteLower.detach`),
`g1_backend.py` (detach instead of stop on exit), `jetson_split.py` and
`g1_lower_transport.py` (a leg-path timing summary). Which of them were present during
the recording is unknown. To run them here, commit them in the teleop repository, push,
and bump the pin.

### Bumping the pin

```bash
git -C g1_walk/teleop-cbf-perception fetch origin <commit>
git -C g1_walk/teleop-cbf-perception checkout --detach <commit>
git -C g1_walk/teleop-cbf-perception submodule update --init --filter=blob:none third_party/g1_walk
# set TELEOP_COMMIT=<commit> in g1_walk/pins.env, then
git add g1_walk/teleop-cbf-perception g1_walk/pins.env
python g1_walk/check_pin.py && bash g1_walk/tests/sim_e2e.sh --python $PY
```

## Known differences from the recording

* The walker runs on the laptop's GPU and reaches the robot over the cable; during the
  recording it ran on the robot's PC2 (Orin NX, GPU) next to the robot.
* The arms: HTD's IK (casadi, legs and waist locked, 4-tap smoothing) replaces the
  teleop's arm engine; the commanded joints differ from the trajectory the policy's
  wrist targets were computed from by the IK's own error (0.14 rad p95 on the test's
  trajectory, offline). The teleop's 8 Hz reference shaper and per-joint speed fraction
  are replaced by the engage glide and a uniform 4 rad/s cap.
* With the IK solved inline (before `ik_worker.py`), `rt/robot/state`, `eef_state` and
  the touch topics arrived at about 34 Hz in the sim test (50 Hz asked): each solve held
  the Python interpreter for about 20 ms, as in HTD's own controller. The solve now runs
  in its own process. policy_rollout steps at 30 Hz.
* The walk command goes to the walker as the policy gives it (clipped); the teleop
  passed stick commands through its walk arbiter's ramp first. The walker slews both
  the same way (0.5 m/s², 1.0 rad/s²).
* The Modbus client library: this environment has pymodbus 3.6.9 (the Inspire SDK's
  pin, which policy_rollout's touch import needs); the teleop env had 2.5.3. The driver
  supports both.
* The recording's `PSI_TORSO_ATTITUDE=yaw`, `PSI_HAND_TYPE=inspire_modbus` and the
  hand addresses are inferred from the data and the operator laptop's TUI profile, not
  recorded.
* In the sim (one run each), a 0.2 m/s walk command barely moved the robot (7 cm in
  8 s, leaning 8° forward) and 0.3 m/s walked (2.2 m in 9 s); the data has |vx| up to
  0.33 m/s. The teleop's own sim walk gates are recorded as failed
  (`docs/G1_WALK_ISSUES.md` in the submodule).

## Stopping, and safety

* Stop in reverse order: policy_rollout, then the controller (`q` or Ctrl-C: the walker
  stands and keeps balancing), then the walker.
* **Stopping the walker** (Ctrl-C or SIGTERM in its terminal) at the pinned commit stops
  its composer and exits, after which **no process commands the motors**. What the G1
  does then is unknown (`G1_WALK_ISSUES.md` item 9). Have it on the gantry or sitting
  first. The teleop's uncommitted release (damp, hold, exit) is not in the pin.
* L2+B on the Unitree remote damps every motor (the robot sinks); A freezes arms and
  waist and stands the walker. Both are read by the walker itself and work with the
  laptop's other processes gone; both latch until the walker restarts.
* Never run two walkers: each publishes `rt/lowcmd`. `pgrep -af g1_jetson_walker` after
  every session.
* The controller sends nothing to the arms before the walker is live, never publishes
  `rt/lowcmd` or `rt/arm_sdk`, and drops a policy frame with a wrong field length or a
  non-finite value.

## Tests

```bash
python -m unittest discover -s g1_walk/tests -p 'test_*.py'    # offline, no DDS
bash g1_walk/tests/sim_e2e.sh --python $PY                      # MuJoCo, about 2 min
```

`sim_e2e.sh` starts the sim host, the walker and the controller, plays a scripted
policy (`tests/sim_drive.py`: hold, arm and finger motions with a torso yaw, a 0.3 m/s
walk, a 0.5 rad/s turn, a silence mid-turn, a reset) and scores it: robot up, arm
command against the trajectory and measured arms against the command, finger
read-back, walk displacement and turn angle, walk zero on silence, arms home after the
reset, exactly one `rt/lowcmd` writer and none on `rt/arm_sdk`, state topic rates,
the controller's detach and every process gone afterwards. Logs and `summary.json` go
to `g1_walk/tests/logs/`.

Run the sim tests and replays on mains power with the performance profile
(`powerprofilesctl set performance`). On battery or in power-saver mode the CPU drops
to about 1.2 GHz: the test then fails its timing checks (state rates, arm tracking,
finger swing) and a replay under-tracks arms and fingers. `check_env.py` reports it;
`controller.py` refuses a real interface in that state unless given `--allow_low_power`.

## Replaying recorded episodes in the sim

To see what a trained policy would do on the robot, replay its command stream for a
validation episode through this stack and compare it with the operator's commands and
with what the real robot measured:

```bash
# 1. on the training machine: the policy's per-step commands on the rollout schedule
#    (async inference + crossfade, at this laptop's inference latency); writes
#    <out>/<checkpoint>/<episode>_commands_<schedule>.npz
python policy_rollout/replay_eval.py --model_path <run dir> --checkpoints <ckpt> \
    --schedules async+crossfade --latency_ms 11 --save_commands
# 2. here: the policy's commands, then the operator's, through sim host + walker + controller
bash g1_walk/replay/sim_replay.sh --commands EP_commands_async+crossfade.npz --source policy   --python $PY
bash g1_walk/replay/sim_replay.sh --commands EP_commands_async+crossfade.npz --source recorded --python $PY
# 3. compare (--real: the processed episode's proprio plus the raw take's odometry)
$PY g1_walk/replay/plot_replay.py --policy g1_walk/replay/out/EP_policy.npz \
    --recorded g1_walk/replay/out/EP_recorded.npz [--real EP_real.npz]
```

Each replay writes `replay/out/EP_<source>.mp4` (the sim's spectator view) and
`EP_<source>.npz` (`rt/lowstate`, `rt/odommodestate`, the controller's `rt/robot/state`
and `eef_state`). The policy sees the recorded episode's images, state and touch, not
the sim's: this is the policy's open-loop output executed by the real controller and
walker, not a closed-loop rollout.

What the sim can and cannot judge: arms, wrist poses, finger closure, torso commands
and turning follow the real robot (episode 98, operator commands: left wrist 1.1 cm
mean from the real one, final yaw within 3°). Forward walking it cannot judge: in both
episodes replayed (98, 93), once the walker has stepped through the episode's turn
(about 0.6 rad/s) or a short forward walk, the sim robot settles into a forward lean
(pitch +5-6°) and keeps walking forward at 0.2-0.5 m/s. It ignores zero and backward
(-0.1 to -0.2 m/s) commands. The real robot, on the same commands, walked forward and
back and ended within 0.2 m of its start. A gentler turn (0.4 rad/s, then zero) with
the arms still, in home or reaching pose, does not trigger it: the robot stands again
within 2 s and walks backward on -0.2 m/s. The cause inside the walker is not known.
