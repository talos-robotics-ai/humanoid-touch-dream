# Policy rollout

Run a trained HTD, MXT, or ACT (HIT) policy alongside the
[robot controller](../robot_control/README.md). HTD and MXT use `--policy_class MXT`;
ACT rollout requires a model trained with `--decoder_only` (HIT) and uses
`--policy_class HIT`. `--touch_dreaming` is only needed during training.

## Install

### Inference-only machine (robot laptop)

For a PC that only runs a trained policy, [`setup_rollout_env.sh`](setup_rollout_env.sh)
creates the environment from [`requirements-rollout.txt`](requirements-rollout.txt):

```bash
bash policy_rollout/setup_rollout_env.sh            # new conda env htd_rollout (torch 2.9.1, CUDA 12.8)
# or, on a PC that already has CUDA PyTorch (e.g. the teleop env), a thin venv on top of it:
bash policy_rollout/setup_rollout_env.sh --base-python ~/anaconda3/envs/teleop/bin/python
```

Then copy the run directory (`policy_last.ckpt`, `all_configs.json`,
`dataset_stats.pkl`) to the PC and check it:

```bash
cd policy_rollout
python check_inference.py --model_path RUN_DIR
```

It loads the checkpoint the way rollout does, captures the CUDA graph and
prints the inference time. To confirm the PC computes the same actions as the
training server, run `--save_reference ref.pt` there and `--reference ref.pt`
here; it reports the largest difference and exits non-zero above `--tolerance`.
Measured: RTX 5070 Laptop GPU 10.7 ms per inference, within 1.3e-5 of the L40S.

Rollout builds the image encoder without ImageNet weights (the checkpoint
replaces them), so it needs no internet access.

### Full installation

Complete the [policy installation](../htd_policy/README.md#installation).
Prepare CycloneDDS and the SDK checkouts from
[controller setup](../robot_control/README.md#install-from-the-repository-root),
and the camera utilities from [collection setup](../data_collect/README.md#requirements-and-installation).
Use a Linux graphical desktop for camera selection and keyboard input.
Then run these commands from the repository root:

```bash
export HTD_ROOT="$(pwd)"
export HTD_DEPS="$(dirname "$HTD_ROOT")/htd-deps"
conda activate htd_policy
python -m pip install -r "$HTD_ROOT/policy_rollout/requirements-hardware.txt"
```

Install the selected hand SDKs into this same `htd_policy` environment using the
[SDK commands in collection setup](../data_collect/README.md#requirements-and-installation):
Unitree and Inspire SDKs for DFTP, or the Inspire F1 SDK for F1.
If your SDK checkouts are elsewhere, set `HTD_DEPS` to their parent directory.

## Prepare the checkpoint and hardware

`--model_path` must point to your training output directory with
`policy_last.ckpt`, `all_configs.json`, and `dataset_stats.pkl` kept together.
Use the original normalization statistics. For HTD/MXT, use the same
embodiment/trunk YAMLs as training, including custom latent dimensions. Rollout
resizes camera images to the input dimensions recorded in the training configuration.

Start the matching hand driver and G1 controller using the
[hardware startup instructions](../robot_control/README.md#hardware-handoff-and-startup).
Complete that handoff before launching the controller; it can move the robot
before its `s` prompt.

Match the cameras and hand type to training:

- By default the head camera must appear as `3D USB Camera`. At the default
  `480p`, it supplies a combined 1280×480 stereo image, rotated 180 degrees by
  rollout. For other head cameras, see [Head camera](#head-camera).
- With `--use_wrist_camera True`, connect two `Global Shutter Camera` devices.
  Label the previewed camera `l` or `r` followed by Enter; the other gets the
  remaining side. Each wrist image is 640×480 at `480p`.
- Match the hand driver, controller's `--hand_type`, and rollout's
  `--inspire_hand_type` (`dftp` or `f1`) to the installed hands and trained model.

Add `--dds_interface YOUR_INTERFACE` if needed. It selects the rollout interface
when `CYCLONEDDS_URI` is unset; the controller and hand driver need matching DDS
network settings.

## Head camera

The policy's head input is one 1280×480 frame holding two 640×480 views side by
side. [`head_camera.py`](head_camera.py) builds that frame from any supported
camera in three independent steps. Each step must match how the training
images were made:

| Option | Values | Default |
| --- | --- | --- |
| `--head_camera_source` | `stereo_usb` (V4L2 stereo camera found by name), `opencv` (any V4L2/OpenCV device), `realsense` (RealSense on this PC, needs `pyrealsense2`), `zmq` (RealSense served over ZMQ REQ/REP, needs `pyzmq`) | `stereo_usb` |
| `--head_camera_device` | device name, `/dev/videoN` or index (`stereo_usb`/`opencv`), serial (`realsense`), address (`zmq`) | `3D USB Camera` / first device / first RealSense / `tcp://192.168.123.164:5556` |
| `--head_camera_layout` | `stereo`: the frame is already a side-by-side pair; `duplicate`: a mono view copied into both halves; `stretch`: a mono view resized to 1280×480 | `stereo` for `stereo_usb`, else `duplicate` |
| `--head_camera_rotate` | `0`, `90`, `180`, `270` (clockwise, applied before the layout) | `180` for `stereo_usb` on G1, else `0` |
| `--head_camera_source_res` | requested source size `WxH` for `opencv`/`realsense` | one view (mono) or the pair (stereo) |

The defaults reproduce HTD's own head camera. Data converted with
htd-dataset-converter from a mono RealSense uses the converter's
`--main-layout` (`duplicate` by default), so pass the same value to
`--head_camera_layout`. For a G1 whose torso RealSense is served by the Jetson
camera server:

```bash
--head_camera_source zmq --head_camera_device tcp://192.168.123.164:5556 \
--head_camera_layout duplicate --head_camera_rotate 0
```

`python test_head_camera.py` checks the layouts against the converter and the
original stereo path, and the ZMQ client against a fake server. No camera is
needed.

## Check a checkpoint offline

[`replay_eval.py`](replay_eval.py) runs a checkpoint on recorded episodes,
loading it exactly as rollout does. By default it uses the run's validation
episodes. For each frame it compares the predicted action chunk with the
recorded commands, and reports the errors as hand position (cm), rotation
(deg), fingertips (mm), torso height and velocity commands. A "freeze"
baseline, which keeps sending the current command, gives scale. It also plots the predicted trajectory, stitched from
chunks the way rollout executes them, over the recording:

```bash
python replay_eval.py --model_path ../runs/run_my_task_HTD \
  --checkpoints policy_best.ckpt policy_last.ckpt --interval 45
```

It also plays each episode from start to end on the rollout schedule (infer
every `--interval` steps, execute that many) and reports, per episode:

- tracking error over the whole task, including where the hand ends up;
- whether each recorded grasp and release happens, and how early or late;
- when walking starts and stops, and the path the velocity commands trace;
- how far the command jumps when rollout switches chunks;
- the measured inference time.

It then replays each episode through rollout's own chunk scheduler for each
schedule in `--schedules` (default: upstream, crossfade, async,
async+crossfade; see [Chunk schedule](#chunk-schedule)) with the inference time
measured on this machine, or `--latency_ms` to simulate the robot computer. For
each schedule it reports tracking, grasp timing, the largest command step
between control steps, and how long commands were held.

Results go to `MODEL_DIR/replay_eval/` (`summary.md`, `metrics.json`, one
plot and `.npz` per episode). This is open loop: it catches broken artifacts,
wrong units and policies that ignore their inputs. It does not measure task
success, because the policy never sees the results of its own commands.

Rollout loads `policy_last.ckpt` unless `--checkpoint` names another file in
the run directory, for example `--checkpoint policy_best.ckpt`.

## Run HTD or MXT

Open a terminal at the repository root. To run your checkpoint from the
[HTD training example](../htd_policy/README.md#train-htd) with DFTP hands and
head/wrist cameras:

```bash
conda activate htd_policy
cd policy_rollout
OMP_NUM_THREADS=1 python policy_rollout.py \
  --model_path ../runs/run_my_task_HTD \
  --policy_class MXT --embodiment g1_inspire_wb \
  --inspire_hand_type dftp \
  --use_wrist_camera True --send_velocity_commands False \
  --device cuda:0 --control_freq 30 --inference_interval 45 \
  --embodiment_config_path ../htd_policy/configs/htd/dftp_nonwalk_latent16.yaml \
  --trunk_config_path ../htd_policy/configs/htd/trunk_nonwalk.yaml
```

For another [model configuration](../htd_policy/README.md#train-htd) or
[MXT baseline](../htd_policy/README.md#mxt-training), select its run directory
and training YAMLs. Set `--use_wrist_camera False` for head-only models and
`--inspire_hand_type f1` for F1 hands. Enable `--send_velocity_commands True`
to send walking commands from a model trained to predict them.

A model trained on data recorded with the teleop stack's G1 walking policy
(the datasets htd-dataset-converter makes)
runs on that walking policy instead of HTD's controller: start the walker and
`g1_walk/controller.py` as [g1_walk/README.md](../g1_walk/README.md) describes,
then this command unchanged.

## Run ACT (HIT)

The [ACT training example](../htd_policy/README.md#act-training) already uses
`--decoder_only` to train HIT. To deploy its checkpoint, start at the repository root:

```bash
conda activate htd_policy
cd policy_rollout
OMP_NUM_THREADS=1 python policy_rollout.py \
  --model_path ../runs/run_my_task_HIT --policy_class HIT \
  --inspire_hand_type dftp \
  --use_wrist_camera True --send_velocity_commands False \
  --device cuda:0 --control_freq 30 --inference_interval 45
```

The observation and action configuration is loaded from the saved training
metadata. For a head-only walking model, set `--use_wrist_camera False` and
`--send_velocity_commands True`.

## Operate rollout

Confirm live camera, robot, hand, and required tactile feedback. Rollout uses
the latest received observations without rejecting stale feedback.
To start, press **`1`**, wait for the controller reset to finish, then press **`2`**.

| Key | Action |
| --- | --- |
| `1` | Request a controller reset and restart the action chunk. |
| `2` | Start after reset, or resume a paused rollout. |
| `3` | Pause command publishing. |
| Ctrl+C | Exit rollout. |

Pausing or exiting rollout leaves the robot controller running with the last
received targets. Press **`2`** to resume a paused rollout.

Keep `--inference_interval` positive and no greater than the checkpoint's action
horizon (60 in the example).

### Chunk schedule

Rollout switches to a new action chunk at chunk step `--inference_interval`.
Two options make the switch smooth:

- **`--crossfade_steps N`** (default 15) blends from the old chunk into the
  new one over `N` control steps instead of jumping. The old chunk must still
  have those steps, so the blend is limited to horizon − interval (15 with a
  60-step horizon and `--inference_interval 45`; none with the upstream
  interval of 60). Rotations blend by slerp. The walk command
  (`velocity_commands`: vx, vy, yaw rate) is not blended: it switches to the
  new chunk's prediction at once, so the crossfade never low-passes the
  walking speed or the turn rate (`chunk_schedule.G1_COMMAND_KINDS`, kind
  `switch`).
- **`--async_inference`** (default True) runs inference in a background
  thread, so commands keep flowing while the next chunk is computed. Rollout
  measures inference time and starts each inference early enough for the
  switch to stay at chunk step `--inference_interval`. It enters the new chunk
  at the step matching the current time. Inferences run every
  `interval − lead` steps, where lead is the inference time plus one step.

### Real-time inference

Inference runs on the GPU only: rollout stops with an error if `--device` is
not an available CUDA device or the model is not on it, and prints the GPU it
uses. RTX 50-series GPUs need PyTorch 2.7 or newer built for CUDA 12.8.

At start-up, before you press `1`, rollout warms the model up and captures
one forward pass as a CUDA graph (`--cuda_graph`, default True). Each
inference then replays that graph instead of launching about 1,200 kernels from
Python. The graph is compared with the normal forward and used only if the
outputs match. Rollout prints the result, for example
`Inference: CUDA graph (matches eager to 0.0e+00); 5.9 ms on NVIDIA L40S`.
If capture fails or an input changes shape, inference falls back to the normal
forward.

The control thread only copies the latest robot state and a `uint8` camera
frame. Moving them to the GPU, reordering the image channels, and inference
happen on the inference thread. `--dream_touch` files are written by a
background thread.

Measured with the rack's shared L40S under load (30 Hz, async, crossfade 15):
inference 6.6 ms median (11 ms max, first chunk 6 ms), control step 5 ms max,
largest gap between commands 35 ms. The same loop with upstream's inference
path took about 360 ms per inference and froze the robot's commands at every
chunk switch.

`--async_inference False --crossfade_steps 0` restores the upstream behaviour:
the control loop stops during each inference and the command jumps to the new
chunk. [`replay_eval.py`](replay_eval.py) compares these schedules on recorded
episodes (`--schedules`, `--latency_ms`).

See [starting, resetting, and stopping the controller](../robot_control/README.md#start-reset-and-stop-the-controller).

Add `--dream_touch True` to record an HTD model's dream predictions and
corresponding sensor observations under `MODEL_DIR/dream_touch_results/TIMESTAMP/`.
