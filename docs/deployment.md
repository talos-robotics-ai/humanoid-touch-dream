# Deployment

The deployment path is being reworked (robot-agnostic core, a safety supervisor that
monitors the robot state, joystick takeover). This page covers the robot-side setup;
the module guides have the current commands.

## Which controller

The training data was recorded with the teleop stack's G1 walking policy driving the
legs, not with HTD's own whole-body controller. A policy trained on it is deployed on
that same walking policy through [`g1_walk/`](../g1_walk/README.md), which replaces
`robot_control/` and its hand driver. Never run both: each one commands the robot.

## Robot-side PC

1. **Environment.** The PC that runs inference needs a CUDA GPU. Either a new conda
   environment, or a thin venv on top of an interpreter that already has CUDA PyTorch
   (the teleop environment on the operator laptop):

   ```bash
   bash policy_rollout/setup_rollout_env.sh
   bash policy_rollout/setup_rollout_env.sh --base-python ~/anaconda3/envs/teleop/bin/python
   ```

   Activate it afterwards (`conda activate htd_rollout`, or
   `source ~/.venvs/htd_rollout/bin/activate` for the venv). For `g1_walk`, also run
   `bash g1_walk/setup.sh --python <its interpreter>` (`~/.venvs/htd_rollout/bin/python`
   for the venv). It fetches the pinned teleop code and SDKs and checks the environment.

2. **Checkpoint.** Copy the run directory, or at least `policy_last.ckpt` (or the
   checkpoint you chose), `all_configs.json` and `dataset_stats.pkl`, and verify it:

   ```bash
   cd policy_rollout && python check_inference.py --model_path RUN_DIR
   ```

3. **Head camera.** The policy expects the training image layout. For a G1 whose
   RealSense is served by the Jetson camera server over ZMQ, with data converted by
   htd-dataset-converter:

   ```bash
   --head_camera_source zmq --head_camera_device tcp://192.168.123.164:5556 \
   --head_camera_layout duplicate --head_camera_rotate 0
   ```

   `python policy_rollout/test_head_camera.py` checks the layouts without a camera.

## Running

Follow [`g1_walk/README.md`](../g1_walk/README.md) for the walker, controller and
rollout commands, first in MuJoCo on one machine, then on the robot. Pass
`--inference_interval 45` to rollout, as the offline evaluation assumes. Rollout options
(`--inference_interval`, `--crossfade_steps`, `--async_inference`, `--checkpoint`,
`--dream_touch`) and the keyboard controls are in
[`policy_rollout/README.md`](../policy_rollout/README.md).
