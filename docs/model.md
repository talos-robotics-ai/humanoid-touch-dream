# The HTD model as trained in this fork

HTD (Humanoid Touch Dream) is a transformer policy that maps the robot's current
observation to a chunk of future whole-body commands, trained with an auxiliary
"touch dreaming" objective that predicts future tactile signals. The code is
`htd_policy/` (model in `htd_policy/detr/models/detr_vae.py`, class `CrossTransformer`,
built by `train_mxt.py --touch_dreaming`). This page describes the configuration used
on our data: embodiment `g1_inspire_wb`, profile
`configs/htd/dftp_walk_latent16.yaml` + `configs/htd/trunk_walk.yaml`, and the
tactile-scale variant `dftp_walk_latent16_tscale.yaml`.

## Inputs

One observation at time *t*, all from the processed HDF5 episode (see
[dataset.md](dataset.md)). Every modality is tokenized separately.

| Modality | Dims | Tokenizer | Tokens |
| --- | --- | --- | --- |
| `main_image` | 480x1280x3 RGB, two 640x480 views side by side | ResNet18, each half encoded separately (`split_dual`), cross-attention to query tokens | 128 |
| `body_pose_proprio` | 6: torso pose (xyz, rpy) in the feet frame | MLP 512-512 + LayerNorm | 4 |
| `eef_pose_proprio` | 12: right and left wrist pose in the feet frame | MLP | 4 |
| `left_/right_eef_to_body_pose_proprio` | 6 each: wrist pose in the torso frame | MLP | 4 + 4 |
| `left_/right_eef_link_proprio` | 15 each: five measured fingertip positions | MLP | 4 + 4 |
| `robot_joint_pos_proprio` | 29 motor positions | MLP | 1 |
| `left_/right_eef_joint_pos_proprio` | 6 each: hand joint angles | MLP | 1 + 1 |
| `left_/right_eef_finger_force_proprio` | 6 each: finger forces | MLP | 4 + 4 |
| 12 tactile patches: `left_/right_{pinky,ring,middle,index,thumb,palm}_tactile` | 185 / 185 / 185 / 185 / 210 / 112 taxels, raw value / 4095 | `FingerPatchTactileEncoder`: a small CNN per taxel patch, concatenated and projected to a 16-dim latent | 2 each |

Proprioception is normalized with the dataset mean and standard deviation
(`dataset_stats.pkl`); images with the ImageNet statistics; tactile inputs are fed as
recorded (see [Tactile input scale](#tactile-input-scale)). Every token gets a
sinusoidal position embedding.

## Trunk

An encoder-decoder transformer (`trunk_walk.yaml`): hidden size 512, 8 heads,
feed-forward 1024, 4 encoder and 4 decoder layers, dropout 0.1. The decoder emits a
fixed number of output tokens per head (24 for each pose head, 30 for each hand head,
15 for the velocity head), which the heads cross-attend to.

## Outputs: the action chunk

Each action head predicts the next **60 steps** (2 s at 30 Hz) of one command group.
The targets are the teleoperator's commands recorded with the observation, normalized
per dimension; the loss is an L1 over the unpadded steps, weighted per head.

| Head | Dims | Meaning | Loss weight |
| --- | --- | --- | --- |
| `body_pose` | 6 | torso height and orientation (x, y are masked out) | 1.0 |
| `left_eef_to_body_pose`, `right_eef_to_body_pose` | 6 each | wrist target in the torso frame (xyz, rpy) | 1.0 |
| `left_eef_link_pose`, `right_eef_link_pose` | 15 each | five fingertip targets | 0.4 |
| `velocity_commands` | 3 | walking command (vx, vy, yaw rate) | 2.0 |

How these become robot commands (frame offsets, rotation remaps, hand keypoints) is
implemented once in `policy_rollout/g1_command.py` and shared by rollout and the
offline evaluation. The rollout commands in this fork pass `--inference_interval 45`
(rollout's own default is 60): 45 of the 60 steps are executed before the next inference,
and consecutive chunks are blended (`policy_rollout/README.md`, "Chunk schedule").

## Touch dreaming

The dream heads read the raw trunk features and predict, for the same 60 future steps,
what the hands will feel:

- `left_/right_eef_finger_force` (6 each): future finger forces, as values, normalized
  like the proprioception. Loss weight 0.8.
- 12 tactile heads, one per patch (16 dims each): the future **latent** of that patch,
  not the raw taxels. The target is produced by an **EMA teacher**: a frozen copy of the
  patch's tactile encoder, updated after every optimizer step as
  `teacher = 0.999 * teacher + 0.001 * student`. The loss is
  `1 - cosine(pred, target) + 1.0 * smooth_l1(|pred|, |target|)`
  (`dream_ema_cosine`, `dream_mag_beta`). Loss weight 1.0.

Dreaming is a training signal only; inference does not run the dream heads unless
rollout is asked to record them (`--dream_touch True`).

Two validation metrics watch for a degenerate teacher: `ema_std_<patch>` (spread of
the teacher latents over the batch) and `ema_cosine_sim_<patch>` (mean pairwise cosine
similarity). A healthy head has `ema_std` clearly above 0 and `ema_cosine_sim` well
below 1.

### Tactile input scale

On DFTP data the taxels arrive as `raw / 4095`, so contact values are sparse and of
order 1e-3. At that scale the encoder's biases dominate its output, the EMA teacher
emits a near-constant latent for every input, and the dream losses collapse to ~0 with
`ema_std = 0` and `ema_cosine_sim = 1`. This happens at random initialization and on
upstream's own DFTP episodes: the cause is the encoder's input scale, not our data.

`FingerPatchTactileEncoder` therefore takes `input_scale` (multiply the taxels before
the CNN) and `input_log1p` (apply `log1p` after scaling). Both default to the upstream
behaviour. `configs/htd/dftp_walk_latent16_tscale.yaml` sets `input_scale: 100.0` on all
twelve patches. The teacher is a copy of the encoder, so student and teacher share the
transform. On our data this restores a non-degenerate teacher for the left thumb,
middle and ring (`ema_std` 0.015-0.13, cosine 0.88-0.91); the left index stays
collapsed, and the right hand barely touches anything in this task.

## Profiles

| Profile | Embodiment YAML | Trunk YAML | Use |
| --- | --- | --- | --- |
| DFTP walk, 16-dim | `configs/htd/dftp_walk_latent16.yaml` | `configs/htd/trunk_walk.yaml` | head-only data with walking: our default |
| DFTP walk, 16-dim, tactile x100 | `configs/htd/dftp_walk_latent16_tscale.yaml` | `configs/htd/trunk_walk.yaml` | same, with the input scale above |
| DFTP walk, 64-dim | `configs/htd/dftp_walk_latent64.yaml` | `configs/htd/trunk_walk.yaml` | larger tactile latents |
| DFTP non-walk, 16/64-dim | `configs/htd/dftp_nonwalk_latent*.yaml` | `configs/htd/trunk_nonwalk.yaml` | upstream's head + wrist cameras, no velocity head |
| F1 hands | `configs/htd/f1.yaml` | `configs/htd/trunk_f1.yaml` | Inspire F1 sensors, raw tactile targets |

The walk profiles have no wrist-camera tokenizer and weight the velocity head 2.0. The
YAML pair of a run is recorded in its `all_configs.json`; rollout and evaluation check
that the YAMLs they are given match it.

## Training artifacts

A run directory (`runs/<prefix>_<task>_HTD/`) contains:

| File | Content |
| --- | --- |
| `policy_last.ckpt` | weights at the last step (what rollout loads by default) |
| `policy_best.ckpt` | weights at the lowest validation loss, written when training finishes (see [training.md](training.md#what-to-watch)) |
| `policy_step_<N>_seed_<s>.ckpt`, `optimizer_step_...` | periodic snapshots every 10k steps |
| `training_last.ckpt`, `optimizer_last.ckpt`, `training_state.json` | resume state for `--auto_resume`: weights, optimizer, step and best validation loss |
| `all_configs.json` | every argument and the resolved model configuration |
| `dataset_stats.pkl` | normalization statistics; must travel with the checkpoint |
| `dataset_split.json` | the training and validation episode paths |
| `config.pkl` | the trainer's configuration object |
| `train_val_seed_<s>.png` | training and validation loss curves |
| `wandb_run.json` | the W&B run id, with `WANDB=1` |

Rollout and evaluation load weights strictly and reject an embodiment or trunk YAML that
differs from the one in `all_configs.json` (`policy_rollout/mxt_loading.py`). They do not
check `dataset_stats.pkl`: a statistics file from another run loads silently and produces
wrong actions, so keep each run's file with its checkpoint.

## Code map

| Path | What it holds |
| --- | --- |
| `htd_policy/train_mxt.py` | training loop, validation, checkpointing, W&B |
| `htd_policy/mxt_utils.py` | HDF5 loading, key-to-path map, split, normalization statistics, velocity rebalancing |
| `htd_policy/policy.py` | `MXTPolicy` wrapper: loss, EMA update, strict (de)serialization |
| `htd_policy/detr/models/detr_vae.py` | `CrossTransformer`: tokenization, trunk, action and dream heads, losses |
| `htd_policy/detr/models/mxt_definitions/tokenizer.py` | image, proprioception and tactile tokenizers |
| `htd_policy/configs/htd/` | the profiles above |
