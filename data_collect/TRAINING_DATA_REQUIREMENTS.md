# Training data requirements for HTD

This document describes the data an HTD (Humanoid Touch Dreaming) policy needs
for training: which sensors must be recorded, how episodes must be collected,
which HDF5 fields the trainer reads, and how to check a dataset before
training.

For the commands themselves, see the [collection guide](README.md) and the
[training guide](../htd_policy/README.md). This document explains what those
commands must produce and why.

Sources of truth in the code:

| Stage | File |
| --- | --- |
| Recording | [htd_data_collect.py](htd_data_collect.py) (`save_to_hdf5_file`, `_episode_metadata`, `get_embodiment_masks`) |
| Robot-field processing | [process_data_g1_inspire_wb.py](data_processing/process_data_g1_inspire_wb.py) |
| Tactile processing | [process_data_inspire_tactile.py](data_processing/process_data_inspire_tactile.py) |
| Dataset contract | [validate_htd_dataset.py](data_processing/validate_htd_dataset.py) |
| Training loader | [mxt_utils.py](../htd_policy/mxt_utils.py) (`KEY_PATH_MAP_G1_INSPIRE_WB`, `EpisodicDataset`) |
| Model inputs/outputs | [configs/htd/](../htd_policy/configs/htd/) (`g1_inspire_wb:` block) |

---

## 1. Summary checklist

A dataset trains HTD correctly only if **all** of the following hold:

- [ ] Recorded on the **G1 with Inspire hands** (`--embodiment g1_inspire_wb`).
      The trainer rejects every other embodiment.
- [ ] **Bimanual** teleoperation (`--manipulate_mode 3`). Masks for both arms
      and hands must be valid.
- [ ] **Tactile recording on** (`--collect_inspire_touch True`), with the
      collector's `--inspire_hand_type` matching the installed hands (`dftp` or
      `f1`).
- [ ] **30 Hz** control/recording frequency (`--control_freq 30`).
- [ ] **Head stereo camera at 480p** (stored as 480×1280×3), and for non-walk
      profiles **both wrist cameras at 480p** (480×640×3 each).
- [ ] Locomotion commands recorded in `actions/velocity_commands` (always
      present for `g1_inspire_wb`). For walk profiles, episodes must actually
      contain walking/turning.
- [ ] Both processors run on a **copy** of the raw data:
      `process_data_g1_inspire_wb.py`, then `process_data_inspire_tactile.py`.
- [ ] `validate_htd_dataset.py` reports **no errors** for the chosen hand and
      camera profile.
- [ ] **At least two episodes.** The last episode(s), in sorted path order, are
      held out for validation.
- [ ] All episodes in the corpus use the **same** hand type, camera layout, and
      body-mask contract. Do not mix DFTP with F1, or head-only with
      head-and-wrists.

---

## 2. Hardware and sensor requirements

| Sensor / source | Required for | Recorded as |
| --- | --- | --- |
| Apple Vision Pro (head, wrists, 25 hand landmarks per hand) | All teleop **targets** (actions) | `actions/*` |
| Stereo head camera (`3D USB Camera`, 480p → 1280×480) | All profiles | `observations/images/main` |
| Left and right wrist cameras (`Global Shutter Camera`, 480p → 640×480) | DFTP non-walk and F1 profiles | `observations/images/wrist_left`, `wrist_right` |
| G1 low state via the WBC controller (torso pose, 29 joint pos/vel, IMU, EEF poses, velocity commands) | Proprioception, locomotion actions | `observations/proprioceptions/*`, `actions/velocity_commands` |
| Inspire hand state (6 joint positions + 6 finger forces per hand, fingertip link poses) | Hand proprioception, **force dreaming** | `.../other/eef_joint_proprio`, `.../other/eef_link_proprio` |
| Inspire tactile arrays (DFTP: 1062 taxels per hand; F1: 34 raw values per hand) | Tactile tokens, **tactile dreaming** | `.../other/left_touch`, `.../other/right_touch` |
| G1 wireless remote (joystick) | Locomotion commands during teleop | Read by the controller ([g1_wholebody.py](../robot_control/g1_wholebody.py)) and recorded as `actions/velocity_commands` |

The robot controller and hand driver must be running and publishing before
collection begins. The collector waits until robot state and hand state have
both arrived. It then always records the **latest** sample, so a stalled driver
produces repeated values instead of an error. The validator detects
repeated DDS transforms (`repeated_dds_transform`). Watch for this.

---

## 3. Collection settings and training profiles

Choose the training profile **before** collecting. The camera layout and hand
type are fixed by the data.

| HTD profile (embodiment + trunk YAML) | Hands | Cameras | Collector flags that must match |
| --- | --- | --- | --- |
| `dftp_nonwalk_latent16/64` + `trunk_nonwalk` | DFTP | head + both wrists | `--inspire_hand_type dftp --use_wrist_camera True` |
| `dftp_walk_latent16/64` + `trunk_walk` | DFTP | head only | `--inspire_hand_type dftp --use_wrist_camera False` |
| `f1` + `trunk_f1` | F1 | head + both wrists | `--inspire_hand_type f1 --use_wrist_camera True` (start the F1 driver with `--touch`) |

The following flags are required in every profile:

```text
--embodiment g1_inspire_wb
--collect_data True
--collect_inspire_touch True
--manipulate_mode 3
--control_freq 30
--head_camera_type 1 --head_camera_res 480p
--wrist_camera_res 480p              # when wrist cameras are used
--image_save_format jpg              # default; png also works, numpy is not "compressed"
```

Notes:

- **Resolution.** Training defaults to `--main_cam_width 1280 --main_cam_height 480`
  and `--wrist_cam_width 640 --wrist_cam_height 480`. The loader resizes
  other sizes, but the validator expects exactly 480×1280 and 480×640. The
  head image is a side-by-side stereo pair. `split_dual: True` in the config
  splits it into left and right views.
- **Wrist camera assignment.** At startup you type `l` or `r` for the displayed
  wrist camera. A swapped assignment silently trains the policy with the wrong
  view per arm. Check the saved videos (`--save_video True`).
- **Frequency.** The dream and action horizon is 60 steps, which is **2 s at
  30 Hz**. Data at any other frequency changes what the model learns. The
  validator rejects a `frequency` attribute other than 30.

---

## 4. Episode file layout

```text
<dataset_dir>/
└── g1_inspire_wb/
    └── YYYYMMDD_HHMMSS/          # one folder per collection session
        ├── episode_1.hdf5
        ├── episode_2.hdf5
        └── ...
```

The loader searches `--dataset_dir` recursively for `*.hdf5` files. It skips
names containing `features` and, with `--skip_mirrored_data`, names containing
`mirror`. A path containing `flipped` is treated as mirrored data, and its
images are flipped horizontally. Do not use these words in normal episode or
folder names.

### 4.1 File attributes (written by the collector)

| Attribute | Required value | Used by |
| --- | --- | --- |
| `embodiment` | `"g1_inspire_wb"` | **Trainer (hard requirement)**, validator |
| `compress` | `True` for jpg/png images | Trainer (decides whether to JPEG/PNG-decode images) |
| `frequency` | `30` | Validator |
| `hand_type` | `"dftp"` or `"f1"` | Tactile processor, validator |
| `tactile_format` | collector: `dftp_normalized_1062_v1` / `f1_force_direction_proximity_29_v1`; after processing: `dftp_normalized_1062_v1` / `f1_normalized_37_v1` | Validator |
| `camera_profile` | `head-and-wrists` or `head-only` | Validator |
| `schema_version` | `htd_v1` | Validator |
| `coordinate_convention` | `htd_g1_legacy_v1` | Validator |
| `sim` | `False` | Validator (simulation data is unsupported) |

Do not edit these attributes by hand. A mismatch between `hand_type` and the
tactile width makes the tactile processor fail.

---

## 5. Recorded fields (raw collector output)

Every temporal dataset has shape `[N, D]`. Here `N` is the episode's frame
count, which must be **identical across all temporal datasets** in the file.
"Right before left" ordering applies to every two-sided field.

Poses are 6-D `[x, y, z, roll, pitch, yaw]` in the "unified" frame:
z points up, and x points along the operator's forward direction at
calibration. The origin is the operator's head position at calibration.

### 5.1 Observations

| HDF5 path | Shape | Content |
| --- | --- | --- |
| `observations/images/main` | `[N]` vlen bytes (jpg/png) or `[N,480,1280,3]` | Head stereo RGB, rotated 180° to match the mount |
| `observations/images/wrist_left`, `wrist_right` | `[N]` vlen bytes or `[N,480,640,3]` | Wrist RGB (only when wrist cameras are used) |
| `observations/head_cam_timestamp` | `[N]` | Host time for each frame. Must increase monotonically |
| `observations/proprioceptions/body_proprio` | `[N,6]` | Robot torso pose |
| `observations/proprioceptions/eef_proprio` | `[N,12]` | Right and left robot wrist pose |
| `observations/proprioceptions/eef_to_body_proprio` | `[N,12]` | Right and left wrist pose relative to the torso |
| `observations/proprioceptions/gripper_proprio` | `[N,2]` | Placeholder (zeros on G1). Not used by HTD |
| `observations/proprioceptions/other/robot_joint_proprio` | `[N,58]` | 29 G1 joint positions, then 29 joint velocities |
| `observations/proprioceptions/other/eef_joint_proprio` | `[N,24]` | Blocks of 6: right joint pos, left joint pos, right finger force, left finger force |
| `observations/proprioceptions/other/eef_link_proprio` | `[N,30]` | Robot fingertip positions (5 tips × xyz), right then left, in the unified EEF frame |
| `observations/proprioceptions/other/imu_proprio` | `[N,9]` | Accelerometer, gyroscope, RPY. Recorded but not used by HTD configs |
| `observations/proprioceptions/other/left_touch`, `right_touch` | `[N,1062]` (DFTP) or `[N,29]` (F1) | Raw tactile per hand (see [section 7](#7-tactile-data-formats)) |

### 5.2 Actions (teleoperation targets)

Actions are the **operator's commanded targets** from the Vision Pro and the
joystick, not the robot's measured state.

| HDF5 path | Shape | Content |
| --- | --- | --- |
| `actions/body` | `[N,6]` | Operator head pose. On the robot it drives torso height (`z + 0.75`) and torso RPY |
| `actions/eef` | `[N,12]` | Operator right and left wrist pose (world/unified frame) |
| `actions/eef_to_body` | `[N,12]` | Operator right and left wrist pose relative to the head. This drives arm IK |
| `actions/eef_joints_target` | `[N,150]` | 2 hands × 25 AVP landmarks × xyz, in each wrist's local frame |
| `actions/velocity_commands` | `[N,3]` | `[lin_vel_x, lin_vel_y, ang_vel_z]` sent to the WBC |
| `actions/delta_body`, `delta_eef`, `gripper`, `delta_gripper` | various | Recorded for compatibility. Not used by HTD |
| `camera_poses/head_camera_to_init` | `[N,6]` | Head camera pose relative to its pose at calibration |

### 5.3 Masks

| Path | Expected value | Meaning |
| --- | --- | --- |
| `masks/act_body` | `[F, F, T, T, T, T]` | The body x/y targets are **excluded** from the loss. Planar motion comes from `velocity_commands` |
| `masks/act_eef` | 12 × `True` | Both arms are active (requires `--manipulate_mode 3`) |
| `masks/img_wrist` | `[T, T]` | Both wrist cameras are valid |
| `masks/proprio_body` | 6 × `True` | |
| `masks/proprio_eef`, `proprio_gripper`, `proprio_other`, `act_gripper`, `img_main` | all `True` | |

A dataset collected with `--manipulate_mode 1` or `2` (single arm) produces
partial masks and fails validation.

---

## 6. Fields created by preprocessing

The trainer reads **per-side split fields** that the collector does not write.
Both processors must run before training. Each one atomically rewrites each
episode in place, so always run them on a copy of the raw recordings.

### 6.1 `process_data_g1_inspire_wb.py`

| Output | Shape | Derived from |
| --- | --- | --- |
| `actions/right_eef_to_body`, `left_eef_to_body` | `[N,6]` | `actions/eef_to_body` |
| `actions/right_eef_link`, `left_eef_link` | `[N,15]` | AVP **fingertip** landmarks (indices 4, 9, 14, 19, 24 = thumb, index, middle, ring, pinky) from `actions/eef_joints_target` |
| `actions/eef_link` | `[N,30]` | Concatenation of the right and left links above |
| `observations/proprioceptions/right_eef_to_body_proprio`, `left_...` | `[N,6]` | `eef_to_body_proprio` |
| `observations/proprioceptions/right_eef_link_proprio`, `left_...` | `[N,15]` | `other/eef_link_proprio` |
| `observations/proprioceptions/right_eef_joint_pos_proprio`, `left_...` | `[N,6]` | `other/eef_joint_proprio` |
| `observations/proprioceptions/right_eef_finger_force_proprio`, `left_...` | `[N,6]` | `other/eef_joint_proprio` |
| `observations/proprioceptions/robot_joint_pos_proprio`, `robot_joint_vel_proprio` | `[N,29]` | `other/robot_joint_proprio` |

### 6.2 `process_data_inspire_tactile.py`

This processor splits each hand's tactile vector into six regions under
`observations/tactile/{left,right}_{pinky,ring,middle,index,thumb,palm}_tactile`:

| Region | DFTP dim | F1 dim |
| --- | --- | --- |
| pinky, ring, middle, index | 185 each | 5 each |
| thumb | 210 | 5 |
| palm | 112 | 12 |
| **Total per hand** | **1062** | **37** |

For F1, it also normalizes `other/left_touch` and `other/right_touch` in place
from 29 to 37 dimensions.

---

## 7. Tactile data formats

### DFTP (1062 values per hand)

The collector divides raw int16 sensor counts by 4095, so every value must lie
in **[0, 1]**. The validator reports `dftp_touch_range_invalid` otherwise. The
per-hand layout, in order:

| Finger (SDK name) | Pads (taxel grid) | Dim |
| --- | --- | --- |
| pinky (`fingerone`) | tip 3×3, top 12×8, palm 10×8 | 185 |
| ring (`fingertwo`) | tip 3×3, top 12×8, palm 10×8 | 185 |
| middle (`fingerthree`) | tip 3×3, top 12×8, palm 10×8 | 185 |
| index (`fingerfour`) | tip 3×3, top 12×8, palm 10×8 | 185 |
| thumb (`fingerfive`) | tip 3×3, top 12×8, middle 3×3, palm 12×8 | 210 |
| palm | 14×8 | 112 |

In HTD, each region is encoded into a 16-D or 64-D latent with 2 tokens. The
dream heads predict the **future latents**, with an EMA target encoder and a
cosine loss.

### F1 (29 values per hand recorded, 37 after processing)

| Stage | Per finger (×5: little, ring, middle, index, thumb) | Per palm zone (×3: left, center, right) |
| --- | --- | --- |
| Recorded (29) | normal force (N), tangential force (N), direction (rad, `-1` = no contact), proximity [0, 1] | normal (N), tangential (N), direction |
| Processed (37) | normal/20, tangential/20, sin(dir), cos(dir), proximity | normal/20, tangential/20, sin(dir), cos(dir) |

With F1, HTD predicts these raw normalized values directly (the `*_tactile_cap`
keys) instead of latents.

---

## 8. What the HTD model consumes

Taken from the `g1_inspire_wb:` block of the HTD YAMLs. With `--touch_dreaming`,
training uses all three groups.

### 8.1 Observation tokens (state at time *t*)

| Config key | HDF5 field | Dim | Profiles |
| --- | --- | --- | --- |
| `main_image` | `images/main` | 480×1280 RGB | all |
| `left_wrist_image`, `right_wrist_image` | `images/wrist_left`, `wrist_right` | 480×640 RGB | DFTP non-walk, F1 |
| `body_pose_proprio` | `proprioceptions/body_proprio` | 6 | all |
| `eef_pose_proprio` | `proprioceptions/eef_proprio` | 12 | all |
| `{left,right}_eef_to_body_pose_proprio` | split field | 6 + 6 | all |
| `{left,right}_eef_link_proprio` | split field | 15 + 15 | all |
| `robot_joint_pos_proprio` | split field | 29 | all |
| `{left,right}_eef_joint_pos_proprio` | split field | 6 + 6 | all |
| `{left,right}_eef_finger_force_proprio` | split field | 6 + 6 | all |
| `{left,right}_{region}_tactile` | `observations/tactile/...` | see [section 7](#7-tactile-data-formats) | DFTP |
| `{left,right}_{region}_tactile_cap` | `observations/tactile/...` | see [section 7](#7-tactile-data-formats) | F1 |

### 8.2 Action targets (chunk *t … t+59*)

| Config key | HDF5 field | Dim | Loss weight |
| --- | --- | --- | --- |
| `body_pose` | `actions/body` (x, y masked out) | 6 | 1.0 |
| `{left,right}_eef_to_body_pose` | split field | 6 + 6 | 1.0 |
| `{left,right}_eef_link_pose` | split field | 15 + 15 | 0.4 |
| `velocity_commands` | `actions/velocity_commands` | 3 | 1.0 non-walk, 2.0 walk, 3.0 F1 |

### 8.3 Touch-dreaming targets (future observations *t+1 … t+60*)

| Config key | HDF5 field | Dim |
| --- | --- | --- |
| `{left,right}_eef_finger_force` | `proprioceptions/{left,right}_eef_finger_force_proprio` | 6 + 6 |
| `{left,right}_{region}_tactile` (DFTP) | `observations/tactile/...` | 16- or 64-D latent per region |
| `{left,right}_{region}_tactile_cap` (F1) | `observations/tactile/...` | raw normalized values |

Dream targets are **future observations from the same episode**. Finger forces
and tactile data must therefore be recorded at every frame and stay
time-aligned with the images and actions. Missing, frozen, or misaligned touch
data does more than weaken an input: it corrupts the dreaming objective.

### 8.4 Normalization

The trainer computes per-dimension mean and std over **all** supplied episodes,
or over `--stats_dir` if given. Every episode must therefore use the same units,
frames, and orderings. A single episode recorded with different conventions
distorts the statistics for the whole corpus. Statistics are saved with the
checkpoint and reused at rollout.

---

## 9. Dataset content and quality guidelines

**Quantity and split**

- You need at least 2 episodes. The trainer holds out
  `max(--min_val_num, floor((1 - --train_ratio) × N))` episodes, taken from the
  **end** of the sorted path list. Session folder names sort by timestamp, so
  the most recent session becomes validation data.
- Episodes shorter than 60 frames (2 s) still work, because chunks are padded.
  However, most of their training samples are then padding. Prefer full task
  demonstrations, each well over 60 frames long.

**Coverage**

- **Contact.** Touch dreaming learns from contact events. Demonstrations should
  contain the real grasps, presses, insertions, and so on that the task
  requires, with consistent contact strategies across episodes. If the tactile
  signals stay near zero for the whole episode, they provide nothing for the
  dream heads to learn from.
- **Locomotion (walk profiles).** Drive the robot with the joystick during
  collection so that `velocity_commands` contains real walking and turning.
  If every episode has zero velocity, the validator reports
  `stationary_velocity_subset`, and the policy will never learn to walk.
  `--vel_rebalance` can oversample the active timesteps when locomotion is rare.
- **Torso motion.** The body action drives torso height and pitch/roll/yaw
  through the WBC. Include the same range of bending and leaning in the
  demonstrations that the task needs at rollout.

**Consistency**

- Use the same hands, camera mounting, wrist-camera left/right assignment,
  and task setup for all episodes in one training corpus.
- Calibrate each episode facing the intended forward direction (index pinch).
  All poses are expressed relative to this calibration frame.
- Avoid pausing mid-episode (pinky pinch). Recording stops while the robot
  holds its last target, so the episode contains a time jump. The validator
  flags this as `timestamp_gap`.
- Discard failed or aborted demonstrations before processing. The trainer
  imitates everything it is given.

---

## 10. Validate before training

```bash
python data_collect/data_processing/validate_htd_dataset.py "$HTD_DATA" \
  --profile dftp --images all --report /path/outside/dataset/report.json
# F1:        --profile f1
# Head-only: --camera-profile head-only
```

The validator is read-only. Fix every **error** before training. The errors
most relevant to the requirements above:

| Check ID | Meaning |
| --- | --- |
| `embodiment_invalid` | Not recorded as `g1_inspire_wb` |
| `frequency_contract_mismatch` | Not recorded at 30 Hz |
| `required_dataset_missing` / `partial_processing` | A processor was not run or did not finish |
| `temporal_length_mismatch` | Datasets in one file have different frame counts |
| `numeric_nonfinite` | NaN or Inf values in a numeric field |
| `mask_value_invalid` | Not bimanual, or wrong body mask |
| `touch_*`, `tactile_split_shape_invalid`, `hand_profile_*` | Tactile missing, wrong hand type, or wrong width |
| `dftp_touch_range_invalid` | DFTP tactile values outside [0, 1] |
| `image_decode_or_shape_invalid` | Wrong resolution or corrupt frames |
| `repeated_dds_transform` | Stale robot or hand state repeated across frames |
| `corpus_mixed_hand_profiles` / `corpus_mixed_body_mask_contracts` | Episodes in the corpus are inconsistent |

Warnings such as `placeholder_gripper` (expected on G1), `timestamp_gap`, and
`stationary_velocity_subset` do not block training. Review them anyway.

---

## 11. Common pitfalls

| Symptom | Cause |
| --- | --- |
| `KeyError` on `.../right_eef_to_body_proprio` or `observations/tactile/...` at train start | A preprocessing step was skipped |
| `unsupported policy embodiment 'human'` | Collected without `--embodiment g1_inspire_wb` |
| Tactile processor: `hand_type metadata ... conflicts with tactile widths` | The collector's `--inspire_hand_type` did not match the hands |
| Model shape mismatch on wrist-image or tactile keys | Training profile does not match the data, e.g. a non-walk YAML on head-only data, or a DFTP YAML on F1 data |
| Policy swaps arm behavior | Wrist cameras were assigned to the wrong side at collector startup |
| Dream losses stay flat or tactile looks frozen | Tactile driver was not streaming: the collector kept the last sample. Check `repeated_dds_transform` and plot `left_touch` |
| Raw recordings changed after processing | Processors rewrite files in place. Always process a copy |
