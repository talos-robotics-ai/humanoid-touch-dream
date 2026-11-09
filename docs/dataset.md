# Dataset: from the Hugging Face repo to a training set

The policies in this fork are trained on
[`Relo02/loco-manipulation-data`](https://huggingface.co/datasets/Relo02/loco-manipulation-data),
a public dataset of whole-body teleoperation takes recorded on a Unitree G1 with
Inspire RH56DFTP hands and a head-mounted RealSense D435i. The repository holds
one folder per task. The task used so far:

| Folder | Prompt | Takes | Frames |
| --- | --- | --- | --- |
| `white_becker_from_sliding_ruller/` | "pick up the white becker from the black sliding ruller and place it into the blue box" | 99 | 111,719 (62.1 min at 30 Hz) |

Each task folder has two views of the same takes:

```text
<task>/
  teleop_raw/manipulation/<subtask>/episode_000000/   the take as recorded by the teleop
      ep.json, frames.parquet, main.mp4, depth.tar     (metadata, 30 Hz state table, colour video, depth PNGs)
  htd/g1_inspire_wb/<task>__manipulation/episode_1.hdf5 ...
                                                       the same take in HTD's raw episode format
```

Raw takes count from 0, HTD episodes from 1: `teleop_raw/.../episode_0000NN` is
`htd/.../episode_{NN+1}.hdf5`. Only the `htd/` files are needed for training.

The HTD files were produced from the teleop takes by
[htd-dataset-converter](https://github.com/talos-robotics-ai/htd-dataset-converter).
They are what HTD's own collector would have written for
`--embodiment g1_inspire_wb --inspire_hand_type dftp --use_wrist_camera False`:
head camera only, DFTP touch, and a placeholder gripper channel. Two
properties of this data drive the training settings in [training.md](training.md):

- **Head-only images.** The mono RealSense frame is duplicated into the two halves of
  HTD's 1280x480 stereo slot (the converter's `--main-layout duplicate`). Rollout must
  build its camera frame the same way (`--head_camera_layout duplicate`).
- **Little walking.** Non-zero forward velocity in about 7% of frames, yaw rate in about
  8%, lateral velocity never. Training uses the walk profile and velocity rebalancing.

## One command

With the `htd_policy` environment active (see [training.md](training.md#environment)) and
the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) installed:

```bash
scripts/prepare_dataset.sh                       # white_becker_from_sliding_ruller, under ~/htd_data
scripts/prepare_dataset.sh --task OTHER_TASK --data-root /data/htd
```

The script downloads the task's `htd/` episodes, links them into a processed copy,
runs HTD's two processors and validates the result. It writes:

```text
<data-root>/
  hf/loco-manipulation-data/<task>/htd/...             the download, untouched (resumable)
  processed/<task>/g1_inspire_wb/<task>__manipulation/episode_N.hdf5
                                                       the training set
  logs/validate_raw_<task>.json, validate_processed_<task>.json
```

Pass `processed/<task>` to training as `DATA`. The steps below are what the script runs.

## Step by step

### 1. Download

HTD episodes of one task (about 15.5 GB for the 99 takes; the processed set below takes
about as much again):

```bash
hf download Relo02/loco-manipulation-data --repo-type dataset \
  --local-dir ~/htd_data/hf/loco-manipulation-data \
  --include "white_becker_from_sliding_ruller/htd/*"
```

The dataset is public; no login is needed. Re-running the command resumes an
interrupted download. Add `--dry-run` to list files and sizes first. The dataset card
on Hugging Face has the patterns for the raw takes and single episodes.

### 2. Make the processed copy

HTD's processors rewrite each episode in place, so they run on a copy. Each processor
writes a new file and renames it over the old one, which makes hard links safe: the
download is never modified, and the processed episodes end up as their own files.

```bash
TASK=white_becker_from_sliding_ruller
SRC=~/htd_data/hf/loco-manipulation-data/$TASK/htd/g1_inspire_wb
DATA=~/htd_data/processed/$TASK
mkdir -p "$DATA"
cp -al "$SRC" "$DATA/"        # cp -a if the two directories are on different filesystems
```

The trainer expects exactly this layout: `<DATA>/g1_inspire_wb/<group>/episode_N.hdf5`.

### 3. Process and validate

From the repository root, in the `htd_policy` environment:

```bash
python data_collect/data_processing/validate_htd_dataset.py "$DATA" \
  --profile dftp --stage collector --camera-profile head-only --images all \
  --report ~/htd_data/logs/validate_raw_$TASK.json
python data_collect/data_processing/process_data_g1_inspire_wb.py --dataset_dir "$DATA"
python data_collect/data_processing/process_data_inspire_tactile.py --dataset_dir "$DATA" --hand_type dftp
python data_collect/data_processing/validate_htd_dataset.py "$DATA" \
  --profile dftp --camera-profile head-only --images all \
  --report ~/htd_data/logs/validate_processed_$TASK.json
```

What the processors add, without changing any recorded value:

- `process_data_g1_inspire_wb.py` splits the two-sided fields into per-hand datasets
  (`actions/left_eef_to_body`, `actions/right_eef_link`,
  `observations/proprioceptions/left_eef_finger_force_proprio`, ...), which are the
  keys the model's tokenizers read.
- `process_data_inspire_tactile.py --hand_type dftp` splits each hand's 1062 taxels
  into the per-patch arrays the tactile tokenizers consume
  (`observations/tactile/<hand>_<patch>_tactile`: pinky, ring, middle and index 185
  each, thumb 210, palm 112).

The validator is read-only and exits non-zero on errors. These **warnings are expected**
for this dataset and are not errors:

- `timestamp_gap` (most episodes): the teleop drops paused stretches, so consecutive
  head-camera timestamps can be more than one period apart;
- `placeholder_gripper` (every episode): the gripper proprioception is all zero; Inspire
  hands have no gripper and no HTD configuration consumes it;
- `dftp_hand_order_provenance_unresolved` and `collector_saver_provenance_unresolved`
  (once per run): the converter does not record the collector code revision and hand
  ordering provenance that the validator looks for.

### 4. Train and validation episodes

The trainer sorts the episode paths as strings and holds out the last
`max(--min_val_num, int((1 - --train_ratio) * N))` of them. With 99 episodes and
`--train_ratio 0.9` the validation set is `episode_91.hdf5` to `episode_99.hdf5`; the run
directory records the exact split in `dataset_split.json`. Normalization statistics are
computed over all episodes unless `--stats_dir` points at training-only directories.

## Adding your own recordings

New takes recorded with
[teleop-cbf-perception](https://github.com/talos-robotics-ai/teleop-cbf-perception) are
converted by htd-dataset-converter (`scripts/record_to_rack.sh` on the operator laptop
syncs each take to the training machine and converts it there). Its output is the same
`htd/g1_inspire_wb/...` layout as the Hugging Face folder, so steps 2 and 3 apply
unchanged. The field-by-field contract of the raw episode format is in that repository's
`docs/htd_raw_format.md`.
