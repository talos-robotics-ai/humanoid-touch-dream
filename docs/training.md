# Training

Train the HTD policy on a processed dataset ([dataset.md](dataset.md)). Training needs
one CUDA GPU; the reference machine is an NVIDIA L40S (46 GB).

## Environment

Once per machine, from the repository root:

```bash
conda create -n htd_policy python=3.10 pip
conda activate htd_policy
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e htd_policy
```

Use another PyTorch wheel if your driver needs it (RTX 50-series GPUs need
PyTorch 2.7 or newer built for CUDA 12.8). Training runs in BF16 by default; pass
`--no-bf16` on GPUs without it. The first run downloads the ImageNet ResNet18 weights
unless they are cached.

Optional: Weights & Biases. Run `wandb login` once; the launcher below enables logging
with `WANDB=1`.

## Launch

[`scripts/train_htd.sh`](../scripts/train_htd.sh) runs `train_mxt.py` with the settings
for our data. Everything is an environment variable with a default:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA` | `~/htd_data/processed/white_becker_from_sliding_ruller` | processed task directory (contains `g1_inspire_wb/`), as `scripts/prepare_dataset.sh` writes it |
| `TASK` | `white_becker` | label used in the run directory name |
| `PREFIX` | `../runs/run` | run directory = `<PREFIX>_<TASK>_HTD`, relative to `htd_policy/` |
| `STEPS` | `80000` | number of optimizer steps |
| `GPU` | `0` | CUDA device index |
| `WORKERS` | cores minus 2 | DataLoader workers |
| `EMB_CONFIG` | `configs/htd/dftp_walk_latent16.yaml` | embodiment profile; `..._tscale.yaml` for tactile x100 |
| `CONDA_ENV` | `htd_policy` | conda env to activate; empty to use the current interpreter |
| `WANDB` | `0` | `1` logs to W&B, project `WANDB_PROJECT` (`htd`), entity `WANDB_ENTITY` if set |

Extra arguments go to `train_mxt.py`. Examples:

```bash
# default: the set prepare_dataset.sh wrote, walk profile, 80k steps, GPU 0 -> runs/run_white_becker_HTD
scripts/train_htd.sh

# tactile input x100 profile into a separate run directory, with W&B
EMB_CONFIG=configs/htd/dftp_walk_latent16_tscale.yaml PREFIX=../runs/tscale WANDB=1 scripts/train_htd.sh

# resume an interrupted run: same variables, plus --auto_resume
scripts/train_htd.sh --auto_resume
```

The launcher expands to:

```bash
cd htd_policy
python train_mxt.py --touch_dreaming \
  --dataset_dir "$DATA" --ckpt_dir "$PREFIX" --task_name "$TASK" \
  --embodiment_config_path "$EMB_CONFIG" --trunk_config_path configs/htd/trunk_walk.yaml \
  --main_cam_width 1280 --main_cam_height 480 \
  --train_ratio 0.9 --min_val_num 1 --vel_rebalance \
  --num_workers "$WORKERS" --num_steps "$STEPS"
```

Why these settings:

- **Walk profile** (`dftp_walk_latent16.yaml` + `trunk_walk.yaml`): the data has no wrist
  cameras and includes walking commands; this profile has only the head-camera tokenizer
  and a velocity head.
- **`--main_cam_width 1280 --main_cam_height 480`**: the recorded image size; the
  duplicated mono frame fills HTD's stereo slot.
- **`--vel_rebalance`**: walking is rare in the data (vx active in ~7% of frames, yaw rate
  in ~8%, vy never); this oversamples timesteps with active velocity commands.
- **`--train_ratio 0.9`**: nine held-out episodes out of 99 give a usable validation
  signal; upstream's 0.99 would leave one.

`--num_steps` is the final step index, not an increment: resuming with the same value
finishes the original schedule. Run `python train_mxt.py --help` for every option.

### On the shared training server

The L40S server has two GPUs and eight cores. Isaac Sim uses GPU 0, so training goes to
GPU 1, with six loader workers. Run inside tmux and log to a file:

```bash
tmux new -s htd_train 'DATA=~/htd_data/processed/white_becker GPU=1 WORKERS=6 WANDB=1 PREFIX=../runs/rack \
  scripts/train_htd.sh 2>&1 | tee ~/htd_data/logs/train_htd.log'
```

`DATA` here is the server's existing processed set, prepared before the script existed;
new sets land at `processed/<task folder>`. Use session names that are not prefixes of each other: `tmux has-session -t htd_train`
also matches `htd_train2`. `-t =htd_train` matches exactly.

## What to watch

The trainer validates every 500 steps on up to 52 batches of the held-out episodes and
logs (console, and W&B with `WANDB=1`):

| Metric (training; validation adds the prefix `val_`) | Meaning |
| --- | --- |
| `all_emb_loss` | total loss: action heads plus dream heads |
| `g1_inspire_wb_<head>`, e.g. `g1_inspire_wb_velocity_commands` | weighted L1 of each action head |
| `g1_inspire_wb_dream_<target>`, e.g. `g1_inspire_wb_dream_left_thumb_tactile` | weighted dream loss per force or tactile target |
| `val_g1_inspire_wb_ema_std_<patch>`, `val_g1_inspire_wb_ema_cosine_sim_<patch>` (validation only) | health of each tactile EMA teacher, see [model.md](model.md#touch-dreaming) |

`policy_best.ckpt` holds the weights with the lowest `val_all_emb_loss` and is written
when training finishes. After `--auto_resume` the best loss is remembered but its weights
are not, so a resumed run writes `policy_best.ckpt` only if a later validation improves on
it. Because the total includes the
dream terms, validation losses are only comparable between runs with the same profile:
a run whose tactile dreaming collapsed to ~0 reports a lower total than a healthy one.

The best and last checkpoints should both be evaluated offline before choosing one
([evaluation.md](evaluation.md)).

## Reference runs

Both on the 99-episode `white_becker` set, 80k steps, batch 16, split 90/9, L40S, about
5.5 hours each (the run is bound by the eight-core data loader, not the GPU).

| Run | Profile | Best `val_all_emb_loss` | Notes |
| --- | --- | --- | --- |
| `runs/rack_white_becker_HTD` | `dftp_walk_latent16` | 5.82 at step 49,500 | tactile dream losses collapsed to ~0 throughout |
| `runs/rack_tscale_white_becker_HTD` | `dftp_walk_latent16_tscale` | 8.03 at step 34,500 | tactile dreaming active on three left-hand patches; loss not comparable with the row above |

Offline, the second run tracks the arm slightly better and walking and turning slightly
worse than the first. The candidates kept for deployment are run 1 `policy_last.ckpt`
and run 2 `policy_step_70000`.

## Outputs

The run directory and each file in it are described in
[model.md](model.md#training-artifacts). Keep `policy_*.ckpt`, `all_configs.json` and
`dataset_stats.pkl` together: evaluation and rollout need all three.
