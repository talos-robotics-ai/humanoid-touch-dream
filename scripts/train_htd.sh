#!/usr/bin/env bash
# Train HTD (MXT + touch dreaming) on a processed dataset with the head-only walk
# profile (docs/training.md). Extra arguments go to train_mxt.py.
#
#   scripts/train_htd.sh [train_mxt.py args]       e.g. --auto_resume to continue a run
#
# Settings, as environment variables [default]:
#   DATA        processed task directory  [~/htd_data/processed/white_becker_from_sliding_ruller]
#   TASK        label in the run directory name  [white_becker]
#   PREFIX      run directory = <PREFIX>_<TASK>_HTD, relative to htd_policy/  [../runs/run]
#   STEPS       optimizer steps  [80000]
#   GPU         CUDA device index  [0]
#   WORKERS     DataLoader workers  [cores - 2]
#   EMB_CONFIG  embodiment YAML, relative to htd_policy/  [configs/htd/dftp_walk_latent16.yaml]
#   CONDA_ENV   conda env to activate; empty = the current python  [htd_policy]
#   WANDB       1 = log to Weights & Biases, project WANDB_PROJECT [htd], entity WANDB_ENTITY  [0]
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=${DATA:-$HOME/htd_data/processed/white_becker_from_sliding_ruller}
TASK=${TASK:-white_becker}
PREFIX=${PREFIX:-../runs/run}
STEPS=${STEPS:-80000}
GPU=${GPU:-0}
CORES=$(nproc 2>/dev/null || echo 4)
WORKERS=${WORKERS:-$(( CORES > 2 ? CORES - 2 : 1 ))}
EMB_CONFIG=${EMB_CONFIG:-configs/htd/dftp_walk_latent16.yaml}
CONDA_ENV=${CONDA_ENV-htd_policy}

case "${1:-}" in -h|--help) sed -n '2,16p' "$0"; exit 0 ;; esac
[[ -d "$DATA/g1_inspire_wb" ]] || { echo "DATA=$DATA has no g1_inspire_wb/: run scripts/prepare_dataset.sh" >&2; exit 1; }
DATA="$(cd "$DATA" && pwd)"    # absolute: the trainer runs from htd_policy/ and records this path

if [[ -n "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null; then
        for d in miniforge3 miniconda3 anaconda3; do
            # shellcheck source=/dev/null
            [[ -f "$HOME/$d/etc/profile.d/conda.sh" ]] && source "$HOME/$d/etc/profile.d/conda.sh" && break
        done
    fi
    command -v conda >/dev/null || { echo "conda not found; install it or set CONDA_ENV= to use the current python" >&2; exit 1; }
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi
cd "$REPO/htd_policy"
export CUDA_VISIBLE_DEVICES="$GPU"

WANDB_ARGS=()
if [[ "${WANDB:-0}" == 1 ]]; then
    WANDB_ARGS=(--wandb --wandb_name "${WANDB_PROJECT:-htd}")
fi

exec python train_mxt.py --touch_dreaming \
  --dataset_dir "$DATA" --ckpt_dir "$PREFIX" --task_name "$TASK" \
  --embodiment_config_path "$EMB_CONFIG" \
  --trunk_config_path configs/htd/trunk_walk.yaml \
  --main_cam_width 1280 --main_cam_height 480 \
  --train_ratio 0.9 --min_val_num 1 \
  --vel_rebalance \
  --num_workers "$WORKERS" --num_steps "$STEPS" \
  "${WANDB_ARGS[@]}" \
  "$@"
