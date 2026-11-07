#!/usr/bin/env bash
# Train the HTD policy (MXT + touch dreaming) on the rack, on GPU 1.
#
# Head-only DFTP data from htd-dataset-converter -> the head-only "walk" profile.
# Run inside tmux on the rack:
#   tmux new -s htd_train 'scripts/rack/train_htd.sh 2>&1 | tee ~/htd_data/logs/train_htd.log'
# Resume after an interruption: the same command with --auto_resume appended.
# Overrides: DATA, TASK, PREFIX, STEPS, GPU, WORKERS, EMB_CONFIG, WANDB(=0 off), WANDB_PROJECT,
# WANDB_ENTITY; extra args go to train_mxt.py.
set -eo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA=${DATA:-$HOME/htd_data/processed/white_becker}
TASK=${TASK:-white_becker}
PREFIX=${PREFIX:-../runs/rack}          # run dir = $REPO/runs/rack_${TASK}_HTD
STEPS=${STEPS:-80000}
WORKERS=${WORKERS:-6}                   # the rack has 8 cores
EMB_CONFIG=${EMB_CONFIG:-configs/htd/dftp_walk_latent16.yaml}   # *_tscale.yaml: tactile input x100

source ~/miniforge3/etc/profile.d/conda.sh
conda activate htd_policy
cd "$REPO/htd_policy"

# GPU 0 belongs to Isaac Sim; torch trains on GPU 1.
export CUDA_VISIBLE_DEVICES=${GPU:-1}

# W&B: on by default (WANDB=0 to disable). The rack's ~/.netrc holds the API key.
WANDB_ARGS=()
if [ "${WANDB:-1}" != 0 ]; then
  export WANDB_ENTITY=${WANDB_ENTITY:-lorenzo1-ortolani-politecnico-di-milano}
  WANDB_ARGS=(--wandb --wandb_name "${WANDB_PROJECT:-htd}")
fi

python train_mxt.py --touch_dreaming \
  --dataset_dir "$DATA" --ckpt_dir "$PREFIX" --task_name "$TASK" \
  --embodiment_config_path "$EMB_CONFIG" \
  --trunk_config_path configs/htd/trunk_walk.yaml \
  --main_cam_width 1280 --main_cam_height 480 \
  --train_ratio 0.9 --min_val_num 1 \
  --vel_rebalance \
  --num_workers "$WORKERS" --num_steps "$STEPS" \
  "${WANDB_ARGS[@]}" \
  "$@"
