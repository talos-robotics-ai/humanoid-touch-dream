#!/usr/bin/env bash
# Download one task of the Hugging Face dataset and turn its HTD episodes into a
# training set (docs/dataset.md).
#
#   scripts/prepare_dataset.sh [--task NAME] [--data-root DIR] [--repo ID] [--skip-download]
#
# Defaults: task white_becker_from_sliding_ruller, data root ~/htd_data, repo
# Relo02/loco-manipulation-data. Writes
#   DIR/hf/<repo name>/<task>/htd/...          the download, left untouched
#   DIR/processed/<task>/g1_inspire_wb/...     the training set (train_htd.sh's DATA)
#   DIR/logs/validate_{raw,processed}_<task>.json
# Disk: about twice the download (15.5 GB for the 99-take task). Needs the hf CLI
# (pip install -U huggingface_hub) and a python with h5py, numpy and opencv (the
# htd_policy environment, or PYTHON=...). Re-running resumes the download.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
TASK=white_becker_from_sliding_ruller
DATA_ROOT="$HOME/htd_data"
HF_REPO=Relo02/loco-manipulation-data
DOWNLOAD=1
PY="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --repo) HF_REPO="$2"; shift 2 ;;
        --skip-download) DOWNLOAD=0; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
say() { echo "[prepare_dataset] $*"; }

"$PY" -c 'import h5py, numpy, cv2' 2>/dev/null \
    || { echo "$PY lacks h5py, numpy or opencv: activate the htd_policy environment" >&2; exit 1; }
mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"    # absolute: the processors run from the repository root
HF_DIR="$DATA_ROOT/hf/${HF_REPO##*/}"
SRC="$HF_DIR/$TASK/htd/g1_inspire_wb"
DATA="$DATA_ROOT/processed/$TASK"
LOGS="$DATA_ROOT/logs"
mkdir -p "$HF_DIR" "$LOGS"

if [[ $DOWNLOAD == 1 ]]; then
    command -v hf >/dev/null || { echo "hf CLI not found: pip install -U huggingface_hub" >&2; exit 1; }
    say "downloading $HF_REPO $TASK/htd into $HF_DIR"
    hf download "$HF_REPO" --repo-type dataset --local-dir "$HF_DIR" --include "$TASK/htd/*"
fi
[[ -d "$SRC" ]] || { echo "no HTD episodes at $SRC" >&2; exit 1; }
say "$(find "$SRC" -name 'episode_*.hdf5' | wc -l) episodes in $SRC"

cd "$REPO_ROOT"
P=data_collect/data_processing
if [[ -e "$DATA/g1_inspire_wb" ]]; then
    say "$DATA/g1_inspire_wb exists: processing it again in place (delete it to restart from the download)"
else
    mkdir -p "$DATA"
    # Hard links make this copy instant and keep the download intact: each processor
    # writes a new file per episode and renames it into place, so the processed set
    # ends up as its own files (about the size of the download). Copy across filesystems.
    say "linking the episodes into $DATA"
    cp -al "$SRC" "$DATA/" 2>/dev/null || { rm -rf "$DATA/g1_inspire_wb"; cp -a "$SRC" "$DATA/"; } \
        || { rm -rf "$DATA/g1_inspire_wb"; exit 1; }
    say "validating the raw episodes"
    "$PY" $P/validate_htd_dataset.py "$DATA" --profile dftp --stage collector --camera-profile head-only \
        --images all --report "$LOGS/validate_raw_$TASK.json" \
        || { echo "raw validation failed: removing $DATA/g1_inspire_wb so the next run starts over" >&2
             rm -rf "$DATA/g1_inspire_wb"; exit 1; }
fi
say "processing"
"$PY" $P/process_data_g1_inspire_wb.py --dataset_dir "$DATA"
"$PY" $P/process_data_inspire_tactile.py --dataset_dir "$DATA" --hand_type dftp
say "validating the processed episodes"
"$PY" $P/validate_htd_dataset.py "$DATA" --profile dftp --camera-profile head-only \
    --images all --report "$LOGS/validate_processed_$TASK.json"
say "done. Train with: DATA=$DATA scripts/train_htd.sh"
