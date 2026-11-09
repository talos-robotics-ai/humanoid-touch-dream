#!/usr/bin/env bash
# Create the Python environment that runs a trained policy on the robot-side PC
# (policy_rollout.py, check_inference.py, replay_eval.py).
#
#   bash policy_rollout/setup_rollout_env.sh [--name ENV] [--torch-index URL]
#       New conda env (default htd_rollout): Python 3.10, PyTorch 2.9.1 for CUDA 12.8,
#       requirements-rollout.txt and htd_policy. About 7 GB.
#   bash policy_rollout/setup_rollout_env.sh --base-python PY [--venv DIR]
#       Thin venv (default ~/.venvs/htd_rollout) on an interpreter that already has
#       CUDA PyTorch, e.g. the teleop conda env; the base env is never modified.
#
# Check the result: python policy_rollout/check_inference.py --model_path RUN_DIR
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
NAME=htd_rollout
VENV="$HOME/.venvs/htd_rollout"
BASE_PY=""
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-python) BASE_PY="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --venv) VENV="$2"; shift 2 ;;
        --torch-index) TORCH_INDEX="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$BASE_PY" ]]; then
    command -v conda >/dev/null || { echo "conda not found (or pass --base-python)" >&2; exit 1; }
    eval "$(conda shell.bash hook)"
    if conda env list | awk '{print $1}' | grep -qx "$NAME"; then
        echo "conda env $NAME exists; updating it"
    else
        conda create -y -n "$NAME" python=3.10 pip
    fi
    conda activate "$NAME"
    PY="$(command -v python)"
    "$PY" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url "$TORCH_INDEX"
else
    [[ -x "$BASE_PY" ]] || { echo "$BASE_PY is not an executable interpreter" >&2; exit 1; }
    "$BASE_PY" - <<'EOF' || { echo "the base interpreter needs CUDA PyTorch working on this GPU" >&2; exit 1; }
import sys, torch
assert sys.version_info[:2] >= (3, 10), sys.version
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(f"base: Python {sys.version.split()[0]}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
EOF
    "$BASE_PY" -m venv --system-site-packages "$VENV"
    PY="$VENV/bin/python"
    "$PY" -m pip install -q --upgrade pip
    if ! "$PY" -c "import torchvision" 2>/dev/null; then
        # torchvision 0.N pairs with torch 2.(N-15); take the wheel for the base's CUDA build.
        read -r TV CU < <("$PY" -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'0.{int(v[1])+15}.*', 'cu'+torch.version.cuda.replace('.', ''))")
        "$PY" -m pip install --no-deps "torchvision==$TV" --index-url "https://download.pytorch.org/whl/$CU"
    fi
fi

"$PY" -m pip install -r "$HERE/requirements-rollout.txt"
"$PY" -m pip install --no-deps -e "$REPO/htd_policy"
"$PY" - <<'EOF'
import torch, torchvision
print(f"ready: torch {torch.__version__}, torchvision {torchvision.__version__}, "
      f"CUDA {'available on ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT available'}")
EOF
echo "interpreter: $PY"
