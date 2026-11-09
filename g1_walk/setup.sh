#!/usr/bin/env bash
# One-time setup of g1_walk: the pinned teleop code (walker and hand driver), the
# Inspire SDK and the Python packages the HTD-side controller needs (requirements.txt).
#
#   bash g1_walk/setup.sh [--python PY]
#
# PY (default: python on PATH) must already be the base environment: the teleop conda
# env at the pinned commit, or a venv layered on it such as the one
# policy_rollout/setup_rollout_env.sh --base-python makes. Needs git 2.36 or newer
# (partial clones). Ends by running check_pin.py and check_env.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
SUB=g1_walk/teleop-cbf-perception
PY="python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PY="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
# shellcheck source=pins.env
source "$HERE/pins.env"
say() { echo "[g1_walk setup] $*"; }

say "teleop-cbf-perception at ${TELEOP_COMMIT:0:8} (partial clone)"
git -C "$REPO" submodule update --init --filter=blob:none "$SUB"
say "third_party/g1_walk (the walking checkpoint)"
git -C "$REPO/$SUB" submodule update --init --filter=blob:none third_party/g1_walk

if "$PY" -c 'import unitree_sdk2py' 2>/dev/null; then
    say "unitree_sdk2py already installed; check_pin.py reports its commit"
else
    say "unitree_sdk2py from the pinned third_party/unitree_sdk2_python"
    git -C "$REPO/$SUB" submodule update --init --filter=blob:none third_party/unitree_sdk2_python
    "$PY" -m pip install -e "$REPO/$SUB/third_party/unitree_sdk2_python"
fi

SDK="$HERE/.deps/inspire_hands_unitree_sdk"
if [[ ! -d "$SDK/.git" ]]; then
    say "cloning the Inspire hand SDK"
    git clone --filter=blob:none --no-checkout "$INSPIRE_SDK_URL" "$SDK"
fi
git -C "$SDK" fetch --quiet origin "$INSPIRE_SDK_COMMIT"
git -C "$SDK" checkout --quiet --detach "$INSPIRE_SDK_COMMIT"
say "Inspire hand SDK at ${INSPIRE_SDK_COMMIT:0:8}"

# --no-deps on the two editable packages: their declared pins (numpy 1.x for
# dex_retargeting's tree, PyQt5 for the SDK's viewer) would otherwise replace
# what the base environment and requirements.txt already pin.
"$PY" -m pip install -r "$HERE/requirements.txt"
"$PY" -m pip install --no-deps -e "$SDK/dftp/inspire_hand_sdk"
"$PY" -m pip install --no-deps -e "$REPO/robot_control/dex-retargeting"

"$PY" "$HERE/check_pin.py"
"$PY" "$HERE/check_env.py"
say "done"
