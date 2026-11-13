#!/usr/bin/env bash
# Replay a recorded command stream (policy_rollout/replay_eval.py --save_commands) through the g1_walk stack in a
# reconstructed scene, and judge it like a rollout:
#
#   bash deploy/tests/sim_scene_replay.sh --commands EP_commands_SCHEDULE.npz --scene g1_walk/scene_sim/out/NAME \
#       [--source recorded|policy] [--python PY] [--tag NAME]
#
# --source recorded (default) sends the operator's commands of that episode: when the scene was built from the
# same episode, the robot should repeat the task, which tests the scene and the physics without any policy.
# --source policy sends what the policy answered to the recorded observations (open loop).
# Logs in deploy/tests/logs/replay_<time>_<tag>/: spectator.mp4, objects.jsonl, contacts.jsonl, touch.jsonl,
# result.json (scene_result.py), replay.npz (the robot log of g1_walk/replay/replay_commands.py).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MOD="$REPO/g1_walk"
PY="python"; COMMANDS=""; SCENE=""; SOURCE="recorded"; TAG=""; SIM="mujoco"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commands) COMMANDS="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        --tag) TAG="_$2"; shift 2 ;;
        --sim) SIM="$2"; shift 2 ;;      # mujoco (default) or isaac (PhysX in Isaac Sim, g1_walk/isaac_sim)
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$COMMANDS" && -f "$SCENE/scene.xml" ]] || { echo "--commands FILE and --scene DIR are required" >&2; exit 2; }
SCENE="$(cd "$SCENE" && pwd)"; COMMANDS="$(cd "$(dirname "$COMMANDS")" && pwd)/$(basename "$COMMANDS")"
LOG="$HERE/logs/replay_$(date +%Y%m%d_%H%M%S)_${SOURCE}$TAG"
NAME=scene_replay
TOUCH="$MOD/scene_sim/out/touch_model.npz"
SIM_HOST="$MOD/scene_sim/scene_sim_host.py"
SIM_ENV="PSI_SIM_SCENE=$SCENE/scene.xml PSI_SIM_VIDEO=$LOG/spectator.mp4 PSI_SIM_VIDEO_FPS=30 HTD_SIM_STATE_LOG=$LOG/objects.jsonl"
[[ -f "$TOUCH" ]] && SIM_ENV="$SIM_ENV HTD_SIM_TOUCH=${HTD_SIM_TOUCH-$TOUCH}"
[[ -n "${HTD_SIM_LOG_BODIES:-}" ]] && SIM_ENV="$SIM_ENV HTD_SIM_LOG_BODIES=$HTD_SIM_LOG_BODIES"
if [[ $SIM == isaac ]]; then
    SIM_HOST="$MOD/isaac_sim/isaac_sim_host.py"; SIM_PY="${ISAAC_PY:-$HOME/isaacsim/python.sh}"
    SIM_ENV="$SIM_ENV PYTHONPATH=${ISAAC_EXTRA:-$HOME/isaac_extra}"; SIM_ARGS="--sim-dt ${SIM_DT:-0.004} --state-hz ${STATE_HZ:-75}"; SIM_WAIT_S=180
fi
# shellcheck source=../../g1_walk/tests/sim_stack.sh
source "$MOD/tests/sim_stack.sh"
stack_start || exit 1
T0=$(tail -1 "$LOG/objects.jsonl" 2>/dev/null | "$PY" -c "import json,sys; print(json.loads(sys.stdin.read() or '{\"t\":0}')['t'])")
say "replaying $SOURCE commands of $(basename "$COMMANDS") (sim t $T0)"
"$PY" "$MOD/replay/replay_commands.py" --commands "$COMMANDS" --source "$SOURCE" --out "$LOG/replay.npz" 2>&1 | tee "$LOG/replay.log"
RC=${PIPESTATUS[0]}
stop_pid "$SIM_PID" "sim host" 10 || STACK_CLEAN=1
stack_cleanup
"$PY" "$HERE/scene_result.py" --log "$LOG" --scene "$SCENE" --t_start "$T0" || RC=1
say "done: $LOG"
[[ $RC == 0 && $STACK_CLEAN == 0 ]]
