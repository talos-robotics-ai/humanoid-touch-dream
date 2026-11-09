#!/usr/bin/env bash
# Replay one recorded episode's command stream through the g1_walk stack in MuJoCo
# and record a video:
#
#   bash g1_walk/replay/sim_replay.sh --commands EP_commands_SCHEDULE.npz \
#       [--source policy|recorded] [--out DIR] [--python PY]
#
# The stream comes from policy_rollout/replay_eval.py --save_commands. --source
# policy (default) replays what the HTD policy's rollout schedule sends; recorded
# replays the operator's commands for the same episode (the baseline). Brings up
# the sim host, the walker and the controller (tests/sim_stack.sh), runs
# replay_commands.py, and tears everything down. Writes DIR/<episode>_<source>.mp4
# (the sim's spectator view) and DIR/<episode>_<source>.npz (the robot log);
# compare runs with plot_replay.py.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$(dirname "$HERE")"
PY="python" COMMANDS="" SOURCE="policy" OUT="$HERE/out"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commands) COMMANDS="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$COMMANDS" ]] || { echo "--commands FILE is required" >&2; exit 2; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
EPISODE="$(basename "$COMMANDS" | sed 's/_commands_.*//')"
TAG="${EPISODE}_${SOURCE}"
LOG="$OUT/logs/${TAG}_$(date +%Y%m%d_%H%M%S)"
NAME=sim_replay
SIM_ENV="PSI_SIM_VIDEO=$OUT/$TAG.mp4 PSI_SIM_VIDEO_FPS=30"
# shellcheck source=../tests/sim_stack.sh
source "$MOD/tests/sim_stack.sh"
stack_start || exit 1

say "replaying $SOURCE commands of $EPISODE"
"$PY" "$HERE/replay_commands.py" --commands "$COMMANDS" --source "$SOURCE" --out "$OUT/$TAG.npz" \
    2>&1 | tee "$LOG/replay.log"
RC=${PIPESTATUS[0]}
# Stop the sim host first: it closes the video on its way out, so the clip ends
# with the robot still on the walker. The usual order (controller, walker, sim)
# would film the last second with no process commanding the legs, and the robot
# collapsing. In the sim nothing is lost by it; stack_cleanup then stops the rest.
stop_pid "$SIM_PID" "sim host" 10 || STACK_CLEAN=1
stack_cleanup
say "replay $([[ $RC == 0 ]] && echo done || echo FAILED), teardown $([[ $STACK_CLEAN == 0 ]] && echo clean || echo UNCLEAN)"
say "video: $OUT/$TAG.mp4   log: $OUT/$TAG.npz"
[[ $RC == 0 && $STACK_CLEAN == 0 ]]
