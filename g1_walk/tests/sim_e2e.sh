#!/usr/bin/env bash
# End-to-end test in MuJoCo on one machine, no hardware:
#
#   the pinned teleop's G1 sim host (Inspire hands emulated over Modbus)
#   + the walker (g1_walk/run_walker.sh --iface lo, network on the GPU)
#   + the HTD-side controller (g1_walk/controller.py --iface lo)
#   + tests/sim_drive.py playing policy_rollout's part and scoring the run.
#
#   bash g1_walk/tests/sim_e2e.sh [--python PY]
#
# Logs and summary.json go to g1_walk/tests/logs/<time>/. Every process this
# starts is stopped on exit, on Ctrl-C and on failure -- controller, then
# walker, then sim host -- and the run fails if any outlives its stop: a stray
# walker keeps publishing rt/lowcmd. Exit status 0 only when every criterion
# of sim_drive.py passed, the controller ended the session on the policy's
# silence and exited cleanly, and nothing was left running.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$(dirname "$HERE")"
PY="python"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PY="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
LOG="$HERE/logs/$(date +%Y%m%d_%H%M%S)"
NAME=sim_e2e
# shellcheck source=sim_stack.sh
source "$HERE/sim_stack.sh"
stack_start || exit 1

say "driving the choreography"
"$PY" "$HERE/sim_drive.py" --log "$LOG" 2>&1 | tee "$LOG/drive.log"
DRIVE=${PIPESTATUS[0]}

SILENCE=0
grep -q "session ended (no policy frame" "$LOG/controller.log" || { say "FAIL: no session end on silence"; SILENCE=1; }
t0=$(date +%s.%N)
stop_pid "$CTRL_PID" controller 10; CTRL_RC=$?
say "controller stopped in $(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s"
grep -q "detaching from the walker" "$LOG/controller.log" || { say "FAIL: the controller did not detach"; CTRL_RC=1; }
stack_cleanup
say "drive $([[ $DRIVE == 0 ]] && echo PASS || echo FAIL), silence $([[ $SILENCE == 0 ]] && echo PASS || echo FAIL), controller shutdown $([[ $CTRL_RC == 0 ]] && echo PASS || echo FAIL), teardown $([[ $STACK_CLEAN == 0 ]] && echo PASS || echo FAIL)"
[[ $DRIVE == 0 && $SILENCE == 0 && $CTRL_RC == 0 && $STACK_CLEAN == 0 ]]
