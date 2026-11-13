# Bring-up and teardown of the MuJoCo stack on one machine, sourced by
# sim_e2e.sh and sim_replay.sh:
#
#   the pinned teleop's G1 sim host (Inspire hands emulated over Modbus)
#   + the walker (g1_walk/run_walker.sh --iface lo, network on the GPU)
#   + the HTD-side controller (g1_walk/controller.py --iface lo --supervisor off)
#
# The caller sets MOD (the g1_walk directory), PY, LOG and NAME (log prefix),
# optionally SIM_ENV (extra VAR=value words for the sim host, e.g. PSI_SIM_VIDEO) and
# SIM_HOST (a wrapper run in place of sim/sim_host.py, e.g. scene_sim/scene_sim_host.py), SIM_PY (its
# interpreter when not PY, e.g. Isaac Sim's python.sh), SIM_ARGS (extra host arguments), SIM_PREFIX (words put before the host's command, e.g. taskset -c 0,1) and SIM_WAIT_S,
# then calls `stack_start`. Every process started is stopped on exit, Ctrl-C and
# failure -- controller, then walker, then sim host -- and STACK_CLEAN is set to 1
# if any outlives its stop: a stray walker keeps publishing rt/lowcmd.

TELEOP="$MOD/teleop-cbf-perception/teleop"
STACK_PATTERNS=('[s]im/sim_host.py' '[s]im_host.py --iface' '[g]1_jetson_walker.py' '[g]1_walk/controller.py')
SIM_PID="" WALKER_PID="" CTRL_PID=""
STACK_CLEAN=0

say() { echo "[$NAME] $*"; }

stop_pid() {   # pid, name, seconds to wait after SIGTERM before SIGKILL
    local pid="$1" name="$2" wait_s="$3" i
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM "$pid" 2>/dev/null
    for ((i = 0; i < wait_s * 10; i++)); do kill -0 "$pid" 2>/dev/null || return 0; sleep 0.1; done
    say "$name did not stop in ${wait_s} s: SIGKILL"; kill -KILL "$pid" 2>/dev/null
    return 1
}

stack_cleanup() {
    trap - EXIT INT TERM
    stop_pid "$CTRL_PID" controller 10 || STACK_CLEAN=1
    stop_pid "$WALKER_PID" walker 15 || STACK_CLEAN=1
    stop_pid "$SIM_PID" "sim host" 10 || STACK_CLEAN=1
    sleep 0.5
    local pat
    for pat in "${STACK_PATTERNS[@]}"; do
        if pgrep -f "$pat" >/dev/null; then
            say "left running ($pat), killing:"; pgrep -af "$pat"; pkill -KILL -f "$pat"; STACK_CLEAN=1
        fi
    done
}

wait_for() {   # file, pattern, seconds, pid
    local i
    for ((i = 0; i < $3 * 2; i++)); do
        grep -q "$2" "$1" 2>/dev/null && return 0
        kill -0 "$4" 2>/dev/null || { say "process died; end of $1:"; tail -20 "$1"; return 1; }
        sleep 0.5
    done
    say "no '$2' in $1 within $3 s"; tail -20 "$1"; return 1
}

stack_start() {
    local pat
    for pat in "${STACK_PATTERNS[@]}"; do
        if pgrep -f "$pat" >/dev/null; then
            say "FAILED: already running ($pat): stop it first"; pgrep -af "$pat"; return 1
        fi
    done
    trap 'stack_cleanup; exit 1' INT TERM
    trap stack_cleanup EXIT
    mkdir -p "$LOG"
    say "logs in $LOG"
    say "sim host (G1 + Inspire, walker stance, pelvis band released at the first session)"
    # shellcheck disable=SC2086  # SIM_ENV is a list of VAR=value words
    (cd "$TELEOP" && exec env PYTHONUNBUFFERED=1 MUJOCO_GL=egl PSI_SIM_HOLD_RELEASE=session \
        PSI_SIM_BASE_METER_S=1 PSI_SIM_RENDER_FPS=5 ${SIM_ENV:-} \
        ${SIM_PREFIX:-} "${SIM_PY:-$PY}" "${SIM_HOST:-sim/sim_host.py}" --iface lo --domain 0 --lower jetson --inspire ${SIM_ARGS:-}) > "$LOG/sim.log" 2>&1 &
    SIM_PID=$!
    wait_for "$LOG/sim.log" "Publishing rt/lowstate" "${SIM_WAIT_S:-60}" "$SIM_PID" || return 1

    say "walker"
    PYTHONUNBUFFERED=1 bash "$MOD/run_walker.sh" --iface lo --python "$PY" > "$LOG/walker.log" 2>&1 &
    WALKER_PID=$!
    wait_for "$LOG/walker.log" "^READY:" 120 "$WALKER_PID" || return 1

    say "controller (handover runs now)"
    env PYTHONUNBUFFERED=1 PSI_INSPIRE_LEFT_IP=127.0.0.1 PSI_INSPIRE_RIGHT_IP=127.0.0.2 \
        "$PY" "$MOD/controller.py" --iface lo --supervisor off < /dev/null > "$LOG/controller.log" 2>&1 &
    CTRL_PID=$!
    wait_for "$LOG/controller.log" "^READY:" 120 "$CTRL_PID" || return 1
}
