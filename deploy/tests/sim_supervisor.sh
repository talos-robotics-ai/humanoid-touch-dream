#!/usr/bin/env bash
# MuJoCo smoke test of the deployment supervisor, the UI and the latency budget, on one
# machine, no hardware:
#
#   the pinned teleop's G1 sim host + the walker + g1_walk/controller.py --supervisor on
#   + policy_rollout.py --supervisor True (static head camera) + deploy/ui/server.py
#
#   bash deploy/tests/sim_supervisor.sh --model_path RUN_DIR [--python PY] [--running_ref_ms MS]
#
# Checks, each printed PASS/FAIL:
#   1. HTDPOLICY READY TO DEPLOY within 90 s of the controller's READY line.
#   2. Latency, in READY: 20 s with no WebSocket client (status_probe.py, straight from
#      DDS), then 20 s through the UI (ws_probe.py): the controller's tick_max_ms < 10 in
#      both, |tick_p50 with - without| < 1 ms, tick_max with - without < 2 ms, rollout's
#      loop_max_ms < 16 in both; then RUNNING with the UI connected: tick_p50 within 2 ms
#      of a reference run's (--running_ref_ms, default 36; RUNNING ticks carry today's IK
#      solve). A RUNNING window with no client, before check 3, is recorded as this run's
#      own reference (reported, not gating).
#   3. deploy/ui/cmd.py start / stop / rearm / reset give the expected supervisor log lines
#      in time.
#   4. Teardown: every process stopped in time, nothing left, no rt/lowcmd writer left.
# Logs and summary.json go to deploy/tests/logs/<time>/. Exit 0 only if all four pass.
# Every process started is stopped on exit, Ctrl-C and failure: UI, rollout, controller,
# walker, sim host.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MOD="$REPO/g1_walk"
PY="python"
MODEL=""
REF_RUNNING_P50_MS=36.0     # RUNNING tick_p50 of the implementer's reference run, ms
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        --running_ref_ms) REF_RUNNING_P50_MS="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODEL" && -d "$MODEL" ]] || { echo "pass --model_path RUN_DIR (a directory)" >&2; exit 2; }
MODEL="$(cd "$MODEL" && pwd)"
export PYTHONPATH=""        # a ROS PYTHONPATH in the shell breaks the stack's imports
export DISPLAY="${DISPLAY:-:1}"   # rollout's keyboard listener (pynput) needs an X display
PORT=8765
WINDOW_S=20
RUN_S=10

LOG="$HERE/logs/$(date +%Y%m%d_%H%M%S)"
NAME=sim_supervisor
# shellcheck source=../../g1_walk/tests/sim_stack.sh
source "$MOD/tests/sim_stack.sh"      # say, stop_pid, wait_for, stack_cleanup, STACK_PATTERNS
# A python process running the script, never a shell whose command line names it.
OWN_PATTERNS=('^[^ ]*python[0-9.]* policy_rollout\.py' '^[^ ]*python[0-9.]* [^ ]*deploy/ui/server\.py')
ROLLOUT_PID="" UI_PID="" PROBE_PID=""
CLEANED=0 TEARDOWN_RC=0

cleanup() {
    [[ $CLEANED == 1 ]] && return 0
    CLEANED=1
    trap - EXIT INT TERM
    [[ -n "$PROBE_PID" ]] && kill -TERM "$PROBE_PID" 2>/dev/null
    stop_pid "$UI_PID" "UI server" 5 || TEARDOWN_RC=1
    stop_pid "$ROLLOUT_PID" rollout 15 || TEARDOWN_RC=1
    stack_cleanup               # controller, walker, sim host; their leftovers
    [[ $STACK_CLEAN == 0 ]] || TEARDOWN_RC=1
    local pat
    for pat in "${OWN_PATTERNS[@]}"; do
        if pgrep -f "$pat" >/dev/null; then
            say "left running ($pat), killing:"; pgrep -af "$pat"; pkill -KILL -f "$pat"; TEARDOWN_RC=1
        fi
    done
}

now() { echo "$EPOCHREALTIME"; }
since() { awk -v a="$1" -v b="$EPOCHREALTIME" 'BEGIN{printf "%.2f", b-a}'; }
nlines() { local n; n=$(wc -l < "$1" 2>/dev/null) || n=0; echo "${n:-0}"; }

wait_new() {   # file, from line, fixed string, seconds, t0 -> prints seconds since t0; 1 on timeout
    local deadline
    deadline=$(awk -v a="$5" -v s="$4" 'BEGIN{printf "%.3f", a+s}')
    while :; do
        if tail -n "+$(($2 + 1))" "$1" 2>/dev/null | grep -qF -- "$3"; then since "$5"; return 0; fi
        awk -v d="$deadline" -v b="$EPOCHREALTIME" 'BEGIN{exit !(b > d)}' && { echo "timeout"; return 1; }
        sleep 0.05
    done
}

ui_cmd() { "$PY" "$REPO/deploy/ui/cmd.py" --iface lo "$1" >> "$LOG/cmd.log" 2>&1; }

lowcmd_writers() { "$PY" "$HERE/status_probe.py" --iface lo --count_writers rt/lowcmd --seconds 3 2>/dev/null | tail -1; }

# -- preflight ---------------------------------------------------------------------------------
for pat in "${STACK_PATTERNS[@]}" "${OWN_PATTERNS[@]}"; do
    if pgrep -f "$pat" >/dev/null; then
        say "FAILED: already running ($pat): stop it first"; pgrep -af "$pat"; exit 1
    fi
done
if "$PY" -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)"; then
    say "FAILED: port $PORT is in use"; exit 1
fi
mkdir -p "$LOG"
say "logs in $LOG"
W0=$(lowcmd_writers)
[[ "$W0" == 0 ]] || { say "FAILED: rt/lowcmd already has writers ($W0): another sim is running"; exit 1; }

read -r EMB_YAML TRUNK_YAML < <("$PY" - "$MODEL" "$REPO/htd_policy" <<'PYEOF'
import json, os, sys
c = json.load(open(os.path.join(sys.argv[1], "all_configs.json")))
print(*(p if os.path.isabs(p) else os.path.join(sys.argv[2], p)
        for p in (c["embodiment_config_path"], c["trunk_config_path"])))
PYEOF
)
[[ -f "$EMB_YAML" && -f "$TRUNK_YAML" ]] || { say "FAILED: the run's YAMLs not found ($EMB_YAML, $TRUNK_YAML)"; exit 1; }

trap 'cleanup; exit 1' INT TERM
trap cleanup EXIT

# -- bring-up (as sim_stack.sh's stack_start, the controller under the supervisor) -------------
T_START=$(now)
say "sim host (G1 + Inspire, walker stance, pelvis band released at the first session)"
(cd "$TELEOP" && exec env PYTHONUNBUFFERED=1 MUJOCO_GL=egl PSI_SIM_HOLD_RELEASE=session \
    PSI_SIM_BASE_METER_S=1 PSI_SIM_RENDER_FPS=5 \
    "$PY" sim/sim_host.py --iface lo --domain 0 --lower jetson --inspire) > "$LOG/sim.log" 2>&1 &
SIM_PID=$!
wait_for "$LOG/sim.log" "Publishing rt/lowstate" 60 "$SIM_PID" || exit 1

say "walker"
PYTHONUNBUFFERED=1 bash "$MOD/run_walker.sh" --iface lo --python "$PY" > "$LOG/walker.log" 2>&1 &
WALKER_PID=$!
wait_for "$LOG/walker.log" "^READY:" 120 "$WALKER_PID" || exit 1

say "controller --supervisor on (handover runs now)"
env PYTHONUNBUFFERED=1 PSI_INSPIRE_LEFT_IP=127.0.0.1 PSI_INSPIRE_RIGHT_IP=127.0.0.2 \
    "$PY" "$MOD/controller.py" --iface lo --supervisor on --walk on --manual_walk off \
    --log_file "$LOG/deploy.log" < /dev/null > "$LOG/controller.log" 2>&1 &
CTRL_PID=$!
wait_for "$LOG/controller.log" "^READY:" 120 "$CTRL_PID" || exit 1
T_CTRL_READY=$(now)
say "controller READY after $(since "$T_START") s of bring-up"

say "releasing the sim's pelvis band"
"$PY" -c "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'{}', ('127.0.0.1', 14651))"

say "rollout --supervisor True (static head camera)"
(cd "$REPO/policy_rollout" && exec env OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 "$PY" policy_rollout.py \
    --model_path "$MODEL" --policy_class MXT --embodiment g1_inspire_wb --inspire_hand_type dftp \
    --use_wrist_camera False --send_velocity_commands True --device cuda:0 --control_freq 30 \
    --inference_interval 45 --dds_interface lo --head_camera_source static --supervisor True \
    --embodiment_config_path "$EMB_YAML" --trunk_config_path "$TRUNK_YAML") \
    < /dev/null > "$LOG/rollout.log" 2>&1 &
ROLLOUT_PID=$!

say "UI server on 127.0.0.1:$PORT"
"$PY" "$REPO/deploy/ui/server.py" --iface lo --port "$PORT" < /dev/null > "$LOG/ui.log" 2>&1 &
UI_PID=$!
wait_for "$LOG/ui.log" "HTD UI on" 30 "$UI_PID" || exit 1

# -- check 1: READY TO DEPLOY ------------------------------------------------------------------
C1=FAIL
T1=$(wait_new "$LOG/controller.log" 0 "HTDPOLICY READY TO DEPLOY" 90 "$T_CTRL_READY") && C1=PASS
if [[ $C1 == FAIL ]]; then
    say "no HTDPOLICY READY TO DEPLOY within 90 s; gating:"
    grep -E "\[supervisor\]|standing|policy" "$LOG/controller.log" | tail -10
    tail -15 "$LOG/rollout.log"
fi
say "check 1: HTDPOLICY READY TO DEPLOY ${T1} s after the controller's READY: $C1"

# -- check 2 (READY windows) -------------------------------------------------------------------
if [[ $C1 == PASS ]]; then
    sleep 3     # the 1 s timing windows hold READY ticks only
    say "window 1: ${WINDOW_S} s in READY, no WebSocket client (DDS reader)"
    "$PY" "$HERE/status_probe.py" --iface lo --seconds "$WINDOW_S" --out "$LOG/ready_without_ui.json" 2>/dev/null
    say "window 2: ${WINDOW_S} s in READY through the UI's WebSocket"
    "$PY" "$HERE/ws_probe.py" --port "$PORT" --seconds "$WINDOW_S" --out "$LOG/ready_with_ui.json"
fi

# -- a RUNNING window without the UI (an in-run reference for check 2, not gating) -------------
declare -A CT=()
if [[ $C1 == PASS ]]; then
    say "RUNNING ${RUN_S} s with no WebSocket client (DDS reader; reference only)"
    "$PY" "$HERE/status_probe.py" --iface lo --seconds $((RUN_S + 1)) --out "$LOG/running_without_ui.json" \
        > "$LOG/running_ref_probe.log" 2>&1 &
    PROBE_PID=$!
    sleep 0.5
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd start
    CT[ref_start_running]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] READY -> RUNNING" 2 "$t")
    sleep "$(awk -v a="$t" -v b="$EPOCHREALTIME" -v s="$RUN_S" 'BEGIN{d=a+s-b; printf "%.2f", (d > 0 ? d : 0)}')"
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd stop
    CT[ref_stop_hold]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] RUNNING -> HOLD" 2 "$t")
    wait "$PROBE_PID"; PROBE_PID=""
    sleep 1
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd rearm
    CT[ref_rearm_ready]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] STABILIZING -> READY" 10 "$t")
    say "reference session: start ${CT[ref_start_running]} s, stop ${CT[ref_stop_hold]} s, rearm -> READY ${CT[ref_rearm_ready]} s"
    sleep 3
fi

# -- check 3 (commands) and the RUNNING window with the UI -------------------------------------
C3=FAIL
if [[ $C1 == PASS ]]; then
    say "RUNNING ${RUN_S} s through the UI's WebSocket"
    "$PY" "$HERE/ws_probe.py" --port "$PORT" --seconds $((RUN_S + 1)) --out "$LOG/running_with_ui.json" \
        > "$LOG/running_probe.log" 2>&1 &
    PROBE_PID=$!
    sleep 0.5
    ok=1
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd start
    CT[start_running]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] READY -> RUNNING" 2 "$t") || ok=0
    say "start -> RUNNING: ${CT[start_running]} s"
    sleep "$(awk -v a="$t" -v b="$EPOCHREALTIME" -v s="$RUN_S" 'BEGIN{d=a+s-b; printf "%.2f", (d > 0 ? d : 0)}')"
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd stop
    CT[stop_hold]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] RUNNING -> HOLD" 2 "$t") || ok=0
    say "stop -> HOLD: ${CT[stop_hold]} s"
    wait "$PROBE_PID"; PROBE_PID=""
    sleep 1
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd rearm
    CT[rearm_stabilizing]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] HOLD -> STABILIZING" 2 "$t") || ok=0
    CT[rearm_ready]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] STABILIZING -> READY" 10 "$t") || ok=0
    say "rearm -> STABILIZING: ${CT[rearm_stabilizing]} s, -> READY: ${CT[rearm_ready]} s"
    sleep 1
    n=$(nlines "$LOG/controller.log"); t=$(now); ui_cmd reset
    CT[reset_glide]=$(wait_new "$LOG/controller.log" "$n" "arms gliding home" 2 "$t") || ok=0
    CT[reset_ready]=$(wait_new "$LOG/controller.log" "$n" "[supervisor] STABILIZING -> READY" 20 "$t")
    say "reset -> arms gliding home: ${CT[reset_glide]} s, back in READY: ${CT[reset_ready]} s (informational)"
    [[ $ok == 1 ]] && C3=PASS
fi
say "check 3: start/stop/rearm/reset: $C3"

# -- check 4: teardown -------------------------------------------------------------------------
W_UP=$(lowcmd_writers)
say "rt/lowcmd writers before teardown: $W_UP"
t=$(now)
cleanup
T_TEARDOWN=$(since "$t")
W_DOWN=$(lowcmd_writers)
LEFT=$(pgrep -af "[s]im/sim_host.py|[g]1_jetson_walker.py|[g]1_walk/controller.py|${OWN_PATTERNS[0]}|${OWN_PATTERNS[1]}" || true)
C4=FAIL
[[ $TEARDOWN_RC == 0 && "$W_UP" -ge 1 && "$W_DOWN" == 0 && -z "$LEFT" ]] && C4=PASS
say "check 4: teardown in ${T_TEARDOWN} s, rt/lowcmd writers $W_UP -> $W_DOWN, left: ${LEFT:-none}: $C4"

# -- check 2 verdict and summary.json ----------------------------------------------------------
CT_JSON=$(for k in "${!CT[@]}"; do printf '"%s": "%s",' "$k" "${CT[$k]}"; done)
"$PY" - "$LOG" <<PYEOF
import json, os, sys
log = sys.argv[1]
def load(name):
    p = os.path.join(log, name)
    return json.load(open(p))["summary"] if os.path.exists(p) else None
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
without, with_ui, running = load("ready_without_ui.json"), load("ready_with_ui.json"), load("running_with_ui.json")
running_ref = load("running_without_ui.json")
def field(s, state, f, stat):
    try: return s["by_state"][state][f][stat]
    except (TypeError, KeyError): return None
def only_ready(s):
    return s is not None and set(s["states"]) == {"READY"}
c2, why = "PASS", []
lat = {}
for tag, s in (("without_ui", without), ("with_ui", with_ui)):
    lat[tag] = {k: field(s, "READY", f, st) for k, f, st in (
        ("tick_p50_ms", "tick_p50_ms", "p50"), ("tick_max_ms", "tick_max_ms", "max"),
        ("loop_p50_ms", "policy.loop_p50_ms", "p50"), ("loop_max_ms", "policy.loop_max_ms", "max"),
        ("inference_ms_p50", "policy.inference_ms", "p50"), ("n_supervisor", None, None))
        if f is not None}
    lat[tag]["states"] = s["states"] if s else None
    if not only_ready(s):
        why.append(f"{tag}: not READY throughout ({lat[tag]['states']})"); continue
    if not lat[tag]["tick_max_ms"] < 10: why.append(f"{tag}: tick_max {lat[tag]['tick_max_ms']} >= 10 ms")
    if not lat[tag]["loop_max_ms"] < 16: why.append(f"{tag}: loop_max {lat[tag]['loop_max_ms']} >= 16 ms")
a, b = lat["without_ui"], lat["with_ui"]
if not why:
    lat["d_tick_p50_ms"] = b["tick_p50_ms"] - a["tick_p50_ms"]
    lat["d_tick_max_ms"] = b["tick_max_ms"] - a["tick_max_ms"]
    if not abs(lat["d_tick_p50_ms"]) < 1: why.append(f"|d tick_p50| {lat['d_tick_p50_ms']:.2f} >= 1 ms")
    if not lat["d_tick_max_ms"] < 2: why.append(f"d tick_max {lat['d_tick_max_ms']:.2f} >= 2 ms")
r = {"tick_p50_ms": field(running, "RUNNING", "tick_p50_ms", "p50"),
     "tick_max_ms": field(running, "RUNNING", "tick_max_ms", "max"),
     "loop_max_ms": field(running, "RUNNING", "policy.loop_max_ms", "max"),
     "n_supervisor": field(running, "RUNNING", "tick_p50_ms", "n"),
     "states": running["states"] if running else None, "reference_p50_ms": $REF_RUNNING_P50_MS}
lat["running_with_ui"] = r
lat["running_without_ui_reference"] = {   # same run, no WebSocket client: informational, not gating
    "tick_p50_ms": field(running_ref, "RUNNING", "tick_p50_ms", "p50"),
    "tick_max_ms": field(running_ref, "RUNNING", "tick_max_ms", "max"),
    "loop_max_ms": field(running_ref, "RUNNING", "policy.loop_max_ms", "max"),
    "n_supervisor": field(running_ref, "RUNNING", "tick_p50_ms", "n"),
    "states": running_ref["states"] if running_ref else None}
if r["tick_p50_ms"] is None: why.append("no settled RUNNING samples")
elif not abs(r["tick_p50_ms"] - $REF_RUNNING_P50_MS) < 2:
    why.append(f"RUNNING tick_p50 {r['tick_p50_ms']:.2f} not within 2 ms of $REF_RUNNING_P50_MS")
if why: c2 = "FAIL"
fmt = lambda x: "n/a" if x is None else f"{x:.2f}"
print(f"[sim_supervisor] READY without UI: tick p50 {fmt(a.get('tick_p50_ms'))} max {fmt(a.get('tick_max_ms'))} ms, "
      f"loop max {fmt(a.get('loop_max_ms'))} ms; with UI: tick p50 {fmt(b.get('tick_p50_ms'))} max "
      f"{fmt(b.get('tick_max_ms'))} ms, loop max {fmt(b.get('loop_max_ms'))} ms; RUNNING with UI: tick p50 "
      f"{fmt(r['tick_p50_ms'])} max {fmt(r['tick_max_ms'])} ms, loop max {fmt(r['loop_max_ms'])} ms")
rr = lat["running_without_ui_reference"]
print(f"[sim_supervisor] RUNNING without UI (this run, reference only): tick p50 {fmt(rr['tick_p50_ms'])} "
      f"max {fmt(rr['tick_max_ms'])} ms, loop max {fmt(rr['loop_max_ms'])} ms")
print(f"[sim_supervisor] check 2: latency: {c2}" + (" (" + "; ".join(why) + ")" if why else ""))
checks = {"1_ready_to_deploy": "$C1", "2_latency": c2, "3_commands": "$C3", "4_teardown": "$C4"}
summary = {
    "checks": checks, "failures_2": why,
    "ready_to_deploy_s_after_controller_ready": num("${T1:-}"),
    "latency": lat,
    "command_timings_s": {k: num(v) if num(v) is not None else v for k, v in {${CT_JSON%,}}.items()},
    "teardown": {"seconds": num("$T_TEARDOWN"), "lowcmd_writers_before": num("$W_UP"),
                 "lowcmd_writers_after": num("$W_DOWN"), "left_running": """$LEFT""".strip()},
}
json.dump(summary, open(os.path.join(log, "summary.json"), "w"), indent=1)
open(os.path.join(log, "check2.txt"), "w").write(c2)
PYEOF
C2=$(cat "$LOG/check2.txt" 2>/dev/null || echo FAIL)
say "1 ready-to-deploy $C1, 2 latency $C2, 3 commands $C3, 4 teardown $C4 (summary: $LOG/summary.json)"
[[ $C1 == PASS && $C2 == PASS && $C3 == PASS && $C4 == PASS ]]
