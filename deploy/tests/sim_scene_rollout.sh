#!/usr/bin/env bash
# Closed-loop run of an HTD checkpoint in a reconstructed scene, on one machine plus a render host:
#
#   the teleop's MuJoCo sim host with the scene's objects (g1_walk/scene_sim/scene_sim_host.py: physics, DDS, state
#   published for the renderer) + the walker + g1_walk/controller.py --supervisor on
#   + policy_rollout.py --supervisor True, its head camera read over ZMQ from the Isaac Sim renderer that mirrors
#     the simulation (real2sim/isaac/head_camera.py --serve --stream, on the render host), through a local relay
#     that holds the newest frame (g1_walk/scene_sim/camera_relay.py: --camera is the renderer's frame stream)
#
#   bash deploy/tests/sim_scene_rollout.sh --model_path RUN_DIR --scene g1_walk/scene_sim/out/NAME \
#        [--python PY] [--seconds 60] [--render_host dev-lorenzo] [--render_dir Desktop/real2sim_rgbd] \
#        [--render_out outputs/NAME] [--camera tcp://HOST:5558] [--state tcp://HOST:5557] [--no_render_start] [--tag T]
#
# The scene folder comes from g1_walk/scene_sim/build_mjcf.py (scene.xml, scene_frame.json). With --render_host the
# renderer is started there over ssh in tmux session htd_render (its scene_frame.json is refreshed first) and stopped
# at the end; --no_render_start uses one that already runs. Sequence: bring-up, wait for HTDPOLICY READY TO DEPLOY,
# `start`, run --seconds, `stop`, teardown. Writes deploy/tests/logs/scene_<time>[_tag]/: the processes' logs,
# objects.jsonl (object poses at 10 Hz), bodies.jsonl (every body's pose at HTD_SIM_STATE_HZ, for real2sim's head_camera.py --playback: first- and
# third-person clips rendered after the run; SPECTATOR=1 adds MuJoCo's third-person view, spectator.mp4, recorded in the
# host at the cost of its real time), head_camera.mp4 (what the policy
# saw, fetched from the render host) and result.json (did the vessel end up inside the bin: see scene_result.py).
# Exit 0 when the run completed (whatever the task outcome), 1 when the stack failed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MOD="$REPO/g1_walk"
PY="python"; MODEL=""; SCENE=""; SECONDS_RUN=60; TAG=""
RENDER_HOST=""; RENDER_DIR="Desktop/real2sim_rgbd"; RENDER_OUT=""; RENDER_START=1
CAMERA=""; STATE=""; SIM="mujoco"; ISAAC_PY="${ISAAC_PY:-$HOME/isaacsim/python.sh}"; ISAAC_EXTRA="${ISAAC_EXTRA:-$HOME/isaac_extra}"; SIM_DT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL="$2"; shift 2 ;;
        --scene) SCENE="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        --seconds) SECONDS_RUN="$2"; shift 2 ;;
        --render_host) RENDER_HOST="$2"; shift 2 ;;
        --render_dir) RENDER_DIR="$2"; shift 2 ;;
        --render_out) RENDER_OUT="$2"; shift 2 ;;
        --camera) CAMERA="$2"; shift 2 ;;
        --state) STATE="$2"; shift 2 ;;
        --no_render_start) RENDER_START=0; shift ;;
        --sim) SIM="$2"; shift 2 ;;          # mujoco (default) or isaac: PhysX in Isaac Sim integrates the dynamics
        --sim_dt) SIM_DT="$2"; shift 2 ;;
        --tag) TAG="_$2"; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODEL" && -d "$MODEL" ]] || { echo "pass --model_path RUN_DIR (a directory)" >&2; exit 2; }
[[ -n "$SCENE" && -f "$SCENE/scene.xml" ]] || { echo "pass --scene DIR (with scene.xml; see build_mjcf.py)" >&2; exit 2; }
MODEL="$(cd "$MODEL" && pwd)"; SCENE="$(cd "$SCENE" && pwd)"
[[ -n "$RENDER_HOST" ]] && { CAMERA="${CAMERA:-tcp://$RENDER_HOST:5558}"; STATE="${STATE:-tcp://$RENDER_HOST:5557}"; }
[[ -n "$CAMERA" && -n "$STATE" ]] || { echo "pass --render_host HOST, or --camera and --state addresses" >&2; exit 2; }
RELAY_ADDR="tcp://127.0.0.1:5559"   # the relay's REP socket: where rollout reads the head camera
# --render_host local: the renderer runs on this machine (the whole run on the GPU rack), no ssh
RENDER_LOCAL=0; [[ "$RENDER_HOST" == local ]] && { RENDER_LOCAL=1; CAMERA="tcp://127.0.0.1:5558"; STATE="tcp://127.0.0.1:5557"; RENDER_DIR="${RENDER_DIR/#\~/$HOME}"; [[ "$RENDER_DIR" = /* ]] || RENDER_DIR="$HOME/$RENDER_DIR"; }
RENDER_OUT="${RENDER_OUT:-outputs/$(basename "$SCENE")}"
export PYTHONPATH=""        # a ROS PYTHONPATH in the shell breaks the stack's imports

LOG="$HERE/logs/scene_$(date +%Y%m%d_%H%M%S)$TAG"
NAME=scene_rollout
# shellcheck source=../../g1_walk/tests/sim_stack.sh
source "$MOD/tests/sim_stack.sh"      # say, stop_pid, wait_for, stack_cleanup, STACK_PATTERNS, TELEOP
OWN_PATTERNS=('^[^ ]*python[0-9.]* policy_rollout\.py' '[s]cene_sim/scene_sim_host.py' '[i]saac_sim/isaac_sim_host.py' '[s]cene_sim/camera_relay.py')
ROLLOUT_PID=""; RELAY_PID=""; CLEANED=0; RC=1

render_ssh() {
    if [[ $RENDER_LOCAL == 1 ]]; then bash -c "export LC_ALL=C; $*"; return; fi
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$RENDER_HOST" "export LC_ALL=C; $*" 2> >(grep -v setlocale >&2)
}

render_stop() {
    [[ -n "$RENDER_HOST" && $RENDER_START == 1 ]] || return 0
    # SIGTERM to the renderer (it writes its record on the way out), then the session goes with it
    render_ssh "pkill -TERM -f '[h]ead_camera.py --out $RENDER_OUT --serve' ; for i in \$(seq 40); do pgrep -f '[h]ead_camera.py --out $RENDER_OUT --serve' >/dev/null || break; sleep 0.5; done; tmux kill-session -t htd_render 2>/dev/null; true"
}

cleanup() {
    [[ $CLEANED == 1 ]] && return 0
    CLEANED=1
    trap - EXIT INT TERM
    stop_pid "$ROLLOUT_PID" rollout 15
    stop_pid "$RELAY_PID" "camera relay" 5
    stack_cleanup               # controller, walker, sim host
    local pat
    for pat in "${OWN_PATTERNS[@]}"; do
        if pgrep -f "$pat" >/dev/null; then
            say "left running ($pat), killing:"; pgrep -af "$pat"; pkill -KILL -f "$pat"
        fi
    done
    render_stop
}
nlines() { local n; n=$(wc -l < "$1" 2>/dev/null) || n=0; echo "${n:-0}"; }
ui_cmd() { "$PY" "$REPO/deploy/ui/cmd.py" --iface lo "$1" >> "$LOG/cmd.log" 2>&1; }

for pat in "${STACK_PATTERNS[@]}" "${OWN_PATTERNS[@]}"; do
    if pgrep -f "$pat" >/dev/null; then
        say "FAILED: already running ($pat): stop it first"; pgrep -af "$pat"; exit 1
    fi
done
mkdir -p "$LOG"
say "logs in $LOG"

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

# -- the renderer ------------------------------------------------------------------------------
if [[ -n "$RENDER_HOST" && $RENDER_START == 1 ]]; then
    say "renderer on $RENDER_HOST ($RENDER_DIR, $RENDER_OUT)"
    { if [[ $RENDER_LOCAL == 1 ]]; then cp "$SCENE/scene_frame.json" "$RENDER_DIR/$RENDER_OUT/scene_frame.json"; else
        scp -q "$SCENE/scene_frame.json" "$RENDER_HOST:$RENDER_DIR/$RENDER_OUT/scene_frame.json" 2>/dev/null; fi; } \
        || { say "FAILED: cannot copy scene_frame.json to $RENDER_HOST:$RENDER_DIR/$RENDER_OUT"; exit 1; }
    render_ssh "cd $RENDER_DIR && tmux kill-session -t htd_render 2>/dev/null; rm -f $RENDER_OUT/serve.log; \
        G=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1); \
        tmux new -d -s htd_render \"CUDA_VISIBLE_DEVICES=\$G ${RENDER_PREFIX:-} nice -n ${RENDER_NICE:-0} ~/isaacsim/python.sh real2sim/isaac/head_camera.py --out $RENDER_OUT --serve tcp://0.0.0.0:5556 --stream tcp://0.0.0.0:5558 --state tcp://0.0.0.0:5557 --frame $RENDER_OUT/scene_frame.json --jpeg-quality ${JPEG_QUALITY:-80} ${RENDER_ARGS:-} > $RENDER_OUT/serve.log 2>&1\""
    ok=0
    for _ in $(seq 60); do
        render_ssh "grep -q 'serving the head camera' $RENDER_DIR/$RENDER_OUT/serve.log 2>/dev/null" && { ok=1; break; }
        sleep 3
    done
    [[ $ok == 1 ]] || { say "FAILED: the renderer did not come up in 180 s:"; render_ssh "tail -5 $RENDER_DIR/$RENDER_OUT/serve.log"; exit 1; }
fi

# -- bring-up (as sim_supervisor.sh, the sim host wrapped and given the scene) -----------------------
say "sim host ($SIM): $SCENE/scene.xml, state to $STATE"
SIM_ARGS=(--iface lo --domain 0 --lower jetson --inspire)
if [[ $SIM == isaac ]]; then
    # the same host, run by Isaac Sim's python with PhysX behind mj_step (g1_walk/isaac_sim); a 4 ms step keeps it
    # at real time on a shared machine
    SIM_CMD=("$ISAAC_PY" "$MOD/isaac_sim/isaac_sim_host.py"); SIM_PP="$ISAAC_EXTRA"; SIM_ARGS+=(--sim-dt "${SIM_DT:-0.004}" --state-hz "${STATE_HZ:-75}")
else
    SIM_CMD=("$PY" "$MOD/scene_sim/scene_sim_host.py"); SIM_PP=""; [[ -n "$SIM_DT" ]] && SIM_ARGS+=(--sim-dt "$SIM_DT")
fi
(cd "$TELEOP" && exec env PYTHONUNBUFFERED=1 MUJOCO_GL=egl PSI_SIM_HOLD_RELEASE=session PYTHONPATH="$SIM_PP" \
    PSI_SIM_BASE_METER_S=1 PSI_SIM_RENDER_FPS=5 PSI_SIM_SCENE="$SCENE/scene.xml" \
    $([[ "${SPECTATOR:-0}" == 1 ]] && echo PSI_SIM_VIDEO="$LOG/spectator.mp4" PSI_SIM_VIDEO_FPS=30) \
    HTD_SIM_TOUCH="${HTD_SIM_TOUCH-$([[ -f $MOD/scene_sim/out/touch_model.npz ]] && echo $MOD/scene_sim/out/touch_model.npz)}" HTD_SIM_STATE_PUB="$STATE" HTD_SIM_STATE_HZ="${HTD_SIM_STATE_HZ:-30}" HTD_SIM_STATE_LOG="$LOG/objects.jsonl" \
    ${SIM_PREFIX:-} "${SIM_CMD[@]}" "${SIM_ARGS[@]}") > "$LOG/sim.log" 2>&1 &
SIM_PID=$!
wait_for "$LOG/sim.log" "Publishing rt/lowstate" 180 "$SIM_PID" || exit 1

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

say "releasing the sim's pelvis band"
"$PY" -c "import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b'{}', ('127.0.0.1', 14651))"

say "camera relay: $CAMERA -> $RELAY_ADDR"
PYTHONUNBUFFERED=1 nice -n "${ROLLOUT_NICE:-0}" "$PY" "$MOD/scene_sim/camera_relay.py" --stream "$CAMERA" --serve "$RELAY_ADDR" > "$LOG/relay.log" 2>&1 &
RELAY_PID=$!
wait_for "$LOG/relay.log" "^READY:" 20 "$RELAY_PID" || exit 1

say "rollout --supervisor True, head camera $RELAY_ADDR"
(cd "$REPO/policy_rollout" && exec env OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 nice -n "${ROLLOUT_NICE:-0}" "$PY" policy_rollout.py \
    --model_path "$MODEL" --policy_class MXT --embodiment g1_inspire_wb --inspire_hand_type dftp \
    --use_wrist_camera False --send_velocity_commands True --device cuda:0 --control_freq 30 \
    --inference_interval 45 --dds_interface lo --head_camera_source zmq --head_camera_device "$RELAY_ADDR" \
    --head_camera_layout duplicate --supervisor True --desired_stream_fps "${STREAM_FPS:-60}" \
    --embodiment_config_path "$EMB_YAML" --trunk_config_path "$TRUNK_YAML") \
    < /dev/null > "$LOG/rollout.log" 2>&1 &
ROLLOUT_PID=$!

say "waiting for HTDPOLICY READY TO DEPLOY"
wait_for "$LOG/controller.log" "HTDPOLICY READY TO DEPLOY" 120 "$ROLLOUT_PID" || { tail -15 "$LOG/rollout.log"; exit 1; }
sleep 2
n=$(nlines "$LOG/controller.log")
ok=0
for attempt in $(seq 15); do      # a start sent while the standing check flickers is ignored: send it again
    ui_cmd start
    for _ in $(seq 20); do
        tail -n "+$((n + 1))" "$LOG/controller.log" | grep -qF "[supervisor] READY -> RUNNING" && { ok=1; break 2; }
        sleep 0.1
    done
done
[[ $ok == 1 ]] || { say "FAILED: start did not reach RUNNING:"; tail -8 "$LOG/controller.log"; exit 1; }
T_START_SIM=$(tail -1 "$LOG/objects.jsonl" 2>/dev/null | "$PY" -c "import json,sys; print(json.loads(sys.stdin.read() or '{\"t\":0}')['t'])")
say "RUNNING for ${SECONDS_RUN} s (sim t at start ${T_START_SIM})"
for ((i = 0; i < SECONDS_RUN; i++)); do
    sleep 1
    kill -0 "$ROLLOUT_PID" 2>/dev/null && kill -0 "$CTRL_PID" 2>/dev/null && kill -0 "$SIM_PID" 2>/dev/null \
        || { say "a process died during the run"; break; }
    if ((i % 10 == 9)); then
        say "t+$((i + 1)) s: $(tail -n "+$((n + 1))" "$LOG/controller.log" | grep -F "[supervisor]" | tail -1)"
    fi
done
ui_cmd stop
sleep 1
tail -n "+$((n + 1))" "$LOG/controller.log" | grep -F "[supervisor]" > "$LOG/supervisor_transitions.log"
say "supervisor transitions during the run:"; cat "$LOG/supervisor_transitions.log"
RC=0
cleanup

# -- what the policy saw, and the verdict ---------------------------------------------------------
if [[ -n "$RENDER_HOST" ]]; then
    mkdir -p "$LOG/head_frames"
    if [[ $RENDER_LOCAL == 1 ]]; then cp -r "$RENDER_DIR/$RENDER_OUT/isaac_serve/." "$LOG/head_frames/" 2>/dev/null
    else rsync -az "$RENDER_HOST:$RENDER_DIR/$RENDER_OUT/isaac_serve/" "$LOG/head_frames/" 2>/dev/null; fi
    if [[ $RENDER_LOCAL == 1 ]]; then cp "$RENDER_DIR/$RENDER_OUT/serve.log" "$LOG/render.log" 2>/dev/null
    else rsync -az "$RENDER_HOST:$RENDER_DIR/$RENDER_OUT/serve.log" "$LOG/render.log" 2>/dev/null; fi
    if ls "$LOG/head_frames/frames/"*.jpg >/dev/null 2>&1; then
        ffmpeg -v error -y -framerate 15 -pattern_type glob -i "$LOG/head_frames/frames/*.jpg" \
            -c:v libx264 -pix_fmt yuv420p "$LOG/head_camera.mp4" && rm -rf "$LOG/head_frames/frames"
    fi
fi
"$PY" "$HERE/scene_result.py" --log "$LOG" --scene "$SCENE" --t_start "${T_START_SIM:-0}"
say "done: $LOG"
exit $RC
