#!/usr/bin/env bash
# on the rack: one closed-loop policy rollout, physics and camera both in Isaac Sim.  usage: isaac_rollout.sh RUN_NAME TAG [seconds]
# The repo checkout on the rack must keep its .git (the walker checks its pinned teleop commit). RUNS: the folder of
# training run directories [~/Desktop/humanoid-touch-dream/runs]. Prints the outcome, the real-time factor and the
# walker/relay health; the logs are in deploy/tests/logs/scene_*_TAG.
export LC_ALL=C
cd "$(dirname "$0")/../../.."   # the repo root
RUN=$1; TAG=$2; SEC=${3:-90}
pkill -9 -f "[i]saac_sim/isaac_sim_host.py" 2>/dev/null; sleep 1
env HTD_SIM_TOUCH_HZ=50 SIM_PREFIX="${SIM_PREFIX-}" RENDER_PREFIX="${RENDER_PREFIX-}" RENDER_NICE=${RENDER_NICE:-15} ROLLOUT_NICE=${ROLLOUT_NICE:-8} RENDER_ARGS="${RENDER_ARGS:---max-fps 10 --supersample 1}" STREAM_FPS=${STREAM_FPS:-20} PYTHONPATH="" ISAAC_EXTRA="$HOME/Desktop/unitree_sdk2_python:$HOME/isaac_extra" HTD_SIM_LOG_BODIES=left_base_link,pelvis,left_index_2,left_thumb_4 \
  timeout 900 bash deploy/tests/sim_scene_rollout.sh --sim isaac --sim_dt ${SIM_DT:-0.006667} --model_path ${RUNS:-$HOME/Desktop/humanoid-touch-dream/runs}/$RUN \
  --scene ${SCENE:-g1_walk/scene_sim/out/unified_ep30} --python ~/miniforge3/envs/htd_stack/bin/python --seconds $SEC \
  --render_host local --render_dir Desktop/real2sim_rgbd --tag $TAG 2>&1 | grep -E "HOLD|FAILED|lifted|max_lift|max_travel|final_|in_bin|tilt|bin_moved|success|done|vessel_in" -A0 | tr -d '\n' | sed 's/\[scene_rollout\]/\n[scene_rollout]/g'; echo
L=$(ls -dt deploy/tests/logs/scene_*_$TAG | head -1); echo $L
grep -E "real-time factor" $L/sim.log | awk '{print $4}' | sort -n | awk '{a[NR]=$1} END {print "rtf median", a[int(NR/2)+1], "min", a[1], "max", a[NR]}'
grep -E "band let go|Traceback" $L/sim.log | sed "s/.*py stderr\]: //" | tail -3 | cut -c1-160
tail -2 $L/relay.log | cut -c1-140
