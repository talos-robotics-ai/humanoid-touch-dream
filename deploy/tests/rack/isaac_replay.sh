#!/usr/bin/env bash
# on the rack: one operator replay, physics in PhysX (SIM=isaac, the default) or MuJoCo (SIM=mujoco); prints the
# result and the host's pace.  usage: isaac_replay.sh TAG [env...]
export LC_ALL=C
cd "$(dirname "$0")/../../.."   # the repo root
TAG=$1; shift
pkill -9 -f "[i]saac_sim/isaac_sim_host.py|[s]cene_sim/scene_sim_host.py" 2>/dev/null; sleep 1
env HTD_SIM_TOUCH_HZ=50 SIM_PREFIX="${SIM_PREFIX-}" PYTHONPATH="" ISAAC_EXTRA="$HOME/Desktop/unitree_sdk2_python:$HOME/isaac_extra" HTD_SIM_LOG_BODIES=left_base_link,pelvis,left_index_2,left_thumb_4 "$@" \
  timeout 600 bash deploy/tests/sim_scene_replay.sh --sim ${SIM:-isaac} --commands "g1_walk/scene_sim/out/commands/episode_31_commands_async+crossfade.npz" \
  --scene ${SCENE:-g1_walk/scene_sim/out/unified_ep30} --python ~/miniforge3/envs/htd_stack/bin/python --tag $TAG 2>&1 | grep -E "lifted|max_lift|max_travel|final_|in_bin|tilt|bin_moved|FAILED" | tr -d '\n'; echo
L=$(ls -dt deploy/tests/logs/replay_*_$TAG | head -1); echo $L
grep -E "real-time factor" $L/sim.log | awk '{print $4}' | sort -n | awk '{a[NR]=$1} END {print "rtf median", a[int(NR/2)+1], "min", a[1], "max", a[NR], "n", NR}'
grep -E "\[isaac_sim\] t=|band (let|kept)|letting go|Traceback" $L/sim.log | sed "s/.*py stderr\]: //" | tail -${TAILN:-4} | cut -c1-210
