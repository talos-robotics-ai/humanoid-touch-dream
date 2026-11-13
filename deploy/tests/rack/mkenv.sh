#!/usr/bin/env bash
# on the rack: the deploy stack's Python environment (teleop conda env + what g1_walk/setup.sh and the rollout add)
export LC_ALL=C
set -x
R=$(cd "$(dirname "$0")/../../.." && pwd); T=$R/g1_walk/teleop-cbf-perception
CONDA=~/miniforge3/bin/conda
$CONDA env list | grep -q "^htd_stack " || $CONDA env create -y -n htd_stack -f $T/environment.yml || exit 1
PY=~/miniforge3/envs/htd_stack/bin/python
$PY -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 || exit 1
grep -v "^torchvision\|^torch=" $T/requirements.txt > /tmp/req_teleop.txt
$PY -m pip install -r /tmp/req_teleop.txt || echo "TELEOP REQ FAILED"
$PY -c 'import unitree_sdk2py' 2>/dev/null || $PY -m pip install -e $T/third_party/unitree_sdk2_python || echo "SDK FAILED"
grep -v "^pymodbus" $R/g1_walk/requirements.txt > /tmp/req_g1walk.txt
$PY -m pip install -r /tmp/req_g1walk.txt || echo "G1WALK REQ FAILED"
$PY -m pip install --no-deps -e $R/g1_walk/.deps/inspire_hands_unitree_sdk/dftp/inspire_hand_sdk || echo "INSPIRE FAILED"
$PY -m pip install --no-deps -e $R/robot_control/dex-retargeting || echo "DEX FAILED"
[ -f $R/policy_rollout/requirements-rollout.txt ] && { grep -v "^torch" $R/policy_rollout/requirements-rollout.txt > /tmp/req_roll.txt; $PY -m pip install -r /tmp/req_roll.txt || echo "ROLLOUT REQ FAILED"; }
$PY -m pip install --no-deps -e $R/htd_policy 2>/dev/null || $PY -m pip install --no-deps -e $R || echo "HTD_POLICY INSTALL?"
$PY -c "import torch,mujoco,pinocchio,cyclonedds,unitree_sdk2py,zmq,logging_mp;print('ENV_OK',torch.__version__,torch.cuda.is_available(),mujoco.__version__)"
echo MKENV_DONE
