# Documentation

This fork of [humanoid-touch-dream](https://github.com/chrisyrniu/humanoid-touch-dream)
trains HTD on our own teleoperation data and deploys it on the Unitree G1 through the
teleop stack's walking policy. These pages cover that pipeline end to end; the module
READMEs (`htd_policy/`, `policy_rollout/`, `g1_walk/`, `data_collect/`,
`robot_control/`) remain the reference for each component.

| Page | Content |
| --- | --- |
| [dataset.md](dataset.md) | the Hugging Face dataset, download, processing into a training set, the train/validation split |
| [model.md](model.md) | the HTD model as configured for our data: inputs, trunk, action chunk, touch dreaming, the tactile input scale, run artifacts |
| [training.md](training.md) | environment, the training launcher and its settings, metrics, reference runs |
| [evaluation.md](evaluation.md) | open-loop replay on the validation episodes, inference check, MuJoCo and Genesis replays, unit tests |
| [deployment.md](deployment.md) | the deployment supervisor and its states, remote buttons and sticks, the monitoring UI, sessions in MuJoCo and on the robot, troubleshooting, measured latency |

## Pipeline

```text
teleop takes ──htd-dataset-converter──► HTD raw episodes ──► Hugging Face (Relo02/loco-manipulation-data)
                                                                      │  scripts/prepare_dataset.sh
                                                                      ▼
                                                     processed/<task>/g1_inspire_wb/*.hdf5
                                                                      │  scripts/train_htd.sh
                                                                      ▼
                                                     runs/<prefix>_<task>_HTD/  (checkpoints, configs, stats)
                                                          │                        │
                                   policy_rollout/replay_eval.py          policy_rollout/check_inference.py
                                   g1_walk/replay, g1_walk/genesis        policy_rollout/policy_rollout.py + g1_walk/
                                   (validation data, simulation)          + deploy/ supervisor and UI (robot)
```

## Environments

| Environment | Created by | Used for | Runs on |
| --- | --- | --- | --- |
| `htd_policy` (conda) | [training.md](training.md#environment) | dataset processing, training, `replay_eval.py` | training server |
| `htd_rollout` (conda or venv) | `policy_rollout/setup_rollout_env.sh` | `policy_rollout.py`, `check_inference.py`, `replay_eval.py`, the UI (`deploy/ui/server.py`, `deploy/ui/cmd.py`), tests | operator laptop |
| teleop env + `g1_walk/setup.sh` | `g1_walk/setup.sh` | the walking policy, `g1_walk/controller.py`, MuJoCo replays (also run from the `htd_rollout` venv layered on it) | operator laptop |
| `htd_data` (conda) | `data_collect/README.md` | upstream's Vision Pro collector only; not needed for our data | collection PC |

## Repository layout

| Path | Content |
| --- | --- |
| `htd_policy/` | model, training, configs (upstream, with the tactile input scale option and GPU-only forward-pass changes) |
| `policy_rollout/` | inference on the robot: fast GPU inference, chunk scheduling, head camera sources, offline evaluation |
| `g1_walk/` | running the policy on the teleop's G1 walking policy (this fork) |
| `deploy/` | the deployment supervisor (states, standing check, remote buttons and sticks, G1 thresholds in `g1_spec.py`), its DDS messages, the monitoring UI and its tests (this fork) |
| `data_collect/` | upstream's collector and the data processors and validator we reuse |
| `robot_control/` | upstream's controller, kinematics and assets (the URDFs and hand retargeting are used by `g1_walk`) |
| `scripts/` | `prepare_dataset.sh`, `train_htd.sh` |
| `docs/` | these pages |
