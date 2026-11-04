"""STAGE 3 — RL in Genesis (CLF + AMO + AnyTracker, then AnyAdapter + world model).

    config        typed knobs -> the cfg dicts the environments consume (robot facts
                  come from the RobotSpec; only task-level knobs live here)
    gait_library  batched torch evaluator of the stage-1 GaitCycles
    clf           Control Lyapunov Function tracking reward
    kinematics    batched torch FK over the MJCF tree (rigid-body targets for g_t)
    canonical/gt  Any2Track canonicalised errors and the dense target assembler
    envs          HzdEnv -> AnyTrackEnv -> AdapterEnv
    nets          AdapterActor (frozen base + zero-init adapter), WorldModel, MBPPO
    trainers      AnyTrackTrainer / AnyAdapterTrainer (+ PipelineStage wrappers)
"""
