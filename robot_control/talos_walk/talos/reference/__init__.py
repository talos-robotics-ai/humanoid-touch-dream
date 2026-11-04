"""STAGE 2 — walking-AMO reference generator ("phi gen").

    GaitNominal              numpy evaluator of the stage-1 library + stance-frame world
                             reconstruction (moving CoM / swing-foot targets)
    ReferenceDatasetBuilder  samples AMO commands, solves whole-body references over a
                             cycle with the shared IK, prunes with inverse dynamics
    PhiWalk / PhiTrainer     the residual MLP  phi(q_upper, rpy, h, phase, v) -> Δq_lower
"""
