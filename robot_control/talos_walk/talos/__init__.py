"""talos — hardware-agnostic humanoid walking platform.

Pipeline (one command per stage, every command takes ``--robot <name>``):

    stage 1  gaitgen    HZD gait library (CasADi/Pinocchio/IPOPT NLP + IK march)  [env amo_walk]
    stage 2  reference  walking-AMO reference generator phi_walk                   [env amo_walk]
    stage 3  rl         AnyTracker (3b) then AnyAdapter + world model (3c)         [env g1_lip]
    deploy   web tele-op panel / offline player

Robots are plug-ins under ``talos/robots/<name>/`` (see ``talos.core.robot.RobotSpec``).
"""

__version__ = "0.1.0"
