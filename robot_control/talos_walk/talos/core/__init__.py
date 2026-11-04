"""Core abstractions shared by every stage: robot description, gait cycle schema,
mirror (left/right relabel) map, workspace layout and the stage interfaces."""

from talos.core.robot import (RobotSpec, JointLayout, FootGeometry, BodyGeometry,   # noqa: F401
                              ControlTables, GaitGenSettings, CommandEnvelope,
                              TrackingLayout, Assets, joint_key)
from talos.core.registry import RobotRegistry, get_robot                            # noqa: F401
from talos.core.workspace import Workspace                                          # noqa: F401
from talos.core.gait import GaitCycle, GaitYamlCodec, load_gait_dir                 # noqa: F401
from talos.core.mirror import MirrorMap                                             # noqa: F401
from talos.core.interfaces import (GaitGenerator, ReferenceSolver, FeasibilityFilter,  # noqa: F401
                                   Trainer, PipelineStage, StageResult)
