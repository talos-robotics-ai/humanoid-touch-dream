"""Thresholds and button bindings of the G1 deployment, in one place."""
import os
import sys
from dataclasses import dataclass

_POLICY_ROLLOUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policy_rollout")
if _POLICY_ROLLOUT not in sys.path:
    sys.path.insert(0, _POLICY_ROLLOUT)
from g1_command import EEF_POSITION_OFFSET, TORSO_HEIGHT_OFFSET  # noqa: E402


@dataclass(frozen=True)
class StandingSpec:
    stand_height: float = 0.78
    height_tol: float = 0.06
    tilt_max: float = 0.10
    gyro_max: float = 0.25
    leg_dq_max: float = 0.5
    lowstate_max_age: float = 0.1
    hold_s: float = 2.0
    leg_joints: tuple = tuple(range(12))


@dataclass(frozen=True)
class RemoteSpec:
    # Bits of the teleop KeyMap: start=2, L1=1, X=10, Y=11.
    button_start: int = 2
    button_stop: int = 10
    button_reset: int = 11
    button_rearm_modifier: int = 1
    deadband: float = 0.3
    vx_forward_max: float = 0.30
    vx_backward_max: float = 0.20
    wz_max: float = 0.50
    release_s: float = 1.0
    rearm_hold_s: float = 1.0
    debounce_s: float = 0.05


@dataclass(frozen=True)
class SupervisorSpec:
    ui_command_max_age: float = 0.5
    policy_status_max_age: float = 1.0
    standing_hold_s: float = 2.0
    manual_release_s: float = 1.0
    start_frame_margin_s: float = 0.15


@dataclass(frozen=True)
class PlotSpec:
    eef_offset: tuple = tuple(EEF_POSITION_OFFSET)
    torso_height_offset: float = TORSO_HEIGHT_OFFSET


@dataclass(frozen=True)
class G1DeploySpec:
    standing: StandingSpec
    remote: RemoteSpec
    supervisor: SupervisorSpec
    plot: PlotSpec


G1_SPEC = G1DeploySpec(standing=StandingSpec(), remote=RemoteSpec(),
                       supervisor=SupervisorSpec(), plot=PlotSpec())
