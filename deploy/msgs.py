"""DDS messages and topics of the deployment supervisor and its UI."""
from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import int32, sequence, uint8

TOPIC_SUPERVISOR = "rt/htd/supervisor"
TOPIC_POLICY_STATUS = "rt/htd/policy_status"
TOPIC_PREDICTED_CHUNK = "rt/htd/predicted_chunk"
TOPIC_CAMERA = "rt/htd/camera"
TOPIC_UI_COMMAND = "rt/htd/ui_command"

UI_COMMANDS = ("start", "stop", "rearm", "reset")
RECENT_LOG_LINES = 12


@dataclass
class SupervisorStatus(IdlStruct):
    timestamp: float
    state: str
    since_s: float
    reason: str
    walker_live: bool
    standing: bool
    standing_for_s: float
    policy_ready: bool
    frames_fresh: bool
    walk_enabled: bool
    manual_walk_enabled: bool
    fault: str
    gating: sequence[str]
    stick_lx: float
    stick_ly: float
    stick_rx: float
    stick_ry: float
    buttons: sequence[str]
    manual_vx: float
    manual_wz: float
    tick_p50_ms: float
    tick_max_ms: float
    log_line: str
    recent_log: sequence[str]       # the last RECENT_LOG_LINES supervisor log lines, oldest first


@dataclass
class PolicyStatus(IdlStruct):
    timestamp: float
    ready: bool
    publishing: bool
    inference_ms: float
    chunks: int32
    overrun_steps: int32
    robot_state_age_s: float
    eef_state_age_s: float
    camera_age_s: float
    loop_p50_ms: float
    loop_max_ms: float
    interval: int32
    horizon: int32
    checkpoint: str
    detail: str


@dataclass
class PredictedChunk(IdlStruct):
    # Row-major horizon x D with D = 6, 6, 6, 15, 15, 3 in field order.
    timestamp: float
    t_obs: int32
    horizon: int32
    interval: int32
    inference_ms: float
    body_pose: sequence[float]
    left_eef_to_body: sequence[float]
    right_eef_to_body: sequence[float]
    left_eef_link: sequence[float]
    right_eef_link: sequence[float]
    velocity_commands: sequence[float]


@dataclass
class CameraFrame(IdlStruct):
    timestamp: float
    width: int32
    height: int32
    jpeg: sequence[uint8]


@dataclass
class UiCommand(IdlStruct):
    timestamp: float
    nonce: int32
    command: str
    source: str
