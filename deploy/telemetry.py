"""What policy_rollout/policy_rollout.py publishes under --supervisor: status, predicted chunks, camera frames."""
import cv2
import numpy as np

from deploy.controller_bridge import TickTimer as LoopTimer
from deploy.msgs import CameraFrame, PolicyStatus, PredictedChunk

__all__ = ['LoopTimer', 'camera_message', 'chunk_message', 'is_ready', 'policy_status_message']

# Command group -> width, in PredictedChunk field order.
CHUNK_GROUPS = (('body_pose', 6), ('left_eef_to_body', 6), ('right_eef_to_body', 6),
                ('left_eef_link', 15), ('right_eef_link', 15), ('velocity_commands', 3))

MAX_AGE_S = 1.0
MISSING_AGE_S = 1e9    # age reported for an input never received (finite: the UI serialises to JSON)


def chunk_message(traj: dict, t_obs: int, interval: int, inference_ms: float, stamp: float) -> PredictedChunk:
    flat = {}
    horizon = None
    for key, width in CHUNK_GROUPS:
        if key not in traj:
            raise ValueError(f"chunk is missing {key}")
        a = np.asarray(traj[key], dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != width:
            raise ValueError(f"{key} has shape {a.shape}, expected (H, {width})")
        if horizon is None:
            horizon = a.shape[0]
        elif a.shape[0] != horizon:
            raise ValueError(f"{key} has horizon {a.shape[0]}, expected {horizon}")
        flat[key] = a.reshape(-1).tolist()
    return PredictedChunk(timestamp=float(stamp), t_obs=int(t_obs), horizon=int(horizon), interval=int(interval),
                          inference_ms=float(inference_ms), **flat)


def camera_message(rgb: np.ndarray, stamp: float, max_width: int = 640, quality: int = 70) -> CameraFrame:
    h, w = rgb.shape[:2]
    if w > max_width:
        h, w = max(1, round(h * max_width / w)), max_width
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return CameraFrame(timestamp=float(stamp), width=int(w), height=int(h), jpeg=list(buf.tobytes()))


def policy_status_message(ready: bool, publishing: bool, inference_ms: float, chunks: int, overrun_steps: int,
                          ages: tuple, loop_stats: tuple, interval: int, horizon: int, checkpoint: str,
                          detail: str, stamp: float) -> PolicyStatus:
    robot_age, eef_age, camera_age = ages
    p50, mx = loop_stats
    return PolicyStatus(timestamp=float(stamp), ready=bool(ready), publishing=bool(publishing),
                        inference_ms=float(inference_ms), chunks=int(chunks), overrun_steps=int(overrun_steps),
                        robot_state_age_s=float(robot_age), eef_state_age_s=float(eef_age),
                        camera_age_s=float(camera_age), loop_p50_ms=float(p50), loop_max_ms=float(mx),
                        interval=int(interval), horizon=int(horizon), checkpoint=str(checkpoint), detail=str(detail))


def is_ready(warmed_up: bool, ages: tuple, last_inference_ms, budget_ms: float) -> tuple:
    if not warmed_up:
        return False, "warming up"
    for age, missing in zip(ages, ("no robot state", "no hand state", "no camera frame")):
        if not age <= MAX_AGE_S:            # also catches NaN
            return False, missing
    if last_inference_ms is not None and not last_inference_ms <= budget_ms:
        return False, "inference slow"
    return True, "ready"
