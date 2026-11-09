"""`server.py --demo`: animated fake telemetry, so the page can be checked without the robot."""
import collections
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np

from deploy.msgs import RECENT_LOG_LINES, UI_COMMANDS, PredictedChunk
from deploy.ui.server import NonceSource, jsonable, measured_entry, predicted_entry
from g1_command import (EEF_POSITION_OFFSET, TORSO_HEIGHT_OFFSET, UNIFIED_TO_UNITREE_EEF_ROT_REMAP,  # noqa: E402
                        rot_mat_to_rpy, rpy_to_rot_mat)    # on sys.path via deploy.g1_spec

# (state, seconds): one cycle of ~20 s.
CYCLE = (("STABILIZING", 4.0), ("READY", 4.0), ("RUNNING", 6.0), ("HOLD", 3.0), ("READY", 3.0))
GATING = {"STABILIZING": ["standing %.1f / 2.0 s"], "READY": ["press Start"], "RUNNING": [],
          "HOLD": ["re-arm: L1+Start 1 s, or Reset"]}
HORIZON, INTERVAL, CONTROL_HZ = 60, 45, 30.0


def _pose(t):
    """Synthetic hands, fingers, torso and walk at time t (s)."""
    w = 2 * math.pi * 0.25
    left = [0.30 + 0.05 * math.sin(w * t), 0.20 + 0.03 * math.cos(w * t), -0.10 + 0.04 * math.sin(0.5 * w * t),
            0.1 * math.sin(w * t), 0.2, 0.3]
    right = [0.30 + 0.04 * math.cos(w * t), -0.20, -0.12 + 0.03 * math.sin(w * t), 0.0, 0.1 * math.cos(w * t), 0.0]
    grip = 0.07 + 0.02 * math.sin(0.7 * w * t)
    return SimpleNamespace(left=left, right=right, grip=grip, height=0.78 + 0.01 * math.sin(w * t),
                           vx=0.15 + 0.1 * math.sin(0.3 * w * t), wz=0.2 * math.sin(0.2 * w * t))


class DemoTelemetry:
    def __init__(self):
        self._t0 = time.monotonic()
        self._nonces = NonceSource()
        self._log = "[supervisor] demo started"
        self._recent = collections.deque([self._log], maxlen=RECENT_LOG_LINES)
        self._chunk = None          # (PredictedChunk, monotonic arrival)
        self._jpeg, self._jpeg_t, self._seq = None, -1.0, 0

    def _phase(self, t):
        t = t % sum(d for _, d in CYCLE)
        for state, d in CYCLE:
            if t < d:
                return state, t
            t -= d

    def _robot(self, t):
        p = _pose(t)
        rs = SimpleNamespace(torso_pose=[0, 0, p.height, 0.01 * math.sin(t), 0.02 * math.cos(t), 0],
                             left_end_effector_to_torso_pose=p.left, right_end_effector_to_torso_pose=p.right,
                             velocity_commands=[p.vx, 0.0, p.wz])
        eef = SimpleNamespace(hand_link_poses=[p.grip + 0.01, 0, 0] * 5 + [p.grip, 0, 0] * 5)
        return rs, eef

    def _make_chunk(self, t):
        rows = [_pose(t + k / CONTROL_HZ) for k in range(HORIZON)]
        def action(pose, side):
            # the action whose wrist target is `pose` (undo g1_command's offset and IK remap)
            R = rpy_to_rot_mat(pose[3:]) @ UNIFIED_TO_UNITREE_EEF_ROT_REMAP[side].T
            return list(np.array(pose[:3]) - EEF_POSITION_OFFSET) + list(rot_mat_to_rpy(R))

        def flat(f):
            return [float(v) for r in rows for v in f(r)]
        return PredictedChunk(
            timestamp=time.time(), t_obs=int(t * CONTROL_HZ), horizon=HORIZON, interval=INTERVAL, inference_ms=42.0,
            body_pose=flat(lambda r: [0, 0, r.height - TORSO_HEIGHT_OFFSET, 0, 0, 0]),
            left_eef_to_body=flat(lambda r: action(r.left, "left")),
            right_eef_to_body=flat(lambda r: action(r.right, "right")),
            left_eef_link=flat(lambda r: [r.grip, 0, 0] * 5), right_eef_link=flat(lambda r: [r.grip + 0.01, 0, 0] * 5),
            velocity_commands=flat(lambda r: [r.vx, 0.0, r.wz]))

    def _frame(self, t, state):
        img = np.zeros((120, 160, 3), np.uint8)
        img[:] = (40 + int(40 * math.sin(t)), 60, 90)
        cv2.circle(img, (80 + int(50 * math.sin(t)), 60), 14, (255, 255, 255), -1)
        cv2.putText(img, state, (6, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return cv2.imencode(".jpg", img)[1].tobytes()

    def stop(self):
        pass

    def snapshot(self) -> dict:
        now = time.monotonic()
        t = now - self._t0
        state, since = self._phase(t)
        if self._chunk is None or now - self._chunk[1] >= INTERVAL / CONTROL_HZ:
            self._chunk = (self._make_chunk(t), now)
        if t - self._jpeg_t >= 0.2:
            self._jpeg, self._jpeg_t, self._seq = self._frame(t, state), t, self._seq + 1
        rs, eef = self._robot(t)
        stabilizing = state == "STABILIZING"
        supervisor = dict(
            timestamp=time.time(), state=state, since_s=since, reason="demo", walker_live=True,
            standing=not stabilizing or since > 2.0, standing_for_s=min(since, 2.0), policy_ready=since > 1.0 or not stabilizing,
            frames_fresh=True, walk_enabled=state == "RUNNING", manual_walk_enabled=True, fault="",
            gating=[g % since if "%" in g else g for g in GATING.get(state, [])],
            stick_lx=0.3 * math.sin(t), stick_ly=0.3 * math.cos(t), stick_rx=0.0, stick_ry=0.0,
            buttons=["L1"] if int(t) % 5 == 0 else [], manual_vx=0.0, manual_wz=0.0,
            tick_p50_ms=0.8, tick_max_ms=2.1, log_line=self._log,
            recent_log=list(self._recent), ik_p50_ms=2.8, ik_max_ms=3.4)
        policy = dict(timestamp=time.time(), ready=supervisor["policy_ready"], publishing=state == "RUNNING",
                      inference_ms=42.0, chunks=int(t / 1.5), overrun_steps=0, robot_state_age_s=0.02,
                      eef_state_age_s=0.02, camera_age_s=0.1, loop_p50_ms=1.2, loop_max_ms=4.0,
                      interval=INTERVAL, horizon=HORIZON, checkpoint="demo.pt",
                      detail="ready" if supervisor["policy_ready"] else "warming up")
        return jsonable({
            "supervisor": supervisor, "policy": policy, "measured": measured_entry(rs, eef),
            "predicted": predicted_entry(self._chunk[0], rs, self._chunk[1]),
            "ages": {"robot_state": 0.02, "eef_state": 0.02, "supervisor": 0.1, "policy_status": 0.2,
                     "predicted_chunk": now - self._chunk[1], "camera": t - self._jpeg_t, "ui_command": None},
            "dds_alive": True,
            "camera_seq": self._seq,
        })

    def camera(self):
        return self._jpeg

    def send_command(self, name: str, source: str = "ui") -> int:
        if name not in UI_COMMANDS:
            raise ValueError(f"unknown command {name!r}")
        nonce = self._nonces.next()
        self._log = f"[supervisor] demo: {name} from {source} (nonce {nonce})"
        self._recent.append(self._log)
        return nonce
