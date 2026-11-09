"""Turn a stream of predicted action chunks into one command per control step.

Shared by ``policy_rollout.py`` (on the robot) and ``replay_eval.py`` (offline),
so the schedule evaluated offline is the one the robot runs. No robot or torch
dependencies.

Time is counted in control steps ``t``. A chunk predicted from the observation
taken at step ``t_obs`` gives the command for step ``t_obs + i`` at index ``i``.

Two options on top of the plain "execute ``interval`` steps, then switch":

crossfade_steps
    When a new chunk is adopted, blend from the old chunk's continuation into
    the new one over this many steps instead of jumping. Positions blend
    linearly, rotations by slerp. The old chunk must still have steps left, so
    the blend is limited to ``chunk_len - interval`` steps.

async_mode
    Inference runs while commands keep flowing. The next inference starts
    ``lead_steps`` before the switch (the measured inference time plus a
    margin), so the switch still happens at chunk step ``interval`` and the new
    chunk is entered at the step matching the current time (its first
    ``lead_steps`` steps are already in the past). Steps between inferences
    become ``interval - lead_steps``.
"""
import math

import numpy as np

# Command layouts used by the G1 whole-body rollout: (x, y, z, roll, pitch, yaw) poses
# blend their rotation by slerp; 'switch' commands are never blended -- the walk command
# (vx, vy, yaw rate) goes from one chunk's value to the next at the switch, as the policy
# predicted it, so the crossfade does not low-pass the walking speed; everything else
# blends linearly.
G1_COMMAND_KINDS = {'body_pose': 'pose6', 'left_eef_to_body': 'pose6', 'right_eef_to_body': 'pose6',
                    'velocity_commands': 'switch'}


# ---------------------------------------------------------------- rotations
# ZYX roll/pitch/yaw, R = Rz(yaw) Ry(pitch) Rx(roll), as policy_rollout.rpy_to_rot_mat.

def rpy_to_quat(rpy):
    r, p, y = rpy[..., 0] / 2, rpy[..., 1] / 2, rpy[..., 2] / 2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr], axis=-1)


def quat_to_rpy(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))], axis=-1)


def slerp(q0, q1, a):
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    return (math.sin((1 - a) * theta) * q0 + math.sin(a * theta) * q1) / math.sin(theta)


def blend(kind, old, new, a):
    """(1 - a) * old + a * new, with slerp for the rotation of a pose6; ``new`` as is
    for a 'switch'."""
    old, new = np.asarray(old, dtype=np.float64), np.asarray(new, dtype=np.float64)
    if kind == 'switch':
        return new.copy()
    out = (1 - a) * old + a * new
    if kind == 'pose6':
        out[3:6] = quat_to_rpy(slerp(rpy_to_quat(old[3:6]), rpy_to_quat(new[3:6]), a))
    return out


# ---------------------------------------------------------------- scheduler

class ChunkScheduler:
    """Decide when to infer and what to command at each control step.

    Per control step the caller does::

        if sched.should_request(t):
            gen = sched.mark_requested(t)
            ... run inference on the observation taken now (now or in a worker) ...
            sched.submit(chunk, t_obs=t, t_now=current_t, generation=gen, latency_steps=...)
        cmd = sched.command(t)      # None until the first chunk arrives

    ``chunk`` is a dict of (chunk_len, D) arrays.
    """

    def __init__(self, interval, crossfade_steps=0, async_mode=False, kinds=None, latency_margin_steps=1):
        if interval < 1:
            raise ValueError(f'interval must be >= 1, got {interval}')
        if crossfade_steps < 0:
            raise ValueError(f'crossfade_steps must be >= 0, got {crossfade_steps}')
        self.interval = int(interval)
        self.crossfade_requested = int(crossfade_steps)
        self.async_mode = bool(async_mode)
        self.kinds = dict(kinds or {})
        self.margin = int(latency_margin_steps)
        self.latency_steps = None  # running estimate of inference time, in control steps
        self.generation = 0
        self.reset()

    def reset(self):
        """Forget all chunks; results of inferences already in flight will be ignored."""
        self.generation += 1
        self.cur = None    # (chunk, t_obs, chunk_len)
        self.prev = None   # chunk being faded out
        self.fade_start = None
        self.crossfade = 0
        self.pending = None
        self.next_request = 0
        self.overrun_steps = 0  # steps commanded past the end of a chunk (last step held)
        self.switches = 0

    @property
    def lead_steps(self):
        if not self.async_mode or self.latency_steps is None:
            return 0
        return max(0, min(self.interval - 1, math.ceil(self.latency_steps) + self.margin))

    def should_request(self, t):
        return self.pending is None and t >= self.next_request

    def mark_requested(self, t):
        self.pending = t
        return self.generation

    def cancel_request(self, generation=None):
        """An inference failed: allow a new request right away."""
        if generation is None or generation == self.generation:
            self.pending = None

    def submit(self, chunk, t_obs, t_now, generation=None, latency_steps=None):
        """Adopt a chunk predicted from the observation at t_obs. Returns False if stale."""
        if generation is not None and generation != self.generation:
            return False
        if latency_steps is not None:
            self.latency_steps = (float(latency_steps) if self.latency_steps is None
                                  else 0.7 * self.latency_steps + 0.3 * float(latency_steps))
        length = len(next(iter(chunk.values())))
        if self.interval > length:
            raise ValueError(f'inference interval {self.interval} exceeds the chunk length {length}')
        self.crossfade = min(self.crossfade_requested, length - self.interval)
        if self.cur is not None and self.crossfade > 0:
            self.prev, self.fade_start = self.cur, t_now
        else:
            self.prev = None
        self.cur = ({k: np.asarray(v) for k, v in chunk.items()}, int(t_obs), length)
        self.pending = None
        self.next_request = int(t_obs) + self.interval - self.lead_steps
        self.switches += 1
        return True

    def command(self, t):
        """The command for control step t, or None before the first chunk."""
        if self.cur is None:
            return None
        chunk, t0, length = self.cur
        i = t - t0
        if i >= length:
            self.overrun_steps += 1
            i = length - 1
        i = max(i, 0)
        out = {k: v[i] for k, v in chunk.items()}
        if self.prev is not None:
            k = t - self.fade_start
            if k >= self.crossfade:
                self.prev = None
            elif k >= 0:
                pchunk, pt0, plen = self.prev
                j = min(max(t - pt0, 0), plen - 1)
                a = (k + 1) / (self.crossfade + 1)
                out = {key: blend(self.kinds.get(key, 'linear'), pchunk[key][j], val, a)
                       for key, val in out.items()}
        return out

    def describe(self):
        mode = 'async' if self.async_mode else 'blocking'
        fade = f'crossfade {self.crossfade_requested} steps' if self.crossfade_requested else 'no crossfade'
        return f'{mode} inference, switch at chunk step {self.interval}, {fade}'
