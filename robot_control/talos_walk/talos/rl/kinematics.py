"""Batched forward kinematics in pure torch (GPU, all envs at once), robot-agnostic.

Needed because the Any2Track-style dense target includes **rigid-body targets** (link
positions/orientations) of the *reference* pose. That configuration is never instantiated
in the simulator, so Genesis cannot report it — we evaluate it ourselves from the
reference joint angles.

The tree is exported from the **MJCF** (the model Genesis loads) by
``talos.tools.export_kintree`` into ``data/<name>/kintree.npz``, so the FK agrees with
Genesis' own link poses to float precision. (Building it from the URDF can silently
disagree when the URDF and MJCF are different revisions.)

MuJoCo convention, quaternions (w,x,y,z):
    T_body = T_parent · (body_pos, body_quat) · Rot(jnt_axis, q)
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------- quaternion utils (w,x,y,z)
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    h = 0.5 * angle
    return torch.cat([torch.cos(h).unsqueeze(-1), axis * torch.sin(h).unsqueeze(-1)], dim=-1)


class RobotKinematics:
    """Batched FK over the MJCF body tree; returns poses of the tracked links."""

    def __init__(self, kintree_path, device, dtype=torch.float32):
        d = np.load(kintree_path, allow_pickle=True)
        t = lambda a, dt=dtype: torch.as_tensor(a, device=device, dtype=dt)
        self.device, self.dtype = device, dtype
        self.joint_names = [str(s) for s in d["joint_names"]]
        self.body_names = [str(s) for s in d["body_names"]]
        self.link_names = [str(s) for s in d["link_names"]]
        self.body_parent = t(d["body_parent"], torch.long)
        self.body_pos = t(d["body_pos"])
        self.body_quat = t(d["body_quat"])
        self.body_jnt = t(d["body_jnt"], torch.long)
        self.body_jnt_axis = t(d["body_jnt_axis"])
        self.body_jnt_pos = t(d["body_jnt_pos"])
        self.link_body = t(d["link_body"], torch.long)
        self.n_bodies = len(self.body_names)
        self.n_links = len(self.link_names)
        self.n_joints = len(self.joint_names)
        bp = self.body_parent.cpu().numpy()
        assert all(bp[i] < i for i in range(self.n_bodies)), "body tree not topologically sorted"
        self._has_jnt_off = bool((self.body_jnt_pos.abs() > 1e-9).any())

    def forward(self, root_pos, root_quat, q):
        """root_pos[B,3], root_quat[B,4] (w,x,y,z), q[B,A] -> link_pos[B,L,3], link_quat[B,L,4]."""
        B = q.shape[0]
        bp = torch.empty((B, self.n_bodies, 3), device=q.device, dtype=q.dtype)
        bq = torch.empty((B, self.n_bodies, 4), device=q.device, dtype=q.dtype)
        for i in range(self.n_bodies):
            par = int(self.body_parent[i])
            if par < 0:
                # root body IS the floating base link (Genesis get_pos/get_quat)
                bp[:, i], bq[:, i] = root_pos, root_quat
                continue
            p_p, q_p = bp[:, par], bq[:, par]
            p_i = p_p + quat_rotate(q_p, self.body_pos[i].expand(B, 3))
            q_i = quat_mul(q_p, self.body_quat[i].expand(B, 4))
            j = int(self.body_jnt[i])
            if j >= 0:
                qr = quat_from_axis_angle(self.body_jnt_axis[i].expand(B, 3), q[:, j])
                if self._has_jnt_off:   # rotate about a point offset from the body origin
                    off = self.body_jnt_pos[i].expand(B, 3)
                    p_i = p_i + quat_rotate(q_i, off - quat_rotate(qr, off))
                q_i = quat_mul(q_i, qr)
            bp[:, i], bq[:, i] = p_i, q_i
        return bp[:, self.link_body], bq[:, self.link_body]
