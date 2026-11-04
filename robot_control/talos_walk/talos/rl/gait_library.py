"""TorchGaitLibrary — batched (GPU) evaluator of the stage-1 GaitCycles.

Given a commanded forward velocity ``vx`` and a per-environment gait clock ``t``,
:meth:`reference` returns desired joint positions/velocities for the FULL robot: library
joints follow the optimised gait (legs *and* arms swing), joints the library does not
cover hold their default angle. Cycles are linearly blended by velocity bracket centre —
with the march at conditioner 0 the blend spans 0 .. max speed. Reads the same YAML
schema as the numpy ``GaitNominal`` (via ``GaitYamlCodec``) and uses the same MirrorMap.
"""

from __future__ import annotations

import math

import torch

from talos.core.gait import load_gait_dir


class TorchGaitLibrary:
    def __init__(self, gait_dir: str, joint_names: list[str], default_pos, device):
        self.device = torch.device(device)
        self.joint_names = list(joint_names)
        self.default_pos = torch.as_tensor(default_pos, dtype=torch.float32, device=self.device)
        cycles = load_gait_dir(gait_dir)
        if not cycles:
            raise FileNotFoundError(f"no walk/march GaitCycles in {gait_dir}")
        self.cycles = cycles
        self.joint_order = cycles[0].joint_order
        self.degree = cycles[0].degree
        self.n_lib = len(self.joint_order)
        f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        self.vels = f32([c.v for c in cycles])
        self.T = f32([c.T for c in cycles])
        self.coeffs_pos = torch.stack([f32(c.pos) for c in cycles])      # [K, J, d+1]
        self.coeffs_vel = torch.stack([f32(c.vel) for c in cycles])
        self._binom = f32([math.comb(self.degree, i) for i in range(self.degree + 1)])
        self.period = float(2.0 * self.T.mean())
        mm = cycles[0].mirror()
        self.mirror_perm = torch.as_tensor(mm.perm, dtype=torch.long, device=self.device)
        self.mirror_sign = f32(mm.sign)
        self.lib_to_policy = torch.tensor([self.joint_names.index(n) for n in self.joint_order],
                                          dtype=torch.long, device=self.device)

    def _bezier(self, tau: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        d = ctrl.shape[-1] - 1
        i = torch.arange(d + 1, device=self.device)
        w = self._binom * tau.unsqueeze(1) ** i * (1 - tau).unsqueeze(1) ** (d - i)
        return torch.einsum("nd,njd->nj", w, ctrl)

    def _interp_by_velocity(self, vx: torch.Tensor):
        vx = torch.clamp(vx, float(self.vels[0]), float(self.vels[-1]))
        hi = torch.searchsorted(self.vels, vx).clamp(1, len(self.vels) - 1)
        lo = hi - 1
        w = ((vx - self.vels[lo]) / (self.vels[hi] - self.vels[lo]).clamp_min(1e-6)).unsqueeze(-1)
        pos = torch.lerp(self.coeffs_pos[lo], self.coeffs_pos[hi], w.unsqueeze(-1))
        vel = torch.lerp(self.coeffs_vel[lo], self.coeffs_vel[hi], w.unsqueeze(-1))
        T = torch.lerp(self.T[lo], self.T[hi], w.squeeze(-1))
        return pos, vel, T

    def reference(self, vx: torch.Tensor, t: torch.Tensor):
        """(q_des [N,A], qd_des [N,A], phase [N] in [0,1), domain [N] 0=right stance)."""
        N = vx.shape[0]
        pos_c, vel_c, T = self._interp_by_velocity(vx)
        period = 2.0 * T
        phase = torch.remainder(t, period) / period
        domain = (phase >= 0.5).long()
        tau = torch.where(domain == 1, (phase - 0.5) / 0.5, phase / 0.5)
        q_lib, qd_lib = self._bezier(tau, pos_c), self._bezier(tau, vel_c)
        mirror = domain.bool().unsqueeze(-1)
        q_lib = torch.where(mirror, self.mirror_sign * q_lib[:, self.mirror_perm], q_lib)
        qd_lib = torch.where(mirror, self.mirror_sign * qd_lib[:, self.mirror_perm], qd_lib)
        q_des = self.default_pos.unsqueeze(0).repeat(N, 1)
        qd_des = torch.zeros((N, len(self.joint_names)), device=self.device)
        q_des[:, self.lib_to_policy] = q_lib
        qd_des[:, self.lib_to_policy] = qd_lib
        return q_des, qd_des, phase, domain
