"""GaitCycle — the half-periodic single-support gait schema every stage exchanges.

One cycle = one single-support half of duration ``T`` (right foot planted, left foot
swinging); the other half is the MirrorMap relabel. Per joint: degree-d Bezier control
points for position AND velocity. ``conditioner`` is the velocity bracket the cycle
covers ([v-0.1, v+0.1]); the library interpolates cycles by bracket centre.

The YAML layout is the published Caltech-AMBER ``robot_rl`` trajectory schema (so any
library in that schema — e.g. the shipped G1 seeds — is readable), extended with
``gait_mode`` / ``realised_speed`` / ``provenance`` metadata.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from talos.core.bezier import bezier_eval, fit_bezier
from talos.core.mirror import MirrorMap


@dataclass
class GaitCycle:
    name: str
    joint_order: list[str]
    T: float                                 # single-support duration (s)
    pos: np.ndarray                          # [J, d+1] position control points
    vel: np.ndarray                          # [J, d+1] velocity control points
    conditioner: tuple[float, float]         # velocity bracket
    degree: int = 5
    gait_mode: str = "walk"                  # walk | march | (future: stair, ...)
    ref_frame: str = ""                      # stance (right) foot link
    realised_speed: float | None = None      # measured forward speed of the solution
    provenance: dict = field(default_factory=dict)

    # --------------------------------------------------------------- props
    @property
    def v(self) -> float:
        return 0.5 * (self.conditioner[0] + self.conditioner[1])

    @property
    def period(self) -> float:
        return 2.0 * self.T

    @property
    def n_joints(self) -> int:
        return len(self.joint_order)

    def mirror(self) -> MirrorMap:
        return MirrorMap.from_names(self.joint_order)

    # ----------------------------------------------------------- evaluate
    def half(self, tau: float):
        """(q, qd) of the base (right-stance) half at tau in [0,1], library joint order."""
        return bezier_eval(tau, self.pos), bezier_eval(tau, self.vel)

    def at_phase(self, phase: float):
        """(q, qd, domain) at full-cycle phase in [0,1); domain 1 = mirrored half."""
        phase = float(phase) % 1.0
        domain = 1 if phase >= 0.5 else 0
        tau = (phase - 0.5) / 0.5 if domain else phase / 0.5
        q, qd = self.half(tau)
        if domain:
            mm = self.mirror()
            q, qd = mm.apply(q), mm.apply(qd)
        return q, qd, domain

    def sample_half(self, n: int):
        """[n, J] positions and velocities on a uniform tau grid."""
        taus = np.linspace(0.0, 1.0, n)
        return bezier_eval(taus, self.pos), bezier_eval(taus, self.vel), taus

    # ------------------------------------------------------------ build
    @classmethod
    def from_samples(cls, name, joint_order, T, q_half, qd_half, conditioner,
                     degree=5, pin_end_slopes=True, **meta) -> "GaitCycle":
        """Fit Beziers to [M, J] sampled half-cycle positions/velocities."""
        q_half, qd_half = np.asarray(q_half), np.asarray(qd_half)
        taus = np.linspace(0, 1, q_half.shape[0])
        pos, vel = [], []
        for j in range(q_half.shape[1]):
            if pin_end_slopes:
                pos.append(fit_bezier(q_half[:, j], degree, taus,
                                      d0=float(qd_half[0, j]) * T, d1=float(qd_half[-1, j]) * T))
            else:
                pos.append(fit_bezier(q_half[:, j], degree, taus))
            vel.append(fit_bezier(qd_half[:, j], degree, taus))
        return cls(name=name, joint_order=list(joint_order), T=float(T),
                   pos=np.array(pos), vel=np.array(vel),
                   conditioner=(float(conditioner[0]), float(conditioner[1])),
                   degree=degree, **meta)


def _plain(obj):
    """Recursively turn tuples/numpy scalars into plain YAML-safe types."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    return obj


class GaitYamlCodec:
    """Read/write GaitCycle <-> the half-periodic YAML schema."""

    @staticmethod
    def is_gait_doc(d: dict) -> bool:
        return isinstance(d, dict) and d.get("type") == "half_periodic" and "single_support" in d

    @classmethod
    def load(cls, path: str | os.PathLike) -> GaitCycle | None:
        with open(path) as f:
            d = yaml.safe_load(f)
        if not cls.is_gait_doc(d):
            return None
        ss = d["single_support"]
        jo = list(ss["joint_order"])
        bz = ss["bezier_coeffs"]
        pos = np.array([bz["joints"][n] for n in jo], dtype=float)
        vel = np.array([bz["joint_vels"][n] for n in jo], dtype=float)
        cond = d["conditioner"]
        return GaitCycle(
            name=str(d.get("name", Path(path).stem)), joint_order=jo, T=float(ss["T"][0]),
            pos=pos, vel=vel, conditioner=(float(cond[0]), float(cond[-1])),
            degree=int(ss.get("spline_order", pos.shape[1] - 1)),
            gait_mode=str(d.get("gait_mode", "walk")), ref_frame=str(ss.get("ref_frame", "")),
            realised_speed=d.get("realised_speed"), provenance=dict(d.get("provenance", {})),
        )

    @staticmethod
    def dump(cycle: GaitCycle, path: str | os.PathLike) -> Path:
        doc = {
            "type": "half_periodic",
            "name": cycle.name,
            "domain_sequence": ["single_support"],
            "conditioner": [float(cycle.conditioner[0]), float(cycle.conditioner[1])],
            "gait_mode": cycle.gait_mode,
            "realised_speed": (None if cycle.realised_speed is None else float(cycle.realised_speed)),
            "provenance": _plain(cycle.provenance),
            "single_support": {
                "T": [float(cycle.T)],
                "spline_order": int(cycle.degree),
                "joint_order": list(cycle.joint_order),
                "bezier_coeffs": {
                    "joints": {n: [float(x) for x in cycle.pos[i]] for i, n in enumerate(cycle.joint_order)},
                    "joint_vels": {n: [float(x) for x in cycle.vel[i]] for i, n in enumerate(cycle.joint_order)},
                },
                "ref_frame": cycle.ref_frame,
                "ref_frame_domain": "single_support",
                "reference_frame_type": "body",
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(doc, f, default_flow_style=None, sort_keys=False)
        return path


def load_gait_dir(gait_dir: str | os.PathLike, modes: tuple[str, ...] = ("walk", "march")) -> list[GaitCycle]:
    """All cycles in a directory, one per velocity bracket, sorted by speed."""
    cycles: dict[float, GaitCycle] = {}
    for f in sorted(glob.glob(os.path.join(str(gait_dir), "*.yaml"))):
        c = GaitYamlCodec.load(f)
        if c is None or (modes and c.gait_mode not in modes):
            continue
        cycles[round(c.v, 3)] = c
    return [cycles[k] for k in sorted(cycles)]
