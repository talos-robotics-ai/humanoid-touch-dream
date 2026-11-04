"""MirrorMap — the half-periodic left/right relabel, shared by every stage.

A half-periodic gait stores ONE single-support half (right foot planted, left swinging);
the other half is its sagittal mirror: swap left<->right joints and flip the sign of
every roll and yaw joint (and of the base lateral/roll/yaw coordinates).

This map is needed identically in the NLP (periodicity constraint), the feasibility
filter (full-cycle reconstruction), the numpy gait evaluator (stage 2) and the torch gait
library (stage 3) — so it lives here, once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MirrorMap:
    names: tuple[str, ...]
    perm: np.ndarray        # mirrored[i] = sign[i] * value[perm[i]]
    sign: np.ndarray

    @staticmethod
    def mirror_name(name: str) -> str:
        return name.replace("left", "\0").replace("right", "left").replace("\0", "right")

    @classmethod
    def from_names(cls, names) -> "MirrorMap":
        names = tuple(names)
        perm, sign = [], []
        for n in names:
            m = cls.mirror_name(n)
            perm.append(names.index(m) if m in names else names.index(n))
            sign.append(-1.0 if ("roll" in n or "yaw" in n) else 1.0)
        return cls(names, np.array(perm, dtype=int), np.array(sign, dtype=float))

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Mirror a [..., J] joint array."""
        return self.sign * np.asarray(values)[..., self.perm]

    def is_symmetric(self) -> bool:
        """True when every joint has a counterpart (self-mapped central joints allowed)."""
        return bool(np.all(self.perm[self.perm] == np.arange(len(self.names))))

    # floating-base (pinocchio xyzw quaternion) sagittal mirror, for completeness
    @staticmethod
    def mirror_base_q(q_base: np.ndarray) -> np.ndarray:
        x, y, z, qx, qy, qz, qw = q_base
        return np.array([x, -y, z, -qx, qy, -qz, qw])

    @staticmethod
    def mirror_base_v(v_base: np.ndarray) -> np.ndarray:
        vx, vy, vz, wx, wy, wz = v_base
        return np.array([vx, -vy, vz, -wx, wy, -wz])
