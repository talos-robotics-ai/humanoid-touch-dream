"""RobotSpec — the ONE place hardware facts live.

A robot is described by a handful of small, composable value objects:

    Assets           URDF (Pinocchio/CasADi, stages 1-2) + MJCF (Genesis, stage 3)
    JointLayout      canonical policy joint order  [legs(12) | waist | arms(2xk) | aux...]
    FootGeometry     sole-carrying links, sole centre/extents, stance width
    BodyGeometry     root/torso/knee links, standing + walking pelvis height, mass
    ControlTables    per-joint defaults, PD gains, torque limits, CLF weights
    GaitGenSettings  stage-1 knobs (speeds, half-period, seed retarget, NLP bounds)
    CommandEnvelope  stage-2/3 AMO command ranges (torso rpy/height, arm amplitudes)
    TrackingLayout   rigid-body targets for the AnyTracker layer

Every stage reads robot facts from the spec and ONLY from the spec. Joint tables are
keyed by ``joint_key(name)`` ('left_hip_pitch_joint' -> 'hip_pitch'); a full joint name
in a table overrides its key, so per-side asymmetries stay expressible.

Adding a robot = one folder ``talos/robots/<name>/`` with ``assets/`` and a
``build_spec() -> RobotSpec``. No pipeline code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def joint_key(joint_name: str) -> str:
    """'left_hip_pitch_joint' -> 'hip_pitch' (side- and suffix-stripped table key)."""
    return joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")


def regex_lookup(patterns: dict[str, float], names: Iterable[str]) -> list[float]:
    """Resolve a {regex: value} table over joint names (fullmatch, last pattern wins)."""
    names = list(names)
    vals = [0.0] * len(names)
    for pattern, val in patterns.items():
        for i, name in enumerate(names):
            if re.fullmatch(pattern, name):
                vals[i] = val
    return vals


class JointTable(dict):
    """{joint_key or full joint name: value}. ``resolve(names)`` -> list in that order."""

    def resolve(self, names: Iterable[str]) -> list:
        out = []
        for n in names:
            if n in self:
                out.append(self[n])
            elif joint_key(n) in self:
                out.append(self[joint_key(n)])
            else:
                raise KeyError(f"no table entry for joint '{n}' (key '{joint_key(n)}')")
        return out


# ----------------------------------------------------------------------------- assets
@dataclass(frozen=True)
class Assets:
    urdf: Path          # Pinocchio / CasADi model (stages 1-2)
    mjcf: Path          # Genesis / MuJoCo model (stage 3, kintree FK)

    def check(self) -> None:
        for p in (self.urdf, self.mjcf):
            if not Path(p).exists():
                raise FileNotFoundError(f"robot asset missing: {p}")


# ------------------------------------------------------------------------ joint layout
@dataclass(frozen=True)
class JointLayout:
    """Canonical policy joint order and its body-part partition.

    ``names[:n_lower]``              lower body (legs + waist): the AMO residual acts here
    ``names[n_lower:n_lower+n_arms]`` arm command joints (tele-op)
    ``names[n_lower+n_arms:]``       aux joints (e.g. head): actuated, CLF-held, no command
    """
    names: tuple[str, ...]
    n_lower: int
    n_arms: int

    def __post_init__(self):
        if len(set(self.names)) != len(self.names):
            raise ValueError("duplicate joint names in layout")
        if self.n_lower + self.n_arms > len(self.names):
            raise ValueError("n_lower + n_arms exceeds the number of joints")

    @property
    def n_joints(self) -> int:
        return len(self.names)

    @property
    def n_aux(self) -> int:
        return self.n_joints - self.n_lower - self.n_arms

    @property
    def lower(self) -> tuple[str, ...]:
        return self.names[: self.n_lower]

    @property
    def arms(self) -> tuple[str, ...]:
        return self.names[self.n_lower: self.n_lower + self.n_arms]

    @property
    def aux(self) -> tuple[str, ...]:
        return self.names[self.n_lower + self.n_arms:]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def indices(self, names: Iterable[str]) -> list[int]:
        return [self.names.index(n) for n in names]

    def select(self, pattern: str) -> list[int]:
        """Indices of joints whose name fullmatches ``pattern``."""
        return [i for i, n in enumerate(self.names) if re.fullmatch(pattern, n)]

    def mirror(self):
        """MirrorMap over this layout (left<->right relabel with roll/yaw sign flips)."""
        from talos.core.mirror import MirrorMap
        return MirrorMap.from_names(self.names)


# ------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class FootGeometry:
    """Sole geometry in the FOOT LINK frame (the link the spec pins to the ground).

    ``sole_center`` is the sole-plane centre expressed in the foot-link frame (its z is
    minus the sole offset); ``half_extents`` the (x, y) support-polygon half sizes about
    that centre. ``stance_half_width`` is |y| of the foot-link origin at neutral stance.
    """
    links: tuple[str, str]                  # (left, right)
    sole_center: tuple[float, float, float]
    half_extents: tuple[float, float]
    stance_half_width: float

    @property
    def sole_offset(self) -> float:
        """Height of the foot-link origin above the sole plane."""
        return -self.sole_center[2]

    def cop_bounds_x(self) -> tuple[float, float]:
        cx, hx = self.sole_center[0], self.half_extents[0]
        return cx - hx, cx + hx

    def cop_bounds_y(self) -> tuple[float, float]:
        cy, hy = self.sole_center[1], self.half_extents[1]
        return cy - hy, cy + hy

    def link(self, side: str) -> str:
        return self.links[0] if side == "left" else self.links[1]


@dataclass(frozen=True)
class BodyGeometry:
    root_link: str
    torso_link: str
    knee_links: tuple[str, str]
    standing_height: float      # pelvis height, straight legs, soles on the ground (m)
    walk_height: float          # pelvis height at the default bent-knee pose (m)
    total_mass: float           # kg

    @property
    def termination_height(self) -> float:
        return round(0.63 * self.standing_height, 3)

    @property
    def impact_force_threshold(self) -> float:
        """Foot contact force above ~1.4x body weight counts as a slam (N)."""
        return round(1.4 * self.total_mass * 9.81, 0)


# ------------------------------------------------------------------------ control
@dataclass(frozen=True)
class ControlTables:
    default_angles: JointTable
    kp: JointTable
    kd: JointTable
    torque_limits: JointTable
    clf_qpv: JointTable         # (pos, vel) CLF weights per joint
    clf_r: JointTable           # CLF input weight per joint

    def validate(self, names: Iterable[str]) -> None:
        names = list(names)
        for tbl in (self.default_angles, self.kp, self.kd, self.torque_limits,
                    self.clf_qpv, self.clf_r):
            tbl.resolve(names)


# ------------------------------------------------------------------------ stage knobs
@dataclass(frozen=True)
class GaitGenSettings:
    speeds: tuple[float, ...]                   # library walking speeds (m/s)
    half_period: float                          # single-support duration T (s)
    seed_dir: Path                              # name-matched warm-start gaits
    seed_scale: dict[str, float] = field(default_factory=dict)   # per-joint seed sign/scale
    effort_overrides: dict[str, float] = field(default_factory=dict)
    height_bounds: tuple[float, float] = (0.4, 1.2)
    clearance: float = 0.06                     # swing apex clearance (m)
    # NLP strategy: "seed" pins all seed joints (always converges; speed = realised),
    # "cascade" = seed solve then a legs-free re-solve with exact step length.
    mode: str = "cascade"
    seed_source: str = "library"                # "library" | "ik" (see SeedRetargeter)
    march_lift: float = 0.06                    # in-place march apex (m)
    nlp_nodes: int = 16
    nlp_reg: float = 0.3


@dataclass(frozen=True)
class CommandEnvelope:
    roll: tuple[float, float] = (-0.30, 0.30)
    pitch: tuple[float, float] = (-0.30, 0.45)
    yaw: tuple[float, float] = (-0.60, 0.60)
    height: tuple[float, float] = (-0.10, 0.05)
    stand_roll: tuple[float, float] = (-0.40, 0.40)
    stand_pitch: tuple[float, float] = (-0.40, 0.70)
    stand_yaw: tuple[float, float] = (-0.60, 0.60)
    stand_height: tuple[float, float] = (-0.22, 0.06)
    arm_amplitude: dict[str, float] = field(default_factory=dict)   # {regex: rad}
    # Arm command range. False (legacy): uniform in +-arm_amplitude about 0, clipped to the
    # joint limits -- on joints whose travel is one-sided (an elbow is [-2.36, 0]) half the
    # samples pile up at the stop and most of the travel is never commanded. True: uniform
    # over each joint's own travel, inset by arm_range_margin of that travel at both ends
    # (0.05 = the dof_pos_limits soft band, so a command never sits in the penalty zone).
    arm_full_range: bool = False
    arm_range_margin: float = 0.05
    # {regex: (lower, upper)} tightens a joint's usable travel BEFORE the margin, for
    # travel the model allows but the robot cannot use (no self-collision in training:
    # a G1 shoulder may roll 1.59 rad inward, straight through the torso).
    arm_range_limit: dict[str, tuple[float, float]] = field(default_factory=dict)

    def arm_bounds(self, lo, hi):
        """Per-joint (lower, upper) the arm command is drawn from, given the joint limits."""
        import numpy as np
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        pad = self.arm_range_margin * (hi - lo)
        return lo + pad, hi - pad

    def walk_ranges(self) -> dict[str, tuple[float, float]]:
        return dict(torso_roll=self.roll, torso_pitch=self.pitch, torso_yaw=self.yaw,
                    base_height=self.height)

    def stand_ranges(self) -> dict[str, tuple[float, float]]:
        return dict(torso_roll=self.stand_roll, torso_pitch=self.stand_pitch,
                    torso_yaw=self.stand_yaw, base_height=self.stand_height)


@dataclass(frozen=True)
class TrackingLayout:
    """Rigid-body targets for the AnyTracker layer.

    Conventional order: 0 root, 1 torso, 2-4 left leg (hip, knee, foot),
    5-7 right leg, 8-10 left arm (shoulder, elbow, wrist), 11-13 right arm.
    """
    links: tuple[str, ...]
    feet: tuple[int, ...]
    lower: tuple[int, ...]
    upper: tuple[int, ...]


# ------------------------------------------------------------------------ the spec
@dataclass(frozen=True)
class RobotSpec:
    name: str
    assets: Assets
    joints: JointLayout
    feet: FootGeometry
    body: BodyGeometry
    control: ControlTables
    gaitgen: GaitGenSettings
    envelope: CommandEnvelope
    tracking: TrackingLayout
    description: str = ""

    # ------------------------------------------------------------- validation
    def validate(self) -> "RobotSpec":
        self.control.validate(self.joints.names)
        if self.feet.links[0] == self.feet.links[1]:
            raise ValueError("foot links must differ")
        for idx in (self.tracking.feet, self.tracking.lower, self.tracking.upper):
            if max(idx) >= len(self.tracking.links):
                raise ValueError("tracking group index out of range")
        return self

    # -------------------------------------------------------------- shortcuts
    @property
    def joint_names(self) -> list[str]:
        return list(self.joints.names)

    @property
    def n_joints(self) -> int:
        return self.joints.n_joints

    @property
    def n_lower(self) -> int:
        return self.joints.n_lower

    @property
    def n_arms(self) -> int:
        return self.joints.n_arms

    @property
    def urdf(self) -> Path:
        return self.assets.urdf

    @property
    def mjcf(self) -> Path:
        return self.assets.mjcf

    def default_angles(self) -> list[float]:
        return self.control.default_angles.resolve(self.joints.names)

    def per_joint(self, table: JointTable) -> list:
        return table.resolve(self.joints.names)

    def arm_amplitudes(self) -> list[float]:
        return regex_lookup(self.envelope.arm_amplitude, self.joints.arms)

    def arm_command_bounds(self) -> tuple[list[float], list[float]]:
        """Per-arm-joint (lower, upper) of the full-range arm command: URDF joint limits,
        tightened by ``envelope.arm_range_limit``, inset by ``arm_range_margin``. The single
        source for the phi_walk dataset, the RL env and the tele-op panel."""
        import xml.etree.ElementTree as ET
        lim = {j.get("name"): j.find("limit") for j in ET.parse(str(self.urdf)).getroot().findall("joint")}
        lo, hi = [], []
        for n in self.joints.arms:
            l, h = float(lim[n].get("lower")), float(lim[n].get("upper"))
            for pattern, (pl, ph) in self.envelope.arm_range_limit.items():
                if re.fullmatch(pattern, n):
                    l, h = max(l, pl), min(h, ph)
            pad = self.envelope.arm_range_margin * (h - l)
            lo.append(l + pad); hi.append(h - pad)
        return lo, hi

    @property
    def base_height_target(self) -> float:
        return round(self.body.walk_height, 3)
