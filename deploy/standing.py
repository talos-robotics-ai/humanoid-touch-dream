"""Decides when the robot is standing correctly enough for the policy to start."""
from dataclasses import dataclass

from deploy.g1_spec import StandingSpec


@dataclass(frozen=True)
class StandingSample:
    walker_live: bool
    torso_height: float
    roll: float
    pitch: float
    gyro_norm: float
    leg_dq_max: float
    lowstate_age_s: float


@dataclass(frozen=True)
class StandingReport:
    standing: bool
    held_s: float
    failing: tuple


class StandingCheck:
    def __init__(self, spec: StandingSpec):
        self.spec = spec
        self._pass_since = None

    def reset(self) -> None:
        self._pass_since = None

    def update(self, sample: StandingSample, now: float) -> StandingReport:
        s = self.spec
        checks = (
            ("walker", not sample.walker_live),
            ("height", abs(sample.torso_height - s.stand_height) > s.height_tol),
            ("tilt", abs(sample.roll) > s.tilt_max or abs(sample.pitch) > s.tilt_max),
            ("gyro", sample.gyro_norm > s.gyro_max),
            ("legs", sample.leg_dq_max > s.leg_dq_max),
            ("stale", sample.lowstate_age_s > s.lowstate_max_age),
        )
        failing = tuple(name for name, bad in checks if bad)
        if failing:
            self._pass_since = None
            return StandingReport(False, 0.0, failing)
        if self._pass_since is None:
            self._pass_since = now
        held = now - self._pass_since
        return StandingReport(held >= s.hold_s, held, ())
