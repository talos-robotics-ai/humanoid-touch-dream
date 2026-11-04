"""Stage interfaces (abstract base classes).

    GaitGenerator      RobotSpec + request  -> GaitCycle            (stage 1)
    ReferenceSolver    nominal + AMO command -> whole-body config   (stage 2)
    FeasibilityFilter  reference cycle       -> accept / reject     (stage 1+2)
    Trainer            robot + run name      -> checkpoint dir      (stage 3)
    PipelineStage      the unit the Pipeline orchestrates

Each stage's concrete classes live in their own sub-package; this module has no heavy
imports so the interfaces can be used from either conda environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from talos.core.gait import GaitCycle
from talos.core.robot import RobotSpec


@dataclass
class StageResult:
    ok: bool
    artifacts: list[Path] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class GaitGenerator(ABC):
    """Produces one verified GaitCycle for a request (speed / gait mode)."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec

    @abstractmethod
    def generate(self, vx: float, **kw) -> GaitCycle | None:
        """Return the cycle, or None if the solve/verification failed."""


class ReferenceSolver(ABC):
    """Solves one whole-body reference configuration for an AMO command."""

    @abstractmethod
    def solve(self, *args, **kw):
        ...


class FeasibilityFilter(ABC):
    """Accepts or rejects a reference cycle on dynamic grounds."""

    @abstractmethod
    def check(self, *args, **kw):
        ...


class Trainer(ABC):
    """A stage-3 training run (AnyTracker, AnyAdapter)."""

    def __init__(self, spec: RobotSpec, exp_name: str):
        self.spec, self.exp_name = spec, exp_name

    @abstractmethod
    def build_env(self, num_envs: int, **kw):
        ...

    @abstractmethod
    def train(self, num_envs: int, max_iterations: int, **kw) -> Path:
        ...


class PipelineStage(ABC):
    name: str = "stage"
    conda_env: str = ""

    def __init__(self, spec: RobotSpec):
        self.spec = spec

    @abstractmethod
    def artifacts(self) -> list[Path]:
        """Files this stage produces (used to decide whether it already ran)."""

    @abstractmethod
    def run(self, **kw) -> StageResult:
        ...

    def is_done(self) -> bool:
        arts = self.artifacts()
        return bool(arts) and all(Path(a).exists() for a in arts)
