"""Workspace — where per-robot artifacts live.

    <root>/gait_libraries/<robot>/   stage-1 output (GaitCycle YAMLs)
    <root>/data/<robot>/             stage-2 output (references/, phi_walk.pt) + kintree.npz
    <root>/logs/<robot>/<exp>/       stage-3 runs (cfgs.pkl, model_*.pt)

``root`` defaults to the repository root (``$TALOS_WORKSPACE`` overrides it) so the
package can be installed anywhere while artifacts stay together.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]          # .../talos
REPO_DIR = PACKAGE_DIR.parent                              # .../talos_walk


@dataclass(frozen=True)
class Workspace:
    root: Path = Path(os.environ.get("TALOS_WORKSPACE", str(REPO_DIR)))

    def for_robot(self, name: str) -> "RobotPaths":
        return RobotPaths(self.root, name)


@dataclass(frozen=True)
class RobotPaths:
    root: Path
    robot: str

    @property
    def gait_dir(self) -> Path:
        return self.root / "gait_libraries" / self.robot

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / self.robot

    @property
    def reference_dataset_dir(self) -> Path:
        return self.data_dir / "references"

    @property
    def phi_ckpt(self) -> Path:
        return self.data_dir / "phi_walk.pt"

    @property
    def kintree(self) -> Path:
        return self.data_dir / "kintree.npz"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs" / self.robot

    def run_dir(self, exp_name: str) -> Path:
        return self.log_dir / exp_name

    def ensure(self) -> "RobotPaths":
        for d in (self.gait_dir, self.data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
