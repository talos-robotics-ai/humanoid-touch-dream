"""PhiWalk — the walking-AMO residual MLP, and its supervised trainer.

    phi_walk(q_upper, rpy, h, phase, v) -> Δq_lower

One network is the whole reference generator at deploy time:
    q_lower_ref(phase) = q_lower_HZD(v, phase) + phi_walk(q_upper, rpy, h, phase, v)

The module is torch-only (no pinocchio) so the SAME class loads the checkpoint inside the
RL environment (``talos.rl.amo_phi``) and in the offline trainer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PHI_HIDDEN = (256, 256, 128)


def phi_dims(spec) -> tuple[int, int]:
    """(input_dim, output_dim) = (n_arms + 7, n_lower) — torch-only, usable in the RL env."""
    return spec.n_arms + 3 + 1 + 2 + 1, spec.n_lower


class PhiWalk(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=PHI_HIDDEN, activation: str = "elu"):
        super().__init__()
        act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(input_dim))
        self.register_buffer("input_std", torch.ones(input_dim))

    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8))

    def update_normalizer(self, data: torch.Tensor):
        self.input_mean.copy_(data.mean(0))
        self.input_std.copy_(data.std(0).clamp(min=1e-4))

    @staticmethod
    def features(q_upper, rpy, h, phase, v, stand=None) -> torch.Tensor:
        """[B, n_upper+7] feature tensor, identical layout to the dataset.

        ``stand`` (bool [B], v2 datasets only -- see ``stand_zero_phase``): double-support
        STANDING rows are encoded with the phase vector (sin, cos) = (0, 0), a point no
        walking phase can produce, so "standing" and "marching in place at v = 0" are
        different inputs instead of one input with two contradictory targets.
        """
        arg = 2.0 * math.pi * phase
        s, c = torch.sin(arg).reshape(-1, 1), torch.cos(arg).reshape(-1, 1)
        if stand is not None:
            keep = (~stand).to(s.dtype).reshape(-1, 1)
            s, c = s * keep, c * keep
        return torch.cat([q_upper, rpy, h.reshape(-1, 1), s, c, v.reshape(-1, 1)], dim=-1)

    @torch.no_grad()
    def residual(self, q_upper, rpy, h, phase, v, stand=None) -> torch.Tensor:
        return self.forward(self.features(q_upper, rpy, h, phase, v, stand))

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, input_dim: int, output_dim: int, device="cpu", frozen: bool = True) -> "PhiWalk":
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        m = cls(input_dim, output_dim)
        m.load_state_dict(sd)
        m.eval().to(device)
        if frozen:
            for p in m.parameters():
                p.requires_grad_(False)
        return m


def phi_meta_path(ckpt) -> Path:
    return Path(str(ckpt) + ".meta.json")


def stand_zero_phase(ckpt) -> bool:
    """True if this checkpoint was trained on a v2 dataset (zero-phase standing rows)."""
    import json
    p = phi_meta_path(ckpt)
    return p.exists() and json.loads(p.read_text()).get("stand_encoding") == "zero_phase"


class PhiTrainer:
    def __init__(self, input_dim: int, output_dim: int, lr: float = 3e-4, batch_size: int = 1024,
                 epochs: int = 120, val_split: float = 0.1, device: str = "cpu", seed: int = 0):
        self.in_dim, self.out_dim = input_dim, output_dim
        self.lr, self.bs, self.epochs, self.val_split = lr, batch_size, epochs, val_split
        self.device, self.seed = torch.device(device), seed

    def fit(self, data_dir: Path, ckpt: Path, verbose: bool = True) -> dict:
        from torch.utils.data import DataLoader, TensorDataset, random_split
        torch.manual_seed(self.seed)
        d = Path(data_dir)
        X = torch.from_numpy(np.load(d / "commands.npy").astype(np.float32))
        Y = torch.from_numpy(np.load(d / "references.npy").astype(np.float32))
        if X.shape[1] != self.in_dim or Y.shape[1] != self.out_dim:
            raise ValueError(f"dataset dims {tuple(X.shape[1:])},{tuple(Y.shape[1:])} != "
                             f"phi dims ({self.in_dim}, {self.out_dim})")
        ds = TensorDataset(X, Y)
        n_val = max(1, int(len(ds) * self.val_split)); n_train = len(ds) - n_val
        tr, va = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed))
        bs = min(self.bs, max(1, n_train))
        tl = DataLoader(tr, batch_size=bs, shuffle=True); vl = DataLoader(va, batch_size=bs)
        model = PhiWalk(self.in_dim, self.out_dim)
        model.update_normalizer(X)
        model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.MSELoss()
        best, best_state = float("inf"), None
        for ep in range(1, self.epochs + 1):
            model.train(); t_loss = 0.0
            for xb, yb in tl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                t_loss += loss.item() * len(xb)
            model.eval(); v_loss = 0.0
            with torch.no_grad():
                for xb, yb in vl:
                    v_loss += crit(model(xb.to(self.device)), yb.to(self.device)).item() * len(xb)
            t_loss /= n_train; v_loss /= n_val
            sched.step()
            if v_loss < best:
                best, best_state = v_loss, {k: v.clone() for k, v in model.state_dict().items()}
            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"[phi] epoch {ep:>4} train {t_loss:.6f} val {v_loss:.6f} "
                      f"({math.degrees(math.sqrt(v_loss)):.3f} deg RMSE)", flush=True)
        model.load_state_dict(best_state)
        model.save(ckpt)
        rmse_deg = math.degrees(math.sqrt(best))
        if verbose:
            print(f"[phi] best val RMSE {rmse_deg:.3f} deg -> {ckpt}")
        return dict(val_mse=best, val_rmse_deg=rmse_deg, samples=len(ds), ckpt=str(ckpt))
