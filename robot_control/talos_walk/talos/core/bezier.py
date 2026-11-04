"""Bernstein–Bezier evaluation and least-squares fitting (numpy).

The HZD virtual constraints are Bezier polynomials; the gait library stores degree-5
control points per joint for positions and velocities separately.
"""

from __future__ import annotations

import math

import numpy as np


def bernstein_matrix(taus: np.ndarray, degree: int) -> np.ndarray:
    """[len(taus), degree+1] Bernstein basis."""
    taus = np.asarray(taus, dtype=float)
    Phi = np.zeros((len(taus), degree + 1))
    for i in range(degree + 1):
        Phi[:, i] = math.comb(degree, i) * taus**i * (1.0 - taus) ** (degree - i)
    return Phi


def bezier_eval(tau: float | np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """Evaluate [J, d+1] control points at scalar tau -> [J] (or [..., J] for array tau)."""
    ctrl = np.asarray(ctrl, dtype=float)
    d = ctrl.shape[-1] - 1
    if np.ndim(tau) == 0:
        w = bernstein_matrix(np.array([float(tau)]), d)[0]
        return ctrl @ w
    W = bernstein_matrix(np.asarray(tau).ravel(), d)          # [M, d+1]
    return (W @ ctrl.T).reshape(*np.shape(tau), ctrl.shape[0])


def fit_bezier(values: np.ndarray, degree: int = 5, taus: np.ndarray | None = None,
               d0: float | None = None, d1: float | None = None) -> np.ndarray:
    """Least-squares control points for a [M] sampled curve; endpoints pinned exactly.

    ``d0``/``d1`` optionally pin the endpoint tau-derivatives (dB/dtau at 0 and 1):
    P1 and P_{d-1} then follow from d/dtau = degree*(P1 - P0). Pinning the endpoint
    slope matters whenever the true velocity is near zero at the boundary — a value-only
    fit leaks phantom endpoint velocity into the fitted curve.
    """
    values = np.asarray(values, dtype=float)
    M = values.shape[0]
    taus = np.linspace(0, 1, M) if taus is None else np.asarray(taus)
    Phi = bernstein_matrix(taus, degree)
    if d0 is None and d1 is None:
        P, *_ = np.linalg.lstsq(Phi, values, rcond=None)
        P[0], P[-1] = values[0], values[-1]
        return P
    fixed = {0: values[0], degree: values[-1]}
    if d0 is not None:
        fixed[1] = values[0] + d0 / degree
    if d1 is not None:
        fixed[degree - 1] = values[-1] - d1 / degree
    free = [i for i in range(degree + 1) if i not in fixed]
    rhs = values - sum(Phi[:, i] * v for i, v in fixed.items())
    x, *_ = np.linalg.lstsq(Phi[:, free], rhs, rcond=None)
    P = np.zeros(degree + 1)
    for i, v in fixed.items():
        P[i] = v
    P[free] = x
    return P
