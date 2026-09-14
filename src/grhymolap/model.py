"""
Core GRHyMoLAP model equations.

Public functions take/return plain NumPy arrays (no I/O, no hidden state),
so they're easy to test and easy to call from a calibration loop. The
sequential loops (state depends on the previous timestep, so they can't be
vectorized) are JIT-compiled with numba, since they're re-run on every
optimizer iteration during calibration.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["net_fluxes", "percolation", "simulate_streamflow"]

#: Parameter order used everywhere params are passed as a vector.
PARAM_NAMES = ("MU", "LAMBDA", "X1", "gamma")


def net_fluxes(P, PET):
    """Split P, PET into net precipitation (Pn) and net PET (En), each >= 0."""
    P = np.asarray(P, dtype=float)
    PET = np.asarray(PET, dtype=float)
    Pn = np.maximum(0.0, P - PET)
    En = np.maximum(0.0, PET - P)
    return Pn, En


@njit(cache=True)
def _percolation_core(Pn, En, X1):
    # Production-store soil moisture accounting (GR-family). State S depends
    # on its own previous value, hence the explicit loop.
    n = Pn.shape[0]
    S = np.zeros(n)
    Perc = np.zeros(n)

    S[0] = X1 / 2.0
    ratio = (4.0 / 9.0) * (S[0] / X1)
    Perc[0] = S[0] * (1 - (1 + ratio ** 4) ** (-0.25))

    for i in range(1, n):
        fill_ratio = (S[i - 1] / X1) ** 2
        Ps = (
            X1 * (1 - fill_ratio) * np.tanh(Pn[i] / X1)
            / (1 + (S[i - 1] / X1) * np.tanh(Pn[i] / X1))
        )
        Es = (
            S[i - 1] * (2 - S[i - 1] / X1) * np.tanh(En[i] / X1)
            / (1 + (1 - S[i - 1] / X1) * np.tanh(En[i] / X1))
        )
        S[i] = S[i - 1] + Ps - Es
        ratio = (4.0 / 9.0) * (S[i] / X1)
        Perc[i] = S[i] * (1 - (1 + ratio ** 4) ** (-0.25))
        S[i] = S[i] - Perc[i]

    return Perc


def percolation(Pn, En, X1):
    """Percolation out of the production store at each timestep."""
    Pn = np.ascontiguousarray(Pn, dtype=np.float64)
    En = np.ascontiguousarray(En, dtype=np.float64)
    return _percolation_core(Pn, En, float(X1))


@njit(cache=True)
def _simulate_core(MU, LAMBDA, X1, gamma, Q0, Pn, En):
    # Recession + Runoff generation, one timestep at a time.
    n = Pn.shape[0]
    Q = np.zeros(n)
    Q[0] = Q0
    Perc = _percolation_core(Pn, En, X1)

    for t in range(n - 1):
        Q[t + 1] = max(
            0.0,
            Q[t] - (MU / LAMBDA) * Q[t] ** (2 * MU - 1)
            + gamma * Perc[t + 1] * Pn[t + 1],
        )
    return Q


def simulate_streamflow(params, Q0, Pn, En):
    """Simulate streamflow from Q0 given params = (MU, LAMBDA, X1, gamma)."""
    MU, LAMBDA, X1, gamma = (float(p) for p in params)
    Pn = np.ascontiguousarray(Pn, dtype=np.float64)
    En = np.ascontiguousarray(En, dtype=np.float64)
    return _simulate_core(MU, LAMBDA, X1, gamma, float(Q0), Pn, En)
