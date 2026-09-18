"""
Core GRHyMoLAP model equations.

Public functions take/return plain NumPy arrays. The sequential loops
are JIT-compiled with numba because they are re-run during calibration.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["net_fluxes", "percolation", "simulate_streamflow"]

PARAM_NAMES = ("MU", "LAMBDA", "X1", "gamma")


def net_fluxes(P, PET):
    """Split P, PET into net precipitation (Pn) and net PET (En), each >= 0."""
    P = np.asarray(P, dtype=float)
    PET = np.asarray(PET, dtype=float)
    Pn = np.maximum(0.0, P - PET)
    En = np.maximum(0.0, PET - P)
    return Pn, En


@njit(cache=True)
def _percolation_core(Pn, En, X1, S0):
    n = Pn.shape[0]
    S = np.zeros(n)
    Perc = np.zeros(n)

    S[0] = S0

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

    return Perc, S[n - 1]


def percolation(Pn, En, X1):
    """Percolation out of the production store at each timestep."""
    Pn = np.ascontiguousarray(Pn, dtype=np.float64)
    En = np.ascontiguousarray(En, dtype=np.float64)

    Perc, _ = _percolation_core(Pn, En, float(X1), float(X1) / 2.0)

    return Perc


@njit(cache=True)
def _simulate_core(MU, LAMBDA, X1, gamma, Q0, S0, Pn, En):
    n = Pn.shape[0]
    Q = np.zeros(n)

    Q[0] = Q0

    Perc, S_final = _percolation_core(Pn, En, X1, S0)

    for t in range(n - 1):
        Q[t + 1] = max(
            0.0,
            Q[t]
            - (MU / LAMBDA) * Q[t] ** (2 * MU - 1)
            + gamma * Perc[t + 1] * Pn[t + 1],
        )

    return Q, Q[n - 1], S_final


def simulate_streamflow(params, Q0, Pn, En, S0=None, return_state=False):
    """Simulate streamflow from Q0 and optionally return the final state."""
    MU, LAMBDA, X1, gamma = (float(p) for p in params)

    Pn = np.ascontiguousarray(Pn, dtype=np.float64)
    En = np.ascontiguousarray(En, dtype=np.float64)

    if S0 is None:
        S0 = X1 / 2.0

    Q, Q_final, S_final = _simulate_core(
        MU,
        LAMBDA,
        X1,
        gamma,
        float(Q0),
        float(S0),
        Pn,
        En,
    )

    if return_state:
        return Q, Q_final, S_final

    return Q
