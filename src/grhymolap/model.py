"""
Core GRHyMoLAP model equations.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["percolation", "simulate_streamflow"]


@njit
def _percolation_core(Pn, En, X1):
    n = len(Pn)
    S = np.zeros(n)
    Perc = np.zeros(n)

    S[0] = X1 / 2.0
    ratio = (4.0 / 9.0) * (S[0] / X1)
    Perc[0] = S[0] * (1 - (1 + ratio**4) ** (-0.25))

    for i in range(1, n):
        temp = (S[i-1] / X1) ** 2

        frac = Pn[i] / X1
        Ps = X1 * (1 - temp) * np.tanh(frac) / (1 + (S[i-1] / X1) * np.tanh(frac))

        frac = En[i] / X1
        Es = S[i-1] * (2 - S[i-1]/X1) * np.tanh(frac) / (1 + (1 - S[i-1]/X1) * np.tanh(frac))

        S[i] = S[i-1] + Ps - Es

        ratio = (4.0 / 9.0) * (S[i] / X1)
        Perc[i] = S[i] * (1 - (1 + ratio**4) ** (-0.25))

        S[i] -= Perc[i]

    return Perc


def percolation(Pn, En, X1):
    Pn = np.asarray(Pn, dtype=np.float64)
    En = np.asarray(En, dtype=np.float64)
    return _percolation_core(Pn, En, float(X1))


@njit
def _simulate_core(MU, LAMBDA, X1, GAMMA, Q0, Pn, En):
    N = len(Pn)

    Q = np.zeros(N)
    Q[0] = Q0

    Perc = _percolation_core(Pn, En, X1)

    for t in range(N-1):
        Q[t+1] = max(
            0.0,
            Q[t] - (MU / LAMBDA) * (Q[t])**(2*MU - 1)
            + GAMMA * Perc[t+1] * Pn[t+1]
        )

    return Q


def simulate_streamflow(params, Q0, Pn, En):
    params = np.asarray(params, dtype=np.float64)
    Pn = np.asarray(Pn, dtype=np.float64)
    En = np.asarray(En, dtype=np.float64)

    MU, LAMBDA, X1, GAMMA = params

    return _simulate_core(MU, LAMBDA, X1, GAMMA, Q0, Pn, En)
