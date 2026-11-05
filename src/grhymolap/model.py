
"""
Core GRHyMoLAP model equations.

The package provides only the mathematical computations of the model.
Calibration, objective functions, optimization, parameter bounds,
initial guesses, and warm-up procedures are defined by the user.

Public functions use NumPy arrays. Sequential computations are
JIT-compiled with Numba.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "net_fluxes",
    "percolation",
    "simulate_streamflow",
]


def net_fluxes(P, PET):
    """
    Compute net precipitation (Pn) and net evapotranspiration (En).

    Parameters
    ----------
    P : array_like
        Precipitation.
    PET : array_like
        Potential evapotranspiration.

    Returns
    -------
    Pn : numpy.ndarray
        Net precipitation.
    En : numpy.ndarray
        Net evapotranspiration.
    """

    P = np.asarray(P, dtype=np.float64)
    PET = np.asarray(PET, dtype=np.float64)

    if P.ndim != 1 or PET.ndim != 1:
        raise ValueError("P and PET must be one-dimensional arrays.")

    if len(P) != len(PET):
        raise ValueError("P and PET must have the same length.")

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
    Perc[0] = S[0] * (1.0 - (1.0 + ratio ** 4) ** (-0.25))

    for i in range(1, n):

        fill_ratio = (S[i - 1] / X1) ** 2

        Ps = (
            X1 * (1.0 - fill_ratio) * np.tanh(Pn[i] / X1)
            / (1.0 + (S[i - 1] / X1) * np.tanh(Pn[i] / X1))
        )

        Es = (
            S[i - 1] * (2.0 - S[i - 1] / X1)
            * np.tanh(En[i] / X1)
            / (1.0 + (1.0 - S[i - 1] / X1) * np.tanh(En[i] / X1))
        )

        S[i] = S[i - 1] + Ps - Es

        ratio = (4.0 / 9.0) * (S[i] / X1)
        Perc[i] = S[i] * (1.0 - (1.0 + ratio ** 4) ** (-0.25))

        S[i] = S[i] - Perc[i]

    return Perc


def percolation(Pn, En, X1, S0=None):
    """
    Compute percolation from the production store.

    Parameters
    ----------
    Pn : array_like
        Net precipitation.
    En : array_like
        Net evapotranspiration.
    X1 : float
        Production store capacity parameter.
    S0 : float, optional
        Initial production store. Default: X1 / 2.

    Returns
    -------
    Perc : numpy.ndarray
        Percolation at each timestep.
    """

    Pn = np.ascontiguousarray(Pn, dtype=np.float64)
    En = np.ascontiguousarray(En, dtype=np.float64)

    if Pn.ndim != 1 or En.ndim != 1:
        raise ValueError("Pn and En must be one-dimensional arrays.")

    if len(Pn) != len(En):
        raise ValueError("Pn and En must have the same length.")

    if len(Pn) == 0:
        raise ValueError("Pn and En must not be empty.")

    X1 = float(X1)

    if X1 <= 0:
        raise ValueError("X1 must be strictly positive.")

    if S0 is None:
        S0 = X1 / 2.0

    Perc = _percolation_core(
        Pn,
        En,
        X1,
        float(S0),
    )

    return Perc


@njit(cache=True)
def _simulate_core(MU, LAMBDA, X1, gamma, Q0, S0, Pn, En):

    n = Pn.shape[0]

    Q = np.zeros(n)

    Q[0] = Q0

    Perc = _percolation_core(Pn, En, X1, S0)

    for t in range(n - 1):

        Q[t + 1] = max(
            0.0,
            Q[t]
            - (MU / LAMBDA) * Q[t] ** (2.0 * MU - 1.0)
            + gamma * Perc[t + 1] * Pn[t + 1],
        )

    return Q


def simulate_streamflow(params, Q0, Pn, En, S0=None):
    """
    Simulate streamflow using the GRHyMoLAP model.

    Parameters
    ----------
    params : array_like
        Model parameters ordered as (MU, LAMBDA, X1, gamma).
    Q0 : float
        Initial streamflow state.
    Pn : array_like
        Net precipitation.
    En : array_like
        Net evapotranspiration.
    S0 : float, optional
        Initial production store. Default: X1 / 2.

    Returns
    -------
    Q : numpy.ndarray
        Simulated streamflow.
    """

    MU, LAMBDA, X1, gamma = (
        float(p) for p in params
    )

    Pn = np.ascontiguousarray(Pn, dtype=np.float64)
    En = np.ascontiguousarray(En, dtype=np.float64)

    if Pn.ndim != 1 or En.ndim != 1:
        raise ValueError("Pn and En must be one-dimensional arrays.")

    if len(Pn) != len(En):
        raise ValueError("Pn and En must have the same length.")

    if len(Pn) == 0:
        raise ValueError("Pn and En must not be empty.")

    if LAMBDA <= 0:
        raise ValueError("LAMBDA must be strictly positive.")

    if X1 <= 0:
        raise ValueError("X1 must be strictly positive.")
        
    if gamma < 0:
        raise ValueError("X1 must be positive.")
        
    if MU < 0.5:
        raise ValueError("MU must be >= 0.5.")

    if S0 is None:
        S0 = X1 / 2.0

    Q = _simulate_core(
        MU,
        LAMBDA,
        X1,
        gamma,
        float(Q0),
        float(S0),
        Pn,
        En,
    )

    return Q
