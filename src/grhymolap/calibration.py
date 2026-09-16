"""
GRHyMoLAP calibration using multi-start Nelder-Mead optimization.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .metrics import rmse
from .model import simulate_streamflow

__all__ = ["calibrate"]


DEFAULT_INITIAL_GUESSES = [
    [1.0, 8, 150, 0.1],
    [0.6, 2, 400, 1],
    [1.4, 15, 300, 0.5],
    [1.0, 10, 1000, 0.3],
    [1.8, 5, 800, 0.5],
]

DEFAULT_BOUNDS = [
    (0.5, 3.5),
    (0.01, 300.0),
    (0.0001, 5000.0),
    (0.0, 20),
]


def objective(params, Q0, Pn_train, En_train, Q_obs_train):
    Q_sim = simulate_streamflow(np.array(params), Q0, Pn_train, En_train)
    value = rmse(Q_obs_train, Q_sim)

    if np.isfinite(value):
        return value

    return 1e9


def calibrate(
    Q0,
    Pn,
    En,
    Q_obs,
    train_mask=None,
    initial_guesses=None,
    bounds=None,
    maxiter=2500,
):
    if train_mask is None:
        train_mask = np.ones(len(Q_obs), dtype=bool)

    Pn_train = Pn[train_mask]
    En_train = En[train_mask]
    Q_obs_train = Q_obs[train_mask]

    if initial_guesses is None:
        initial_guesses = DEFAULT_INITIAL_GUESSES

    if bounds is None:
        bounds = DEFAULT_BOUNDS

    best_res = None
    best_val = float("inf")

    for guess in initial_guesses:
        res = minimize(
            objective,
            guess,
            bounds=bounds,
            args=(Q0, Pn_train, En_train, Q_obs_train),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "disp": False},
        )

        if res.fun < best_val:
            best_val = res.fun
            best_res = res

    return best_res.x, best_res
