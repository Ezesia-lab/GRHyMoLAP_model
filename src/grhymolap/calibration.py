"""
Calibration for the GRHyMoLAP model.

Design goals:
- pick an objective by name (default NSE) or supply your own
- pick an optimizer by name (default Nelder-Mead) or supply your own
- everything is a plain callable, so this composes with outside tooling
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .metrics import OBJECTIVES
from .model import simulate_streamflow

__all__ = [
    "calibrate",
    "DEFAULT_BOUNDS",
    "DEFAULT_INITIAL_GUESSES",
    "OPTIMIZERS",
]

DEFAULT_BOUNDS = [(0.5, 3.5), (0.01, 300.0), (0.0001, 5000.0), (0.0, 20)]

DEFAULT_INITIAL_GUESSES = [
    [1.0, 8, 150, 0.1],
    [0.6, 2, 400, 1],
    [1.4, 15, 300, 0.5],
    [1.0, 10, 1000, 0.3],
    [1.8, 5, 800, 0.5],
]

OPTIMIZERS = {
    "nelder-mead": {
        "scipy_method": "Nelder-Mead",
        "needs_bounds": True,
        "global": False,
    },
    "l-bfgs-b": {
        "scipy_method": "L-BFGS-B",
        "needs_bounds": True,
        "global": False,
    },
    "differential_evolution": {
        "global": True,
    },
}


def _build_objective(objective, Q0, Pn, En, Q_obs, calibration_mask):
    if objective not in OBJECTIVES:
        raise ValueError(
            f"Unknown objective '{objective}'. Options: {list(OBJECTIVES)}"
        )

    metric_func, direction = OBJECTIVES[objective]
    sign = 1.0 if direction == "min" else -1.0

    def _objective(params):
        Qsim = simulate_streamflow(params, Q0, Pn, En)
        value = metric_func(Q_obs[calibration_mask], Qsim[calibration_mask])

        if value is None or not np.isfinite(value):
            return 1e9

        return sign * value

    return _objective


def calibrate(
    Q0: float,
    Pn: np.ndarray,
    En: np.ndarray,
    Q_obs: np.ndarray,
    calibration_mask: np.ndarray,
    objective: str = "nse",
    optimizer: str = "nelder-mead",
    bounds: Sequence[tuple] | None = None,
    initial_guesses: Sequence[Sequence[float]] | None = None,
    custom_objective: Callable | None = None,
    custom_optimizer: Callable | None = None,
    optimizer_kwargs: dict | None = None,
):
    """Calibrate GRHyMoLAP parameters on selected timesteps."""

    bounds = list(bounds) if bounds is not None else DEFAULT_BOUNDS

    if initial_guesses is None:
        initial_guesses = DEFAULT_INITIAL_GUESSES

    initial_guesses = list(initial_guesses)

    if optimizer_kwargs is None:
        optimizer_kwargs = {"options": {"maxiter": 2500, "disp": False}}
    else:
        optimizer_kwargs = dict(optimizer_kwargs)

    if not np.any(calibration_mask):
        raise ValueError("calibration_mask has no True entries to calibrate on.")

    if custom_objective is not None:
        def obj_fn(params):
            return custom_objective(
                params,
                Q0,
                Pn,
                En,
                Q_obs,
                calibration_mask,
            )
    else:
        obj_fn = _build_objective(
            objective,
            Q0,
            Pn,
            En,
            Q_obs,
            calibration_mask,
        )

    if custom_optimizer is not None:
        return np.asarray(
            custom_optimizer(
                obj_fn,
                initial_guesses,
                bounds,
                optimizer_kwargs,
            )
        )

    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Options: {list(OPTIMIZERS)}"
        )

    spec = OPTIMIZERS[optimizer]

    if spec["global"]:
        res = differential_evolution(
            obj_fn,
            bounds=bounds,
            **optimizer_kwargs,
        )
        return res.x

    best_x, best_objective = None, float("inf")

    for guess in initial_guesses:
        res = minimize(
            obj_fn,
            guess,
            method=spec["scipy_method"],
            bounds=bounds if spec["needs_bounds"] else None,
            **optimizer_kwargs,
        )

        if res.fun < best_objective:
            best_objective, best_x = res.fun, res.x

    return best_x
