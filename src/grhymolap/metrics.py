"""
Goodness-of-fit metrics for GRHyMoLAP.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["nse", "rmse"]


@njit
def nse(obs, sim):
    n = len(obs)
    mean_obs = 0.0
    count = 0

    for i in range(n):
        if not np.isnan(obs[i]) and not np.isnan(sim[i]):
            mean_obs += obs[i]
            count += 1

    if count == 0:
        return np.nan

    mean_obs /= count

    num = 0.0
    den = 0.0

    for i in range(n):
        if not np.isnan(obs[i]) and not np.isnan(sim[i]):
            num += (sim[i] - obs[i]) ** 2
            den += (obs[i] - mean_obs) ** 2

    if den == 0:
        return np.nan

    return 1.0 - num / den


@njit
def rmse(obs, sim):
    n = len(obs)
    mse = 0.0
    count = 0

    for i in range(n):
        if not np.isnan(obs[i]) and not np.isnan(sim[i]):
            diff = sim[i] - obs[i]
            mse += diff * diff
            count += 1

    if count == 0:
        return np.nan

    return np.sqrt(mse / count)
