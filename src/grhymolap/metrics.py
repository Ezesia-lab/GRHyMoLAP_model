"""
Goodness-of-fit metrics for streamflow simulation.

Every function has the signature ``metric(obs, sim) -> float`` and
drops NaN pairs before computing, so callers don't need to pre-clean
their arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "nse", "nnse", "rmse", "mae", "kge", "lognse", "pbias", "OBJECTIVES",
]


def _clean(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return pd.DataFrame({"obs": obs, "sim": sim}).dropna()


def nse(obs, sim) -> float:
    """Nash-Sutcliffe Efficiency. 1 = perfect, higher is better."""
    df = _clean(obs, sim)
    if df.empty or df["obs"].var() == 0:
        return np.nan
    resid = (df["sim"] - df["obs"]) ** 2
    total = (df["obs"] - df["obs"].mean()) ** 2
    return 1 - np.sum(resid) / np.sum(total)


def nnse(obs, sim=None) -> float:
    """Normalized NSE in (0, 1]. Takes (obs, sim), or a raw NSE value."""
    nse_val = obs if sim is None else nse(obs, sim)
    if nse_val is None or np.isnan(nse_val):
        return np.nan
    return 1.0 / (2.0 - nse_val)


def rmse(obs, sim) -> float:
    """Root mean squared error. Lower is better."""
    df = _clean(obs, sim)
    if df.empty:
        return np.nan
    return float(np.sqrt(np.mean((df["sim"] - df["obs"]) ** 2)))


def mae(obs, sim) -> float:
    """Mean absolute error. Lower is better."""
    df = _clean(obs, sim)
    if df.empty:
        return np.nan
    return float(np.mean(np.abs(df["sim"] - df["obs"])))


def kge(obs, sim) -> float:
    """Kling-Gupta Efficiency (Gupta et al., 2009). 1 = perfect."""
    df = _clean(obs, sim)
    if df.empty or df["obs"].std() == 0 or df["sim"].std() == 0:
        return np.nan
    if df["obs"].mean() == 0:
        return np.nan
    r = df["obs"].corr(df["sim"])
    alpha = df["sim"].std() / df["obs"].std()
    beta = df["sim"].mean() / df["obs"].mean()
    err = (r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2
    return float(1 - np.sqrt(err))


def lognse(obs, sim, epsilon: float | None = None) -> float:
    """NSE on log-transformed flows — emphasizes low-flow performance."""
    df = _clean(obs, sim)
    if df.empty:
        return np.nan
    default_eps = 0.01 * max(df["obs"].mean(), 1e-6)
    eps = default_eps if epsilon is None else epsilon
    return nse(np.log(df["obs"] + eps), np.log(df["sim"] + eps))


def pbias(obs, sim) -> float:
    """Percent bias. 0 = no bias; positive = model overestimates."""
    df = _clean(obs, sim)
    if df.empty or df["obs"].sum() == 0:
        return np.nan
    return float(100 * (df["sim"].sum() - df["obs"].sum()) / df["obs"].sum())


def _abs_pbias(obs, sim) -> float:
    val = pbias(obs, sim)
    return np.nan if np.isnan(val) else abs(val)


# name -> (metric_func, "max" or "min"): the direction calibration should
# optimize toward, and what GRHyMoLAP.fit()/.score() report.
OBJECTIVES = {
    "nse": (nse, "max"),
    "kge": (kge, "max"),
    "lognse": (lognse, "max"),
    "rmse": (rmse, "min"),
    "mae": (mae, "min"),
    "pbias": (_abs_pbias, "min"),
}
