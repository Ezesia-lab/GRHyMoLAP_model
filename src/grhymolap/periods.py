"""
Warm-up / train / validation period handling.

Supports two ways of specifying train/validation splits, either of
which can be combined with a fixed warm-up length:

- ratio-based: ``train_ratio=0.7`` splits whatever comes after warm-up
- date-based: ``train_period=("1980-01-01", "2005-12-31")`` /
  ``val_period=("2006-01-01", "2014-12-31")``, resolved against a
  ``dates`` index.

Both return boolean masks over the full series so the model can always
be simulated continuously (state is never reset mid-series), while
scoring only looks at the relevant mask.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["resolve_periods"]


def resolve_periods(
    n: int,
    dates=None,
    n_warmup: int = 0,
    train_ratio: float | None = None,
    train_period: tuple | None = None,
    val_period: tuple | None = None,
):
    """Compute warm-up / train / validation masks for a series of length n.

    Parameters
    ----------
    n : int
        Length of the series.
    dates : array-like of datetimes, optional
        Required only if ``train_period``/``val_period`` are given.
    n_warmup : int, default 0
        Leading timesteps excluded from both scoring masks (the model
        still runs through them to let internal state settle).
    train_ratio : float, optional
        Fraction of the post-warmup series used for training, rest is
        validation. Ignored if ``train_period``/``val_period`` given.
    train_period, val_period : (start, end), optional
        Explicit date ranges (inclusive), resolved against ``dates``.

    Returns
    -------
    warmup_mask, train_mask, val_mask : np.ndarray[bool], length n
    """
    warmup_mask = np.zeros(n, dtype=bool)
    warmup_mask[: min(n_warmup, n)] = True

    if train_period is not None or val_period is not None:
        if dates is None:
            raise ValueError(
                "`dates` is required to use train_period/val_period."
            )
        idx = pd.DatetimeIndex(dates)
        if len(idx) != n:
            raise ValueError(
                f"`dates` length ({len(idx)}) != series length ({n})."
            )

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)

        if train_period is not None:
            t0 = pd.Timestamp(train_period[0])
            t1 = pd.Timestamp(train_period[1])
            train_mask = np.asarray((idx >= t0) & (idx <= t1))
        if val_period is not None:
            v0 = pd.Timestamp(val_period[0])
            v1 = pd.Timestamp(val_period[1])
            val_mask = np.asarray((idx >= v0) & (idx <= v1))
    else:
        ratio = 0.7 if train_ratio is None else train_ratio
        remaining = np.flatnonzero(~warmup_mask)
        split = int(len(remaining) * ratio)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        train_mask[remaining[:split]] = True
        val_mask[remaining[split:]] = True

    # Warm-up always wins, regardless of how train/val were specified.
    train_mask &= ~warmup_mask
    val_mask &= ~warmup_mask
    return warmup_mask, train_mask, val_mask
