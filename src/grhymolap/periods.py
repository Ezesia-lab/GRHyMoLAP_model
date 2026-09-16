"""
Warm-up, calibration, and validation period handling.

Calibration and validation periods are specified explicitly using dates.
A fixed warm-up length is applied at the start of the calibration period.
The warm-up period is simulated to allow internal states to settle but
excluded from the calibration objective and performance scores.

Both return boolean masks over the full series so the model can always
be simulated continuously (state is never reset between periods).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["resolve_periods"]


def resolve_periods(
    n: int,
    dates=None,
    n_warmup: int = 0,
    calibration_period: tuple | None = None,
    validation_period: tuple | None = None,
):
    """Compute warm-up, calibration, and validation masks for a series.

    Parameters
    ----------
    n : int
        Length of the series.
    dates : array-like of datetimes
        Dates corresponding to the series. Required when
        calibration_period or validation_period is specified.
    n_warmup : int, default 0
        Leading timesteps of the calibration period that are simulated
        to allow internal states to settle but excluded from the
        calibration objective and performance scores.
    calibration_period : (start, end), optional
        Explicit calibration period, inclusive. Requires ``dates``.
    validation_period : (start, end), optional
        Explicit validation period, inclusive. Requires ``dates``.

    Returns
    -------
    warmup_mask, calibration_mask, validation_mask : np.ndarray[bool], length n
    """
    warmup_mask = np.zeros(n, dtype=bool)
    calibration_mask = np.zeros(n, dtype=bool)
    validation_mask = np.zeros(n, dtype=bool)

    if calibration_period is None and validation_period is None:
        raise ValueError(
            "At least one of `calibration_period` or `validation_period` "
            "must be specified."
        )

    if dates is None:
        raise ValueError(
            "`dates` is required to specify calibration_period or "
            "validation_period."
        )

    idx = pd.DatetimeIndex(dates)

    if len(idx) != n:
        raise ValueError(
            f"`dates` length ({len(idx)}) != series length ({n})."
        )

    if calibration_period is not None:
        c0 = pd.Timestamp(calibration_period[0])
        c1 = pd.Timestamp(calibration_period[1])
        calibration_mask = np.asarray((idx >= c0) & (idx <= c1))

        calibration_indices = np.flatnonzero(calibration_mask)

        if len(calibration_indices) == 0:
            raise ValueError(
                "The specified calibration_period does not overlap with "
                "the provided dates."
            )

        n_warmup_actual = min(n_warmup, len(calibration_indices))
        warmup_indices = calibration_indices[:n_warmup_actual]
        warmup_mask[warmup_indices] = True

    if validation_period is not None:
        v0 = pd.Timestamp(validation_period[0])
        v1 = pd.Timestamp(validation_period[1])
        validation_mask = np.asarray((idx >= v0) & (idx <= v1))

        if not validation_mask.any():
            raise ValueError(
                "The specified validation_period does not overlap with "
                "the provided dates."
            )

    # Warm-up is simulated as part of the calibration period but excluded
    # from the calibration objective and performance scores.
    calibration_mask &= ~warmup_mask

    return warmup_mask, calibration_mask, validation_mask
