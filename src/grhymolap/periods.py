"""
Calibration / validation period handling.
"""

from __future__ import annotations

import numpy as np

__all__ = ["resolve_periods"]


def resolve_periods(n, train_ratio=0.6):
    b1 = int(n * train_ratio)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)

    train_mask[:b1] = True
    val_mask[b1:] = True

    return train_mask, val_mask
