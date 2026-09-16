"""
Public GRHyMoLAP interface.
"""

from __future__ import annotations

import numpy as np

from .calibration import calibrate
from .metrics import nse, rmse
from .model import percolation, simulate_streamflow
from .periods import resolve_periods

__all__ = ["GRHyMoLAP"]


class GRHyMoLAP:

    def __init__(
        self,
        initial_guesses=None,
        bounds=None,
        maxiter=2500,
    ):
        self.initial_guesses = initial_guesses
        self.bounds = bounds
        self.maxiter = maxiter

    def fit(self, P, PET, Q, train_ratio=0.6, Q0=None):
        P = np.asarray(P, dtype=np.float64)
        PET = np.asarray(PET, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        Pn = np.maximum(0.0, P - PET)
        En = np.maximum(0.0, PET - P)

        N = len(Q)

        if N == 0 or np.all(np.isnan(Q)):
            raise ValueError("No valid observed streamflow data.")

        if Q0 is None:
            Q0 = Q[0]

        train_mask, val_mask = resolve_periods(
            N,
            train_ratio=train_ratio,
        )

        params, result = calibrate(
            Q0,
            Pn,
            En,
            Q,
            train_mask=train_mask,
            initial_guesses=self.initial_guesses,
            bounds=self.bounds,
            maxiter=self.maxiter,
        )

        Qsim = simulate_streamflow(
            params,
            Q0,
            Pn,
            En,
        )

        self.params_ = np.asarray(params)
        self.Q_sim_ = Qsim
        self.Pn_ = Pn
        self.En_ = En
        self.Q0_ = Q0

        self.train_mask_ = train_mask
        self.val_mask_ = val_mask

        self.train_scores_ = {
            "nse": nse(Q[train_mask], Qsim[train_mask]),
            "rmse": rmse(Q[train_mask], Qsim[train_mask]),
        }

        self.val_scores_ = {
            "nse": nse(Q[val_mask], Qsim[val_mask]),
            "rmse": rmse(Q[val_mask], Qsim[val_mask]),
        }

        self.calibration_result_ = result

        self.P_ = P
        self.PET_ = PET
        self.Q_ = Q

        return self

    def simulate(self, P=None, PET=None, Q0=None):
        if not hasattr(self, "params_"):
            raise RuntimeError("Call fit() before simulate().")

        if P is None and PET is None:
            Pn = self.Pn_
            En = self.En_
            Q0_ = self.Q0_ if Q0 is None else float(Q0)
        else:
            P = np.asarray(P, dtype=np.float64)
            PET = np.asarray(PET, dtype=np.float64)

            Pn = np.maximum(0.0, P - PET)
            En = np.maximum(0.0, PET - P)

            Q0_ = self.Q0_ if Q0 is None else float(Q0)

        return simulate_streamflow(
            self.params_,
            Q0_,
            Pn,
            En,
        )
