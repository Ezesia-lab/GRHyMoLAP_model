"""
Public interface: ``GRHyMoLAP().fit(P, PET, Q).simulate()``.
"""

from __future__ import annotations

import numpy as np

from .calibration import calibrate
from .metrics import OBJECTIVES
from .model import net_fluxes, simulate_streamflow
from .periods import resolve_periods

__all__ = ["GRHyMoLAP"]


def _score_all(obs, sim) -> dict:
    return {name: func(obs, sim) for name, (func, _) in OBJECTIVES.items()}


class GRHyMoLAP:
    """A single-basin GRHyMoLAP rainfall-runoff model.

    Examples
    --------
    >>> model = GRHyMoLAP(n_warmup=365)
    >>> model.fit(
    ...     P, PET, Q,
    ...     dates=dates,
    ...     calibration_period=("2000-01-01", "2010-12-31"),
    ...     validation_period=("2011-01-01", "2014-12-31"),
    ... )
    >>> model.calibration_scores_["nse"], model.validation_scores_["nse"]
    >>> Qsim_future = model.simulate(P_new, PET_new)

    Parameters
    ----------
    n_warmup : int, default 365
        Timesteps at the start of the calibration period that are
        simulated to allow the internal states to settle but excluded
        from the calibration objective and performance scores.

    objective : str, default "nse"
        Calibration objective — one of "nse", "kge", "lognse", "rmse",
        "mae", "pbias".

    optimizer : str, default "nelder-mead"
        Calibration optimizer — one of "nelder-mead", "l-bfgs-b",
        "differential_evolution".

    bounds, initial_guesses, custom_objective, custom_optimizer,
    optimizer_kwargs :
        Passed straight through to
        :func:`grhymolap.calibration.calibrate`; see its docstring
        for details, including signatures for the custom hooks.
    """

    def __init__(
        self,
        n_warmup: int = 365,
        objective: str = "nse",
        optimizer: str = "nelder-mead",
        bounds=None,
        initial_guesses=None,
        custom_objective=None,
        custom_optimizer=None,
        optimizer_kwargs=None,
    ):
        self.n_warmup = n_warmup
        self.objective = objective
        self.optimizer = optimizer
        self.bounds = bounds
        self.initial_guesses = initial_guesses
        self.custom_objective = custom_objective
        self.custom_optimizer = custom_optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def fit(
        self,
        P,
        PET,
        Q,
        dates=None,
        calibration_period: tuple | None = None,
        validation_period: tuple | None = None,
        Q0: float | None = None,
    ) -> "GRHyMoLAP":
        """Calibrate the model.

        ``calibration_period`` and ``validation_period`` define explicit
        date ranges for calibration and validation. The warm-up period
        is simulated within the calibration period but excluded from
        the calibration objective and performance scores.

        ``dates`` is required when either period is specified.
        """
        P, PET, Q = (np.asarray(a, dtype=float) for a in (P, PET, Q))
        n = len(Q)
        Pn, En = net_fluxes(P, PET)
        Q0_ = float(Q[0]) if Q0 is None else float(Q0)

        warmup_mask, calibration_mask, validation_mask = resolve_periods(
            n,
            dates=dates,
            n_warmup=self.n_warmup,
            calibration_period=calibration_period,
            validation_period=validation_period,
        )

        params, _ = calibrate(
            Q0_,
            Pn,
            En,
            Q,
            calibration_mask=calibration_mask,
            objective=self.objective,
            optimizer=self.optimizer,
            bounds=self.bounds,
            initial_guesses=self.initial_guesses,
            custom_objective=self.custom_objective,
            custom_optimizer=self.custom_optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
        )

        Qsim = simulate_streamflow(params, Q0_, Pn, En)

        self.params_ = np.asarray(params)
        self.warmup_mask_ = warmup_mask
        self.calibration_mask_ = calibration_mask
        self.validation_mask_ = validation_mask
        self.Q_sim_ = Qsim
        self.calibration_scores_ = _score_all(Q[calibration_mask], Qsim[calibration_mask])
        self.validation_scores_ = (
            _score_all(Q[validation_mask], Qsim[validation_mask])
            if validation_mask.any()
            else {}
        )

        self._Pn, self._En, self._Q0 = Pn, En, Q0_
        return self

    def simulate(
        self, P=None, PET=None, Q0: float | None = None
    ) -> np.ndarray:
        """Simulate streamflow; with no args, replays the fit() data.

        Pass new ``P``/``PET`` (e.g. a held-out period, or forcing
        from an independent period) to run the fitted parameters on
        different data.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("Call fit() before simulate().")

        if P is None and PET is None:
            Pn, En, Q0_ = self._Pn, self._En, self._Q0
        else:
            Pn, En = net_fluxes(P, PET)
            Q0_ = self._Q0 if Q0 is None else float(Q0)

        return simulate_streamflow(self.params_, Q0_, Pn, En)

    def score(self, P, PET, Q, metric: str = "nse") -> float:
        """Simulate on (P, PET) and score against observed Q."""
        Q = np.asarray(Q, dtype=float)
        Qsim = self.simulate(P, PET, Q0=Q[0])
        func, _ = OBJECTIVES[metric]
        return func(Q, Qsim)
