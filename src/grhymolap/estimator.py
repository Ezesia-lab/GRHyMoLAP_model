"""
Public, sklearn-style interface: ``GRHyMoLAP().fit(P, PET, Q).simulate()``.
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
    >>> model = GRHyMoLAP(n_warmup=0)
    >>> model.fit(P, PET, Q, dates=dates, train_ratio=0.6)
    >>> model.train_scores_["nse"], model.val_scores_["nse"]
    >>> Qsim_future = model.simulate(P_new, PET_new)

    Parameters
    ----------
    n_warmup : int, default 0
        Timesteps at the start of the record excluded from scoring.
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
        n_warmup: int = 0,
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
        train_ratio: float | None = 0.6,
        train_period: tuple | None = None,
        val_period: tuple | None = None,
        Q0: float | None = None,
    ) -> "GRHyMoLAP":
        """Calibrate the model.

        ``train_ratio`` (post-warmup ratio split) and
        ``train_period``/``val_period`` (explicit date ranges,
        requires ``dates``) are both supported — pass whichever fits
        your workflow. Defaults to ``train_ratio=0.6`` if neither is
        given.
        """
        P, PET, Q = (np.asarray(a, dtype=float) for a in (P, PET, Q))
        n = len(Q)
        Pn, En = net_fluxes(P, PET)
        Q0_ = float(Q[0]) if Q0 is None else float(Q0)

        warmup_mask, train_mask, val_mask = resolve_periods(
            n,
            dates=dates,
            n_warmup=self.n_warmup,
            train_ratio=train_ratio,
            train_period=train_period,
            val_period=val_period,
        )

        params, _ = calibrate(
            Q0_,
            Pn,
            En,
            Q,
            train_mask=train_mask,
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
        self.train_mask_ = train_mask
        self.val_mask_ = val_mask
        self.Q_sim_ = Qsim
        self.train_scores_ = _score_all(Q[train_mask], Qsim[train_mask])
        self.val_scores_ = (
            _score_all(Q[val_mask], Qsim[val_mask]) if val_mask.any() else {}
        )

        self._Pn, self._En, self._Q0 = Pn, En, Q0_
        return self

    def simulate(
        self, P=None, PET=None, Q0: float | None = None
    ) -> np.ndarray:
        """Simulate streamflow; with no args, replays the fit() data.

        Pass new ``P``/``PET`` (e.g. a held-out period, or forcing
        from an ML component) to run the fitted parameters on
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
