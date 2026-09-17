"""
Public interface: ``GRHyMoLAP().fit(P, PET, Q, Q0).simulate()``.
"""

from __future__ import annotations

import numpy as np

from .calibration import calibrate
from .metrics import OBJECTIVES
from .model import net_fluxes, simulate_streamflow

__all__ = ["GRHyMoLAP"]


def _score_all(obs, sim) -> dict:
    return {name: func(obs, sim) for name, (func, _) in OBJECTIVES.items()}


class GRHyMoLAP:
    """A single-basin GRHyMoLAP rainfall-runoff model.

    Examples
    --------
    >>> model = GRHyMoLAP(n_warmup=365)
    >>> model.fit(P_cal, PET_cal, Q_cal, Q0=Q_cal[0])
    >>> model.calibration_scores_["nse"]
    >>> Qsim_val = model.simulate(P_val, PET_val, Q0=Q_val[0])
    >>> model.score(P_val, PET_val, Q_val, Q0=Q_val[0], metric="nse")

    Parameters
    ----------
    n_warmup : int, default 365
        Timesteps at the start of the calibration series that are
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

    def fit(self, P, PET, Q, Q0) -> "GRHyMoLAP":
        """Calibrate the model on an independent calibration series.

        Parameters
        ----------
        P, PET, Q : array-like
            Precipitation, potential evapotranspiration, and observed
            streamflow for the calibration series.
        Q0 : float
            Initial streamflow state used to start the simulation.

        Returns
        -------
        GRHyMoLAP
            Fitted model instance.
        """
        P, PET, Q = (np.asarray(a, dtype=float) for a in (P, PET, Q))

        if not (len(P) == len(PET) == len(Q)):
            raise ValueError("P, PET, and Q must have the same length.")

        if len(Q) == 0:
            raise ValueError("P, PET, and Q must not be empty.")

        if self.n_warmup < 0:
            raise ValueError("n_warmup must be non-negative.")

        n = len(Q)
        n_warmup = min(self.n_warmup, n)

        warmup_mask = np.zeros(n, dtype=bool)
        warmup_mask[:n_warmup] = True
        calibration_mask = ~warmup_mask

        if not np.any(calibration_mask):
            raise ValueError(
                "n_warmup must be smaller than the length of the calibration series."
            )

        Pn, En = net_fluxes(P, PET)
        Q0_ = float(Q0)

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
        self.Q_sim_ = Qsim
        self.calibration_scores_ = _score_all(
            Q[calibration_mask], Qsim[calibration_mask]
        )

        return self

    def simulate(self, P, PET, Q0) -> np.ndarray:
        """Simulate streamflow on an independent forcing series.

        Parameters
        ----------
        P, PET : array-like
            Precipitation and potential evapotranspiration.
        Q0 : float
            Initial streamflow state used to start the simulation.

        Returns
        -------
        np.ndarray
            Simulated streamflow.
        """
        if not hasattr(self, "params_"):
            raise RuntimeError("Call fit() before simulate().")

        Pn, En = net_fluxes(P, PET)
        return simulate_streamflow(self.params_, float(Q0), Pn, En)

    def score(self, P, PET, Q, Q0, metric: str = "nse") -> float:
        """Simulate and score an independent observed series.

        Parameters
        ----------
        P, PET, Q : array-like
            Precipitation, potential evapotranspiration, and observed
            streamflow for the independent series.
        Q0 : float
            Initial streamflow state used to start the simulation.
        metric : str, default "nse"
            Metric to compute. Must be one of ``OBJECTIVES``.

        Returns
        -------
        float
            Performance score.
        """
        Q = np.asarray(Q, dtype=float)
        Qsim = self.simulate(P, PET, Q0=Q0)
        func, _ = OBJECTIVES[metric]
        return func(Q, Qsim)
