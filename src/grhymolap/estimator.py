"""
Public interface: ``GRHyMoLAP().fit(P, PET, Q).simulate()``.
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
    >>> model = GRHyMoLAP()
    >>> model.fit(P, PET, Q)
    >>> model.calibration_scores_
    >>> Qsim = model.simulate(P_next, PET_next)

    Parameters
    ----------
    objective : str, default "nse"
        Calibration objective — one of "nse", "kge", "lognse", "rmse",
        "mae", "pbias".

    optimizer : str, default "nelder-mead"
        Calibration optimizer — one of "nelder-mead", "l-bfgs-b",
        "differential_evolution".
    """

    def __init__(
        self,
        objective: str = "nse",
        optimizer: str = "nelder-mead",
        bounds=None,
        initial_guesses=None,
        custom_objective=None,
        custom_optimizer=None,
        optimizer_kwargs=None,
    ):
        self.objective = objective
        self.optimizer = optimizer
        self.bounds = bounds
        self.initial_guesses = initial_guesses
        self.custom_objective = custom_objective
        self.custom_optimizer = custom_optimizer
        self.optimizer_kwargs = optimizer_kwargs

    def fit(self, P, PET, Q) -> "GRHyMoLAP":
        """Calibrate the model on a supplied series."""

        P, PET, Q = (np.asarray(a, dtype=float) for a in (P, PET, Q))

        if not (len(P) == len(PET) == len(Q)):
            raise ValueError("P, PET, and Q must have the same length.")

        if len(Q) == 0:
            raise ValueError("P, PET, and Q must not be empty.")

        Q0 = float(Q[0])
        calibration_mask = np.ones(len(Q), dtype=bool)

        Pn, En = net_fluxes(P, PET)

        params = calibrate(
            Q0,
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

        Qsim, Q_final, S_final = simulate_streamflow(
            params,
            Q0,
            Pn,
            En,
            return_state=True,
        )

        self.params_ = np.asarray(params)
        self.calibration_mask_ = calibration_mask
        self.Q_sim_ = Qsim
        self.calibration_scores_ = _score_all(Q, Qsim)

        self.Q_state_ = Q_final
        self.S_state_ = S_final

        return self

    def simulate(self, P, PET) -> np.ndarray:
        """Continue the simulation from the current model state."""

        if not hasattr(self, "params_"):
            raise RuntimeError("Call fit() before simulate().")

        Pn, En = net_fluxes(P, PET)

        Qsim, Q_final, S_final = simulate_streamflow(
            self.params_,
            self.Q_state_,
            Pn,
            En,
            S0=self.S_state_,
            return_state=True,
        )

        self.Q_state_ = Q_final
        self.S_state_ = S_final

        return Qsim

    def score(self, P, PET, Q, metric: str = "nse") -> float:
        """Continue the simulation and calculate one performance metric."""

        Q = np.asarray(Q, dtype=float)
        Qsim = self.simulate(P, PET)

        if len(Q) != len(Qsim):
            raise ValueError("Q and simulated streamflow must have the same length.")

        if metric not in OBJECTIVES:
            raise ValueError(
                f"Unknown metric '{metric}'. Options: {list(OBJECTIVES)}"
            )

        func, _ = OBJECTIVES[metric]

        return func(Q, Qsim)
