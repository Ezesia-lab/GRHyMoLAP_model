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
    >>> model.fit(P, PET, Q, Q0=Q[0])
    >>> model.calibration_scores_
    >>> Qsim = model.simulate(P_next, PET_next)

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
        """Calibrate the model on a supplied series."""

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

        params = calibrate(
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

        Qsim, Q_final, S_final = simulate_streamflow(
            params,
            Q0_,
            Pn,
            En,
            return_state=True,
        )

        self.params_ = np.asarray(params)
        self.warmup_mask_ = warmup_mask
        self.calibration_mask_ = calibration_mask
        self.Q_sim_ = Qsim
        self.calibration_scores_ = _score_all(
            Q[calibration_mask],
            Qsim[calibration_mask],
        )

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

        if metric not in OBJECTIVES:
            raise ValueError(
                f"Unknown metric '{metric}'. Options: {list(OBJECTIVES)}"
            )

        func, _ = OBJECTIVES[metric]

        return func(Q, Qsim)
