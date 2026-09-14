import numpy as np
import pandas as pd

from grhymolap import GRHyMoLAP
from grhymolap.model import net_fluxes, simulate_streamflow
from grhymolap.periods import resolve_periods


def _make_synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.gamma(2.0, 3.0, n)
    PET = np.full(n, 3.0) + rng.normal(0, 0.3, n)
    PET = np.clip(PET, 0, None)
    Pn, En = net_fluxes(P, PET)
    true_params = [1.1, 6.0, 140.0, 0.3]
    Q = simulate_streamflow(true_params, Q0=1.0, Pn=Pn, En=En)
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    return P, PET, Q, dates, true_params


def test_fit_recovers_near_perfect_nse_on_noise_free_data():
    P, PET, Q, dates, _ = _make_synthetic()
    model = GRHyMoLAP(n_warmup=50)
    model.fit(P, PET, Q, dates=dates, train_ratio=0.7)
    assert model.train_scores_["nse"] > 0.9
    assert model.val_scores_["nse"] > 0.8


def test_simulate_matches_fit_length():
    P, PET, Q, dates, _ = _make_synthetic()
    model = GRHyMoLAP(n_warmup=50).fit(P, PET, Q, dates=dates, train_ratio=0.7)
    Qsim = model.simulate()
    assert Qsim.shape == Q.shape


def test_date_based_periods_supported():
    P, PET, Q, dates, _ = _make_synthetic()
    model = GRHyMoLAP(n_warmup=50)
    model.fit(
        P, PET, Q, dates=dates,
        train_period=("2000-01-01", "2001-01-01"),
        val_period=("2001-01-02", "2001-08-23"),
    )
    assert model.train_scores_ and model.val_scores_


def test_resolve_periods_masks_are_disjoint_and_exclude_warmup():
    n = 100
    warmup, train, val = resolve_periods(n, n_warmup=20, train_ratio=0.7)
    assert not np.any(warmup & train)
    assert not np.any(warmup & val)
    assert not np.any(train & val)
    assert warmup.sum() == 20
