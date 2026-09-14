import numpy as np

from grhymolap.metrics import kge, mae, nnse, nse, pbias, rmse


def test_nse_perfect_fit():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(obs, obs) == 1.0


def test_nse_handles_nan():
    obs = np.array([1.0, np.nan, 3.0])
    sim = np.array([1.0, 2.0, 3.0])
    assert nse(obs, sim) == 1.0


def test_nnse_bounds():
    assert nnse(1.0) == 1.0
    assert 0 < nnse(0.0) <= 1.0


def test_rmse_mae_zero_for_perfect_fit():
    obs = np.array([1.0, 2.0, 3.0])
    assert rmse(obs, obs) == 0.0
    assert mae(obs, obs) == 0.0


def test_kge_perfect_fit():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(kge(obs, obs) - 1.0) < 1e-9


def test_pbias_zero_for_perfect_fit():
    obs = np.array([1.0, 2.0, 3.0])
    assert pbias(obs, obs) == 0.0
