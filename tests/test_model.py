import numpy as np

from grhymolap.model import net_fluxes, percolation, simulate_streamflow


def test_net_fluxes_basic():
    P = np.array([10.0, 2.0, 5.0])
    PET = np.array([3.0, 4.0, 5.0])
    Pn, En = net_fluxes(P, PET)
    np.testing.assert_allclose(Pn, [7.0, 0.0, 0.0])
    np.testing.assert_allclose(En, [0.0, 2.0, 0.0])


def test_percolation_shape_and_nonnegative():
    rng = np.random.default_rng(0)
    Pn = rng.uniform(0, 10, 100)
    En = rng.uniform(0, 5, 100)
    perc = percolation(Pn, En, X1=150.0)
    assert perc.shape == (100,)
    assert np.all(perc >= 0)


def test_simulate_streamflow_starts_at_q0_and_is_nonnegative():
    rng = np.random.default_rng(1)
    Pn = rng.uniform(0, 10, 50)
    En = rng.uniform(0, 5, 50)
    Q0 = 2.5
    Qsim = simulate_streamflow([1.0, 8.0, 150.0, 0.1], Q0, Pn, En)
    assert Qsim[0] == Q0
    assert np.all(Qsim >= 0)
    assert Qsim.shape == Pn.shape
