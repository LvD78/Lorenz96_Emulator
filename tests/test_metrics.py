import numpy as np

from l96emu.lorenz96 import integrate, rk4_step
from l96emu.metrics import rmse_by_lead, rollout, valid_time


def test_rollout_matches_direct_integration():
    rng = np.random.default_rng(0)
    x0 = 8.0 + rng.standard_normal((2, 40))
    traj = rollout(lambda x: rk4_step(x, 0.05), x0, 10)
    np.testing.assert_allclose(traj, integrate(x0, 10, 0.05))


def test_persistence_rollout_is_constant():
    x0 = np.arange(40.0)
    traj = rollout(lambda x: x, x0, 5)
    assert traj.shape == (6, 40)
    np.testing.assert_array_equal(traj, np.broadcast_to(x0, traj.shape))


def test_rmse_by_lead():
    truth = np.zeros((3, 2, 4))
    forecast = np.zeros((3, 2, 4))
    forecast[1] = 2.0
    np.testing.assert_allclose(rmse_by_lead(forecast, truth), [0.0, 2.0, 0.0])


def test_valid_time():
    rmse = np.array([0.0, 0.1, 0.4, 0.6, 0.3])
    assert valid_time(rmse, scale=1.0, threshold=0.5) == 3
    assert valid_time(rmse, scale=10.0, threshold=0.5) == 4  # never exceeded
    assert valid_time(np.array([0.0, np.inf]), scale=1.0) == 1  # blow-up counts
