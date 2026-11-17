import numpy as np
import pytest

from l96emu.data import Normaliser, make_pairs, split_trajectories


def test_normaliser_roundtrip_and_moments():
    rng = np.random.default_rng(0)
    x = 3.0 + 2.0 * rng.standard_normal((100, 40))
    norm = Normaliser.fit(x)
    z = norm.encode(x)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std() - 1.0) < 1e-12
    np.testing.assert_allclose(norm.decode(z), x)


def test_split_is_disjoint_and_complete():
    traj = np.arange(10)[:, None, None] * np.ones((10, 5, 4))
    train, val, test = split_trajectories(traj)
    assert (len(train), len(val), len(test)) == (8, 1, 1)
    ids = np.concatenate([train[:, 0, 0], val[:, 0, 0], test[:, 0, 0]])
    np.testing.assert_array_equal(np.sort(ids), np.arange(10))


def test_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_trajectories(np.zeros((10, 5, 4)), (0.5, 0.2, 0.2))


def test_pairs_do_not_cross_trajectory_boundaries():
    n_traj, n_time, k = 3, 6, 4
    # Encode (trajectory, time) in the values so pairs can be checked.
    traj = (100 * np.arange(n_traj)[:, None] + np.arange(n_time)[None, :])[..., None]
    traj = np.repeat(traj, k, axis=-1).astype(float)
    x, y = make_pairs(traj)
    assert x.shape == y.shape == (n_traj * (n_time - 1), k)
    np.testing.assert_array_equal(y - x, 1.0)  # target is always the next time step
