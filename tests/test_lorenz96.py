import numpy as np
import pytest

from l96emu.lorenz96 import generate_trajectories, integrate, rk4_step, tendency


def test_fixed_point_is_stationary():
    x = np.full(40, 8.0)
    np.testing.assert_allclose(tendency(x, forcing=8.0), 0.0)


def test_advection_term_conserves_energy():
    # The quadratic term alone neither creates nor destroys energy sum(x^2)/2,
    # so its projection onto the state must vanish.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 40))
    advection = tendency(x, forcing=0.0) + x
    np.testing.assert_allclose((x * advection).sum(axis=-1), 0.0, atol=1e-10)


def test_batched_and_single_tendency_agree():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 40))
    for i in range(3):
        np.testing.assert_allclose(tendency(x)[i], tendency(x[i]))


def test_rk4_is_fourth_order():
    rng = np.random.default_rng(3)
    x0 = 8.0 + rng.standard_normal(40)
    t_end = 0.1
    reference = integrate(x0, 1, t_end, substeps=512)[-1]

    errors = []
    for n in (4, 8):
        h = t_end / n
        x = x0.copy()
        for _ in range(n):
            x = rk4_step(x, h)
        errors.append(np.abs(x - reference).max())
    order = np.log2(errors[0] / errors[1])
    assert 3.5 < order < 4.5


def test_trajectories_shape_and_reproducibility():
    a = generate_trajectories(3, 20, n_vars=40, spinup_steps=50, seed=7)
    b = generate_trajectories(3, 20, n_vars=40, spinup_steps=50, seed=7)
    assert a.shape == (3, 21, 40)
    np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_attractor_statistics():
    # Known climate of Lorenz-96 with K=40, F=8: mean about 2.3, std about 3.6.
    traj = generate_trajectories(4, 2000, seed=0)
    assert 2.0 < traj.mean() < 2.7
    assert 3.3 < traj.std() < 3.9
