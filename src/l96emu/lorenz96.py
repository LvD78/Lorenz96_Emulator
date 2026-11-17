"""Lorenz-96 model and a fourth-order Runge-Kutta integrator.

The Lorenz-96 system is a standard toy model for atmospheric dynamics:

    dx_k/dt = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + F,    k = 0, ..., K-1,

with periodic boundary conditions. The quadratic term mimics advection, -x_k is
damping and F is a constant forcing. With K = 40 and F = 8 the dynamics are
chaotic, and one model time unit corresponds to roughly 5 days in the
atmosphere (Lorenz 1996), so dt = 0.05 is about 6 hours.
"""

from __future__ import annotations

import numpy as np


def tendency(x: np.ndarray, forcing: float = 8.0) -> np.ndarray:
    """Return dx/dt for a state (or a batch of states) along the last axis."""
    x_p1 = np.roll(x, -1, axis=-1)  # x_{k+1}
    x_m1 = np.roll(x, 1, axis=-1)  # x_{k-1}
    x_m2 = np.roll(x, 2, axis=-1)  # x_{k-2}
    return (x_p1 - x_m2) * x_m1 - x + forcing


def rk4_step(x: np.ndarray, dt: float, forcing: float = 8.0) -> np.ndarray:
    """Advance the state by one classical Runge-Kutta step of size dt."""
    k1 = tendency(x, forcing)
    k2 = tendency(x + 0.5 * dt * k1, forcing)
    k3 = tendency(x + 0.5 * dt * k2, forcing)
    k4 = tendency(x + dt * k3, forcing)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(
    x0: np.ndarray,
    n_steps: int,
    dt: float,
    forcing: float = 8.0,
    substeps: int = 1,
) -> np.ndarray:
    """Integrate from x0 and return the trajectory, including x0.

    The output has shape (n_steps + 1, *x0.shape). With substeps > 1 each
    output interval dt is split into smaller RK4 steps for extra accuracy.
    """
    h = dt / substeps
    out = np.empty((n_steps + 1, *x0.shape), dtype=np.float64)
    out[0] = x = np.asarray(x0, dtype=np.float64)
    for n in range(n_steps):
        for _ in range(substeps):
            x = rk4_step(x, h, forcing)
        out[n + 1] = x
    return out


def generate_trajectories(
    n_traj: int,
    n_steps: int,
    n_vars: int = 40,
    dt: float = 0.05,
    forcing: float = 8.0,
    spinup_steps: int = 1000,
    substeps: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Generate independent trajectories on the attractor.

    Each trajectory starts from the fixed point x = F plus a small random
    perturbation and is run for spinup_steps before recording, so that all
    recorded states lie on the chaotic attractor. Returns an array of shape
    (n_traj, n_steps + 1, n_vars).
    """
    rng = np.random.default_rng(seed)
    x0 = forcing + 0.01 * rng.standard_normal((n_traj, n_vars))
    # All trajectories are integrated together as one batch.
    spun_up = integrate(x0, spinup_steps, dt, forcing, substeps)[-1]
    traj = integrate(spun_up, n_steps, dt, forcing, substeps)
    return np.transpose(traj, (1, 0, 2))
