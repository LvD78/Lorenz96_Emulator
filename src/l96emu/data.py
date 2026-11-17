"""Turn trajectories into training data for a one-step emulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Normaliser:
    """Standardise states with a single mean and std shared by all variables.

    Lorenz-96 is statistically the same at every grid point, so one mean and
    one standard deviation are enough and keep the model translation
    equivariant.
    """

    mean: float
    std: float

    @classmethod
    def fit(cls, x: np.ndarray) -> Normaliser:
        return cls(mean=float(x.mean()), std=float(x.std()))

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


def split_trajectories(
    traj: np.ndarray, fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split along the trajectory axis into train, validation and test sets.

    Splitting whole trajectories, rather than individual time steps, avoids
    near-duplicate neighbouring states ending up in both train and test.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    n = traj.shape[0]
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    if n_train == 0 or n_val == 0 or n - n_train - n_val == 0:
        raise ValueError("not enough trajectories for a three-way split")
    return traj[:n_train], traj[n_train : n_train + n_val], traj[n_train + n_val :]


def make_pairs(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (x_t, x_{t+1}) pairs from trajectories of shape (n_traj, T, K)."""
    inputs = traj[:, :-1].reshape(-1, traj.shape[-1])
    targets = traj[:, 1:].reshape(-1, traj.shape[-1])
    return inputs, targets
