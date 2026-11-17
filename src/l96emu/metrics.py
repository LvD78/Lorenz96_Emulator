"""Forecast rollouts and verification metrics (NumPy only)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# One Lorenz-96 model time unit corresponds to about 5 days in the atmosphere.
MTU_IN_DAYS = 5.0

StepFn = Callable[[np.ndarray], np.ndarray]


def rollout(step: StepFn, x0: np.ndarray, n_steps: int) -> np.ndarray:
    """Apply step repeatedly, feeding each output back in as the next input.

    Returns an array of shape (n_steps + 1, *x0.shape) that starts with x0.
    """
    out = np.empty((n_steps + 1, *x0.shape), dtype=np.float64)
    out[0] = x = x0
    for n in range(n_steps):
        x = step(x)
        out[n + 1] = x
    return out


def rmse_by_lead(forecast: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Root-mean-square error for each lead time (first axis)."""
    if forecast.shape != truth.shape:
        raise ValueError(f"shape mismatch: {forecast.shape} vs {truth.shape}")
    err2 = (forecast - truth) ** 2
    return np.sqrt(err2.reshape(err2.shape[0], -1).mean(axis=1))


def valid_time(rmse: np.ndarray, scale: float, threshold: float = 0.5) -> int:
    """Number of steps before the normalised RMSE first exceeds threshold.

    The RMSE is divided by scale, usually the climatological standard
    deviation. If the threshold is never exceeded, the full length is returned.
    A non-finite RMSE (a forecast that blew up) counts as exceeding it.
    """
    exceeded = ~(rmse / scale <= threshold)
    if not exceeded.any():
        return len(rmse) - 1
    return int(np.argmax(exceeded))
