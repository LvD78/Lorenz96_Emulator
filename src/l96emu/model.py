"""Neural network emulator for one Lorenz-96 time step."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from l96emu.data import Normaliser


@dataclass
class ModelConfig:
    channels: int = 32
    kernel_size: int = 5
    n_layers: int = 4


class ResidualCNN(nn.Module):
    """1D convolutional network with periodic (circular) padding.

    The network maps a normalised state of shape (batch, K) to the normalised
    increment x_{t+1} - x_t. Circular padding matches the periodic boundary of
    Lorenz-96, and because the same kernels are applied at every grid point
    the network is translation equivariant, just like the true dynamics.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        k = self.config.kernel_size
        if k % 2 == 0:
            raise ValueError("kernel_size must be odd so that padding is symmetric")
        if self.config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        layers: list[nn.Module] = []
        in_ch = 1
        for _ in range(self.config.n_layers - 1):
            layers.append(
                nn.Conv1d(in_ch, self.config.channels, k, padding=k // 2, padding_mode="circular")
            )
            layers.append(nn.GELU())
            in_ch = self.config.channels
        layers.append(nn.Conv1d(in_ch, 1, k, padding=k // 2, padding_mode="circular"))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.unsqueeze(1)).squeeze(1)


@dataclass
class Emulator:
    """Wraps the network so it can be used as a NumPy step function."""

    net: ResidualCNN
    normaliser: Normaliser
    increment_std: float

    def step(self, x: np.ndarray) -> np.ndarray:
        """Predict x_{t+1} from x_t for a single state (K,) or a batch (B, K)."""
        z = self.normaliser.encode(np.atleast_2d(x))
        self.net.eval()
        with torch.no_grad():
            dz = self.net(torch.as_tensor(z, dtype=torch.float32)).numpy()
        x_next = self.normaliser.decode(z + self.increment_std * dz.astype(np.float64))
        return x_next.reshape(np.shape(x))

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": asdict(self.net.config),
                "mean": self.normaliser.mean,
                "std": self.normaliser.std,
                "increment_std": self.increment_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Emulator:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        net = ResidualCNN(ModelConfig(**ckpt["config"]))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return cls(
            net=net,
            normaliser=Normaliser(mean=ckpt["mean"], std=ckpt["std"]),
            increment_std=ckpt["increment_std"],
        )
