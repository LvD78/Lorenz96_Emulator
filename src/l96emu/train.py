"""Generate Lorenz-96 data and train the one-step emulator.

Usage:
    l96-train --out runs/default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from l96emu.data import Normaliser, make_pairs, split_trajectories
from l96emu.lorenz96 import generate_trajectories
from l96emu.model import Emulator, ModelConfig, ResidualCNN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("runs/default"), help="output directory")
    p.add_argument("--n-traj", type=int, default=100, help="number of trajectories")
    p.add_argument("--n-steps", type=int, default=1000, help="time steps per trajectory")
    p.add_argument("--forcing", type=float, default=8.0, help="Lorenz-96 forcing F")
    p.add_argument("--dt", type=float, default=0.05, help="time step between saved states")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.dt <= 0:
        p.error("--dt must be positive")
    return args


def to_training_arrays(traj: np.ndarray, normaliser: Normaliser) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised inputs and normalised one-step increments."""
    x, y = make_pairs(traj)
    zx, zy = normaliser.encode(x), normaliser.encode(y)
    return zx, zy - zx


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.n_traj} trajectories of {args.n_steps} steps, "
        f"F = {args.forcing}, dt = {args.dt} ..."
    )
    traj = generate_trajectories(
        args.n_traj, args.n_steps, dt=args.dt, forcing=args.forcing, seed=args.seed
    )
    train, val, test = split_trajectories(traj)
    # dt and forcing are saved with the data so that evaluation uses the same values.
    np.savez_compressed(
        args.out / "data.npz",
        train=train,
        val=val,
        test=test,
        dt=args.dt,
        forcing=args.forcing,
    )

    normaliser = Normaliser.fit(train)
    zx_train, dz_train = to_training_arrays(train, normaliser)
    zx_val, dz_val = to_training_arrays(val, normaliser)
    # Increments are much smaller than the state itself, so the network is
    # trained on increments rescaled to unit variance.
    increment_std = float(dz_train.std())

    def tensor(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(tensor(zx_train), tensor(dz_train / increment_std)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    x_val, y_val = tensor(zx_val), tensor(dz_val / increment_std)

    config = ModelConfig(args.channels, args.kernel_size, args.n_layers)
    net = ResidualCNN(config)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {config} ({n_params} parameters)")

    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        net.train()
        total, count = 0.0, 0
        for xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimiser.step()
            total += loss.item() * len(xb)
            count += len(xb)
        scheduler.step()
        train_loss = total / count

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(x_val), y_val).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:3d}   train {train_loss:.3e}   val {val_loss:.3e}")

        if val_loss < best_val:
            best_val = val_loss
            Emulator(net, normaliser, increment_std).save(args.out / "model.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation loss {best_val:.3e}. Saved model to {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
