"""Evaluate a trained emulator with autoregressive forecasts on the test set.

The emulator is compared with three baselines:
  * persistence:   the forecast keeps the initial state unchanged
  * climatology:   the forecast is the long-term mean state
  * Euler physics: the true equations, but integrated with one crude
                   forward-Euler step per time step

Usage:
    l96-evaluate --run runs/default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, also works on CI servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from l96emu.lorenz96 import tendency  # noqa: E402
from l96emu.metrics import MTU_IN_DAYS, rmse_by_lead, rollout, valid_time  # noqa: E402
from l96emu.model import Emulator  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/default"), help="training output dir")
    p.add_argument("--out", type=Path, default=Path("figures"), help="where to save figures")
    p.add_argument("--lead-steps", type=int, default=200, help="forecast length in steps")
    p.add_argument("--n-init", type=int, default=100, help="number of initial conditions")
    return p.parse_args(argv)


def initial_conditions(
    test: np.ndarray, n_init: int, lead_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pick start states from the test trajectories, evenly spread in time.

    Returns x0 with shape (n_init, K) and the true continuation with shape
    (lead_steps + 1, n_init, K).
    """
    n_traj, n_time, _ = test.shape
    if lead_steps >= n_time:
        raise ValueError("test trajectories are shorter than the forecast length")
    stride = max(lead_steps // 4, 1)
    starts = [(i, t) for t in range(0, n_time - lead_steps, stride) for i in range(n_traj)]
    if len(starts) < n_init:
        raise ValueError(f"only {len(starts)} initial conditions available, asked for {n_init}")
    chosen = starts[:n_init]
    truth = np.stack([test[i, t : t + lead_steps + 1] for i, t in chosen], axis=1)
    return truth[0], truth


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    emulator = Emulator.load(args.run / "model.pt")
    data = np.load(args.run / "data.npz")
    clim_mean = float(data["train"].mean())
    clim_std = float(data["train"].std())
    # dt and forcing come from the run itself. Older runs did not save them and
    # always used dt = 0.05 and F = 8.
    dt = float(data["dt"]) if "dt" in data.files else 0.05
    forcing = float(data["forcing"]) if "forcing" in data.files else 8.0
    days_per_step = dt * MTU_IN_DAYS

    x0, truth = initial_conditions(data["test"], args.n_init, args.lead_steps)

    with np.errstate(over="ignore", invalid="ignore"):
        forecasts = {
            "Emulator (CNN)": rollout(emulator.step, x0, args.lead_steps),
            "Persistence": rollout(lambda x: x, x0, args.lead_steps),
            "Euler physics": rollout(lambda x: x + dt * tendency(x, forcing), x0, args.lead_steps),
            "Climatology": np.full_like(truth, clim_mean),
        }
        rmse = {name: rmse_by_lead(f, truth) for name, f in forecasts.items()}

    lead_days = np.arange(args.lead_steps + 1) * days_per_step

    # Summary: how long does each forecast stay useful?
    summary = {}
    print(f"{'forecast':<16} {'RMSE @ 1 day':>13} {'valid time':>12}")
    # The step closest to a 1-day lead, at least one step and within the forecast.
    one_day = min(max(int(round(1.0 / days_per_step)), 1), args.lead_steps)
    for name, r in rmse.items():
        vt_days = valid_time(r, clim_std) * days_per_step
        summary[name] = {"rmse_1day": float(r[one_day]), "valid_time_days": vt_days}
        print(f"{name:<16} {r[one_day]:13.3f} {vt_days:10.2f} d")
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: RMSE against lead time
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in rmse.items():
        ax.plot(lead_days, r, label=name)
    ax.axhline(0.5 * clim_std, color="grey", ls=":", lw=1, label="valid-time threshold")
    ax.set_xlabel("lead time [days]")
    ax.set_ylabel("RMSE")
    ax.set_ylim(0, 1.6 * clim_std)
    ax.set_xlim(0, lead_days[-1])
    ax.legend()
    ax.set_title(f"Lorenz-96 forecast error, mean over {args.n_init} test cases")
    fig.tight_layout()
    fig.savefig(args.out / "rmse_vs_lead.png", dpi=150)
    plt.close(fig)

    # Figure 2: Hovmoeller diagram of one forecast
    case = 0
    panels = {
        "Truth": truth[:, case],
        "Emulator": forecasts["Emulator (CNN)"][:, case],
        "Emulator - Truth": forecasts["Emulator (CNN)"][:, case] - truth[:, case],
    }
    vmax = np.abs(truth[:, case]).max()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for ax, (title, field) in zip(axes, panels.items(), strict=True):
        im = ax.pcolormesh(
            np.arange(field.shape[1]),
            lead_days,
            field,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("grid point k")
    axes[0].set_ylabel("lead time [days]")
    fig.colorbar(im, ax=axes, shrink=0.9)
    fig.savefig(args.out / "hovmoeller.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures and metrics to {args.out}/")


if __name__ == "__main__":
    main()
