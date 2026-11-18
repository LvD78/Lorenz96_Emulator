"""Visualise Lorenz-96 trajectories in 3D and the structure and weights of the emulator CNN.

Figures written to --out:
  trajectory_3d.png       one trajectory: (x0, x1, x2) and the top 3 PCA directions
  spacetime_surface.png   the state as a 3D surface over grid point and time
  receptive_field.png     which input points one output point of the CNN depends on
  network_graph.png       channels as nodes, connections coloured by weight (needs torch)
  kernels.png             learned kernels of the first and last layer       (needs torch)
  forecast_3d.png         truth vs emulator forecast in the PCA view        (needs torch)

Usage (after `pip install -e .` and `l96-train`):
    l96-plot --run runs/default --out figures
The time step and forcing are read from the run's data.npz. Without a run folder, a short
trajectory is generated with dt = 0.05 and F = 8, and only the first three figures are made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from l96emu.lorenz96 import generate_trajectories  # noqa: E402
from l96emu.metrics import MTU_IN_DAYS  # noqa: E402

DEFAULT_DT = 0.05


# ----------------------------------------------------------------------------- data


def load_trajectories(run: Path | None) -> tuple[np.ndarray, float]:
    """Test trajectories of a training run and their time step, or a few fresh ones."""
    if run is not None and (run / "data.npz").exists():
        data = np.load(run / "data.npz")
        dt = float(data["dt"]) if "dt" in data.files else DEFAULT_DT
        return data["test"], dt
    print("No run folder found. Generating 3 trajectories of 400 steps with dt = 0.05, F = 8.")
    return generate_trajectories(3, 400, dt=DEFAULT_DT, seed=0), DEFAULT_DT


def pca_basis(states: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, the n leading principal directions (rows) and their explained variance fractions."""
    mean = states.mean(axis=0)
    _, s, vt = np.linalg.svd(states - mean, full_matrices=False)
    var = s**2 / np.sum(s**2)
    return mean, vt[:n], var[:n]


def coloured_line_3d(
    ax, xyz: np.ndarray, days_per_step: float, cmap: str = "viridis", lw: float = 1.0
):
    """Draw a 3D line whose colour runs along time."""
    seg = np.stack([xyz[:-1], xyz[1:]], axis=1)
    lc = Line3DCollection(seg, cmap=cmap, linewidth=lw)
    lc.set_array(np.arange(len(seg)) * days_per_step)
    ax.add_collection(lc)
    for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lo, hi = xyz[:, i].min(), xyz[:, i].max()
        pad = 0.05 * (hi - lo)
        setter(lo - pad, hi + pad)
    return lc


# ----------------------------------------------------------------------------- trajectories


def plot_trajectory_3d(traj: np.ndarray, days_per_step: float, out: Path, days: float = 50) -> None:
    n = min(int(round(days / days_per_step)) + 1, traj.shape[1])
    one = traj[0, :n]
    mean, basis, var = pca_basis(traj.reshape(-1, traj.shape[-1]))
    pcs = (one - mean) @ basis.T

    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    lc = coloured_line_3d(ax1, one[:, :3], days_per_step)
    ax1.set_xlabel("$x_0$")
    ax1.set_ylabel("$x_1$")
    ax1.set_zlabel("$x_2$")
    ax1.set_title("Three neighbouring grid points")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    coloured_line_3d(ax2, pcs, days_per_step)
    ax2.set_xlabel(f"PC 1 ({var[0]:.0%})")
    ax2.set_ylabel(f"PC 2 ({var[1]:.0%})")
    ax2.set_zlabel(f"PC 3 ({var[2]:.0%})")
    ax2.set_title("Top 3 principal directions of all 40 variables")

    fig.colorbar(lc, ax=[ax1, ax2], shrink=0.7, label="time [days]")
    fig.suptitle(f"One Lorenz-96 trajectory over {days:g} days (40-dim. state, two 3D views)")
    fig.savefig(out / "trajectory_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spacetime_surface(
    traj: np.ndarray, days_per_step: float, out: Path, days: float = 20
) -> None:
    n = min(int(round(days / days_per_step)) + 1, traj.shape[1])
    field = traj[0, :n]
    k = np.arange(field.shape[1])
    t = np.arange(n) * days_per_step
    kk, tt = np.meshgrid(k, t)
    vmax = np.abs(field).max()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        kk, tt, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rstride=1, cstride=1, linewidth=0
    )
    ax.set_xlabel("grid point k")
    ax.set_ylabel("time [days]")
    ax.set_zlabel("$x_k$")
    ax.view_init(elev=35, azim=-60)
    fig.colorbar(surf, shrink=0.6, pad=0.1)
    ax.set_title(f"Lorenz-96 state as a surface over space and time ({days:g} days)")
    fig.savefig(out / "spacetime_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- network structure


def plot_receptive_field(out: Path, n_layers: int = 4, kernel: int = 5, k_vars: int = 40) -> None:
    """Every connection that leads to one output point, layer by layer."""
    half = kernel // 2
    centre = k_vars // 2
    names = ["input $x_k$"] + [f"conv {i + 1}" for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    # positions that influence the centre output, walking back from the output layer
    active = [set() for _ in range(n_layers + 1)]
    active[n_layers] = {centre}
    for layer in range(n_layers, 0, -1):
        for p in active[layer]:
            active[layer - 1].update((p + d) % k_vars for d in range(-half, half + 1))

    segs = []
    for layer in range(n_layers, 0, -1):
        for p in active[layer]:
            for d in range(-half, half + 1):
                segs.append([((p + d) % k_vars, layer - 1), (p, layer)])
    ax.add_collection(LineCollection(segs, colors="tab:blue", alpha=0.18, linewidths=0.8))

    for layer in range(n_layers + 1):
        on = np.array(sorted(active[layer]))
        off = np.array([p for p in range(k_vars) if p not in active[layer]])
        ax.scatter(off, np.full(off.size, layer), s=14, color="lightgrey", zorder=3)
        ax.scatter(on, np.full(on.size, layer), s=22, color="tab:blue", zorder=4)
    ax.scatter([centre], [n_layers], s=90, color="tab:red", zorder=5, label="one output point")

    ax.set_yticks(range(n_layers + 1), names)
    ax.set_xlabel("grid point k")
    ax.set_xlim(-1, k_vars)
    ax.set_ylim(-0.5, n_layers + 0.5)
    ax.legend(loc="upper left")
    width = len(active[0])
    ax.set_title(
        f"Receptive field: {n_layers} layers of kernel size {kernel} let one output see "
        f"{width} of the {k_vars} input points"
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out / "receptive_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_network_graph(
    weights: list[np.ndarray], out: Path, keep: float = 0.05, title_note: str = ""
) -> None:
    """Channels as nodes, one column per layer; an edge = one input channel -> output channel.

    Each edge stands for a whole kernel (k weights). Its width shows the kernel's size
    (L2 norm), its colour the sign of the kernel's sum. Between the 32-channel layers
    (1024 edges each) only the strongest `keep` fraction is drawn, otherwise the picture
    is solid ink; the 32 edges into the first and out of the last layer are all drawn.
    """
    sizes = [weights[0].shape[1]] + [w.shape[0] for w in weights]
    xs = np.arange(len(sizes)) * 3.0

    def ys(n: int) -> np.ndarray:
        return np.linspace(-1, 1, n) if n > 1 else np.zeros(1)

    fig, ax = plt.subplots(figsize=(12, 7))
    for li, w in enumerate(weights):
        strength = np.linalg.norm(w, axis=2)  # (out, in)
        sign = np.sign(w.sum(axis=2))
        thresh = np.quantile(strength, 1 - keep) if strength.size > 100 else -np.inf
        y_in, y_out = ys(w.shape[1]), ys(w.shape[0])
        smax = strength.max()
        for o in range(w.shape[0]):
            for i in range(w.shape[1]):
                if strength[o, i] < thresh:
                    continue
                ax.plot(
                    [xs[li], xs[li + 1]],
                    [y_in[i], y_out[o]],
                    color="tab:red" if sign[o, i] > 0 else "tab:blue",
                    lw=0.3 + 2.2 * strength[o, i] / smax,
                    alpha=0.25 + 0.6 * strength[o, i] / smax,
                    zorder=1,
                )
    labels = ["input"] + [
        f"conv {i + 1}" + (" + GELU" if i < len(weights) - 1 else "") for i in range(len(weights))
    ]
    for x, n, lab in zip(xs, sizes, labels, strict=True):
        ax.scatter(
            np.full(n, x), ys(n), s=40 if n > 1 else 160, color="white", edgecolor="black", zorder=3
        )
        ax.text(x, 1.12, f"{lab}\n{n} channel{'s' if n > 1 else ''}", ha="center", va="bottom")

    ax.plot([], [], color="tab:red", label="kernel sum > 0")
    ax.plot([], [], color="tab:blue", label="kernel sum < 0")
    ax.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    ax.set_axis_off()
    n_par = sum(w.size + w.shape[0] for w in weights)
    ax.set_title(
        f"Channel graph of the CNN ({n_par:,} parameters; between hidden layers only the "
        f"strongest {keep:.0%} of connections are shown){title_note}",
        pad=40,
    )
    fig.savefig(out / "network_graph.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kernels(weights: list[np.ndarray], out: Path, title_note: str = "") -> None:
    """The kernels of the first layer (input -> 32) and last layer (32 -> output) as heatmaps."""
    first = weights[0][:, 0, :]  # (32, k): one 5-point stencil per hidden channel
    last = weights[-1][0, :, :]  # (32, k): how each channel's neighbourhood feeds the output
    k = first.shape[1]
    offsets = np.arange(k) - k // 2

    fig, axes = plt.subplots(1, 2, figsize=(9, 7), sharey=True, layout="constrained")
    for ax, mat, title in zip(
        axes,
        (first, last),
        ("first layer: input -> channel", "last layer: channel -> output"),
        strict=True,
    ):
        v = np.abs(mat).max()
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(k), [f"k{o:+d}" if o else "k" for o in offsets])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.6)
    axes[0].set_ylabel("hidden channel")
    fig.suptitle(f"Convolution kernels (each row = 5 weights over neighbours){title_note}")
    fig.savefig(out / "kernels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_3d(
    traj: np.ndarray, step, days_per_step: float, out: Path, days: float = 15
) -> None:
    """Truth and emulator forecast from the same start, in the PCA view."""
    from l96emu.metrics import rollout

    n = min(int(round(days / days_per_step)), traj.shape[1] - 1)
    truth = traj[0, : n + 1]
    forecast = rollout(step, truth[0], n)
    mean, basis, _ = pca_basis(traj.reshape(-1, traj.shape[-1]))
    t_p, f_p = (truth - mean) @ basis.T, (forecast - mean) @ basis.T

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection="3d")
    ax.plot(*t_p.T, color="black", lw=1.2, label="truth")
    ax.plot(*f_p.T, color="tab:orange", lw=1.2, label="emulator")
    ax.scatter(*t_p[0], color="tab:green", s=50, label="start")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend()
    ax.set_title(f"Emulator vs truth over {days:g} days")
    fig.savefig(out / "forecast_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main


def conv_weights_from_state_dict(state_dict) -> list[np.ndarray]:
    """The conv weight tensors in layer order, as NumPy arrays of shape (out, in, k)."""
    keys = sorted(
        (k for k in state_dict if k.endswith(".weight")), key=lambda k: int(k.split(".")[1])
    )
    return [state_dict[k].detach().cpu().numpy() for k in keys]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/default"), help="training output dir")
    p.add_argument("--out", type=Path, default=Path("figures"), help="where to save figures")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    traj, dt = load_trajectories(args.run)
    days_per_step = dt * MTU_IN_DAYS
    plot_trajectory_3d(traj, days_per_step, args.out)
    plot_spacetime_surface(traj, days_per_step, args.out)

    model_path = args.run / "model.pt"
    try:
        from l96emu.model import Emulator
    except ImportError:
        Emulator = None
    if Emulator is not None and model_path.exists():
        emu = Emulator.load(model_path)
        cfg = emu.net.config
        plot_receptive_field(args.out, cfg.n_layers, cfg.kernel_size, traj.shape[-1])
        weights = conv_weights_from_state_dict(emu.net.state_dict())
        plot_network_graph(weights, args.out)
        plot_kernels(weights, args.out)
        plot_forecast_3d(traj, emu.step, days_per_step, args.out)
    else:
        plot_receptive_field(args.out)
        print("No trained model (or no torch): skipped network_graph, kernels and forecast_3d.")
    print(f"Saved figures to {args.out}/")


if __name__ == "__main__":
    main()
