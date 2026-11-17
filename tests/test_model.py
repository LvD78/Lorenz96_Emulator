import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from l96emu.data import Normaliser  # noqa: E402
from l96emu.model import Emulator, ModelConfig, ResidualCNN  # noqa: E402


def make_emulator(seed: int = 0) -> Emulator:
    torch.manual_seed(seed)
    net = ResidualCNN(ModelConfig(channels=8, kernel_size=5, n_layers=3))
    return Emulator(net, Normaliser(mean=2.3, std=3.6), increment_std=0.3)


def test_output_shape():
    net = ResidualCNN()
    assert net(torch.randn(7, 40)).shape == (7, 40)


def test_even_kernel_is_rejected():
    with pytest.raises(ValueError):
        ResidualCNN(ModelConfig(kernel_size=4))


def test_translation_equivariance():
    # Shifting the input around the circle must shift the output the same way.
    net = ResidualCNN()
    z = torch.randn(3, 40)
    shifted_out = net(torch.roll(z, shifts=5, dims=-1))
    torch.testing.assert_close(shifted_out, torch.roll(net(z), shifts=5, dims=-1))


def test_step_keeps_shape_for_single_and_batch():
    emu = make_emulator()
    x = 8.0 + np.random.default_rng(0).standard_normal((4, 40))
    assert emu.step(x).shape == (4, 40)
    assert emu.step(x[0]).shape == (40,)
    np.testing.assert_allclose(emu.step(x)[0], emu.step(x[0]), rtol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    emu = make_emulator()
    emu.save(tmp_path / "model.pt")
    loaded = Emulator.load(tmp_path / "model.pt")
    x = 8.0 + np.random.default_rng(1).standard_normal((2, 40))
    np.testing.assert_allclose(loaded.step(x), emu.step(x))


def test_training_and_evaluation_smoke(tmp_path):
    # Runs the full pipeline on a tiny problem to catch wiring errors.
    from l96emu import evaluate, train

    run = tmp_path / "run"
    train.main(
        [
            "--out",
            str(run),
            "--n-traj",
            "10",
            "--n-steps",
            "60",
            "--epochs",
            "2",
            "--channels",
            "8",
            "--n-layers",
            "2",
        ]
    )
    history = json.loads((run / "history.json").read_text())
    assert len(history) == 2
    assert all(np.isfinite(h["val_loss"]) for h in history)
    data = np.load(run / "data.npz")
    assert float(data["dt"]) == 0.05
    assert float(data["forcing"]) == 8.0

    figs = tmp_path / "figs"
    evaluate.main(["--run", str(run), "--out", str(figs), "--lead-steps", "20", "--n-init", "3"])
    assert (figs / "rmse_vs_lead.png").exists()
    assert (figs / "hovmoeller.png").exists()
    metrics = json.loads((figs / "metrics.json").read_text())
    assert set(metrics) == {"Emulator (CNN)", "Persistence", "Euler physics", "Climatology"}
