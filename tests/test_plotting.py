"""Optional plotting works without a display or a terminal control sample."""

import importlib

import numpy as np
import pytest

from pendulum_ilqr.controls.ilqr import alilqr, ilqr


@pytest.fixture
def plotting(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    pytest.importorskip("matplotlib")
    module = importlib.import_module("pendulum_ilqr.simulation.plotting")
    yield module
    module.plt.close("all")


@pytest.mark.parametrize(
    ("solver", "plot_name", "axes_shape"),
    [(ilqr, "plot_ilqr", (2, 2)), (alilqr, "plot_alilqr", (3, 2))],
)
def test_zero_update_history_saves_a_headless_plot(
    solver, plot_name, axes_shape, plotting, tmp_path
):
    _, _, info = solver(np.zeros(2), np.zeros((3, 1)), np.eye(2), np.eye(1), np.zeros(2), 0.04)
    output = tmp_path / "convergence.png"
    figure, axes = getattr(plotting, plot_name)(
        info["xs"], info["us"], 0.04, info=info, save_path=output, show=False
    )
    assert axes.shape == axes_shape
    assert len(figure.axes) == axes_shape[0] * axes_shape[1]
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("plot_name", ["plot_ilqr", "plot_alilqr"])
def test_plot_rejects_a_terminal_control_sample(plot_name, plotting):
    _, _, info = ilqr(np.zeros(2), np.zeros((3, 1)), np.eye(2), np.eye(1), np.zeros(2), 0.04)
    with pytest.raises(ValueError, match="terminal control"):
        getattr(plotting, plot_name)(info["xs"], np.zeros((1, 4, 1)), 0.04, info=info, show=False)
