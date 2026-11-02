"""Attitude plots preserve quaternion components and frame conventions."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rocket_landing.demo import run_demo
from rocket_landing.dynamics import make_state


@pytest.fixture
def plotting(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    pytest.importorskip("matplotlib")
    module = importlib.import_module("rocket_landing.plotting")
    yield module
    module.plt.close("all")


def test_quaternion_panels_show_all_stored_components_with_separate_scales(plotting):
    states, controls, summary = run_demo()
    _, axes = plotting.plot_trajectory(states, controls, summary["dt_s"])

    scalar, vector = axes[3]
    assert "quaternion" in scalar.get_title().lower()
    assert "quaternion" in vector.get_title().lower()
    assert not scalar.get_shared_y_axes().joined(scalar, vector)
    lines = [*scalar.lines, *vector.lines]
    assert [line.get_label() for line in lines] == [r"$q_w$", r"$q_x$", r"$q_y$", r"$q_z$"]
    for index, line in enumerate(lines):
        assert_allclose(line.get_xdata(), np.arange(len(states)) * summary["dt_s"])
        assert_allclose(line.get_ydata(), states[:, 6 + index])


def test_nose_arrows_convert_passive_attitude_back_to_world_coordinates(plotting, monkeypatch):
    from mpl_toolkits.mplot3d import Axes3D

    # A +90-degree rotation about body/world Y points the nose along world +X.
    # Using C_BW instead of its transpose would incorrectly draw it along -X.
    state = make_state(quaternion=(np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0))
    states = np.tile(state, (2, 1))
    captured_directions = []
    original_quiver = Axes3D.quiver

    def capture_quiver(axis, x, y, z, u, v, w, **kwargs):
        captured_directions.append(np.column_stack((u, v, w)))
        return original_quiver(axis, x, y, z, u, v, w, **kwargs)

    monkeypatch.setattr(Axes3D, "quiver", capture_quiver)
    _, axes = plotting.plot_trajectory(states, np.zeros((1, 3)), 0.02)

    assert len(captured_directions) == 1
    assert_allclose(captured_directions[0], [[1.0, 0.0, 0.0]] * 2, atol=1e-14)
    assert_allclose(axes[2, 0].lines[0].get_ydata(), [90.0, 90.0])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("states", np.zeros(14), "states"),
        ("states", np.zeros((2, 13)), "states"),
        ("states", np.zeros((1, 14)), "states"),
        ("controls", np.zeros((2, 3)), "terminal control"),
        ("controls", np.zeros((1, 2)), "controls"),
        ("states", np.full((2, 14), np.nan), "finite"),
        ("controls", np.full((1, 3), np.inf), "finite"),
        ("dt", 0.0, "positive scalar"),
    ],
)
def test_invalid_trajectories_fail_before_creating_a_figure(plotting, field, value, message):
    arguments = {"states": np.tile(make_state(), (2, 1)), "controls": np.zeros((1, 3)), "dt": 0.02}
    arguments[field] = value
    existing_figures = plotting.plt.get_fignums()

    with pytest.raises(ValueError, match=message):
        plotting.plot_trajectory(**arguments)

    assert plotting.plt.get_fignums() == existing_figures
