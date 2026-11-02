"""Portable open-loop examples, artifact exports, and optional plotting."""

import builtins
import csv
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from rocket_landing import demo
from rocket_landing.attitude import rotation_matrix
from rocket_landing.dynamics import RocketLandingPlant
from rocket_landing.simulation import rollout


def reject_nonfinite_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


@pytest.mark.parametrize("scenario", ["maneuver", "braking", "hover", "coast"])
def test_scenarios_export_finite_results_without_importing_plotting(
    scenario, tmp_path, monkeypatch, capsys
):
    original_import = builtins.__import__

    def import_without_plotting(name, *args, **kwargs):
        if name.startswith("matplotlib") or name == "rocket_landing.plotting":
            pytest.fail("The default rocket demo must run without plotting dependencies.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_plotting)
    output = tmp_path / "nested" / scenario
    assert demo.main(["--scenario", scenario, "--output-dir", str(output)]) == 0
    assert f"Scenario: {scenario} (open loop)" in capsys.readouterr().out

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8"),
        parse_constant=reject_nonfinite_constant,
    )
    assert summary["scenario"] == scenario
    assert summary["mass_depletion_enabled"] is False
    assert summary["minimum_altitude_m"] > 0.0
    assert summary["maximum_quaternion_norm_error"] < 1e-12
    assert summary["duration_s"] == pytest.approx(summary["dt_s"] * summary["steps"])

    with (output / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == [
            "time_s",
            "x_m",
            "y_m",
            "z_m",
            "vx_m_s",
            "vy_m_s",
            "vz_m_s",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx_rad_s",
            "wy_rad_s",
            "wz_rad_s",
            "mass_kg",
            "Tx_N",
            "Ty_N",
            "Tz_N",
        ]
    assert len(rows) == summary["steps"] + 1
    controls = ["Tx_N", "Ty_N", "Tz_N"]
    assert all(rows[-1][name] == "" for name in controls)
    assert all(np.isfinite(float(row[name])) for row in rows[:-1] for name in controls)
    state_columns = reader.fieldnames[1:15]
    states = np.array([[float(row[name]) for name in state_columns] for row in rows])
    assert np.all(np.isfinite(states))
    assert_allclose(states[0], summary["initial_state"])
    assert_allclose(states[-1], summary["final_state"])
    assert float(rows[-1]["time_s"]) == pytest.approx(summary["duration_s"])
    assert_allclose(states[:, 13], 1000.0)
    if scenario == "hover":
        assert_allclose(states, np.tile(states[0], (len(states), 1)), atol=1e-12)
    elif scenario == "braking":
        assert states[0, 5] < states[-1, 5] < 0.0
    elif scenario == "coast":
        assert states[-1, 5] < states[0, 5]
        assert_allclose(summary["thrust_body_N"], np.zeros(3))


def test_default_maneuver_descends_translates_and_rotates_without_tumbling():
    states, controls, summary = demo.run_demo()
    assert summary["scenario"] == "maneuver"
    assert np.all(np.isfinite(states))
    assert np.all(states[:, 2] > 0.0)
    assert np.all(states[:, 5] < 0.0)
    assert states[-1, 2] < states[0, 2] - 20.0
    assert -3.0 < states[-1, 5] < -0.5
    assert np.all(np.abs(states[-1, :2] - states[0, :2]) > 2.0)

    nose_vertical = np.array([rotation_matrix(q)[2, 2] for q in states[:, 6:10]])
    tilt = np.rad2deg(np.arccos(np.clip(nose_vertical, -1.0, 1.0)))
    angular_speed = np.rad2deg(np.linalg.norm(states[:, 10:13], axis=1))
    assert 4.0 < tilt.max() < 12.0
    assert tilt[-1] < 3.0
    assert angular_speed.max() < 12.0
    assert angular_speed[-1] < 1.0
    assert_allclose(np.linalg.norm(states[:, 6:10], axis=1), 1.0, atol=1e-12)
    assert_allclose(states[:, 13], 1000.0)

    # The exported commands fully determine the flight, including its attitude.
    replay = rollout(RocketLandingPlant(), states[0], controls, summary["dt_s"])
    assert_allclose(replay, states)


def test_maneuver_thrust_causes_the_lateral_motion_rotation_and_braking():
    states, controls, summary = demo.run_demo("maneuver")
    plant = RocketLandingPlant()
    axial_only = controls.copy()
    axial_only[:, :2] = 0.0
    unsteered = rollout(plant, states[0], axial_only, summary["dt_s"])
    assert_allclose(unsteered[:, :2], 0.0, atol=1e-12)
    assert_allclose(unsteered[:, 7:13], 0.0, atol=1e-12)
    assert np.linalg.norm(states[-1, :2]) > 5.0
    assert np.max(np.abs(states[:, 7:10])) > 0.02

    # Compare the first second only, while both flights are still above ground.
    steps = round(1.0 / summary["dt_s"])
    unpowered = rollout(plant, states[0], np.zeros_like(controls[:steps]), summary["dt_s"])
    assert np.all(unpowered[:, 2] > 0.0)
    assert states[steps, 5] > unpowered[-1, 5] + 5.0


def test_maneuver_summary_describes_every_held_thrust_command():
    _, controls, summary = demo.run_demo("maneuver")
    assert "thrust_body_N" not in summary  # This flight has no single constant input.
    reconstructed = np.zeros_like(controls)
    previous_end = 0
    for phase in summary["thrust_schedule"]:
        start = round(phase["start_s"] / summary["dt_s"])
        end = round(phase["end_s"] / summary["dt_s"])
        assert start == previous_end
        assert end > start
        reconstructed[start:end] = phase["thrust_body_N"]
        previous_end = end
    assert previous_end == len(controls)
    assert_allclose(reconstructed, controls)


def test_invalid_scenario_is_rejected_by_api_and_cli(capsys):
    with pytest.raises(ValueError, match="scenario"):
        demo.run_demo("unknown")
    with pytest.raises(SystemExit) as exit_info:
        demo.main(["--scenario", "unknown"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_linearization_export_matches_the_saved_short_rollout(tmp_path, monkeypatch):
    states, controls, summary = demo.run_demo("braking")
    states, controls = states[:3], controls[:2]
    summary.update(
        steps=2,
        duration_s=2 * summary["dt_s"],
        final_state=states[-1].tolist(),
        minimum_altitude_m=float(states[:, 2].min()),
    )
    monkeypatch.setattr(demo, "run_demo", lambda scenario: (states, controls, summary))

    assert demo.main(["--linearize", "--output-dir", str(tmp_path)]) == 0

    with np.load(tmp_path / "linearizations.npz", allow_pickle=False) as saved:
        assert saved["A"].shape == (2, 13, 13)
        assert saved["B"].shape == (2, 13, 3)
        assert np.all(np.isfinite(saved["A"]))
        assert np.all(np.isfinite(saved["B"]))
        assert_allclose(saved["states"], states)
        assert_allclose(saved["controls"], controls)
        assert float(saved["dt"]) == pytest.approx(summary["dt_s"])
        a, b = RocketLandingPlant().linearize_discrete(states[0], controls[0], summary["dt_s"])
        assert_allclose(saved["A"][0], a)
        assert_allclose(saved["B"][0], b)
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["linearization_shapes"] == {"A": [2, 13, 13], "B": [2, 13, 3]}


def test_plot_option_saves_a_headless_png_and_closes_its_figure(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    existing_figures = plt.get_fignums()
    output = tmp_path / "nested" / "trajectory.png"

    assert demo.main(["--plot", str(output)]) == 0

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert matplotlib.get_backend().lower() == "agg"
    assert plt.get_fignums() == existing_figures


def test_plot_option_explains_a_missing_plot_dependency(tmp_path, monkeypatch, capsys):
    original_import = builtins.__import__

    def import_without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)
    with pytest.raises(SystemExit) as exit_info:
        demo.main(["--plot", str(tmp_path / "plot.png")])
    assert exit_info.value.code == 2
    assert 'python -m pip install ".[plot]"' in capsys.readouterr().err
