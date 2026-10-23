"""The command-line example produces portable, finite results by default."""

import builtins
import csv
import json

import numpy as np
import pytest

from pendulum_ilqr import demo
from pendulum_ilqr.solver import alilqr


def _reject_nonfinite_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


@pytest.mark.parametrize("method", ["ilqr", "alilqr"])
def test_cli_exports_finite_results_without_loading_plotting(method, tmp_path, monkeypatch, capsys):
    output = tmp_path / "nested" / "results"
    original_import = builtins.__import__

    def import_without_plotting(name, *args, **kwargs):
        if name.startswith("matplotlib") or name == "pendulum_ilqr.plotting":
            pytest.fail("A default CLI run must not import plotting or require a display.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_plotting)
    assert demo.main(["--method", method, "--output-dir", str(output)]) == 0
    assert "Status: converged" in capsys.readouterr().out

    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_constant,
    )
    history = json.loads(
        (output / "iterations.json").read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_constant,
    )
    assert summary["method"] == method
    assert summary["result"]["status"] == "converged"
    assert summary["result"]["converged"] is True
    assert summary["result"]["final_cost"] < summary["result"]["initial_cost"]
    assert history

    with (output / "trajectory.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == [
            "time_s",
            "angle_rad",
            "angular_velocity_rad_s",
            "control_rad_s2",
        ]
    steps, dt = summary["problem"]["steps"], summary["problem"]["dt_s"]
    assert len(rows) == steps + 1
    assert rows[-1]["control_rad_s2"] == ""
    assert all(row["control_rad_s2"] != "" for row in rows[:-1])
    assert float(rows[-1]["time_s"]) == pytest.approx(steps * dt)
    states = np.array(
        [[float(row["angle_rad"]), float(row["angular_velocity_rad_s"])] for row in rows]
    )
    assert np.isfinite(states).all()
    assert states[-1].tolist() == pytest.approx(summary["result"]["terminal_state"])


def test_invalid_method_is_rejected_by_python_api_and_cli(capsys):
    with pytest.raises(ValueError, match="method"):
        demo.run_demo("unknown")
    with pytest.raises(SystemExit) as exit_info:
        demo.main(["--method", "unknown"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_demo_problem_converges_with_an_integer_initial_penalty():
    states, controls, info = alilqr(**demo.demo_problem(), mu_init=10, record_history=False)
    assert info["converged"], info["status"]
    assert info["outer_iterations"] > 1  # Exercise penalty growth after the first solve.
    assert np.issubdtype(info["mu_t"].dtype, np.floating)
    assert np.isfinite(states).all()
    assert np.isfinite(controls).all()
    assert info["constraint_violation"] <= info["constraint_tol"]
