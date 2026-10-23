"""Reproducible pendulum example and command-line exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pendulum_ilqr.solver import alilqr, ilqr, total_cost


def demo_problem() -> dict:
    """Return fresh inputs for a 1.2 s braking maneuver near the angle bound."""
    return {
        "x_0": np.array([0.2, -2.0]),
        "u_t": np.zeros((30, 1)),
        "W": np.diag([20.0, 1.0]),
        "R": np.array([[0.02]]),
        "W_f": np.diag([100.0, 10.0]),
        "x_ref": np.zeros(2),
        "dt": 0.04,
    }


def run_demo(method: str = "alilqr") -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Solve the documented problem and return trajectories, info, and a summary."""
    if method not in {"alilqr", "ilqr"}:
        raise ValueError("method must be 'alilqr' or 'ilqr'.")
    problem = demo_problem()
    solve = alilqr if method == "alilqr" else ilqr
    states, controls, info = solve(**problem)
    initial_cost = total_cost(
        info["xs"][0],
        info["us"][0],
        problem["W"],
        problem["R"],
        problem["W_f"],
        problem["x_ref"],
    )
    summary = {
        "method": method,
        "problem": {
            "steps": len(controls),
            "dt_s": problem["dt"],
            "initial_state": problem["x_0"].tolist(),
            "reference_state": problem["x_ref"].tolist(),
            "state_weights": problem["W"].tolist(),
            "control_weights": problem["R"].tolist(),
            "terminal_weights": problem["W_f"].tolist(),
            "angle_lower_bound_rad": 0.0 if method == "alilqr" else None,
        },
        "result": {
            "status": info["status"],
            "converged": bool(info["converged"]),
            "initial_cost": initial_cost,
            "final_cost": info["cost"],
            "minimum_angle_rad": float(states[:, 0].min()),
            "violation_of_zero_angle_bound_rad": float(np.maximum(-states[:, 0], 0).max()),
            "stationarity": info["stationarity"],
            "complementarity": info["complementarity"],
            "accepted_steps": info["accepted_steps"],
            "peak_control_rad_s2": float(np.abs(controls).max()),
            "terminal_state": states[-1].tolist(),
        },
    }
    if method == "alilqr":
        summary["result"]["outer_iterations"] = info["outer_iterations"]
    return states, controls, info, summary


def _export(
    directory: Path, states: np.ndarray, controls: np.ndarray, info: dict, summary: dict
) -> None:
    """Save structured diagnostics and one row per state time; terminal u is blank."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (directory / "iterations.json").write_text(
        json.dumps(info["history"], indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (directory / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "angle_rad", "angular_velocity_rad_s", "control_rad_s2"])
        for k, state in enumerate(states):
            control = float(controls[k, 0]) if k < len(controls) else ""
            writer.writerow([k * info["dt"], *state.tolist(), control])


def main(argv: list[str] | None = None) -> int:
    """Run a deterministic example; return nonzero if the solver does not converge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("alilqr", "ilqr"), default="alilqr")
    parser.add_argument("--output-dir", type=Path, help="save CSV and JSON results here")
    parser.add_argument("--plot", type=Path, help="save a convergence figure (requires plot extra)")
    parser.add_argument(
        "--show", action="store_true", help="open an interactive convergence figure"
    )
    args = parser.parse_args(argv)

    plot_function = None
    if args.plot is not None or args.show:
        try:
            import matplotlib

            if not args.show:
                matplotlib.use("Agg")
            from pendulum_ilqr.plotting import plot_alilqr, plot_ilqr
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            parser.error('plotting requires: python -m pip install ".[plot]"')
        plot_function = plot_alilqr if args.method == "alilqr" else plot_ilqr

    states, controls, info, summary = run_demo(args.method)
    result = summary["result"]
    print(f"Method: {args.method}")
    print(f"Status: {result['status']}")
    print(f"Cost: {result['initial_cost']:.6g} -> {result['final_cost']:.6g}")
    print(f"Minimum angle: {result['minimum_angle_rad']:.6g} rad")
    print(f"Stationarity residual: {result['stationarity']:.3e}")
    if args.method == "alilqr":
        print(f"Constraint violation: {info['constraint_violation']:.3e} rad")
        print(f"Complementarity residual: {result['complementarity']:.3e}")
    print(f"Accepted iLQR updates: {info['accepted_steps']}")
    if args.output_dir is not None:
        _export(args.output_dir, states, controls, info, summary)
        print(f"Results: {args.output_dir}")
    if plot_function is not None:
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_function(
            info["xs"], info["us"], info["dt"], info, save_path=args.plot, show=args.show
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        if args.plot is not None:
            print(f"Figure: {args.plot}")
    return 0 if info["converged"] else 1
