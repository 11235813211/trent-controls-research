"""Run a prescribed rocket descent maneuver; no controller or optimizer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from rocket_landing.attitude import rotation_matrix
from rocket_landing.dynamics import RocketLandingPlant, make_state
from rocket_landing.simulation import linearize_trajectory, rollout

SCENARIOS = ("maneuver", "braking", "hover", "coast")


def run_demo(scenario: str = "maneuver") -> tuple[np.ndarray, np.ndarray, dict]:
    """Propagate a fixed thrust schedule with the default atmospheric plant.

    Maneuver starts upright at 50 m and -10 m/s. Staggered transverse thrust
    pulses tip the vehicle in two axes; opposing pulses limit rotation while
    axial thrust slows the descent. A small, preselected final pulse reduces
    the angular rate induced by aerodynamics. The six-second segment ends
    above ground, still descending and translating, with modest tilt/rates.

    Every input is chosen before rollout; none depends on simulated state.
    These illustrative pulses are not a feedback law or a complete landing.
    Braking, hover, and coast retain their four-second constant-input cases.
    All scenarios use fixed 1000 kg mass and have no engine or contact states.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}.")
    plant = RocketLandingPlant()
    dt = 0.02
    gravity = plant.parameters.gravity
    if scenario == "maneuver":
        state = make_state(position_world=(0.0, 0.0, 50.0))
        weight = state[13] * gravity
        # (end time, phase name, body thrust in N). Steering at the tail also
        # creates a moment: r_T x T. Opposing pulses limit that rotation.
        phases = [
            (0.8, "X steering pulse", [400.0, 0.0, weight]),
            (1.6, "Reverse X / start Y steering", [-400.0, 280.0, weight + 1400.0]),
            (2.4, "Reverse Y steering", [0.0, -280.0, weight + 1400.0]),
            (4.5, "Axial braking", [0.0, 0.0, weight + 2000.0]),
            (6.0, "Small rate-trimming pulse", [80.0, 45.0, weight + 2000.0]),
        ]
    else:
        state = make_state(
            position_world=(0.0, 0.0, 150.0 if scenario == "coast" else 30.0),
            velocity_world=(0.0, 0.0, 0.0 if scenario == "hover" else -10.0),
        )
        acceleration = 2.0 if scenario == "braking" else 0.0
        thrust = state[13] * (gravity + acceleration) if scenario != "coast" else 0.0
        phases = [(4.0, scenario.capitalize(), [0.0, 0.0, thrust])]

    steps = round(phases[-1][0] / dt)
    controls = np.empty((steps, 3))
    schedule = []
    start = 0
    for end_s, name, thrust_body in phases:
        end = round(end_s / dt)
        controls[start:end] = thrust_body
        schedule.append(
            {"name": name, "start_s": start * dt, "end_s": end * dt, "thrust_body_N": thrust_body}
        )
        start = end
    states = rollout(plant, state, controls, dt)
    nose_vertical = np.array([rotation_matrix(q)[2, 2] for q in states[:, 6:10]])
    tilt = np.rad2deg(np.arccos(np.clip(nose_vertical, -1.0, 1.0)))
    angular_speed = np.rad2deg(np.linalg.norm(states[:, 10:13], axis=1))
    summary = {
        "scenario": scenario,
        "description": "Open-loop free-flight segment with a preset thrust schedule; no feedback or touchdown logic.",
        "dt_s": dt,
        "steps": steps,
        "duration_s": dt * steps,
        "initial_state": states[0].tolist(),
        "final_state": states[-1].tolist(),
        "thrust_schedule": schedule,
        "displacement_world_m": (states[-1, :3] - states[0, :3]).tolist(),
        "minimum_altitude_m": float(states[:, 2].min()),
        "maximum_tilt_deg": float(tilt.max()),
        "final_tilt_deg": float(tilt[-1]),
        "maximum_angular_speed_deg_s": float(angular_speed.max()),
        "final_angular_speed_deg_s": float(angular_speed[-1]),
        "maximum_quaternion_norm_error": float(
            np.abs(np.linalg.norm(states[:, 6:10], axis=1) - 1).max()
        ),
        "mass_depletion_enabled": False,
    }
    if len(phases) == 1:
        summary["thrust_body_N"] = controls[0].tolist()
    return states, controls, summary


def main(argv: list[str] | None = None) -> int:
    """Run the plant example and optionally save simulation/linearization artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="maneuver")
    parser.add_argument("--output-dir", type=Path, help="save summary.json and trajectory.csv")
    parser.add_argument("--plot", type=Path, help="save a simulation figure (requires plot extra)")
    parser.add_argument("--show", action="store_true", help="display the simulation figure")
    parser.add_argument(
        "--linearize", action="store_true", help="collect discrete A/B along the rollout"
    )
    args = parser.parse_args(argv)

    plot_function = None
    if args.plot is not None or args.show:
        try:
            import matplotlib

            if not args.show:
                matplotlib.use("Agg")
            from rocket_landing.plotting import plot_trajectory
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            parser.error('plotting requires: python -m pip install ".[plot]"')
        plot_function = plot_trajectory

    states, controls, summary = run_demo(args.scenario)
    dt = summary["dt_s"]
    A = B = None
    if args.linearize:
        A, B = linearize_trajectory(RocketLandingPlant(), states, controls, dt)
        summary["linearization_shapes"] = {"A": list(A.shape), "B": list(B.shape)}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        header = "time_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,qw,qx,qy,qz,wx_rad_s,wy_rad_s,wz_rad_s,mass_kg,Tx_N,Ty_N,Tz_N"
        with (args.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header.split(","))
            for k, state in enumerate(states):
                control = controls[k].tolist() if k < len(controls) else ["", "", ""]
                writer.writerow([k * dt, *state.tolist(), *control])
        if A is not None:
            np.savez_compressed(
                args.output_dir / "linearizations.npz",
                A=A,
                B=B,
                states=states,
                controls=controls,
                dt=dt,
            )
    if plot_function is not None:
        if args.plot is not None:
            args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure, _ = plot_function(states, controls, dt, save_path=args.plot, show=args.show)
        import matplotlib.pyplot as plt

        plt.close(figure)
    print(f"Scenario: {args.scenario} (open loop)")
    print(f"Final altitude: {states[-1, 2]:.6g} m")
    print(f"Final vertical velocity: {states[-1, 5]:.6g} m/s")
    print(
        f"Horizontal displacement: E {summary['displacement_world_m'][0]:.3f} m, N {summary['displacement_world_m'][1]:.3f} m"
    )
    print(
        f"Tilt: peak {summary['maximum_tilt_deg']:.3f} deg, final {summary['final_tilt_deg']:.3f} deg"
    )
    print(f"Final angular speed: {summary['final_angular_speed_deg_s']:.3f} deg/s")
    print(f"Maximum quaternion norm error: {summary['maximum_quaternion_norm_error']:.3e}")
    if A is not None:
        print(f"Local Jacobians: A{A.shape}, B{B.shape}")
    if args.output_dir is not None:
        print(f"Results: {args.output_dir}")
    if args.plot is not None:
        print(f"Figure: {args.plot}")
    return 0
