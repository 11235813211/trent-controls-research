"""Optional simulation figures; never imported by plant or linearization code."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from rocket_landing.attitude import rotation_matrix


def plot_trajectory(states, controls, dt, *, save_path=None, show=False):
    """Plot flight, thrust, and quaternion attitude; return 4-by-2 axes.

    The spatial view uses equal distance scales and sampled body +Z arrows.
    Scalar and vector quaternion components have separate panels so small
    rotations remain visible even when the scalar component stays close to one.
    """
    states, controls = np.asarray(states, dtype=float), np.asarray(controls, dtype=float)
    if states.ndim != 2 or states.shape[1] != 14 or len(states) < 2:
        raise ValueError("states must have shape (N+1,14), with N >= 1.")
    if controls.shape != (len(states) - 1, 3):
        raise ValueError("controls must have shape (N,3); no terminal control.")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise ValueError("Trajectories must be finite.")
    if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive scalar.")
    time = np.arange(len(states)) * dt
    body_axis_world = np.array([rotation_matrix(q).T[:, 2] for q in states[:, 6:10]])
    tilt = np.rad2deg(np.arccos(np.clip(body_axis_world[:, 2], -1, 1)))
    fig = plt.figure(figsize=(13, 15), layout="constrained")
    grid = fig.add_gridspec(4, 2)
    axes = np.empty((4, 2), dtype=object)
    for row, column in np.ndindex(axes.shape):
        axes[row, column] = fig.add_subplot(
            grid[row, column], projection="3d" if (row, column) == (0, 0) else None
        )
    (path, altitude), (velocity, thrust), (attitude, rates), (scalar, vector) = axes

    positions = states[:, :3]
    path.plot(*positions.T, color="tab:blue", label="Mass-center path")
    path.scatter(*positions[0], color="tab:green", s=35, label="Start")
    path.scatter(*positions[-1], color="tab:red", marker="s", s=35, label="End")
    samples = np.unique(np.linspace(0, len(states) - 1, 8, dtype=int))
    nose_length = 2.0
    path.quiver(
        *positions[samples].T,
        *body_axis_world[samples].T,
        length=nose_length,
        color="tab:orange",
        arrow_length_ratio=0.25,
        label="Nose direction (2 m arrows)",
    )
    # Include arrow tips in the bounds. Matching box aspect to the coordinate
    # spans keeps a meter the same length on every axis, including hover cases.
    geometry = np.vstack((positions, positions[samples] + nose_length * body_axis_world[samples]))
    lower, upper = geometry.min(axis=0), geometry.max(axis=0)
    span = max(float(np.max(upper - lower)), 2.0)
    padding = max(span * 0.08, 1.0)
    # Leave enough horizontal room to read the axes during purely vertical
    # flights, while retaining equal physical scales in the 3D projection.
    center = 0.5 * (lower + upper)
    extent = np.maximum(upper - lower + 2 * padding, 0.65 * span)
    lower, upper = center - extent / 2, center + extent / 2
    path.set_xlim(lower[0], upper[0])
    path.set_ylim(lower[1], upper[1])
    path.set_zlim(lower[2], upper[2])
    path.set_box_aspect(upper - lower)
    path.set(
        title="Flight path and attitude (equal distance scales)",
        xlabel="East (m)",
        ylabel="North (m)",
        zlabel="Altitude (m)",
    )
    path.view_init(elev=20, azim=-55)
    for axis in (path.xaxis, path.yaxis, path.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=3))
    path.tick_params(labelsize=8, pad=0)
    path.legend(loc="upper left", bbox_to_anchor=(-0.25, 1), fontsize="small")

    altitude.plot(time, states[:, 2], color="tab:blue")
    altitude.axhline(0, color="0.5", linestyle="--", label="Ground reference")
    altitude.set(title="Altitude", ylabel="Height (m)")
    altitude.legend()
    for j, name in enumerate(("East", "North", "Up")):
        velocity.plot(time, states[:, j + 3], label=name)
    velocity.set(title="World-frame velocity", ylabel="Velocity (m/s)")
    velocity.legend()
    # A separate labeled steering scale keeps small lateral pulses legible
    # beside the much larger axial thrust. Both show the actual held inputs.
    steering = thrust.twinx()
    thrust.stairs(controls[:, 2] / 1000, time, baseline=None, color="tab:green", label="Axial Z")
    for j, name in enumerate(("X", "Y")):
        steering.stairs(
            controls[:, j], time, baseline=None, label=f"Steering {name}", color=f"C{j}"
        )
    for j, name in enumerate(("X", "Y", "Z")):
        rates.plot(time, np.rad2deg(states[:, 10 + j]), label=name)
    thrust.set(title="Commanded body thrust", ylabel="Axial Z force (kN)")
    thrust.set_ylim(
        min(0, float(controls[:, 2].min()) * 0.0011), max(1, float(controls[:, 2].max()) * 0.0011)
    )
    thrust.yaxis.label.set_color("tab:green")
    thrust.tick_params(axis="y", colors="tab:green")
    steering.set_ylabel("Steering X/Y force (N)")
    steering_limit = max(float(np.abs(controls[:, :2]).max()) * 1.2, 1.0)
    steering.set_ylim(-steering_limit, steering_limit)
    rates.set(title="Body angular velocity", ylabel="Rate (deg/s)")
    handles, labels = thrust.get_legend_handles_labels()
    steering_handles, steering_labels = steering.get_legend_handles_labels()
    thrust.legend(handles + steering_handles, labels + steering_labels, loc="lower right")
    rates.legend()
    attitude.plot(time, tilt)
    attitude.set(title="Nose tilt from world vertical", ylabel="Tilt (deg)")

    scalar.plot(time, states[:, 6], color="tab:purple", label=r"$q_w$")
    scalar.set(title=r"Attitude quaternion: scalar $q_w$", ylabel="Component (unitless)")
    scalar.ticklabel_format(axis="y", useOffset=False)
    scalar.legend()
    for j, name in enumerate(("x", "y", "z")):
        vector.plot(time, states[:, 7 + j], label=rf"$q_{name}$")
    vector.set(
        title=r"Attitude quaternion: vector $[q_x, q_y, q_z]$", ylabel="Component (unitless)"
    )
    vector.legend()
    for ax in axes.flat[1:]:
        ax.set_xlabel("Time (s)")
        ax.set_xlim(time[0], time[-1])
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Rocket recovery: open-loop free-flight simulation\n"
        r"Attitude $q_{BW} = [q_w, q_x, q_y, q_z]$: scalar first, world to body"
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
