import matplotlib.pyplot as plt
import numpy as np

def plot_ilqr(xs, us, dt):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9))

    M = xs.shape[0]
    N = xs.shape[1] - 1
    t = np.arange(N + 1) * dt

    for i in range(M):
        # color gradient: black (i=0) -> red (i=M-1)
        frac = i / (M - 1) if M > 1 else 1.0
        color = (frac, 0, 0)

        ax1.plot(t, xs[i, :, 0], color=color)
        ax2.plot(t, xs[i, :, 1], color=color)
        ax3.plot(t, us[i, :, 0], color=color)

    ax1.set_title(f'Position over iterations (n={M-1})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (rad)')

    ax2.set_title(f'Velocity over iterations (n={M-1})')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (rad/s)')

    ax3.set_title(f'Control over iterations (n={M-1})')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Control (rad/s^2)')

    ax4.axis('off')  # unused subplot

    plt.tight_layout()
    plt.show()


def plot_alilqr(xs, us, dt, info, x_ref=None, min_position=None,
                save_path=None, show=True):
    """Plot the histories returned by alilqr(..., record_history=True).

    xs has shape (updates + 1, N + 1, 2); us has shape (updates + 1, N, 1).
    Index zero is the initial rollout, followed by accepted iLQR updates.
    The original cost J need not decrease as constraints are enforced. Each
    augmented-cost segment uses fixed lambda and mu; segments are deliberately
    separate because the objective changes at the outer AL updates.

    Returns (fig, axes). Use show=False for saving or headless execution.
    """
    xs = np.asarray(xs, dtype=float)
    us = np.asarray(us, dtype=float)
    if xs.ndim != 3 or xs.shape[2] != 2 or xs.shape[0] < 1 or xs.shape[1] < 2:
        raise ValueError("xs must have shape (updates + 1, N + 1, 2), with N >= 1.")
    expected_u_shape = (xs.shape[0], xs.shape[1] - 1, 1)
    if us.shape != expected_u_shape:
        raise ValueError(f"us must have shape {expected_u_shape}; no terminal control.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not all(np.all(np.isfinite(a)) for a in (xs, us)):
        raise ValueError("The trajectory histories must be finite.")
    costs = np.asarray(info.get("cost_history", []), dtype=float)
    if costs.shape != (len(xs),):
        raise ValueError("Pass xs, us, and info from the same recorded alilqr solve.")
    if x_ref is None:
        x_ref = info.get("x_ref")
    if x_ref is not None:
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape != (2,):
            raise ValueError("x_ref must have shape (2,).")
    if min_position is None:
        min_position = info.get("min_position", 0.0)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    (ax_angle, ax_cost), (ax_velocity, ax_violation), (ax_control, ax_optimality) = axes
    t = np.arange(xs.shape[1]) * dt
    n_updates = len(xs) - 1
    steps = np.arange(n_updates + 1)

    # Intermediate trajectories first, then initial/final so both remain visible.
    draw_order = list(range(1, len(xs) - 1)) + [0]
    if len(xs) > 1:
        draw_order.append(len(xs) - 1)
    for i in draw_order:
        frac = i / n_updates if n_updates > 0 else 1.0
        color = (frac, 0.0, 0.0)
        final = i == n_updates
        initial = i == 0 and not final
        label = "Final" if final else "Initial" if initial else None
        alpha = 1.0 if final else 0.8 if initial else 0.35
        width = 2.3 if final else 1.5 if initial else 0.8
        ax_angle.plot(t, xs[i, :, 0], color=color, alpha=alpha,
                      linewidth=width, label=label)
        ax_velocity.plot(t, xs[i, :, 1], color=color, alpha=alpha,
                         linewidth=width, label=label)
        # N controls are held on N intervals, whose edges are the N+1 state times.
        ax_control.stairs(us[i, :, 0], t, color=color, alpha=alpha,
                          linewidth=width, label=label)

    bound_label = "Angle lower bound"
    if x_ref is not None:
        if np.isclose(x_ref[0], min_position):
            bound_label = "Target / lower bound"
        else:
            ax_angle.axhline(x_ref[0], color="0.4", linestyle="--", label="Target")
        ax_velocity.axhline(x_ref[1], color="0.4", linestyle="--", label="Target")
    ax_angle.axhline(min_position, color="tab:blue", linestyle="--", label=bound_label)
    lower_limit = ax_angle.get_ylim()[0]
    if lower_limit < min_position:
        ax_angle.axhspan(lower_limit, min_position, color="tab:blue", alpha=0.06)

    ax_angle.set(title="Angle over accepted updates", xlabel="Time (s)", ylabel="Angle (rad)")
    ax_velocity.set(title="Angular velocity over accepted updates", xlabel="Time (s)",
                    ylabel="Angular velocity (rad/s)")
    ax_control.set(title="Control over accepted updates", xlabel="Time (s)",
                   ylabel=r"Control (rad/s$^2$)")
    for ax in (ax_angle, ax_velocity, ax_control):
        ax.set_xlim(t[0], t[-1])
        ax.legend(fontsize=9)

    ax_cost.plot(steps, costs, color="tab:blue", marker=".", linewidth=1.6,
                 label=r"Original cost $J$")
    all_costs = list(costs)
    augmented_label_used = False
    outer_history = info.get("history", [])
    for outer in outer_history:
        accepted = [row for row in outer["inner_history"] if row["accepted"]]
        if not accepted:
            continue
        segment_cost = [accepted[0]["cost_before"]] + [row["cost_after"] for row in accepted]
        segment_steps = outer["step_start"] + np.arange(len(segment_cost))
        label = r"Augmented cost $\mathcal{L}_A$ (each inner solve)" if not augmented_label_used else None
        ax_cost.plot(segment_steps, segment_cost, color="tab:orange", marker=".",
                     linewidth=1.3, label=label)
        augmented_label_used = True
        all_costs.extend(segment_cost)
    if np.all(np.asarray(all_costs) > 0.0):
        ax_cost.set_yscale("log")
        cost_label = "Cost (log scale)"
    else:
        ax_cost.set_yscale("symlog", linthresh=1e-6)
        cost_label = "Cost (symlog scale)"
        if np.all(np.asarray(all_costs) >= 0.0):
            ax_cost.set_ylim(bottom=0.0)
    ax_cost.set(title="Cost convergence", xlabel="Accepted iLQR update", ylabel=cost_label)
    ax_cost.legend(fontsize=8)

    constraint_tol = info.get("constraint_tol", 1e-6)
    gradient_tol = info.get("gradient_tol", 1e-6)
    violations = np.maximum(min_position - xs[:, :, 0], 0.0).max(axis=1)
    ax_violation.plot(steps, violations, color="tab:purple", marker=".", linewidth=1.6)
    ax_violation.axhline(constraint_tol, color="0.4", linestyle="--", label="Feasibility tolerance")
    # symlog shows exact zeros without replacing them with an artificial floor.
    ax_violation.set_yscale("symlog", linthresh=constraint_tol)
    ax_violation.set_ylim(bottom=0.0)
    ax_violation.set(title="Maximum angle-constraint violation", xlabel="Accepted iLQR update",
                      ylabel="Violation (rad)")
    ax_violation.legend(fontsize=9)

    if outer_history:
        outer_steps = [row["outer_iteration"] for row in outer_history]
        ax_optimality.plot(outer_steps, [row["stationarity"] for row in outer_history],
                           "o-", color="tab:green", markersize=4, label="Stationarity")
        ax_optimality.plot(outer_steps, [row["complementarity"] for row in outer_history],
                           "s-", color="tab:red", markersize=4, label="Complementarity")
        if gradient_tol == constraint_tol:
            ax_optimality.axhline(gradient_tol, color="0.4", linestyle="--", label="Tolerance")
        else:
            ax_optimality.axhline(gradient_tol, color="tab:green", linestyle="--", label="Gradient tolerance")
            ax_optimality.axhline(constraint_tol, color="tab:red", linestyle="--", label="Complementarity tolerance")
        ax_optimality.set_yscale("symlog", linthresh=min(gradient_tol, constraint_tol) / 10)
        ax_optimality.set_ylim(bottom=0.0)
        ax_optimality.set_xticks(outer_steps)
        ax_optimality.legend(fontsize=9)
    ax_optimality.set(title="Optimality after multiplier updates", xlabel="Outer AL iteration",
                       ylabel="Residual norm")

    # A boundary means lambda/mu changed; augmented costs across it are different objectives.
    boundaries = sorted({row["step_start"] for row in outer_history[1:]})
    for boundary in boundaries:
        for ax in (ax_cost, ax_violation):
            ax.axvline(boundary, color="0.55", linestyle=":", alpha=0.45, linewidth=0.9)
    for ax in axes.flat:
        ax.grid(True, alpha=0.2)
    for ax in (ax_cost, ax_violation):
        ax.set_xlim(0, max(n_updates, 1))

    fig.suptitle("AL-iLQR: inverted pendulum convergence", fontsize=16, y=0.99)
    fig.text(0.5, 0.948,
             f"{info['status']}  |  {info['outer_iterations']} outer AL iterations  |  "
             f"{n_updates} accepted iLQR updates  |  "
             f"final violation {info['constraint_violation']:.2e} rad",
             ha="center", fontsize=10)
    fig.text(0.5, 0.016,
             "Black to red: initial to final trajectory. Dotted vertical lines: multiplier/penalty updates. "
             "Augmented-cost segments use fixed multipliers and penalties.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes