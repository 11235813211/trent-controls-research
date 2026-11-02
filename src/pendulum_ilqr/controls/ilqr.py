"""Iterative LQR and augmented-Lagrangian iLQR for a nonlinear pendulum.

Both solvers use nonlinear Euler rollouts, an iLQR backward pass, feedback
line search, and a control-gradient convergence check. The constrained solver
enforces ``min_position - x[k, 0] <= 0`` at all N + 1 states, including the
terminal state. The unconstrained solver uses the same numerical machinery
with zero constraint multipliers and penalties.

The discrete objective sums half-weighted state and input quadratics over N
controls, followed by a terminal state quadratic. Costs are not multiplied by
the sample period. These are local optimizers; inspect ``info['converged']``
and the returned residuals before treating a result as a solution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pendulum_ilqr.plants.pendulum import linearized_plant_step, nonlinear_plant

FloatArray = NDArray[np.float64]
SolverInfo = dict[str, Any]


def quadratic_cost_derivatives(
    W: FloatArray,
    R: FloatArray,
    x: FloatArray,
    u: FloatArray,
    x_ref: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Derivatives of ``0.5 * ((x-x_ref).T W (x-x_ref) + u.T R u)``."""
    l_x = W @ (x - x_ref)
    l_u = R @ u
    l_xx = W
    l_uu = R
    l_ux = np.zeros((u.size, x.size))
    return l_x, l_u, l_xx, l_uu, l_ux


def constraint_cost_derivatives(
    x: FloatArray, u: FloatArray, min_position: float = 0.0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the angle-bound residual and its state/control Jacobians."""
    # One constraint: c has one entry, and each Jacobian has one row.
    c = np.array([min_position - x[0]])
    c_x = np.array([[-1.0, 0.0]])
    c_u = np.zeros((1, u.size))
    return c, c_x, c_u


def penalty_matrix(c: FloatArray, lambda_k: FloatArray, mu_k: FloatArray) -> FloatArray:
    """Switch off a strictly satisfied bound with a zero multiplier."""
    weights = np.where((c < 0.0) & (lambda_k == 0.0), 0.0, mu_k)
    return np.diag(weights)


def init_total_cost_derivatives(
    x_N: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
    lambda_N: FloatArray,
    mu_N: FloatArray,
    min_position: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Initialize the terminal value gradient and Hessian."""
    c, c_x, _ = constraint_cost_derivatives(x_N, np.zeros(1), min_position)
    I_mu = penalty_matrix(c, lambda_N, mu_N)
    p_N = W_f @ (x_N - x_ref) + c_x.T @ (lambda_N + I_mu @ c)
    P_N = W_f + c_x.T @ I_mu @ c_x
    return p_N, P_N


def _make_positive_definite(A: FloatArray, eps: float = 1e-6) -> tuple[FloatArray, float]:
    """Symmetrize a square matrix and shift its smallest eigenvalue to eps."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square.")
    A_sym = 0.5 * (A + A.T)
    min_eig = np.min(np.linalg.eigvalsh(A_sym))
    correction = max(0.0, eps - min_eig)
    return A_sym + correction * np.eye(A.shape[0]), correction


def alilqr_step(
    x_k: FloatArray,
    u_k: FloatArray,
    p_k1: FloatArray,
    P_k1: FloatArray,
    lambda_k: FloatArray,
    mu_k: FloatArray,
    W: FloatArray,
    R: FloatArray,
    x_ref: FloatArray,
    dt: float,
    rho: float = 0.0,
    min_position: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Perform one backward step, returning feedback/feedforward gains and value."""
    l_x, l_u, l_xx, l_uu, l_ux = quadratic_cost_derivatives(W, R, x_k, u_k, x_ref)
    c, c_x, c_u = constraint_cost_derivatives(x_k, u_k, min_position)
    I_mu = penalty_matrix(c, lambda_k, mu_k)
    residual = lambda_k + I_mu @ c
    A_k, B_k = linearized_plant_step(x_k, u_k, dt)

    Q_x = l_x + A_k.T @ p_k1 + c_x.T @ residual
    Q_u = l_u + B_k.T @ p_k1 + c_u.T @ residual
    Q_xx = l_xx + A_k.T @ P_k1 @ A_k + c_x.T @ I_mu @ c_x
    Q_uu = l_uu + B_k.T @ P_k1 @ B_k + c_u.T @ I_mu @ c_u
    Q_ux = l_ux + B_k.T @ P_k1 @ A_k + c_u.T @ I_mu @ c_x
    Q_xu = Q_ux.T

    H_k, _ = _make_positive_definite(Q_uu + rho * np.eye(u_k.size))
    K_k = -np.linalg.solve(H_k, Q_ux)
    d_k = -np.linalg.solve(H_k, Q_u)

    # Use the unregularized Q_uu in the value function updates.
    P_k = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_ux + Q_xu @ K_k
    p_k = Q_x + K_k.T @ Q_uu @ d_k + K_k.T @ Q_u + Q_xu @ d_k
    P_k = 0.5 * (P_k + P_k.T)
    return K_k, d_k, P_k, p_k


def backward_pass(
    x_t: FloatArray,
    u_t: FloatArray,
    lambda_t: FloatArray,
    mu_t: FloatArray,
    W: FloatArray,
    R: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
    dt: float,
    rho: float = 0.0,
    min_position: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    """Build a local feedback policy for N controls and N + 1 states."""
    N, n_u = u_t.shape
    n_x = x_t.shape[1]
    K_t = np.empty((N, n_u, n_x))
    d_t = np.empty((N, n_u))
    p_k1, P_k1 = init_total_cost_derivatives(x_t[N], W_f, x_ref, lambda_t[N], mu_t[N], min_position)
    for k in range(N - 1, -1, -1):
        K_t[k], d_t[k], P_k1, p_k1 = alilqr_step(
            x_t[k], u_t[k], p_k1, P_k1, lambda_t[k], mu_t[k], W, R, x_ref, dt, rho, min_position
        )
    return K_t, d_t


def rollout(x_0: ArrayLike, u_t: ArrayLike, dt: float) -> FloatArray:
    """Roll out N scalar controls to N + 1 nonlinear states.

    Controls may have shape ``(N,)`` or ``(N, 1)``. An empty control sequence
    returns the initial state alone; the optimization APIs require N >= 1.
    """
    x_0 = np.asarray(x_0, dtype=float)
    u_t = np.asarray(u_t, dtype=float)
    if x_0.shape != (2,):
        raise ValueError("x_0 must have shape (2,).")
    if u_t.ndim == 1:
        u_t = u_t[:, None]
    if u_t.ndim != 2 or u_t.shape[1] != 1:
        raise ValueError("u_t must have shape (N,) or (N, 1).")
    if not np.all(np.isfinite(x_0)) or not np.all(np.isfinite(u_t)):
        raise ValueError("Initial state and controls must be finite.")
    _validate_scalar("dt", dt, lower=0.0, strict=True)
    x_t = np.empty((len(u_t) + 1, x_0.size))
    x_t[0] = x_0
    for k in range(len(u_t)):
        x_t[k + 1] = nonlinear_plant(x_t[k], u_t[k], dt)
    return x_t


def total_cost(
    x_t: FloatArray,
    u_t: FloatArray,
    W: FloatArray,
    R: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
) -> float:
    """Evaluate the unaugmented discrete tracking cost, including the terminal."""
    cost = 0.0
    for x, u in zip(x_t[:-1], u_t, strict=True):
        error = x - x_ref
        cost += 0.5 * (error @ W @ error + u @ R @ u)
    error_N = x_t[-1] - x_ref
    return float(cost + 0.5 * error_N @ W_f @ error_N)


def augmented_cost(
    x_t: FloatArray,
    u_t: FloatArray,
    lambda_t: FloatArray,
    mu_t: FloatArray,
    W: FloatArray,
    R: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
    min_position: float = 0.0,
) -> float:
    """Evaluate tracking cost plus the active augmented-Lagrangian terms.

    Inactive terms are skipped before arithmetic. In particular, an
    unconstrained solve must not square large, unpenalized angle residuals.
    """
    c_t = min_position - x_t[:, [0]]
    weights = np.where((c_t < 0.0) & (lambda_t == 0.0), 0.0, mu_t)
    active_multipliers = lambda_t != 0.0
    active_penalties = weights != 0.0
    linear_cost = np.sum(lambda_t[active_multipliers] * c_t[active_multipliers])
    # Scale before squaring: c**2 can overflow even when weight*c**2 is finite.
    scaled_residual = np.sqrt(weights[active_penalties]) * c_t[active_penalties]
    constraint_cost = linear_cost + (0.5 * scaled_residual) @ scaled_residual
    return total_cost(x_t, u_t, W, R, W_f, x_ref) + float(constraint_cost)


def _control_gradient(
    x_t: FloatArray,
    u_t: FloatArray,
    lambda_t: FloatArray,
    mu_t: FloatArray,
    W: FloatArray,
    R: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
    dt: float,
    min_position: float = 0.0,
) -> FloatArray:
    """Exact first derivative of the rollout's augmented cost w.r.t. controls.

    This adjoint recursion checks convergence independently of the regularized
    gains: a large rho can make d small even when the gradient is not small.
    """
    p_next, _ = init_total_cost_derivatives(
        x_t[-1], W_f, x_ref, lambda_t[-1], mu_t[-1], min_position
    )
    gradient = np.empty_like(u_t)
    for k in range(len(u_t) - 1, -1, -1):
        c, c_x, c_u = constraint_cost_derivatives(x_t[k], u_t[k], min_position)
        residual = lambda_t[k] + penalty_matrix(c, lambda_t[k], mu_t[k]) @ c
        A_k, B_k = linearized_plant_step(x_t[k], u_t[k], dt)
        gradient[k] = R @ u_t[k] + c_u.T @ residual + B_k.T @ p_next
        p_next = W @ (x_t[k] - x_ref) + c_x.T @ residual + A_k.T @ p_next
    return gradient


def forward_step(
    x_t: FloatArray,
    u_t: FloatArray,
    K_t: FloatArray,
    d_t: FloatArray,
    dt: float,
    alpha: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    """Roll out a feedback update; alpha scales only the feedforward step."""
    x_trial = np.empty_like(x_t)
    u_trial = np.empty_like(u_t)
    x_trial[0] = x_t[0]
    for k in range(len(u_t)):
        dx = x_trial[k] - x_t[k]
        u_trial[k] = u_t[k] + alpha * d_t[k] + K_t[k] @ dx
        x_trial[k + 1] = nonlinear_plant(x_trial[k], u_trial[k], dt)
    return x_trial, u_trial


def solve_inner_ilqr(
    x_t: FloatArray,
    u_t: FloatArray,
    lambda_t: FloatArray,
    mu_t: FloatArray,
    W: FloatArray,
    R: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
    dt: float,
    max_inner_iters: int = 100,
    gradient_tol: float = 1e-6,
    rho_init: float = 1e-6,
    max_line_search_steps: int = 25,
    min_position: float = 0.0,
    record_history: bool = False,
) -> tuple[FloatArray, FloatArray, SolverInfo]:
    """Optimize a fixed augmented objective with damped iLQR and line search.

    Array arguments are normalized by the public solvers. Failed line searches
    increase damping and retry at the same trajectory. Convergence uses the
    infinity norm of the exact rollout gradient, independently of damping.
    """
    _validate_iterations("max_inner_iters", max_inner_iters)
    _validate_iterations("max_line_search_steps", max_line_search_steps)
    _validate_scalar("gradient_tol", gradient_tol, lower=0.0, strict=True)
    _validate_scalar("rho_init", rho_init, lower=0.0)
    rho = rho_init
    history = []
    trajectory_history = []
    status = "max_inner_iterations"
    for iteration in range(max_inner_iters):
        with np.errstate(over="ignore", invalid="ignore"):
            cost = augmented_cost(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, min_position)
        if not np.isfinite(cost):
            raise ValueError("The initial trajectory produces a non-finite objective.")
        gradient = _control_gradient(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt, min_position)
        if np.max(np.abs(gradient)) <= gradient_tol:
            status = "converged"
            break

        K_t, d_t = backward_pass(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt, rho, min_position)
        accepted = False
        alpha = 1.0
        for _ in range(max_line_search_steps):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    x_trial, u_trial = forward_step(x_t, u_t, K_t, d_t, dt, alpha)
                    trial_cost = augmented_cost(
                        x_trial, u_trial, lambda_t, mu_t, W, R, W_f, x_ref, min_position
                    )
            except (ValueError, FloatingPointError):
                # A divergent trial can overflow before the final state. Public
                # plant validation then rejects that state; backtrack as usual.
                trial_cost = np.inf
            if np.isfinite(trial_cost) and trial_cost < cost:
                accepted = True
                break
            alpha *= 0.5

        history.append(
            {
                "iteration": iteration + 1,
                "cost_before": cost,
                "cost_after": trial_cost if accepted else cost,
                "alpha": alpha if accepted else 0.0,
                "rho": rho,
                "gradient_norm": float(np.max(np.abs(gradient))),
                "accepted": accepted,
            }
        )
        if accepted:
            x_t, u_t = x_trial, u_trial
            if record_history:
                trajectory_history.append((x_t.copy(), u_t.copy()))
            rho = max(rho_init, rho / 10.0)
        else:
            # Recompute the backward pass at the SAME trajectory with more damping.
            rho = max(1e-6, 10.0 * rho)
            if rho > 1e12:
                status = "line_search_failed"
                break

    # The last allowed step may itself have reached the tolerance.
    gradient_norm = float(
        np.max(
            np.abs(_control_gradient(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt, min_position))
        )
    )
    if gradient_norm <= gradient_tol:
        status = "converged"
    return (
        x_t,
        u_t,
        {
            "converged": status == "converged",
            "status": status,
            "gradient_norm": gradient_norm,
            "history": history,
            "trajectory_history": trajectory_history,
        },
    )


def _validate_iterations(name: str, value: int) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_scalar(
    name: str,
    value: float,
    *,
    lower: float | None = None,
    strict: bool = False,
) -> None:
    """Validate numeric scalar settings before any numerical work."""
    try:
        finite = np.ndim(value) == 0 and bool(np.isfinite(value))
    except TypeError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite scalar.")
    if lower is not None and (value <= lower if strict else value < lower):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must satisfy {relation} {lower}.")


def _validate_problem(
    x_0: ArrayLike,
    u_t: ArrayLike,
    W: ArrayLike,
    R: ArrayLike,
    x_ref: ArrayLike,
    dt: float,
    W_f: ArrayLike | None,
    max_inner_iters: int,
    gradient_tol: float,
    rho_init: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Copy inputs, check dimensions/weights, and validate shared settings."""
    x_0 = np.asarray(x_0, dtype=float).copy()
    x_ref = np.asarray(x_ref, dtype=float).copy()
    u_t = np.asarray(u_t, dtype=float).copy()
    if u_t.ndim == 1:
        u_t = u_t[:, None]
    W = np.asarray(W, dtype=float).copy()
    R = np.asarray(R, dtype=float).copy()
    W_f = W.copy() if W_f is None else np.asarray(W_f, dtype=float).copy()
    if x_0.shape != (2,) or x_ref.shape != (2,):
        raise ValueError("x_0 and x_ref must each have shape (2,).")
    if u_t.ndim != 2 or u_t.shape[1] != 1 or len(u_t) == 0:
        raise ValueError("u_t must contain N >= 1 controls, with shape (N, 1).")
    weights = []
    for name, weight, shape in (("W", W, (2, 2)), ("R", R, (1, 1)), ("W_f", W_f, (2, 2))):
        if weight.shape != shape or not np.all(np.isfinite(weight)):
            raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
        if not np.allclose(weight, weight.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} must be symmetric.")
        # Normalize roundoff-level asymmetry so cost and analytic derivatives
        # use exactly the same quadratic form.
        weight = 0.5 * (weight + weight.T)
        minimum = np.min(np.linalg.eigvalsh(weight))
        if minimum < -1e-12 or (name == "R" and minimum <= 0.0):
            raise ValueError("W and W_f must be positive semidefinite; R positive definite.")
        weights.append(weight)
    W, R, W_f = weights
    if not all(np.all(np.isfinite(a)) for a in (x_0, x_ref, u_t)):
        raise ValueError("States, reference, and controls must be finite.")
    _validate_scalar("dt", dt, lower=0.0, strict=True)
    _validate_scalar("gradient_tol", gradient_tol, lower=0.0, strict=True)
    _validate_scalar("rho_init", rho_init, lower=0.0)
    _validate_iterations("max_inner_iters", max_inner_iters)
    return x_0, u_t, W, R, x_ref, W_f


def _initial_rollout(x_0: FloatArray, u_t: FloatArray, dt: float) -> FloatArray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            x_t = rollout(x_0, u_t, dt)
    except (ValueError, FloatingPointError) as exc:
        raise ValueError("The initial controls produce a non-finite trajectory.") from exc
    if not np.all(np.isfinite(x_t)):
        raise ValueError("The initial controls produce a non-finite trajectory.")
    return x_t


def _initial_cost(
    x_t: FloatArray,
    u_t: FloatArray,
    W: FloatArray,
    R: FloatArray,
    W_f: FloatArray,
    x_ref: FloatArray,
) -> float:
    """Reject an overflowing initial objective before recording or optimizing it."""
    with np.errstate(over="ignore", invalid="ignore"):
        cost = total_cost(x_t, u_t, W, R, W_f, x_ref)
    if not np.isfinite(cost):
        raise ValueError("The initial trajectory produces a non-finite objective.")
    return cost


def alilqr(
    x_0: ArrayLike,
    u_t: ArrayLike,
    W: ArrayLike,
    R: ArrayLike,
    x_ref: ArrayLike,
    dt: float,
    W_f: ArrayLike | None = None,
    max_outer_iters: int = 20,
    max_inner_iters: int = 100,
    mu_init: float = 10.0,
    phi: float = 5.0,
    constraint_tol: float = 1e-6,
    gradient_tol: float = 1e-6,
    rho_init: float = 1e-6,
    min_position: float = 0.0,
    record_history: bool = True,
) -> tuple[FloatArray, FloatArray, SolverInfo]:
    """Solve an angle-constrained problem by augmented-Lagrangian iLQR.

    Parameters
    ----------
    x_0, x_ref : arrays of shape (2,)
    u_t : initial controls, shape (N, 1), or a length-N vector
    W, R : running-cost weights, shapes (2, 2) and (1, 1)
    W_f : terminal weight, shape (2, 2); defaults to W
    min_position : bound x[k, 0] >= min_position; use a positive value for a margin
        Despite the historical name, this is an angle bound, in radians.
    max_outer_iters, max_inner_iters : positive integers
        Augmented-Lagrangian iterations and iLQR iterations per outer solve.
    mu_init, phi : floats
        Initial positive penalty and its growth factor (greater than one).
    constraint_tol, gradient_tol : positive floats
        Feasibility/complementarity tolerance and reduced-gradient tolerance.
    rho_init : nonnegative float
        Initial backward-pass damping; failed line searches increase it.
    record_history : save the initial trajectory and every accepted inner update

    Returns
    -------
    x_t : nonlinear state trajectory, shape (N + 1, 2)
    u_t : optimized controls, shape (N, 1)
    info : convergence, multipliers, penalties, residuals, and iteration history
        When record_history=True, info['xs'] and info['us'] contain trajectory
        histories; info['cost_history'] holds the original tracking cost.

    This is a local optimizer. A returned trajectory satisfies the requested
    tolerances only when info['converged'] is True.
    """
    x_0, u_t, W, R, x_ref, W_f = _validate_problem(
        x_0, u_t, W, R, x_ref, dt, W_f, max_inner_iters, gradient_tol, rho_init
    )
    _validate_iterations("max_outer_iters", max_outer_iters)
    _validate_scalar("mu_init", mu_init, lower=0.0, strict=True)
    _validate_scalar("phi", phi, lower=1.0, strict=True)
    _validate_scalar("constraint_tol", constraint_tol, lower=0.0, strict=True)
    _validate_scalar("min_position", min_position)
    if x_0[0] < min_position - constraint_tol:
        raise ValueError("The fixed initial position already violates the constraint.")

    N = len(u_t)
    x_t = _initial_rollout(x_0, u_t, dt)
    initial_cost = _initial_cost(x_t, u_t, W, R, W_f, x_ref)
    lambda_t = np.zeros((N + 1, 1))
    mu_t = np.full((N + 1, 1), mu_init, dtype=float)
    history = []
    converged = False
    accepted_steps = 0
    inner_iterations = 0
    xs = [x_t.copy()] if record_history else []
    us = [u_t.copy()] if record_history else []
    costs = [initial_cost] if record_history else []

    for outer in range(max_outer_iters):
        step_start = accepted_steps
        # Always optimize, even if the initial trajectory is already feasible.
        x_t, u_t, inner = solve_inner_ilqr(
            x_t,
            u_t,
            lambda_t,
            mu_t,
            W,
            R,
            W_f,
            x_ref,
            dt,
            max_inner_iters,
            gradient_tol,
            rho_init,
            min_position=min_position,
            record_history=record_history,
        )
        accepted_steps += sum(row["accepted"] for row in inner["history"])
        inner_iterations += len(inner["history"])
        if record_history:
            for x_saved, u_saved in inner["trajectory_history"]:
                xs.append(x_saved)
                us.append(u_saved)
                costs.append(total_cost(x_saved, u_saved, W, R, W_f, x_ref))
        c_t = min_position - x_t[:, [0]]
        violation = float(np.max(np.maximum(c_t, 0.0)))

        # Projection keeps inequality multipliers nonnegative.
        lambda_t = np.maximum(0.0, lambda_t + mu_t * c_t)

        # Check the ORIGINAL constrained problem after the multiplier update.
        # Setting the penalty weights to zero leaves ell + lambda^T c.
        lagrangian_gradient = _control_gradient(
            x_t, u_t, lambda_t, np.zeros_like(mu_t), W, R, W_f, x_ref, dt, min_position
        )
        stationarity = float(np.max(np.abs(lagrangian_gradient)))
        complementarity = float(np.max(np.abs(lambda_t * c_t)))
        history.append(
            {
                "outer_iteration": outer + 1,
                "cost": total_cost(x_t, u_t, W, R, W_f, x_ref),
                "constraint_violation": violation,
                "stationarity": stationarity,
                "complementarity": complementarity,
                "mu": float(np.max(mu_t)),
                "inner_converged": inner["converged"],
                "inner_status": inner["status"],
                "inner_history": inner["history"],
                "inner_gradient_norm": inner["gradient_norm"],
                "step_start": step_start,
                "step_end": accepted_steps,
            }
        )
        converged = (
            violation <= constraint_tol
            and stationarity <= gradient_tol
            and complementarity <= constraint_tol
        )
        if converged:
            break
        if outer + 1 < max_outer_iters:
            # Reuse the current trajectory and controls for the next inner solve.
            mu_t *= phi

    info = {
        "method": "alilqr",
        "converged": converged,
        "status": "converged" if converged else "max_outer_iterations",
        "outer_iterations": len(history),
        "inner_iterations": inner_iterations,
        "cost": total_cost(x_t, u_t, W, R, W_f, x_ref),
        "augmented_cost": augmented_cost(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, min_position),
        "constraint_violation": violation,
        "stationarity": stationarity,
        "gradient_norm": stationarity,
        "complementarity": complementarity,
        "lambda_t": lambda_t.copy(),
        "mu_t": mu_t.copy(),
        "history": history,
        "accepted_steps": accepted_steps,
        "constraint_tol": constraint_tol,
        "gradient_tol": gradient_tol,
        "min_position": min_position,
        "x_ref": x_ref.copy(),
        "dt": dt,
    }
    if record_history:
        info["xs"] = np.stack(xs)
        info["us"] = np.stack(us)
        info["cost_history"] = np.asarray(costs)
    return x_t, u_t, info


def ilqr(
    x_0: ArrayLike,
    u_t: ArrayLike,
    W: ArrayLike,
    R: ArrayLike,
    x_ref: ArrayLike,
    dt: float,
    W_f: ArrayLike | None = None,
    max_inner_iters: int = 100,
    gradient_tol: float = 1e-6,
    rho_init: float = 1e-6,
    record_history: bool = True,
) -> tuple[FloatArray, FloatArray, SolverInfo]:
    """Minimize the unconstrained nonlinear tracking cost with iterative LQR.

    Parameters
    ----------
    x_0, x_ref : array-like, shape (2,)
        Initial state and fixed reference ``[angle, angular_velocity]``.
    u_t : array-like, shape (N, 1) or (N,)
        Initial control sequence, with N >= 1 angular-acceleration inputs.
    W, R : array-like, shapes (2, 2) and (1, 1)
        Symmetric running-cost weights. W is positive semidefinite and R is
        positive definite. The objective uses half-weighted quadratics.
    dt : positive float
        Sample period in seconds for explicit Euler integration.
    W_f : array-like, shape (2, 2), optional
        Positive semidefinite terminal weight; defaults to W.
    max_inner_iters : positive integer
        Maximum number of iLQR updates, including rejected line searches.
    gradient_tol : positive float
        Convergence threshold for the control-gradient infinity norm.
    rho_init : nonnegative float
        Initial damping applied in the backward pass.
    record_history : bool
        Store the initial trajectory and every accepted update.

    Returns
    -------
    x_t, u_t, info : tuple
        States of shape ``(N + 1, 2)``, controls of shape ``(N, 1)``, and
        convergence diagnostics. When requested, ``info['xs']`` and
        ``info['us']`` contain trajectory histories and ``cost_history``
        contains their objective values. ``history`` uses the same outer/inner
        structure as :func:`alilqr`, with one unconstrained inner solve.

    Notes
    -----
    No state or control bounds are enforced. In particular, negative angles
    are allowed at the initial and subsequent states. The bound diagnostics
    and multiplier arrays in ``info`` are zero for interface compatibility.
    """
    x_0, u_t, W, R, x_ref, W_f = _validate_problem(
        x_0, u_t, W, R, x_ref, dt, W_f, max_inner_iters, gradient_tol, rho_init
    )
    x_t = _initial_rollout(x_0, u_t, dt)
    initial_cost = _initial_cost(x_t, u_t, W, R, W_f, x_ref)
    lambda_t = np.zeros((len(u_t) + 1, 1))
    mu_t = np.zeros_like(lambda_t)
    xs = [x_t.copy()] if record_history else []
    us = [u_t.copy()] if record_history else []
    costs = [initial_cost] if record_history else []

    x_t, u_t, inner = solve_inner_ilqr(
        x_t,
        u_t,
        lambda_t,
        mu_t,
        W,
        R,
        W_f,
        x_ref,
        dt,
        max_inner_iters=max_inner_iters,
        gradient_tol=gradient_tol,
        rho_init=rho_init,
        record_history=record_history,
    )
    accepted_steps = sum(row["accepted"] for row in inner["history"])
    cost = total_cost(x_t, u_t, W, R, W_f, x_ref)
    if record_history:
        for x_saved, u_saved in inner["trajectory_history"]:
            xs.append(x_saved)
            us.append(u_saved)
            costs.append(total_cost(x_saved, u_saved, W, R, W_f, x_ref))

    info = {
        "method": "ilqr",
        "converged": inner["converged"],
        "status": inner["status"],
        "outer_iterations": 1,
        "inner_iterations": len(inner["history"]),
        "cost": cost,
        "augmented_cost": cost,
        "constraint_violation": 0.0,
        "stationarity": inner["gradient_norm"],
        "gradient_norm": inner["gradient_norm"],
        "complementarity": 0.0,
        "lambda_t": lambda_t,
        "mu_t": mu_t,
        "history": [
            {
                "outer_iteration": 1,
                "cost": cost,
                "constraint_violation": 0.0,
                "stationarity": inner["gradient_norm"],
                "complementarity": 0.0,
                "mu": 0.0,
                "inner_converged": inner["converged"],
                "inner_status": inner["status"],
                "inner_history": inner["history"],
                "inner_gradient_norm": inner["gradient_norm"],
                "step_start": 0,
                "step_end": accepted_steps,
            }
        ],
        "accepted_steps": accepted_steps,
        "constraint_tol": 0.0,
        "gradient_tol": gradient_tol,
        "min_position": None,
        "x_ref": x_ref.copy(),
        "dt": dt,
    }
    if record_history:
        info["xs"] = np.stack(xs)
        info["us"] = np.stack(us)
        info["cost_history"] = np.asarray(costs)
    return x_t, u_t, info
