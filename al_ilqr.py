"""AL-iLQR for the two-state plant in inverted_pendulum.py.

The inequality is min_position - x[0] <= 0, including the terminal state.
There are N controls and N + 1 states. Costs track x_ref at every state.
The backward pass follows Howell's Eqs. (5)-(15); functions marked with
steps 5, 6, and 7 implement the forward pass and the two solve loops.
"""

import numpy as np

from inverted_pendulum import G, M, nonlinear_plant
from plotter import plot_alilqr


def linearized_plant_step(x, u, dt):
    """Jacobians of the nonlinear_plant supplied earlier in the conversation."""
    A = np.array([[1.0, dt], [M * G * np.cos(x[0]) * dt, 1.0]])
    B = np.array([[0.0], [dt]])
    return A, B


def quadratic_cost_derivatives(W, R, x, u, x_ref):
    l_x = W @ (x - x_ref)
    l_u = R @ u
    l_xx = W
    l_uu = R
    l_ux = np.zeros((u.size, x.size))
    return l_x, l_u, l_xx, l_uu, l_ux


def constraint_cost_derivatives(x, u, min_position=0.0):
    # One constraint: c has one entry, and each Jacobian has one row.
    c = np.array([min_position - x[0]])
    c_x = np.array([[-1.0, 0.0]])
    c_u = np.zeros((1, u.size))
    return c, c_x, c_u


def penalty_matrix(c, lambda_k, mu_k):
    """Howell Eq. (3): switch off a strictly satisfied, zero-multiplier bound."""
    weights = np.where((c < 0.0) & (lambda_k == 0.0), 0.0, mu_k)
    return np.diag(weights)


def init_total_cost_derivatives(x_N, W_f, x_ref, lambda_N, mu_N,
                                min_position=0.0):
    c, c_x, _ = constraint_cost_derivatives(x_N, np.zeros(1), min_position)
    I_mu = penalty_matrix(c, lambda_N, mu_N)
    p_N = W_f @ (x_N - x_ref) + c_x.T @ (lambda_N + I_mu @ c)
    P_N = W_f + c_x.T @ I_mu @ c_x
    return p_N, P_N


def _make_positive_definite(A, eps=1e-6):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix A must be square.")
    A_sym = 0.5 * (A + A.T)
    min_eig = np.min(np.linalg.eigvalsh(A_sym))
    correction = max(0.0, eps - min_eig)
    return A_sym + correction * np.eye(A.shape[0]), correction


def alilqr_step(x_k, u_k, p_k1, P_k1, lambda_k, mu_k, W, R,
                x_ref, dt, rho=0.0, min_position=0.0):
    """One BACKWARD step; return the policy, rather than evaluating it at x_ref."""
    l_x, l_u, l_xx, l_uu, l_ux = quadratic_cost_derivatives(
        W, R, x_k, u_k, x_ref
    )
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

    # Howell Eqs. (14)-(15), using the unregularized Q_uu in these updates.
    P_k = Q_xx + K_k.T @ Q_uu @ K_k + K_k.T @ Q_ux + Q_xu @ K_k
    p_k = Q_x + K_k.T @ Q_uu @ d_k + K_k.T @ Q_u + Q_xu @ d_k
    P_k = 0.5 * (P_k + P_k.T)
    return K_k, d_k, P_k, p_k


def backward_pass(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt,
                  rho=0.0, min_position=0.0):
    N, n_u = u_t.shape
    n_x = x_t.shape[1]
    K_t = np.empty((N, n_u, n_x))
    d_t = np.empty((N, n_u))
    p_k1, P_k1 = init_total_cost_derivatives(
        x_t[N], W_f, x_ref, lambda_t[N], mu_t[N], min_position
    )
    for k in range(N - 1, -1, -1):
        K_t[k], d_t[k], P_k1, p_k1 = alilqr_step(
            x_t[k], u_t[k], p_k1, P_k1, lambda_t[k], mu_t[k],
            W, R, x_ref, dt, rho, min_position
        )
    return K_t, d_t


def rollout(x_0, u_t, dt):
    x_t = np.empty((len(u_t) + 1, x_0.size))
    x_t[0] = x_0
    for k in range(len(u_t)):
        x_t[k + 1] = nonlinear_plant(x_t[k], u_t[k], dt)
    return x_t


def total_cost(x_t, u_t, W, R, W_f, x_ref):
    cost = 0.0
    for x, u in zip(x_t[:-1], u_t):
        error = x - x_ref
        cost += 0.5 * (error @ W @ error + u @ R @ u)
    error_N = x_t[-1] - x_ref
    return float(cost + 0.5 * error_N @ W_f @ error_N)


def augmented_cost(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref,
                   min_position=0.0):
    c_t = min_position - x_t[:, [0]]
    weights = np.where((c_t < 0.0) & (lambda_t == 0.0), 0.0, mu_t)
    constraint_cost = np.sum(lambda_t * c_t + 0.5 * weights * c_t**2)
    return total_cost(x_t, u_t, W, R, W_f, x_ref) + float(constraint_cost)


def _control_gradient(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt,
                      min_position=0.0):
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


def forward_step(x_t, u_t, K_t, d_t, dt, alpha=1.0):
    """Step 5: nonlinear rollout; scale only d, and measure dx from x_t."""
    x_trial = np.empty_like(x_t)
    u_trial = np.empty_like(u_t)
    x_trial[0] = x_t[0]
    for k in range(len(u_t)):
        dx = x_trial[k] - x_t[k]
        u_trial[k] = u_t[k] + alpha * d_t[k] + K_t[k] @ dx
        x_trial[k + 1] = nonlinear_plant(x_trial[k], u_trial[k], dt)
    return x_trial, u_trial


def solve_inner_ilqr(x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt,
                     max_inner_iters=100, gradient_tol=1e-6, rho_init=1e-6,
                     max_line_search_steps=25, min_position=0.0,
                     record_history=False):
    rho = rho_init
    history = []
    trajectory_history = []
    status = "max_inner_iterations"
    for iteration in range(max_inner_iters):
        gradient = _control_gradient(
            x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt, min_position
        )
        if np.max(np.abs(gradient)) <= gradient_tol:
            status = "converged"
            break

        K_t, d_t = backward_pass(
            x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt, rho, min_position
        )
        cost = augmented_cost(
            x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, min_position
        )
        accepted = False
        alpha = 1.0
        for _ in range(max_line_search_steps):
            with np.errstate(over="ignore", invalid="ignore"):
                x_trial, u_trial = forward_step(x_t, u_t, K_t, d_t, dt, alpha)
                trial_cost = augmented_cost(
                    x_trial, u_trial, lambda_t, mu_t, W, R, W_f, x_ref,
                    min_position
                )
            if np.isfinite(trial_cost) and trial_cost < cost:
                accepted = True
                break
            alpha *= 0.5

        history.append({
            "iteration": iteration + 1, "cost_before": cost,
            "cost_after": trial_cost if accepted else cost,
            "alpha": alpha if accepted else 0.0, "rho": rho,
            "gradient_norm": float(np.max(np.abs(gradient))),
            "accepted": accepted,
        })
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
    gradient_norm = float(np.max(np.abs(_control_gradient(
        x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt, min_position
    ))))
    if gradient_norm <= gradient_tol:
        status = "converged"
    return x_t, u_t, {
        "converged": status == "converged", "status": status,
        "gradient_norm": gradient_norm, "history": history,
        "trajectory_history": trajectory_history,
    }


def alilqr(x_0, u_t, W, R, x_ref, dt, W_f=None, max_outer_iters=20,
            max_inner_iters=100, mu_init=10.0, phi=5.0,
            constraint_tol=1e-6, gradient_tol=1e-6, rho_init=1e-6,
            min_position=0.0, record_history=True):
    """Step 7: warm-start successive inner solves and update lambda and mu.

    Parameters
    ----------
    x_0, x_ref : arrays of shape (2,)
    u_t : initial controls, shape (N, 1), or a length-N vector
    W, R : running-cost weights, shapes (2, 2) and (1, 1)
    W_f : terminal weight, shape (2, 2); defaults to W
    min_position : bound x[k, 0] >= min_position; use a positive value for a margin
    record_history : save the initial trajectory and every accepted inner update

    Returns
    -------
    x_t : nonlinear state trajectory, shape (N + 1, 2)
    u_t : optimized controls, shape (N, 1)
    info : convergence, multipliers, penalties, residuals, and iteration history
        When record_history=True, info['xs'] and info['us'] contain trajectory
        histories for plot_alilqr; info['cost_history'] holds the original cost.

    This is a local optimizer. A returned trajectory satisfies the requested
    tolerances only when info['converged'] is True.
    """
    x_0 = np.asarray(x_0, dtype=float).copy()
    x_ref = np.asarray(x_ref, dtype=float).copy()
    u_t = np.asarray(u_t, dtype=float).copy()
    if u_t.ndim == 1:
        u_t = u_t[:, None]
    W = np.asarray(W, dtype=float)
    R = np.asarray(R, dtype=float)
    W_f = W.copy() if W_f is None else np.asarray(W_f, dtype=float)
    if x_0.shape != (2,) or x_ref.shape != (2,):
        raise ValueError("x_0 and x_ref must each have shape (2,).")
    if u_t.ndim != 2 or u_t.shape[1] != 1 or len(u_t) == 0:
        raise ValueError("u_t must contain N >= 1 controls, with shape (N, 1).")
    for name, weight, shape in (("W", W, (2, 2)), ("R", R, (1, 1)),
                                ("W_f", W_f, (2, 2))):
        if weight.shape != shape or not np.all(np.isfinite(weight)):
            raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
        if not np.allclose(weight, weight.T):
            raise ValueError(f"{name} must be symmetric.")
        minimum = np.min(np.linalg.eigvalsh(weight))
        if minimum < -1e-12 or (name == "R" and minimum <= 0.0):
            raise ValueError("W and W_f must be positive semidefinite; R positive definite.")
    if not all(np.all(np.isfinite(a)) for a in (x_0, x_ref, u_t)):
        raise ValueError("States, reference, and controls must be finite.")
    if not all(np.isfinite(v) for v in (dt, mu_init, phi, constraint_tol,
                                       gradient_tol, rho_init, min_position)):
        raise ValueError("Solver parameters must be finite.")
    if dt <= 0 or mu_init <= 0 or phi <= 1 or rho_init < 0:
        raise ValueError("Require dt > 0, mu_init > 0, phi > 1, and rho_init >= 0.")
    if constraint_tol <= 0 or gradient_tol <= 0:
        raise ValueError("Convergence tolerances must be positive.")
    if any(not isinstance(v, (int, np.integer)) or v < 1
           for v in (max_outer_iters, max_inner_iters)):
        raise ValueError("Iteration limits must be positive integers.")
    if x_0[0] < min_position - constraint_tol:
        raise ValueError("The fixed initial position already violates the constraint.")

    N = len(u_t)
    x_t = rollout(x_0, u_t, dt)
    if not np.all(np.isfinite(x_t)):
        raise ValueError("The initial controls produce a non-finite trajectory.")
    lambda_t = np.zeros((N + 1, 1))
    mu_t = np.full((N + 1, 1), mu_init)
    history = []
    converged = False
    accepted_steps = 0
    xs = [x_t.copy()] if record_history else []
    us = [u_t.copy()] if record_history else []
    costs = [total_cost(x_t, u_t, W, R, W_f, x_ref)] if record_history else []

    for outer in range(max_outer_iters):
        step_start = accepted_steps
        # Always optimize, even if the initial trajectory is already feasible.
        x_t, u_t, inner = solve_inner_ilqr(
            x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, dt,
            max_inner_iters, gradient_tol, rho_init,
            min_position=min_position, record_history=record_history
        )
        accepted_steps += sum(row["accepted"] for row in inner["history"])
        if record_history:
            for x_saved, u_saved in inner["trajectory_history"]:
                xs.append(x_saved)
                us.append(u_saved)
                costs.append(total_cost(x_saved, u_saved, W, R, W_f, x_ref))
        c_t = min_position - x_t[:, [0]]
        violation = float(np.max(np.maximum(c_t, 0.0)))

        # Howell Eq. (17): projection keeps inequality multipliers nonnegative.
        lambda_t = np.maximum(0.0, lambda_t + mu_t * c_t)

        # Check the ORIGINAL constrained problem after the multiplier update.
        # Setting the penalty weights to zero leaves ell + lambda^T c.
        lagrangian_gradient = _control_gradient(
            x_t, u_t, lambda_t, np.zeros_like(mu_t), W, R, W_f,
            x_ref, dt, min_position
        )
        stationarity = float(np.max(np.abs(lagrangian_gradient)))
        complementarity = float(np.max(np.abs(lambda_t * c_t)))
        history.append({
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
            "step_start": step_start, "step_end": accepted_steps,
        })
        converged = (violation <= constraint_tol
                     and stationarity <= gradient_tol
                     and complementarity <= constraint_tol)
        if converged:
            break
        if outer + 1 < max_outer_iters:
            # Howell Eq. (18); reuse the current X, U for the next inner solve.
            mu_t *= phi

    info = {
        "converged": converged,
        "status": "converged" if converged else "max_outer_iterations",
        "outer_iterations": len(history),
        "cost": total_cost(x_t, u_t, W, R, W_f, x_ref),
        "augmented_cost": augmented_cost(
            x_t, u_t, lambda_t, mu_t, W, R, W_f, x_ref, min_position
        ),
        "constraint_violation": violation,
        "stationarity": stationarity,
        "complementarity": complementarity,
        "lambda_t": lambda_t.copy(), "mu_t": mu_t.copy(),
        "history": history,
        "accepted_steps": accepted_steps,
        "constraint_tol": constraint_tol, "gradient_tol": gradient_tol,
        "min_position": min_position, "x_ref": x_ref.copy(), "dt": dt,
    }
    if record_history:
        info["xs"] = np.stack(xs)
        info["us"] = np.stack(us)
        info["cost_history"] = np.asarray(costs)
    return x_t, u_t, info


if __name__ == "__main__":

    # Upright target, with enough initial negative velocity to exercise theta >= 0.
    dt = 0.04
    N = 30
    x_0 = np.array([0.2, -2.0])
    x_ref = np.array([0.0, 0.0])
    u_guess = np.zeros((N, 1))
    W = np.diag([20.0, 1.0])
    W_f = np.diag([100.0, 10.0])
    R = np.array([[0.02]])

    x_t, u_t, info = alilqr(
        x_0, u_guess, W, R, x_ref, dt, W_f=W_f,
        min_position=0.0, record_history=True
    )
    print(f"Status: {info['status']}")
    print(f"Outer AL iterations: {info['outer_iterations']}")
    print(f"Accepted iLQR updates: {info['accepted_steps']}")
    print(f"Final cost: {info['cost']:.6g}")
    print(f"Maximum constraint violation: {info['constraint_violation']:.3e}")
    print(f"Stationarity residual: {info['stationarity']:.3e}")
    print(f"Complementarity residual: {info['complementarity']:.3e}")
    plot_alilqr(info["xs"], info["us"], dt, info=info)