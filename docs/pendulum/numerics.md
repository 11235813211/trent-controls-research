# Numerical method

## Discrete dynamics and derivatives

The model uses forward Euler with a fixed time step:

```text
theta[k+1] = theta[k] + dt * omega[k]
omega[k+1] = omega[k] + dt * (29.43 * sin(theta[k]) + u[k])

A[k] = [[1,                         dt],
        [29.43 * cos(theta[k]) * dt,  1]]
B[k] = [[0], [dt]]
```

The normalized input has angular-acceleration units. A dimensional rigid
pendulum would use a gravity coefficient `m*g*l/I` and torque input divided by
inertia. This project exposes an effective coefficient and does not identify
those physical parameters. The angle is unwrapped, so quadratic tracking
penalizes angle differences directly rather than modulo `2*pi`.

The first angle update depends only on the fixed initial state, not on the
first control. A feasible initial angle therefore does not ensure that every
requested bound is achievable over the horizon.

## Shared iLQR inner solve

Each iteration linearizes the dynamics about the nominal nonlinear trajectory
and uses analytic quadratic cost derivatives. A backward dynamic-programming
pass builds the local control law

```text
u_trial[k] = u_nominal[k] + alpha * d[k] + K[k] @ (x_trial[k] - x_nominal[k])
```

This is iLQR / Gauss–Newton curvature: second derivatives of the dynamics are
omitted. The control Hessian is symmetrized and shifted when necessary to be
positive definite. The policy solve also uses adaptive diagonal damping.
Linear systems are solved directly rather than by explicitly inverting matrices.
The value recursion uses the unregularized control Hessian.

The forward pass rolls out the nonlinear model. Backtracking starts at
`alpha=1` and halves it until a finite, strictly lower objective is found.
Only the feedforward term `d` is scaled. A rejected search increases damping and
recomputes the backward pass at the same trajectory. Accepted steps reduce
damping. The default search permits 25 trials per backward pass.

A separate adjoint recursion evaluates the exact first derivative of the
discretized objective with respect to all controls. Convergence depends on its
infinity norm, rather than on a small policy step: large damping can produce
small steps even when a solution is not stationary.

Unconstrained `ilqr` uses this same implementation with zero multipliers and
penalties. Its accepted cost history is strictly decreasing. The returned
trajectory is always a nonlinear rollout of the returned controls.

## Augmented Lagrangian outer solve

For each state, including the terminal state:

```text
c[k] = min_position - theta[k]
L_A = J + sum(k=0..N) [lambda[k]*c[k] + 1/2*I_mu[k]*c[k]^2]
I_mu[k] = 0       if c[k] < 0 and lambda[k] == 0
          mu[k]   otherwise
```

The inequality is affine in state, so its constraint Hessian is zero. The
Jacobian contribution to the augmented Hessian is included. The inner solve
holds `lambda` and `mu` fixed. After each solve:

```text
lambda[k] = max(0, lambda[k] + mu[k]*c[k])
mu[k]     = phi*mu[k]   if another outer iteration is needed
```

Each outer iteration warm-starts from the previous state and control trajectory.
Defaults are `mu_init=10`, `phi=5`, at most 20 outer iterations, and at most 100
iterations per inner solve. The penalty schedule is deliberately simple;
unbounded penalty growth can make difficult problems ill-conditioned.

The policy, value, multiplier, and penalty updates follow Section II.A,
Eqs. (3), (5)–(18), of [Howell, Jackson, and Manchester (2019)](https://thowell.github.io/documents/altro_iros.pdf).
This implementation uses a standard dense backward pass and adaptive control
damping. It does not include ALTRO's square-root formulation or projected
Newton polishing.

## Stopping criteria

For AL-iLQR, all three checks must pass after updating the multipliers:

| Diagnostic | Definition | Default tolerance |
| --- | --- | --- |
| Primal feasibility | `max(max(c, 0))` | `constraint_tol = 1e-6` rad |
| Stationarity | `max(abs(d(J + lambda*c)/dU))` | `gradient_tol = 1e-6` |
| Complementarity | `max(abs(lambda*c))` | `constraint_tol = 1e-6` |

Stationarity is evaluated in the reduced control space: the dynamics are
eliminated through the rollout and adjoint. Its residual depends on the chosen
units and cost scaling. The original Lagrangian gradient uses no quadratic
penalty term. Projection ensures nonnegative inequality multipliers.

Unconstrained convergence requires only the objective gradient infinity norm.
An iteration limit or failed line search is a nonconverged result. Solver status
and residuals should be inspected together; feasibility alone is insufficient.

## Reproducing and interpreting results

Run the commands in the [README](README.md) to regenerate both examples.
The same zero-control initial guess, model, horizon, and weights are used in
both methods. Minor floating-point differences across NumPy and BLAS versions
can affect the final residuals and accepted-step counts.

The original objective need not decrease between augmented Lagrangian solves.
Only augmented costs evaluated at the same multipliers and penalties are
directly comparable. Plotting separates those segments and marks the outer
updates. State constraints are measured at integration nodes, not at arbitrary
times between nodes.
