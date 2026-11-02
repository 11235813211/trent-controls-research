"""Explicit-Euler dynamics for the normalized inverted pendulum example.

The state is ``[angle, angular_velocity]`` in radians and radians/second.
The angle is measured from the upright equilibrium, so the continuous model is
``theta_ddot = GRAVITY_COEFFICIENT * sin(theta) + u``. Control ``u`` is angular
acceleration in radians/second². The coefficient retains the original project's
value, 29.43 s⁻²; it is a normalized model parameter, not a mass or torque.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

GRAVITY_COEFFICIENT: float = 29.43
"""Destabilizing gravity coefficient in s⁻², preserving the original model."""

FloatArray = NDArray[np.float64]


def _step_inputs(x: ArrayLike, u: ArrayLike, dt: float) -> tuple[FloatArray, float]:
    """Normalize the public step inputs without changing the caller's arrays."""
    state = np.asarray(x, dtype=float)
    control = np.asarray(u, dtype=float)
    if state.shape != (2,):
        raise ValueError("x must have shape (2,): angle and angular velocity.")
    if control.shape not in ((), (1,)):
        raise ValueError("u must be a scalar or have shape (1,).")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(control)):
        raise ValueError("State and control must be finite.")
    if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive scalar.")
    return state, float(control.item())


def nonlinear_plant(x: ArrayLike, u: ArrayLike, dt: float) -> FloatArray:
    """Advance one time step with explicit Euler integration.

    Parameters
    ----------
    x : array-like, shape (2,)
        Current angle and angular velocity.
    u : scalar or array-like, shape (1,)
        Angular-acceleration input, held constant for the time step.
    dt : float
        Positive sample period in seconds.

    Returns
    -------
    numpy.ndarray, shape (2,)
        The next state. Inputs are never modified.
    """
    state, control = _step_inputs(x, u, dt)
    acceleration = GRAVITY_COEFFICIENT * np.sin(state[0]) + control
    return np.array([state[0] + dt * state[1], state[1] + dt * acceleration], dtype=float)


def linearized_plant_step(x: ArrayLike, u: ArrayLike, dt: float) -> tuple[FloatArray, FloatArray]:
    """Return the exact Jacobians of the discrete Euler dynamics.

    The returned matrices have shapes ``A=(2, 2)`` and ``B=(2, 1)``. The
    derivative is evaluated at ``x``; it is independent of ``u`` for this model.
    """
    state, _ = _step_inputs(x, u, dt)
    A = np.array([[1.0, dt], [GRAVITY_COEFFICIENT * np.cos(state[0]) * dt, 1.0]])
    B = np.array([[0.0], [dt]])
    return A, B
