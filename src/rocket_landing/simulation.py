"""Open-loop propagation and Jacobian collection, independent of any solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rocket_landing.attitude import STATE_SIZE, TANGENT_SIZE, state_minus
from rocket_landing.dynamics import (
    CONTROL_SIZE,
    FloatArray,
    RocketLandingPlant,
    _positive_scalar,
)


def _controls_array(controls: ArrayLike) -> FloatArray:
    controls = np.array(controls, dtype=float, copy=True)
    if controls.ndim != 2 or controls.shape[1] != CONTROL_SIZE or not np.all(np.isfinite(controls)):
        raise ValueError("controls must be finite with shape (N, 3), including N=0.")
    return controls


def rollout(
    plant: RocketLandingPlant, initial_state: ArrayLike, controls: ArrayLike, dt: float
) -> FloatArray:
    """Propagate N piecewise-constant thrust vectors into N+1 states.

    This free-flight rollout has no ground contact or touchdown stopping logic.
    A zero-length (0,3) control sequence returns just the validated initial state.
    """
    dt = _positive_scalar("dt", dt)
    controls = _controls_array(controls)
    states = np.empty((len(controls) + 1, STATE_SIZE))
    states[0] = plant.validate_state(initial_state)
    for k, control in enumerate(controls):
        states[k + 1] = plant.step(states[k], control, dt)
    return states


def linearize_trajectory(
    plant: RocketLandingPlant,
    states: ArrayLike,
    controls: ArrayLike,
    dt: float,
    *,
    relative_step: float = 1e-5,
) -> tuple[FloatArray, FloatArray]:
    """Collect local discrete A/B matrices along a dynamically consistent rollout.

    Returns shapes (N,13,13) and (N,13,3). Each A/B pair maps local perturbations
    at states[k] into the left-attitude-error chart at states[k+1]. Antipodal
    quaternion representatives are equivalent. Inconsistent nominal states are
    rejected because they would require a defect term and a chart transformation.
    """
    dt = _positive_scalar("dt", dt)
    relative_step = _positive_scalar("relative_step", relative_step)
    controls = _controls_array(controls)
    states = np.array(states, dtype=float, copy=True)
    if states.shape != (len(controls) + 1, STATE_SIZE):
        raise ValueError("states must have shape (N+1, 14) for N controls.")
    for k in range(len(states)):
        states[k] = plant.validate_state(states[k])
    A = np.empty((len(controls), TANGENT_SIZE, TANGENT_SIZE))
    B = np.empty((len(controls), TANGENT_SIZE, CONTROL_SIZE))
    for k, control in enumerate(controls):
        predicted = plant.step(states[k], control, dt)
        if not np.allclose(state_minus(predicted, states[k + 1]), 0.0, rtol=0.0, atol=1e-7):
            raise ValueError("states must be dynamically consistent; generate them with rollout.")
        A[k], B[k] = plant.linearize_discrete(states[k], control, dt, relative_step=relative_step)
    return A, B
