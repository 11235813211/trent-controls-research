"""Numerical checks for the discrete nonlinear pendulum model."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pendulum_ilqr.dynamics import (
    GRAVITY_COEFFICIENT,
    linearized_plant_step,
    nonlinear_plant,
)


@pytest.mark.parametrize(
    ("state", "control", "dt"),
    [([0.0, 0.0], 0.0, 0.01), ([0.7, -1.2], 2.3, 0.04), ([-1.1, 0.4], -3.0, 0.1)],
)
def test_discrete_jacobians_match_central_differences(state, control, dt):
    state = np.array(state)
    control = np.array([control])
    a, b = linearized_plant_step(state, control, dt)
    epsilon = 1e-6
    numerical_a = np.column_stack(
        [
            (
                nonlinear_plant(state + epsilon * direction, control, dt)
                - nonlinear_plant(state - epsilon * direction, control, dt)
            )
            / (2 * epsilon)
            for direction in np.eye(2)
        ]
    )
    numerical_b = (
        nonlinear_plant(state, control + epsilon, dt)
        - nonlinear_plant(state, control - epsilon, dt)
    )[:, None] / (2 * epsilon)
    assert_allclose(a, numerical_a, rtol=1e-7, atol=1e-9)
    assert_allclose(b, numerical_b, rtol=1e-7, atol=1e-9)


def test_step_preserves_the_documented_euler_model_and_inputs():
    state = np.array([np.pi / 2, -2.0])
    control = np.array([3.0])
    original_state, original_control = state.copy(), control.copy()
    assert GRAVITY_COEFFICIENT == pytest.approx(29.43)
    assert_allclose(
        nonlinear_plant(state, control, 0.1),
        [np.pi / 2 - 0.2, -2.0 + 0.1 * (29.43 + 3.0)],
    )
    assert_allclose(nonlinear_plant(state, 3.0, 0.1), nonlinear_plant(state, control, 0.1))
    linearized_plant_step(state, control, 0.1)
    assert_array_equal(state, original_state)
    assert_array_equal(control, original_control)
