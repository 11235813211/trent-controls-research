"""Open-loop rollout and trajectory-linearization contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rocket_landing.dynamics import RocketLandingPlant, RocketParameters
from rocket_landing.simulation import linearize_trajectory, rollout


def make_state():
    state = np.zeros(14)
    state[2] = 100.0
    state[6] = 1.0
    state[13] = 1000.0
    return state


def test_rollout_contains_initial_state_and_each_control_interval():
    plant = RocketLandingPlant(RocketParameters(air_density_sea_level=0.0))
    state = make_state()
    state[5] = -10.0
    controls = np.zeros((20, 3))
    dt = 0.05

    states = rollout(plant, state, controls, dt)
    times = np.arange(21) * dt

    assert states.shape == (21, 14)
    assert_array_equal(states[0], state)
    assert_allclose(
        states[:, 2], state[2] + state[5] * times - plant.parameters.gravity * times**2 / 2
    )
    assert_allclose(states[:, 5], state[5] - plant.parameters.gravity * times)
    assert_allclose(np.linalg.norm(states[:, 6:10], axis=1), np.ones(21), atol=1e-14)


def test_rollout_applies_distinct_controls_in_order():
    plant = RocketLandingPlant(
        RocketParameters(air_density_sea_level=0.0, thrust_application_point_body=np.zeros(3))
    )
    state = make_state()
    controls = np.array([[1000.0, 0.0, 0.0], [-2000.0, 0.0, 0.0]])

    states = rollout(plant, state, controls, 0.1)

    assert_allclose(states[:, 3], [0.0, 0.1, -0.1], atol=1e-12)
    assert_allclose(states[:, 0], [0.0, 0.005, 0.005], atol=1e-12)


def test_empty_rollout_and_linearizations_have_explicit_dimensions():
    plant = RocketLandingPlant()
    initial = make_state()
    controls = np.empty((0, 3))

    states = rollout(plant, initial, controls, 0.02)
    a, b = linearize_trajectory(plant, states, controls, 0.02)

    assert states.shape == (1, 14)
    assert_array_equal(states[0], initial)
    assert a.shape == (0, 13, 13)
    assert b.shape == (0, 13, 3)


def test_trajectory_linearizations_match_each_plant_interval():
    plant = RocketLandingPlant()
    controls = np.array([[20.0, -10.0, 10000.0], [-30.0, 15.0, 8000.0]])
    dt = 0.03
    states = rollout(plant, make_state(), controls, dt)

    a, b = linearize_trajectory(plant, states, controls, dt)

    assert a.shape == (2, 13, 13)
    assert b.shape == (2, 13, 3)
    for index in range(len(controls)):
        expected_a, expected_b = plant.linearize_discrete(states[index], controls[index], dt)
        assert_allclose(a[index], expected_a)
        assert_allclose(b[index], expected_b)


def test_simulation_utilities_preserve_caller_arrays():
    plant = RocketLandingPlant()
    initial = make_state()
    controls = np.array([[0.0, 0.0, 10000.0]])
    original_initial, original_controls = initial.copy(), controls.copy()
    states = rollout(plant, initial, controls, 0.01)
    original_states = states.copy()

    linearize_trajectory(plant, states, controls, 0.01)

    assert_array_equal(initial, original_initial)
    assert_array_equal(controls, original_controls)
    assert_array_equal(states, original_states)
    states[0, 0] = 123.0
    assert_array_equal(initial, original_initial)


@pytest.mark.parametrize("controls", [np.zeros(3), np.zeros((2, 2)), np.full((1, 3), np.nan)])
def test_rollout_rejects_malformed_or_nonfinite_controls(controls):
    with pytest.raises(ValueError):
        rollout(RocketLandingPlant(), make_state(), controls, 0.02)


def test_trajectory_linearization_requires_one_more_state_than_controls():
    with pytest.raises(ValueError):
        linearize_trajectory(
            RocketLandingPlant(), np.tile(make_state(), (2, 1)), np.zeros((2, 3)), 0.02
        )


def test_trajectory_linearization_rejects_inconsistent_nominal_states():
    plant = RocketLandingPlant()
    controls = np.zeros((1, 3))
    states = rollout(plant, make_state(), controls, 0.02)
    states[1, 0] += 1.0

    with pytest.raises(ValueError):
        linearize_trajectory(plant, states, controls, 0.02)
