"""Physical invariants and linearization checks for the recovery plant."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rocket_landing.attitude import from_rotvec, state_minus, state_plus
from rocket_landing.dynamics import RocketLandingPlant, RocketParameters


def make_state():
    """An upright, stationary vehicle with SI units throughout."""
    state = np.zeros(14)
    state[6] = 1.0
    state[13] = 1000.0
    return state


def vacuum_plant(**overrides):
    return RocketLandingPlant(RocketParameters(air_density_sea_level=0.0, **overrides))


def test_vacuum_free_fall_matches_the_analytic_solution():
    plant = vacuum_plant()
    state = make_state()
    state[:3] = [12.0, -4.0, 250.0]
    state[3:6] = [3.0, -2.0, -18.0]
    duration = 0.7
    gravity = np.array([0.0, 0.0, -plant.parameters.gravity])

    result = plant.step(state, np.zeros(3), duration)

    assert_allclose(result[:3], state[:3] + state[3:6] * duration + gravity * duration**2 / 2)
    assert_allclose(result[3:6], state[3:6] + gravity * duration)
    assert_array_equal(result[6:], state[6:])


def test_upright_thrust_can_balance_weight():
    plant = vacuum_plant()
    state = make_state()
    state[2] = 100.0
    thrust = np.array([0.0, 0.0, state[13] * plant.parameters.gravity])

    assert_allclose(plant.derivative(state, thrust), np.zeros(14), atol=1e-12)
    assert_allclose(plant.step(state, thrust, 0.5), state, atol=1e-12)


def test_passive_attitude_rotates_body_thrust_into_world():
    plant = vacuum_plant(thrust_application_point_body=np.zeros(3))
    state = make_state()
    state[6:10] = from_rotvec([0.0, np.pi / 2, 0.0])

    derivative = plant.derivative(state, np.array([2000.0, 0.0, 0.0]))
    # A nose tilted toward world +x maps body +x toward world -z.
    assert_allclose(derivative[3:6], [0.0, 0.0, -plant.parameters.gravity - 2.0], atol=1e-12)
    derivative = plant.derivative(state, np.array([0.0, 0.0, 2000.0]))
    assert_allclose(derivative[3:6], [2.0, 0.0, -plant.parameters.gravity], atol=1e-12)


def test_lateral_thrust_at_aft_nozzle_has_correct_pitch_sign():
    plant = vacuum_plant(
        inertia_body=np.diag([2000.0, 3000.0, 1000.0]),
        thrust_application_point_body=np.array([0.0, 0.0, -3.0]),
    )
    state = make_state()
    control = np.array([600.0, 0.0, 0.0])

    assert_allclose(plant.loads(state, control).thrust_moment_body, [0.0, -1800.0, 0.0])
    assert_allclose(plant.derivative(state, control)[10:13], [0.0, -0.6, 0.0])


def test_gravity_has_no_center_of_mass_moment():
    plant = vacuum_plant()
    state = make_state()
    state[6:10] = from_rotvec([0.5, -0.3, 0.8])

    assert_allclose(plant.derivative(state, np.zeros(3))[10:13], np.zeros(3), atol=1e-14)
    assert_allclose(
        plant.loads(state, np.zeros(3)).gravity_force_world,
        [0.0, 0.0, -state[13] * plant.parameters.gravity],
    )


def test_body_drag_opposes_relative_air_velocity():
    plant = RocketLandingPlant(
        RocketParameters(
            drag_area_body=np.array([8.0, 6.0, 1.0]),
            aerodynamic_center_body=np.zeros(3),
            wind_world=np.array([2.0, -1.0, 0.0]),
        )
    )
    state = make_state()
    state[3:6] = [8.0, -4.0, -12.0]
    state[6:10] = from_rotvec([0.2, 0.1, -0.4])

    loads = plant.loads(state, np.zeros(3))

    assert np.dot(loads.aerodynamic_force_body, loads.relative_velocity_body) < 0.0
    assert np.all(loads.aerodynamic_force_body * loads.relative_velocity_body <= 0.0)
    assert_allclose(loads.aerodynamic_moment_body, np.zeros(3))


def test_wind_produces_drag_on_a_stationary_vehicle():
    plant = RocketLandingPlant(
        RocketParameters(wind_world=np.array([12.0, 0.0, 0.0]), drag_area_body=np.ones(3))
    )
    loads = plant.loads(make_state(), np.zeros(3))

    assert_allclose(loads.relative_velocity_body, [-12.0, 0.0, 0.0])
    assert loads.aerodynamic_force_body[0] > 0.0
    assert_allclose(loads.aerodynamic_force_body[1:], [0.0, 0.0])


def test_airflow_vanishes_when_vehicle_matches_wind():
    wind = np.array([12.0, -3.0, 2.0])
    plant = RocketLandingPlant(RocketParameters(wind_world=wind))
    state = make_state()
    state[3:6] = wind
    state[6:10] = from_rotvec([-0.3, 0.4, 0.2])

    loads = plant.loads(state, np.zeros(3))

    assert_allclose(loads.relative_velocity_body, np.zeros(3), atol=1e-14)
    assert_allclose(loads.aerodynamic_force_body, np.zeros(3), atol=1e-14)


def test_rotation_at_offset_aerodynamic_center_creates_damping_moment():
    plant = RocketLandingPlant(
        RocketParameters(
            aerodynamic_center_body=np.array([0.0, 0.0, 2.0]),
            drag_area_body=np.ones(3),
        )
    )
    state = make_state()
    state[11] = 1.0
    loads = plant.loads(state, np.zeros(3))

    assert_allclose(loads.relative_velocity_body, [2.0, 0.0, 0.0])
    assert loads.aerodynamic_force_body[0] < 0.0
    assert loads.aerodynamic_moment_body[1] < 0.0
    assert np.dot(loads.aerodynamic_moment_body, state[10:13]) < 0.0


def test_exponential_atmosphere_drops_by_e_over_one_scale_height():
    plant = RocketLandingPlant()
    surface, elevated = make_state(), make_state()
    elevated[2] = plant.parameters.atmosphere_scale_height

    surface_density = plant.loads(surface, np.zeros(3)).air_density
    elevated_density = plant.loads(elevated, np.zeros(3)).air_density

    assert surface_density == pytest.approx(plant.parameters.air_density_sea_level)
    assert elevated_density / surface_density == pytest.approx(np.exp(-1.0))


def test_torque_free_asymmetric_body_has_gyroscopic_acceleration():
    plant = vacuum_plant(inertia_body=np.diag([2.0, 3.0, 4.0]))
    state = make_state()
    state[10:13] = [1.0, 2.0, 3.0]

    derivative = plant.derivative(state, np.zeros(3))

    assert_allclose(derivative[10:13], [-3.0, 2.0, -0.5])
    # A torque-free rigid body conserves instantaneous rotational kinetic energy.
    assert np.dot(state[10:13], plant.parameters.inertia_body @ derivative[10:13]) == pytest.approx(
        0.0
    )


def test_constant_body_rate_matches_passive_quaternion_evolution():
    plant = vacuum_plant(inertia_body=np.eye(3))
    state = make_state()
    state[12] = 0.6

    result = plant.step(state, np.zeros(3), 0.1)

    assert_allclose(result[6:10], [np.cos(0.03), 0.0, 0.0, np.sin(0.03)], atol=1e-10)
    assert_allclose(result[10:13], state[10:13])
    assert np.linalg.norm(result[6:10]) == pytest.approx(1.0, abs=1e-14)


def test_optional_specific_impulse_consumes_mass_without_engine_state():
    plant = vacuum_plant(specific_impulse=300.0)
    state = make_state()
    thrust = np.array([3000.0, 4000.0, 0.0])
    expected_mass_rate = -5000.0 / (300.0 * 9.80665)

    assert plant.derivative(state, thrust)[13] == pytest.approx(expected_mass_rate)
    assert plant.step(state, thrust, 0.1)[13] == pytest.approx(state[13] + expected_mass_rate * 0.1)
    assert plant.derivative(state, np.zeros(3))[13] == 0.0
    assert vacuum_plant().derivative(state, thrust)[13] == 0.0


def test_normalized_rk4_retains_fourth_order_rotational_accuracy():
    plant = vacuum_plant(inertia_body=np.eye(3))
    state = make_state()
    state[12] = 1.5
    expected = from_rotvec([0.0, 0.0, 0.6])
    coarse = plant.step(state, np.zeros(3), 0.4)
    fine = plant.step(plant.step(state, np.zeros(3), 0.2), np.zeros(3), 0.2)

    coarse_error = np.linalg.norm(coarse[6:10] - expected)
    fine_error = np.linalg.norm(fine[6:10] - expected)

    assert coarse_error > 10.0 * fine_error
    assert fine_error < 1e-6


def test_local_discrete_linearization_predicts_state_and_thrust_perturbations():
    plant = RocketLandingPlant()
    state = make_state()
    state[:6] = [2.0, -3.0, 80.0, 5.0, 1.0, -15.0]
    state[6:10] = from_rotvec([0.1, -0.2, 0.05])
    state[10:13] = [0.2, -0.1, 0.05]
    control = np.array([50.0, -30.0, 10000.0])
    dt = 0.03
    a, b = plant.linearize_discrete(state, control, dt)
    nominal_next = plant.step(state, control, dt)
    delta_state = np.array([0.3, -0.2, 0.1, 0.1, 0.2, -0.3, 0.2, -0.1, 0.1, -0.2, 0.1, 0.3, 2.0])
    delta_control = np.array([20.0, -10.0, 50.0])

    def prediction_error(scale):
        actual_next = plant.step(
            state_plus(state, scale * delta_state), control + scale * delta_control, dt
        )
        actual_delta = state_minus(actual_next, nominal_next)
        return np.linalg.norm(actual_delta - scale * (a @ delta_state + b @ delta_control))

    assert a.shape == (13, 13)
    assert b.shape == (13, 3)
    coarse_error, fine_error = prediction_error(1e-3), prediction_error(5e-4)
    assert fine_error < 0.35 * coarse_error
    assert fine_error < 1e-7


def test_vacuum_linearization_recovers_translation_double_integrator():
    plant = vacuum_plant(thrust_application_point_body=np.zeros(3))
    state = make_state()
    dt = 0.2
    a, b = plant.linearize_discrete(state, np.zeros(3), dt)

    assert_allclose(a[:3, :3], np.eye(3), atol=1e-9)
    assert_allclose(a[:3, 3:6], dt * np.eye(3), atol=1e-9)
    assert_allclose(a[3:6, 3:6], np.eye(3), atol=1e-9)
    assert_allclose(b[:3], dt**2 / (2 * state[13]) * np.eye(3), atol=1e-10)
    assert_allclose(b[3:6], dt / state[13] * np.eye(3), atol=1e-10)
    assert_allclose(b[6:], np.zeros((7, 3)), atol=1e-10)


def test_continuous_linearization_has_ambient_state_dimensions():
    plant = vacuum_plant(thrust_application_point_body=np.zeros(3))
    state = make_state()
    a, b = plant.linearize_continuous(state, np.zeros(3))

    assert a.shape == (14, 14)
    assert b.shape == (14, 3)
    assert np.all(np.isfinite(a))
    assert np.all(np.isfinite(b))
    assert_allclose(a[:3, 3:6], np.eye(3), atol=1e-9)
    assert_allclose(b[3:6], np.eye(3) / state[13], atol=1e-10)


def test_ambient_linearization_ignores_radial_quaternion_scaling():
    plant = RocketLandingPlant()
    state = make_state()
    state[3:6] = [2.0, -1.0, -10.0]
    state[6:10] = from_rotvec([0.2, -0.3, 0.1])
    state[10:13] = [0.1, 0.2, -0.1]

    a, _ = plant.linearize_continuous(state, np.array([20.0, -10.0, 10000.0]))

    assert_allclose(a[:, 6:10] @ state[6:10], np.zeros(14), atol=1e-8)


@pytest.mark.parametrize("method", ["linearize_continuous", "linearize_discrete"])
def test_mass_depletion_linearization_rejects_nondifferentiable_zero_thrust(method):
    plant = vacuum_plant(specific_impulse=300.0)
    arguments = (make_state(), np.zeros(3))
    if method == "linearize_discrete":
        arguments += (0.02,)
    with pytest.raises(ValueError, match="nonzero"):
        getattr(plant, method)(*arguments)


def test_plant_calls_preserve_caller_owned_state_and_control():
    plant = vacuum_plant()
    state = make_state()
    state[3:6] = [1.0, 2.0, -3.0]
    control = np.array([10.0, -20.0, 9000.0])
    original_state, original_control = state.copy(), control.copy()

    plant.loads(state, control)
    plant.derivative(state, control)
    plant.step(state, control, 0.02)
    plant.linearize_continuous(state, control)
    plant.linearize_discrete(state, control, 0.02)

    assert_array_equal(state, original_state)
    assert_array_equal(control, original_control)


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan])
def test_nonpositive_or_nonfinite_mass_is_rejected(mass):
    state = make_state()
    state[13] = mass
    with pytest.raises(ValueError):
        vacuum_plant().derivative(state, np.zeros(3))


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf])
def test_nonpositive_or_nonfinite_step_is_rejected(dt):
    with pytest.raises(ValueError):
        vacuum_plant().step(make_state(), np.zeros(3), dt)


def test_zero_quaternion_and_invalid_control_shape_are_rejected():
    plant = vacuum_plant()
    state = make_state()
    state[6:10] = 0.0
    with pytest.raises(ValueError):
        plant.derivative(state, np.zeros(3))
    with pytest.raises(ValueError):
        plant.derivative(make_state(), np.zeros(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inertia_body": np.diag([1.0, -1.0, 2.0])},
        {"drag_area_body": np.array([1.0, -1.0, 1.0])},
        {"air_density_sea_level": -1.0},
        {"atmosphere_scale_height": 0.0},
        {"specific_impulse": 0.0},
    ],
)
def test_nonphysical_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RocketParameters(**kwargs)


def test_parameters_copy_array_inputs():
    inertia = np.diag([2.0, 3.0, 4.0])
    wind = np.array([3.0, 2.0, 1.0])
    parameters = RocketParameters(inertia_body=inertia, wind_world=wind)
    inertia[0, 0] = -100.0
    wind[:] = 99.0

    assert_allclose(parameters.inertia_body, np.diag([2.0, 3.0, 4.0]))
    assert_allclose(parameters.wind_world, [3.0, 2.0, 1.0])
