"""Checks against passive rotation geometry, independent of quaternion helpers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from rocket_landing.attitude import (
    conjugate,
    from_rotvec,
    multiply,
    normalize_quaternion,
    quaternion_derivative,
    rotation_matrix,
    state_minus,
    state_plus,
    to_rotvec,
)


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _passive_rodrigues(rotation_vector):
    angle = np.linalg.norm(rotation_vector)
    if angle == 0.0:
        return np.eye(3)
    cross = _skew(rotation_vector / angle)
    return np.eye(3) - np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def _state():
    return np.array(
        [2.0, -3.0, 50.0, 0.5, 1.0, -4.0, *from_rotvec([0.3, -0.5, 0.8]), 0.4, -0.7, 0.2, 500.0]
    )


def test_scalar_first_passive_positive_z_rotation():
    quaternion = from_rotvec([0.0, 0.0, np.pi / 2])
    assert_allclose(quaternion, [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)], atol=1e-15)
    assert_allclose(rotation_matrix(quaternion) @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("rotation_vector", [[0.0, 0.0, 0.0], [0.4, -0.6, 1.2], [-2.1, 0.7, 0.9]])
def test_rotation_matrix_matches_passive_rodrigues(rotation_vector):
    matrix = rotation_matrix(from_rotvec(rotation_vector))
    assert_allclose(matrix, _passive_rodrigues(rotation_vector), atol=6e-16)
    assert_allclose(matrix @ matrix.T, np.eye(3), atol=8e-16)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_left_product_has_negative_cross_and_composes_passive_rotations():
    assert_array_equal(multiply([0, 1, 0, 0], [0, 0, 1, 0]), [0, 0, 0, -1])
    first = np.array([0.2, -0.5, 0.7])
    second = np.array([-0.3, 0.4, 0.1])
    product = multiply(from_rotvec(first), from_rotvec(second))
    expected = _passive_rodrigues(first) @ _passive_rodrigues(second)
    assert_allclose(rotation_matrix(product), expected, atol=6e-16)
    assert_allclose(multiply(product, conjugate(product)), [1, 0, 0, 0], atol=3e-16)


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300, -3.0])
def test_normalization_and_matrix_are_scale_invariant(scale):
    quaternion = np.array([1.0, 2.0, -3.0, 4.0])
    original = quaternion.copy()
    expected = quaternion / np.linalg.norm(quaternion)
    assert_allclose(normalize_quaternion(scale * quaternion), np.sign(scale) * expected)
    assert_allclose(rotation_matrix(scale * quaternion), rotation_matrix(quaternion), atol=5e-16)
    assert_array_equal(quaternion, original)


@pytest.mark.parametrize(
    "rotation_vector",
    [[0.0, 0.0, 0.0], [1e-12, -2e-12, 3e-12], [0.2, -0.7, 0.3], [0.0, np.pi - 1e-9, 0.0]],
)
def test_rotation_vector_round_trip_and_antipodal_invariance(rotation_vector):
    quaternion = from_rotvec(rotation_vector)
    assert_allclose(to_rotvec(quaternion), rotation_vector, rtol=2e-15, atol=1e-16)
    assert_allclose(to_rotvec(-quaternion), rotation_vector, rtol=2e-15, atol=1e-16)


def test_principal_logarithm_wraps_large_angles_and_resolves_pi_tie():
    assert_allclose(to_rotvec(from_rotvec([0.0, 0.0, 1.5 * np.pi])), [0, 0, -0.5 * np.pi])
    assert_array_equal(to_rotvec([0, -1, 0, 0]), [np.pi, 0, 0])
    assert_array_equal(to_rotvec([0, 1, 0, 0]), [np.pi, 0, 0])
    assert_array_equal(to_rotvec([-1, 0, 0, 0]), [0, 0, 0])


def test_quaternion_derivative_matches_body_rate_and_passive_matrix_derivative():
    quaternion = from_rotvec([0.4, -0.2, 0.8])
    rate = np.array([0.3, -0.7, 0.2])
    derivative = quaternion_derivative(quaternion, rate)
    expected = 0.5 * np.concatenate(
        ([-np.dot(rate, quaternion[1:])], quaternion[0] * rate - np.cross(rate, quaternion[1:]))
    )
    assert_allclose(derivative, expected, atol=1e-16)
    assert np.dot(quaternion, derivative) == pytest.approx(0.0, abs=5e-17)
    epsilon = 1e-6
    matrix_derivative = (
        rotation_matrix(quaternion + epsilon * derivative)
        - rotation_matrix(quaternion - epsilon * derivative)
    ) / (2.0 * epsilon)
    assert_allclose(matrix_derivative, -_skew(rate) @ rotation_matrix(quaternion), atol=2e-10)
    exact_step_derivative = (
        multiply(from_rotvec(epsilon * rate), quaternion)
        - multiply(from_rotvec(-epsilon * rate), quaternion)
    ) / (2.0 * epsilon)
    assert_allclose(derivative, exact_step_derivative, atol=6e-11)
    assert_allclose(quaternion_derivative(-quaternion, rate), -derivative)
    assert_array_equal(quaternion_derivative(quaternion, [0, 0, 0]), np.zeros(4))


def test_local_state_coordinates_round_trip_without_modifying_inputs():
    state = _state()
    delta = np.array([0.2, 0.3, -0.4, -0.1, 0.2, 0.5, 0.04, -0.03, 0.02, 0.1, 0.2, -0.1, -2.0])
    original_state, original_delta = state.copy(), delta.copy()
    perturbed = state_plus(state, delta)
    assert_allclose(state_minus(perturbed, state), delta, atol=1e-14)
    expected_rotation = _passive_rodrigues(delta[6:9]) @ rotation_matrix(state[6:10])
    assert_allclose(rotation_matrix(perturbed[6:10]), expected_rotation, atol=6e-16)
    assert_allclose(np.linalg.norm(perturbed[6:10]), 1.0)
    assert_array_equal(state, original_state)
    assert_array_equal(delta, original_delta)


def test_state_coordinates_are_antipodal_invariant_and_retain_small_angles():
    reference = _state()
    delta = np.arange(1, 14, dtype=float) * 1e-9
    state = state_plus(reference, delta)
    antipodal_reference, antipodal_state = reference.copy(), state.copy()
    antipodal_reference[6:10] *= -1.0
    antipodal_state[6:10] *= -1.0
    assert_allclose(state_minus(state, antipodal_reference), delta, atol=3e-14)
    assert_allclose(state_minus(antipodal_state, reference), delta, atol=3e-14)
    assert_allclose(state_minus(antipodal_state, antipodal_reference), delta, atol=3e-14)
    assert_allclose(state_minus(reference, antipodal_reference), np.zeros(13), atol=1e-16)
    assert_allclose(state_plus(antipodal_reference, delta)[6:10], -state[6:10], rtol=0, atol=1e-16)


@pytest.mark.parametrize(
    "quaternion", [[0, 0, 0, 0], [1, 0, 0], [1, 0, 0, np.inf], [np.nan, 0, 0, 0]]
)
def test_orientation_operations_reject_invalid_quaternions(quaternion):
    for operation in (normalize_quaternion, rotation_matrix, to_rotvec):
        with pytest.raises(ValueError):
            operation(quaternion)
    with pytest.raises(ValueError):
        quaternion_derivative(quaternion, [0, 0, 0])


@pytest.mark.parametrize("vector", [[0, 0], [0, 0, np.inf], [np.nan, 0, 0]])
def test_rotation_vector_and_angular_rate_validation(vector):
    with pytest.raises(ValueError):
        from_rotvec(vector)
    with pytest.raises(ValueError):
        quaternion_derivative([1, 0, 0, 0], vector)


def test_algebra_does_not_normalize_quaternions_or_reject_zero_rates():
    assert_array_equal(conjugate([2, 3, -4, 5]), [2, -3, 4, -5])
    assert_array_equal(multiply([2, 0, 0, 0], [3, 0, 0, 0]), [6, 0, 0, 0])
    assert_array_equal(multiply([0, 0, 0, 0], [1, 0, 0, 0]), [0, 0, 0, 0])
    with pytest.raises(ValueError):
        multiply([1, 0, 0], [1, 0, 0, 0])
    with pytest.raises(ValueError):
        conjugate([1, 0, 0, np.nan])


def test_state_coordinate_validation_and_nonunit_attitudes():
    state = _state()
    state[6:10] *= 3.0
    normalized = state_plus(state, np.zeros(13))
    assert_allclose(np.linalg.norm(normalized[6:10]), 1.0)
    assert_allclose(state_minus(normalized, state), np.zeros(13), atol=1e-16)
    for invalid in (np.zeros(13), np.full(14, np.nan), np.zeros(14)):
        with pytest.raises(ValueError):
            state_plus(invalid, np.zeros(13))
        with pytest.raises(ValueError):
            state_minus(invalid, state)
        with pytest.raises(ValueError):
            state_minus(state, invalid)
    with pytest.raises(ValueError):
        state_plus(state, np.zeros(14))
    with pytest.raises(ValueError):
        state_plus(state, np.full(13, np.nan))
