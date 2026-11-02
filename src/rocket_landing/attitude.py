"""Quaternion attitude and local state coordinates for the landing plant.

Quaternions are **scalar first**, ``q = [s, vx, vy, vz]``, and use the left
product (the cross term has a minus sign). Attitude is passive: ``C_BW(q)``
maps a world-frame vector into body coordinates. For a positive rotation
vector ``phi``, ``C_BW(from_rotvec(phi)) = exp(-skew(phi))``. Thus
``C_BW(q * p) = C_BW(q) @ C_BW(p)`` with the multiplication used here.

Angular velocity is the body-relative-world rate expressed in body axes, so
``q_dot = 0.5 * [0, omega_B] * q`` and ``C_dot = -skew(omega_B) @ C``.
"Left" describes the quaternion algebra; both coordinate frames are right
handed. These explicit definitions avoid ambiguities between quaternion
conventions discussed by NASA/JPL:
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/q2m_c.html
The SPICE/Hamilton product on that page has the opposite cross-term sign.

The stored state has 14 entries ``[r_W(3), v_W(3), q_BW(4), omega_B(3), m]``.
Local perturbations have 13 entries ``[dr_W(3), dv_W(3), dphi_B(3), domega_B(3), dm]``.
Attitude perturbations multiply on the left, giving three independent attitude
coordinates for subsequent linearization without treating quaternion norm as
an extra degree of freedom. The principal attitude logarithm is discontinuous
at rotations of pi radians, so local models should remain inside that chart.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
STATE_SIZE = 14
TANGENT_SIZE = 13


def _vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def normalize_quaternion(quaternion: ArrayLike) -> FloatArray:
    """Return a unit quaternion without changing its sign or the input array.

    Nonunit inputs are accepted; zero and nonfinite inputs are rejected.
    Scaling before normalization avoids overflow for large finite components.
    """
    quaternion = _vector(quaternion, 4, "quaternion")
    scale = float(np.max(np.abs(quaternion)))
    if scale == 0.0:
        raise ValueError("quaternion must have nonzero norm.")
    scaled = quaternion / scale
    return scaled / np.linalg.norm(scaled)


def conjugate(quaternion: ArrayLike) -> FloatArray:
    """Return ``[s, -v]``; this is the inverse for a unit quaternion.

    This is an algebraic operation: inputs need not be unit length, and a
    zero quaternion is permitted. No normalization is performed.
    """
    result = _vector(quaternion, 4, "quaternion").copy()
    result[1:] *= -1.0
    return result


def multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Return the scalar-first left quaternion product ``left * right``.

    The product is ``[s1*s2 - v1.v2, s1*v2 + s2*v1 - cross(v1, v2)]``.
    Operands are not normalized: pure angular-rate and zero quaternions are
    valid inputs, and the function can be used in differential equations.
    """
    left = _vector(left, 4, "left quaternion")
    right = _vector(right, 4, "right quaternion")
    scalar = left[0] * right[0] - np.dot(left[1:], right[1:])
    vector = left[0] * right[1:] + right[0] * left[1:] - np.cross(left[1:], right[1:])
    return np.concatenate(([scalar], vector))


def rotation_matrix(quaternion: ArrayLike) -> FloatArray:
    """Return the passive rotation ``C_BW`` mapping world vectors to body.

    The input is normalized internally. Antipodal quaternions produce the
    same matrix. Transpose this matrix to map body vectors into the world.
    """
    quaternion = normalize_quaternion(quaternion)
    scalar = quaternion[0]
    vector = quaternion[1:]
    x, y, z = vector
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        (scalar * scalar - np.dot(vector, vector)) * np.eye(3)
        + 2.0 * np.outer(vector, vector)
        - 2.0 * scalar * cross
    )


def from_rotvec(rotation_vector: ArrayLike) -> FloatArray:
    """Convert an axis-angle vector in radians to a unit quaternion.

    Its direction is the positive body-frame rotation axis and its norm is
    the rotation angle. Zero returns the identity. The small-angle branch
    retains the first-order quaternion vector needed by numerical Jacobians.
    """
    rotation_vector = _vector(rotation_vector, 3, "rotation_vector")
    angle = math.hypot(*rotation_vector)
    if not math.isfinite(angle):
        raise ValueError("rotation_vector magnitude must be finite.")
    half_angle = 0.5 * angle
    scale = 0.5 - angle * angle / 48.0 if angle < 1e-8 else math.sin(half_angle) / angle
    return normalize_quaternion(np.concatenate(([math.cos(half_angle)], scale * rotation_vector)))


def to_rotvec(quaternion: ArrayLike) -> FloatArray:
    """Return the principal rotation vector, with norm between zero and pi.

    ``q`` and ``-q`` return the same vector. At exactly pi, where the rotation
    axis is ambiguous, the first nonzero vector component is chosen positive.
    The map is smooth near identity but cannot be continuous across pi.
    """
    quaternion = normalize_quaternion(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    elif quaternion[0] == 0.0:
        nonzero = np.flatnonzero(quaternion[1:])
        if quaternion[1 + nonzero[0]] < 0.0:
            quaternion = -quaternion
    sin_half = math.hypot(*quaternion[1:])
    if sin_half == 0.0:
        return np.zeros(3)
    scale = 2.0 * math.atan2(sin_half, quaternion[0]) / sin_half
    return scale * quaternion[1:]


def quaternion_derivative(quaternion: ArrayLike, angular_velocity: ArrayLike) -> FloatArray:
    """Return ``q_dot`` for body-relative-world angular velocity in body axes.

    ``angular_velocity`` has three entries in radians/second. The attitude
    input is normalized, and no artificial quaternion norm correction is
    added. A numerical integrator should normalize accepted state outputs.
    """
    quaternion = normalize_quaternion(quaternion)
    angular_velocity = _vector(angular_velocity, 3, "angular_velocity")
    return 0.5 * multiply(np.concatenate(([0.0], angular_velocity)), quaternion)


def state_plus(state: ArrayLike, delta: ArrayLike) -> FloatArray:
    """Apply a 13-entry local perturbation to a 14-entry stored state.

    Euclidean components are added. Attitude changes as
    ``q_new = from_rotvec(delta[6:9]) * q``. Body angular-velocity components
    are added in the same body-axis coordinate convention. Inputs are never
    modified. This coordinate helper accepts any finite mass; the plant
    applies its own physical mass checks.
    """
    state = _vector(state, STATE_SIZE, "state")
    delta = _vector(delta, TANGENT_SIZE, "delta")
    quaternion = normalize_quaternion(state[6:10])
    result = state.copy()
    result[:6] += delta[:6]
    result[6:10] = normalize_quaternion(multiply(from_rotvec(delta[6:9]), quaternion))
    result[10:13] += delta[9:12]
    result[13] += delta[12]
    return result


def state_minus(state: ArrayLike, reference: ArrayLike) -> FloatArray:
    """Return the local difference from ``reference`` to ``state``.

    Attitude difference is ``to_rotvec(q_state * conjugate(q_reference))``.
    Therefore ``state_minus(state_plus(x, d), x) == d`` for attitude increments
    with norm below pi, up to roundoff. Quaternion sign does not affect this
    difference. Euclidean state entries use ordinary subtraction.
    """
    state = _vector(state, STATE_SIZE, "state")
    reference = _vector(reference, STATE_SIZE, "reference")
    quaternion = normalize_quaternion(state[6:10])
    reference_quaternion = normalize_quaternion(reference[6:10])
    attitude_difference = to_rotvec(multiply(quaternion, conjugate(reference_quaternion)))
    return np.concatenate(
        (
            state[:6] - reference[:6],
            attitude_difference,
            state[10:14] - reference[10:14],
        )
    )
