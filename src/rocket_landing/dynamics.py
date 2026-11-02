"""Six-degree-of-freedom free-flight plant for terminal rocket descent.

World coordinates are local East-North-Up; body +Z points toward the nose.
State: [r_W(3), v_W(3), q_BW(4), omega_B(3), mass(1)]. The scalar-first,
left-handed passive quaternion maps world-coordinate vectors into the body.
Input: commanded body-frame thrust force in newtons, held over a time step.

The plant combines gravity, a lumped anisotropic aerodynamic force at a center
of pressure, and abstract thrust at an engine application point. It contains no
controller, optimizer, engine state, actuator saturation, or contact model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rocket_landing.attitude import (
    STATE_SIZE,
    TANGENT_SIZE,
    normalize_quaternion,
    quaternion_derivative,
    rotation_matrix,
    state_minus,
    state_plus,
)

FloatArray = NDArray[np.float64]
STANDARD_GRAVITY = 9.80665
CONTROL_SIZE = 3
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
QUATERNION = slice(6, 10)
ANGULAR_VELOCITY = slice(10, 13)
MASS = 13


def _finite_array(name: str, value: ArrayLike, shape: tuple[int, ...]) -> FloatArray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


def _positive_scalar(name: str, value: float, *, zero_allowed: bool = False) -> float:
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar.")
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero_allowed else value <= 0):
        qualifier = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


@dataclass(frozen=True, eq=False)
class RocketParameters:
    """Configurable illustrative vehicle and local atmospheric parameters.

    Array fields are copied and made read-only. Inertia (kg m²), center of mass,
    engine/pressure-center offsets (m), and drag-area products Cd*A (m²) remain
    fixed. Gravity is m/s², wind m/s, density kg/m³, scale height m, and optional
    specific impulse seconds. Values describe no particular flight vehicle.
    """

    gravity: float = STANDARD_GRAVITY
    inertia_body: ArrayLike = field(default_factory=lambda: np.diag([8000.0, 8000.0, 500.0]))
    thrust_application_point_body: ArrayLike = field(
        default_factory=lambda: np.array([0.0, 0.0, -3.0])
    )
    aerodynamic_center_body: ArrayLike = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    drag_area_body: ArrayLike = field(default_factory=lambda: np.array([8.0, 8.0, 0.5]))
    air_density_sea_level: float = 1.225
    atmosphere_scale_height: float = 8500.0
    wind_world: ArrayLike = field(default_factory=lambda: np.zeros(3))
    specific_impulse: float | None = None

    def __post_init__(self) -> None:
        for name, allow_zero in (
            ("gravity", True),
            ("air_density_sea_level", True),
            ("atmosphere_scale_height", False),
        ):
            object.__setattr__(
                self, name, _positive_scalar(name, getattr(self, name), zero_allowed=allow_zero)
            )
        if self.specific_impulse is not None:
            object.__setattr__(
                self,
                "specific_impulse",
                _positive_scalar("specific_impulse", self.specific_impulse),
            )
        for name in (
            "thrust_application_point_body",
            "aerodynamic_center_body",
            "drag_area_body",
            "wind_world",
        ):
            value = _finite_array(name, getattr(self, name), (3,))
            if name == "drag_area_body" and np.any(value < 0):
                raise ValueError("drag_area_body must be nonnegative.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        inertia = _finite_array("inertia_body", self.inertia_body, (3, 3))
        if not np.allclose(inertia, inertia.T, rtol=1e-12, atol=1e-12):
            raise ValueError("inertia_body must be symmetric positive definite.")
        inertia = 0.5 * (inertia + inertia.T)
        if np.min(np.linalg.eigvalsh(inertia)) <= 0:
            raise ValueError("inertia_body must be symmetric positive definite.")
        inertia.setflags(write=False)
        object.__setattr__(self, "inertia_body", inertia)


@dataclass(frozen=True)
class Loads:
    """Separated forces (N), moments about the mass center (N m), and airflow."""

    gravity_force_world: FloatArray
    aerodynamic_force_body: FloatArray
    thrust_force_body: FloatArray
    aerodynamic_moment_body: FloatArray
    thrust_moment_body: FloatArray
    air_density: float
    relative_velocity_body: FloatArray


def make_state(
    *,
    position_world: ArrayLike = (0.0, 0.0, 30.0),
    velocity_world: ArrayLike = (0.0, 0.0, -10.0),
    quaternion: ArrayLike = (1.0, 0.0, 0.0, 0.0),
    angular_velocity_body: ArrayLike = (0.0, 0.0, 0.0),
    mass: float = 1000.0,
) -> FloatArray:
    """Construct a validated state; defaults are an upright descending example."""
    return np.concatenate(
        (
            _finite_array("position_world", position_world, (3,)),
            _finite_array("velocity_world", velocity_world, (3,)),
            normalize_quaternion(quaternion),
            _finite_array("angular_velocity_body", angular_velocity_body, (3,)),
            [_positive_scalar("mass", mass)],
        )
    )


class RocketLandingPlant:
    """Continuous dynamics, normalized RK4 integration, and local Jacobians."""

    state_size = STATE_SIZE
    tangent_size = TANGENT_SIZE
    control_size = CONTROL_SIZE

    def __init__(self, parameters: RocketParameters | None = None) -> None:
        if parameters is not None and not isinstance(parameters, RocketParameters):
            raise TypeError("parameters must be a RocketParameters instance.")
        self.parameters = RocketParameters() if parameters is None else parameters

    @staticmethod
    def validate_state(state: ArrayLike) -> FloatArray:
        """Copy a finite state, normalize its quaternion, and require positive mass."""
        state = _finite_array("state", state, (STATE_SIZE,))
        state[QUATERNION] = normalize_quaternion(state[QUATERNION])
        if state[MASS] <= 0:
            raise ValueError("State mass must be positive.")
        return state

    @staticmethod
    def validate_control(control: ArrayLike) -> FloatArray:
        """Copy body-frame thrust; actuator limits belong outside this plant."""
        return _finite_array("control", control, (CONTROL_SIZE,))

    def loads(self, state: ArrayLike, control: ArrayLike) -> Loads:
        """Evaluate gravity, aerodynamics, thrust, and application-point moments.

        Relative velocity is evaluated at the pressure center, including
        omega cross r. Drag is dissipative in that local air-relative velocity.
        No lift table, fin actuator, independent roll damping, or CFD fit is
        implied by this lumped Cd*A model.
        """
        state, control = self.validate_state(state), self.validate_control(control)
        p = self.parameters
        C_BW = rotation_matrix(state[QUATERNION])
        airflow = C_BW @ (state[VELOCITY] - p.wind_world) + np.cross(
            state[ANGULAR_VELOCITY], p.aerodynamic_center_body
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                density = (
                    0.0
                    if p.air_density_sea_level == 0
                    else float(
                        p.air_density_sea_level * np.exp(-state[2] / p.atmosphere_scale_height)
                    )
                )
                aerodynamic_force = (
                    -0.5 * density * np.linalg.norm(airflow) * p.drag_area_body * airflow
                )
                gravity_force = np.array([0.0, 0.0, -state[MASS] * p.gravity])
                aerodynamic_moment = np.cross(p.aerodynamic_center_body, aerodynamic_force)
                thrust_moment = np.cross(p.thrust_application_point_body, control)
        except FloatingPointError as exc:
            raise ValueError("State or parameters produce non-finite loads.") from exc
        return Loads(
            gravity_force,
            aerodynamic_force,
            control,
            aerodynamic_moment,
            thrust_moment,
            density,
            airflow,
        )

    def derivative(self, state: ArrayLike, control: ArrayLike) -> FloatArray:
        """Return [velocity, acceleration, q_dot, omega_dot, mass_dot].

        Quaternion inputs are normalized, defining a smooth ambient extension
        near unit quaternions. Positive mass is required even with fixed mass.
        Optional depletion changes translational mass only: inertia and load
        offsets remain fixed, and there is no propellant exhaustion/contact law.
        """
        state, control = self.validate_state(state), self.validate_control(control)
        p = self.parameters
        loads = self.loads(state, control)
        rate = np.empty(STATE_SIZE)
        rate[POSITION] = state[VELOCITY]
        C_BW = rotation_matrix(state[QUATERNION])
        rate[VELOCITY] = (
            loads.gravity_force_world
            + C_BW.T @ (loads.aerodynamic_force_body + loads.thrust_force_body)
        ) / state[MASS]
        rate[QUATERNION] = quaternion_derivative(state[QUATERNION], state[ANGULAR_VELOCITY])
        rate[ANGULAR_VELOCITY] = np.linalg.solve(
            p.inertia_body,
            loads.aerodynamic_moment_body
            + loads.thrust_moment_body
            - np.cross(state[ANGULAR_VELOCITY], p.inertia_body @ state[ANGULAR_VELOCITY]),
        )
        rate[MASS] = (
            0.0
            if p.specific_impulse is None
            else -np.linalg.norm(control) / (p.specific_impulse * STANDARD_GRAVITY)
        )
        if not np.all(np.isfinite(rate)):
            raise ValueError("State or parameters produce a non-finite derivative.")
        return rate

    def step(self, state: ArrayLike, control: ArrayLike, dt: float) -> FloatArray:
        """Advance one zero-order-hold thrust interval with normalized RK4.

        No ground clamp or actuator saturation is applied. The free-flight map
        stays suitable for local linearization; touchdown logic is a simulation
        concern. Steps reaching nonpositive mass are rejected.
        """
        dt = _positive_scalar("dt", dt)
        state, control = self.validate_state(state), self.validate_control(control)
        k1 = self.derivative(state, control)
        k2 = self.derivative(state + 0.5 * dt * k1, control)
        k3 = self.derivative(state + 0.5 * dt * k2, control)
        k4 = self.derivative(state + dt * k3, control)
        return self.validate_state(state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    def linearize_continuous(
        self, state: ArrayLike, control: ArrayLike, *, relative_step: float = 1e-5
    ) -> tuple[FloatArray, FloatArray]:
        """Central-difference the normalized ambient derivative: A14x14, B14x3.

        The four quaternion columns differentiate the documented normalized
        extension. For future manifold trajectory optimization, use the
        13-dimensional local coordinates in linearize_discrete instead.
        Steps scale with max(1, abs(component)); reduce them for local studies.
        """
        relative_step = _positive_scalar("relative_step", relative_step)
        state, control = self.validate_state(state), self.validate_control(control)
        if self.parameters.specific_impulse is not None and np.linalg.norm(control) == 0:
            raise ValueError("Mass-flow linearization requires nonzero nominal thrust.")
        A = np.empty((STATE_SIZE, STATE_SIZE))
        B = np.empty((STATE_SIZE, CONTROL_SIZE))
        for j in range(STATE_SIZE):
            epsilon = relative_step * max(1.0, abs(state[j]))
            if j == MASS:
                epsilon = min(epsilon, 0.25 * state[MASS])
            delta = np.zeros(STATE_SIZE)
            delta[j] = epsilon
            A[:, j] = (
                self.derivative(state + delta, control) - self.derivative(state - delta, control)
            ) / (2 * epsilon)
        for j in range(CONTROL_SIZE):
            epsilon = relative_step * max(1.0, abs(control[j]))
            delta = np.zeros(CONTROL_SIZE)
            delta[j] = epsilon
            B[:, j] = (
                self.derivative(state, control + delta) - self.derivative(state, control - delta)
            ) / (2 * epsilon)
        return A, B

    def linearize_discrete(
        self, state: ArrayLike, control: ArrayLike, dt: float, *, relative_step: float = 1e-5
    ) -> tuple[FloatArray, FloatArray]:
        """Return A13x13, B13x3 in left-multiplicative attitude-error coordinates.

        About x_next=step(x,u,dt), the approximation is
        state_minus(step(state_plus(x,dx),u+du,dt),x_next) = A@dx + B@du.
        Finite differences include the full RK4 map, aerodynamic loads, and
        quaternion normalization. The exact zero-thrust mass-flow norm is
        nonsmooth when depletion is enabled; use a nonzero nominal thrust for
        a differentiable mass-flow linearization.
        """
        dt = _positive_scalar("dt", dt)
        relative_step = _positive_scalar("relative_step", relative_step)
        state, control = self.validate_state(state), self.validate_control(control)
        if self.parameters.specific_impulse is not None and np.linalg.norm(control) == 0:
            raise ValueError("Mass-flow linearization requires nonzero nominal thrust.")
        nominal_next = self.step(state, control, dt)
        A = np.empty((TANGENT_SIZE, TANGENT_SIZE))
        B = np.empty((TANGENT_SIZE, CONTROL_SIZE))
        scales = np.concatenate((np.abs(state[:6]), np.ones(3), np.abs(state[10:14])))
        for j in range(TANGENT_SIZE):
            epsilon = relative_step * max(1.0, scales[j])
            if j == TANGENT_SIZE - 1:
                epsilon = min(epsilon, 0.25 * min(state[MASS], nominal_next[MASS]))
            delta = np.zeros(TANGENT_SIZE)
            delta[j] = epsilon
            plus = self.step(state_plus(state, delta), control, dt)
            minus = self.step(state_plus(state, -delta), control, dt)
            A[:, j] = (state_minus(plus, nominal_next) - state_minus(minus, nominal_next)) / (
                2 * epsilon
            )
        for j in range(CONTROL_SIZE):
            epsilon = relative_step * max(1.0, abs(control[j]))
            delta = np.zeros(CONTROL_SIZE)
            delta[j] = epsilon
            plus = self.step(state, control + delta, dt)
            minus = self.step(state, control - delta, dt)
            B[:, j] = (state_minus(plus, nominal_next) - state_minus(minus, nominal_next)) / (
                2 * epsilon
            )
        return A, B
