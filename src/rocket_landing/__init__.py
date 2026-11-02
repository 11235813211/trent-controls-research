"""Rocket recovery free-flight dynamics and simulation, independent of solvers."""

from rocket_landing.attitude import (
    STATE_SIZE,
    TANGENT_SIZE,
    state_minus,
    state_plus,
)
from rocket_landing.dynamics import (
    CONTROL_SIZE,
    RocketLandingPlant,
    RocketParameters,
    make_state,
)
from rocket_landing.simulation import linearize_trajectory, rollout

__all__ = [
    "CONTROL_SIZE",
    "STATE_SIZE",
    "TANGENT_SIZE",
    "RocketLandingPlant",
    "RocketParameters",
    "linearize_trajectory",
    "make_state",
    "rollout",
    "state_minus",
    "state_plus",
]
