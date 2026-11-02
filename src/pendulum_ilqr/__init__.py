"""Nonlinear pendulum trajectory optimization using iLQR and AL-iLQR."""

from pendulum_ilqr.controls.ilqr import alilqr, ilqr
from pendulum_ilqr.plants.pendulum import (
    GRAVITY_COEFFICIENT,
    linearized_plant_step,
    nonlinear_plant,
)

__all__ = [
    "GRAVITY_COEFFICIENT",
    "alilqr",
    "ilqr",
    "linearized_plant_step",
    "nonlinear_plant",
]
