"""Nonlinear pendulum trajectory optimization using iLQR and AL-iLQR."""

from pendulum_ilqr.dynamics import GRAVITY_COEFFICIENT, linearized_plant_step, nonlinear_plant
from pendulum_ilqr.solver import alilqr, ilqr

__all__ = [
    "GRAVITY_COEFFICIENT",
    "alilqr",
    "ilqr",
    "linearized_plant_step",
    "nonlinear_plant",
]
