import numpy as np
from src.core.simulation import IntegrationMethod, Simulation
from src.core.closed_state_space import ClosedStateSpaceModel
from src.core.plotter import plot_outputs
from pprint import pprint
from typing import List


def walker_model(state_: np.ndarray, input_: np.ndarray, plant_state: List = None) -> np.ndarray:
    A = np.array([
        [0, 1],
        [0, 0]
    ])

    B = np.array([
        [[0]],
        [[1]]
    ])

    derivative = A @ state_ + B @ input_
    # print("\t\tA @ Derivative: ", A, state_, A @ state_)
    return derivative 

def walker_output(state: np.ndarray, input: np.ndarray) -> np.ndarray:
    return state

def walker_controller(state: np.ndarray, state_ref: np.ndarray, control_state = None, gains = None) -> np.ndarray:
    return np.array([1]), None

WalkerClosedModel = ClosedStateSpaceModel(walker_model, walker_controller, walker_output)

# Simulation
state_0 = np.array([
    [5],
    [-1]
])

state_ref = np.array([
    [0],
    [0]
])

WalkerSim = Simulation(
    WalkerClosedModel,
    IntegrationMethod.RK4,
    state_0,
    state_ref,
    dt=0.01,
    dt_computer=0.1,
    max_time=5
)

outputs = WalkerSim.simulate()

output_names = ["Position", "Velocity"]

plot_outputs(outputs, output_names)
