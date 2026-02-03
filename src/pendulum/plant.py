import numpy as np
from typing import List

def pendulum_model(state_: np.ndarray, input_: np.ndarray, plant_state: List = None) -> np.ndarray:
    g = plant_state[0]
    m = plant_state[1]
    L = plant_state[2]
    
    theta = state_[0][0]
    omega = state_[1][0]

    u = input_[0][0]
    state_dot = np.array([
        [omega],
        [g / L * np.sin(theta) + u / (m * L)]
    ])
    return state_dot
