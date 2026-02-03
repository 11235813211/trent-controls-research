from typing import Callable
import numpy as np
from enum import Enum, auto

from src.core.closed_state_space import ClosedStateSpaceModel

# TO-DO'S
    # add output flexibility
    # implement model, and controller
    # plot graphs and send

class IntegrationMethod(Enum):
    RK4 = auto()

class Simulation:
    def __init__(
            self, 
            state_space: ClosedStateSpaceModel,
            integration_method: IntegrationMethod, 
            state_0 = np.ndarray, 
            state_ref = np.ndarray, 
            dt=0.001,
            dt_computer=0.01, 
            max_time=10):
        
        self.state_space: ClosedStateSpaceModel = state_space

        self.dt = dt
        self.dt_computer = dt_computer # period of time between input calculations
        self.max_time = max_time

        # make sure dt < dt_computer
        if dt > dt_computer:
            raise ValueError(f"dt cannot be greater than dt_computer: dt={dt}, dt_computer={dt_computer}.")

        self.state_0 = state_0
        self.state_ref = state_ref

        # Time Solver Setup
        self.time_solver = None
        if integration_method == IntegrationMethod.RK4:
            self.time_solver = self.rk4_step
        else:
            raise ValueError(f"invalid integration method supplied: {integration_method}.")
    
    def simulate(self):
        time = 0
        time_input_counter = 0
        state = self.state_0.copy()
        input = self.state_space.controller(state, self.state_ref)

        outputs = []

        outputs.append((time, self.state_space.output(state, input)))

        while time < self.max_time:
            if time_input_counter > self.dt_computer:
                time_input_counter -= self.dt_computer
                input = self.state_space.controller(state, self.state_ref)
                outputs.append((time, self.state_space.output(state, input)))

            state = self.time_solver(state, input, self.dt)

            time += self.dt
            time_input_counter = time
        
        return outputs

    # Time Solvers
    def rk4_step(self, x, u, dt):
        k1 = self.state_space.state_derivative(x, u)
        k2 = self.state_space.state_derivative(x + dt/2 * k1, u)
        k3 = self.state_space.state_derivative(x + dt/2 * k2, u)
        k4 = self.state_space.state_derivative(x + dt * k3, u)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)