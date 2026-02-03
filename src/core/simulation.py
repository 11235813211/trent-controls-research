from typing import Callable
import numpy as np
from enum import Enum, auto

# TO-DO'S
    # add output flexibility
    # implement model, and controller
    # plot graphs and send

class IntegrationMethod(Enum):
    RK4 = auto()

class Simulation:
    def __init__(
            self, 
            plant_model: Callable[[np.ndarray, np.ndarray], np.ndarray], 
            controller: Callable[[np.ndarray, np.ndarray],np.ndarray],
            output: Callable[[np.ndarray, np.ndarray], np.ndarray],
            integration_method: IntegrationMethod, 
            state_0 = np.ndarray, 
            state_ref = np.ndarray, 
            dt=0.001,
            dt_input=0.01, 
            max_time=10):
        self.plant_model = plant_model
        self.controller = controller
        self.output = output

        self.dt = dt
        self.dt_input = dt_input # period of time between input calculations
        self.max_time = max_time

        # make sure dt < dt_input
        if dt > dt_input:
            raise ValueError(f"dt cannot be greater than dt_input: dt={dt}, dt_input={dt_input}.")

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
        input = self.controller(state, self.state_ref)

        outputs = []

        outputs.append((time, self.output(state, input)))

        while time < self.max_time:
            if time_input_counter > self.dt_input:
                time_input_counter -= self.dt_input
                input = self.controller(state, self.state_ref)
                outputs.append((time, self.output(state, input)))

            state = self.time_solver(state, input, self.dt)

            time += self.dt
            time_input_counter = time
        
        return outputs

    # Time Solvers
    def rk4_step(self, x, u, dt):
        k1 = self.plant_model(x, u)
        k2 = self.plant_model(x + dt/2 * k1, u)
        k3 = self.plant_model(x + dt/2 * k2, u)
        k4 = self.plant_model(x + dt * k3, u)
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)