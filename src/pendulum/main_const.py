import numpy as np
from src.core.closed_state_space import ClosedStateSpaceModel
from src.core.simulation import Simulation, IntegrationMethod
from src.core.plotter import plot_outputs

from src.pendulum.controller import pendulum_controller_constant, pendulum_controller_pid
from src.pendulum.output import pendulum_output
from src.pendulum.plant import pendulum_model

print("Initializing variables.")
# INITIALIZATION
dt = 0.01
dt_computer = 0.05
max_time = 15

# Plant State
g = 9.8 # m / s^2
m = 30 # kg
L = 2 # m
plant_state = [g, m, L]

state_0 = np.array([
    [0.01],
    [0]
])

state_ref = np.array([
    [0],
    [0]
])

gains = [[0]]

control_state_0 = None

output_names = ["Angle (rad)", "Angular Velocity (rad/s)"]

# MODEL
print("Creating model.")
PendulumClosedModel_constant = ClosedStateSpaceModel(pendulum_model, pendulum_controller_constant, pendulum_output, plant_state, control_state_0, gains)

print("Creating simulation.")
PendulumSim_constant = Simulation(
    PendulumClosedModel_constant,
    IntegrationMethod.RK4,
    state_0,
    state_ref,
    dt,
    dt_computer,
    max_time
    )

print("Running simulation... ")
outputs = PendulumSim_constant.simulate()


print("Simulation finished!")
plot_outputs(outputs, output_names)


