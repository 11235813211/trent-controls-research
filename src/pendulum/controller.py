import numpy as np
from typing import Tuple, List
import scipy.linalg

# CONSTANT CONTROLLER
def pendulum_controller_constant(state_: np.ndarray, state_ref: np.ndarray, control_state = None, gains = [[1]]) -> Tuple[np.ndarray, np.ndarray]:
    return gains, None

# PID
def pendulum_controller_pid(state_: np.ndarray, state_ref: np.ndarray, control_state: np.ndarray = None, gains: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    previous_state: np.ndarray  = control_state[0]
    err_integral: np.ndarray = control_state[1]
    dt: float = control_state[2]

    kp = gains[0] # array of kp's
    ki = gains[1]
    kd = gains[2]

    err = state_ - state_ref
    err_derivative = (state_ - previous_state) * 1/dt
    err_integral += err
    
    u = [kp @ err + ki @ err_integral + kd @ err_derivative]

    control_state[0] = state_.copy()
    control_state[1] = err_integral.copy()

    return u, control_state

# LQR
def solve_lqr_gains(A, B, Q, R):
    # looked up how to do this on google
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)

    K = np.linalg.inv(R) @ B.T @ P
    return K

def pendulum_controller_lqr(state_: np.ndarray, state_ref: np.ndarray, control_state: List = None, gains: List = None) -> Tuple[np.ndarray, List]:
    K = gains

    err = state_ - state_ref

    return (-1 * K @ err, None)



