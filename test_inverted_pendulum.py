import numpy as np
from inverted_pendulum import *
from plotter import *

class Scenario():
    def __init__(self, N, DT, X_0, U_T, Q, R, X_REF, M):
        self.N = N
        self.DT = DT
        self.X_0 = X_0
        self.U_T = U_T
        self.Q = Q
        self.R = R
        self.X_REF = X_REF
        self.M = M # number of iterations
N_= 50000
SmallStopScenario = Scenario(
    N=N_,
    DT = 10.0/N_,
    X_0=np.array([1, 0]),
    U_T = np.linspace(0, 0, N_+1).reshape(-1, 1), 
    Q = np.array([
        [10000, 0],
        [0, 100]
    ]),
    R = np.array([[0.01]]), 
    X_REF = np.array([0, 0]),
    M=5
)

def test_nonlinear_plant_sim():
    print("\n=== Nonlinear Plant Sim ===")
    x = nonlinear_plant_sim(
        SmallStopScenario.X_0,
        SmallStopScenario.U_T,
        SmallStopScenario.N,
        SmallStopScenario.DT
    )
    print(x)

def test_linear_plant_sim():
    print("\n=== Linear Plant Sim ===")
    x_t = nonlinear_plant_sim(
            SmallStopScenario.X_0,
            SmallStopScenario.U_T,
            SmallStopScenario.N,
            SmallStopScenario.DT
        )
    A_t, B_t = linearized_plant(
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.DT
    )
    # print(A_t)
    # print(B_t)
    dx_0 = x_t[0] - SmallStopScenario.X_REF
    dx = dx_0.copy()
    for i in range(SmallStopScenario.N):
        # print(A_t[i] @ dx, " vs ", B_t[i] * SmallStopScenario.U_T[i])
        dx = A_t[i] @ dx #+ B_t[i] * SmallStopScenario.U_T[i]
        print(dx)

def test_solve_ilqr_gains():
    print("\n=== iLQR Gain Solver ===")
    x_t = nonlinear_plant_sim(
                SmallStopScenario.X_0,
                SmallStopScenario.U_T,
                SmallStopScenario.N,
                SmallStopScenario.DT
            )
    A_t, B_t = linearized_plant(
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.DT
    )
    K_t, Kv_t, Ku_t, V_t = solve_ilqr_gains(
        A_t, 
        B_t,
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.X_REF,
        SmallStopScenario.Q,
        SmallStopScenario.R
    )
    print(K_t)

def test_ilqr_forward_step():
    print("\n=== iLQR Gain Solver ===")
    x_t = nonlinear_plant_sim(
                SmallStopScenario.X_0,
                SmallStopScenario.U_T,
                SmallStopScenario.N,
                SmallStopScenario.DT
            )
    A_t, B_t = linearized_plant(
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.DT
    )
    K_t, Kv_t, Ku_t, V_t = solve_ilqr_gains(
        A_t, 
        B_t,
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.X_REF,
        SmallStopScenario.Q,
        SmallStopScenario.R
    )
    dx_next, du = ilqr_forward_step(np.zeros_like(SmallStopScenario.X_0),SmallStopScenario.U_T[0], K_t[0], Ku_t[0], Kv_t[0], V_t[1], A_t[0], B_t[0])
    print(dx_next, du)

def test_ilqr_step():
    print("\n=== iLQR Step ===")
    x_t = nonlinear_plant_sim(
                SmallStopScenario.X_0,
                SmallStopScenario.U_T,
                SmallStopScenario.N,
                SmallStopScenario.DT
            )
    A_t, B_t = linearized_plant(
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.DT
    )
    K_t, Kv_t, Ku_t, V_t = solve_ilqr_gains(
        A_t, 
        B_t,
        x_t,
        SmallStopScenario.U_T,
        SmallStopScenario.X_REF,
        SmallStopScenario.Q,
        SmallStopScenario.R
    )
    x_t_next, u_t_next = ilqr_step(
        x_t, 
        SmallStopScenario.U_T, SmallStopScenario.Q, SmallStopScenario.R, SmallStopScenario.N, SmallStopScenario.X_REF, SmallStopScenario.DT)
    print(x_t_next, u_t_next)

