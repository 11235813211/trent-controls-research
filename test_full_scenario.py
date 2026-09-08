import numpy as np
from inverted_pendulum import *
from plotter import *
from test_inverted_pendulum import SmallStopScenario, Scenario

def test_solve_ilqr():
    print("\n=== iLQR Solver ===")
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
    xs, us = solve_ilqr(
        SmallStopScenario.X_0,
        SmallStopScenario.U_T,
        SmallStopScenario.Q,
        SmallStopScenario.R,
        SmallStopScenario.N,
        SmallStopScenario.X_REF,
        SmallStopScenario.DT,
        SmallStopScenario.M
    )
    plot_ilqr(xs, us, SmallStopScenario.DT)
    
