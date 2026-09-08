import numpy as np
from pygame import K_t

# global macros on plant dynamics
M = 3 # kg
G = 9.81 # m / s^2

def nonlinear_plant(x, u, dt):
    # u is an array, so extract the scalar
    u_val = u[0] if isinstance(u, np.ndarray) else u 
    
    f = u_val + M*G*np.sin(x[0])
    x_next = np.array([
        x[0] + dt * x[1],
        x[1] + dt * f
    ])
    return x_next

def linearized_plant(x_t_, u_t_, dt):
    N = len(u_t_) - 1
    A_t = np.zeros((N+1, 2, 2))
    B_t = np.zeros((N+1, 2, 1))
    
    for i in range(len(x_t_)):
        u_val = u_t_[i][0] if isinstance(u_t_[i], np.ndarray) else u_t_[i]
        
        A_t[i] = np.array([
            [1, dt],
            [M*G*np.cos(x_t_[i][0])*dt, 1],
        ])
        
        B_t[i] = np.array([
            [0],
            [dt]
        ])
        
    return A_t, B_t

def nonlinear_plant_sim(x_0_, u_t_, N_, dt_):
    x_t = np.zeros((N_+1, 2))
    x_t[0] = x_0_
    for i in range(N_):
        x_t[i+1] = nonlinear_plant(x_t[i], u_t_[i], dt_)
    return x_t

def solve_ilqr_gains(A_t_, B_t_, x_t_, u_t_, x_ref, Q_, R_):
    N = len(u_t_) - 1
    # init
    S_t = np.zeros((N+1, 2, 2))
    V_t = np.zeros((N+1, 2))
    K_t = np.zeros((N+1, 1, 2))
    Kv_t = np.zeros((N+1, 1, 2))
    Ku_t = np.zeros((N+1, 1, 1))

    # initialize S, V @ t=N
    S_t[N] = Q_.copy()
    V_t[N] = Q_ @ (x_t_[N] - x_ref)

    for i in range(N-1, -1, -1):
        
        inv = np.linalg.inv((B_t_[i].T @ S_t[i+1] @ B_t_[i] + R_))
        
        K_t[i]  = inv @ B_t_[i].T @ S_t[i+1] @ A_t_[i]
        Kv_t[i] = inv @ B_t_[i].T
        Ku_t[i] = inv @ R_
        
        S_t[i] = A_t_[i].T @ S_t[i+1] @ (A_t_[i] - B_t_[i] @ K_t[i]) + Q_
        
        V_t[i] = (A_t_[i] - B_t_[i] @ K_t[i]).T @ V_t[i+1] - K_t[i].T @ R_ @ u_t_[i] + Q_ @ x_t_[i]

    return K_t, Kv_t, Ku_t, V_t

def ilqr_forward_step(dx, u, K, Ku, Kv, V_next, A, B):
    du = -1 * K @ dx - Ku @ u - Kv @ V_next
    dx_next = A @ dx + B @ du
    return dx_next, du

def ilqr_step(x_t, u_t, Q, R, N, x_ref, dt):
    x_t_next = x_t.copy()
    u_t_next = u_t.copy()

    A_t, B_t = linearized_plant(x_t, u_t, dt)
    K_t, Kv_t, Ku_t, V_t = solve_ilqr_gains(A_t, B_t, x_t, u_t, x_ref, Q, R)

    dx = np.zeros((N+1, 2))
    du = np.zeros((N+1, 1))

    for i in range(N):
        dx[i+1], du[i] = ilqr_forward_step(dx[i], u_t[i], K_t[i], Ku_t[i], Kv_t[i], V_t[i+1], A_t[i], B_t[i])

    x_t_next = x_t + dx
    u_t_next = u_t + du
    
    return x_t_next, u_t_next

def solve_ilqr(x_0, u_t, 
Q, R, N, x_ref, dt,M):
    xs = np.empty((M+1, N+1, 2))
    us = np.empty((M+1, N+1, 1))

    x_t = nonlinear_plant_sim(x_0, u_t, N, dt)
    xs[0] = x_t
    us[0] = u_t

    for i in range(M):
        xs[i+1], us[i+1] = ilqr_step(xs[i], us[i], Q, R, N, x_ref, dt)

    return xs, us

    



