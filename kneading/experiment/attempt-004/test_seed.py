import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings

warnings.filterwarnings('ignore')
a, b, c_param = 0.2, 0.2, 5.7

def rossler_var(t, state):
    x, y, z = state[0:3]
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c_param)
    
    w11, w12, w13 = state[3:6]
    w21, w22, w23 = state[6:9]
    w31, w32, w33 = state[9:12]
    
    dw11, dw12, dw13 = -w21 - w31, -w22 - w32, -w23 - w33
    dw21, dw22, dw23 = w11 + a*w21, w12 + a*w22, w13 + a*w23
    dw31, dw32, dw33 = z*w11 + (x-c_param)*w31, z*w12 + (x-c_param)*w32, z*w13 + (x-c_param)*w33
    
    return[dx, dy, dz, dw11, dw12, dw13, dw21, dw22, dw23, dw31, dw32, dw33]

N_points = 150
T_max, dt = 120.0, 4.0
steps = int(T_max / dt)

for seed in [42, 43, 44, 45, 46]:
    np.random.seed(seed)
    states = np.zeros((N_points, 12))
    states[:, 0] = np.random.uniform(-10, 15, N_points)
    states[:, 1] = np.random.uniform(-15, 10, N_points)
    states[:, 2] = np.random.uniform(0, 30, N_points)
    states[:, 3] = 1.0; states[:, 7] = 1.0; states[:, 11] = 1.0
    
    log_expansions = np.zeros(N_points)
    for step in range(steps):
        for i in range(N_points):
            sol_ee = solve_ivp(rossler_var, (0, dt), states[i], method='RK45', rtol=1e-5, atol=1e-5)
            states[i] = sol_ee.y[:, -1]
            M = states[i, 3:12].reshape((3, 3))
            norm_M = np.linalg.norm(M, ord=2)
            log_expansions[i] += np.log(norm_M)
            states[i, 3:12] = (M / norm_M).flatten()
            
    curr_T = steps * dt
    max_log = np.max(log_expansions)
    mean_exp = np.mean(np.exp(log_expansions - max_log))
    EE = (max_log + np.log(mean_exp)) / curr_T
    print(f"Seed {seed}, Python EE at T=120: {EE}")
