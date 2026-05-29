import numpy as np
from scipy.integrate import solve_ivp
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

N_points = 1
np.random.seed(42)
states = np.zeros((N_points, 12))
states[:, 0] = np.random.uniform(-10, 15, N_points)
states[:, 1] = np.random.uniform(-15, 10, N_points)
states[:, 2] = np.random.uniform(0, 30, N_points)

states[:, 3] = 1.0; states[:, 7] = 1.0; states[:, 11] = 1.0

np.savetxt("kneading/experiment/attempt-004/single_ic.txt", states)

T_max, dt = 120.0, 4.0
steps = int(T_max / dt)

for step in range(steps):
    sol_ee = solve_ivp(rossler_var, (0, dt), states[0], method='RK45', rtol=1e-5, atol=1e-5)
    states[0] = sol_ee.y[:, -1]
    
    M = states[0, 3:12].reshape((3, 3))
    norm_M = np.linalg.norm(M, ord=2)
    
    if step == 0:
        print(f"Step 1 Norm Python: {norm_M}")
    elif step == 29:
        print(f"Step 30 Norm Python: {norm_M}")
