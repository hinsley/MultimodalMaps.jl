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

states = np.loadtxt("kneading/experiment/attempt-004/single_ic.txt")
y0 = states if states.ndim == 1 else states[0, :]

sol = solve_ivp(rossler_var, (0, 4.0), y0, method='RK45', rtol=1e-5, atol=1e-5)
print("Python end state:", sol.y[:, -1])
