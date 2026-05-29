import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings

warnings.filterwarnings('ignore')

sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz(t, state):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def max_x_event(t, state):
    return state[1] - state[0]
max_x_event.direction = 0

print("Extracting 1D map...")
sol_map = solve_ivp(lorenz, (0, 400), [1.0, 1.0, 1.0], method='RK45', 
                    events=[max_x_event], rtol=1e-8, atol=1e-8, max_step=0.1)

y_ev = sol_map.y_events[0]
is_max = y_ev[:, 2] > (rho - 1)
abs_x_max = np.abs(y_ev[is_max, 0])
t_max = sol_map.t_events[0][is_max]

x_n, x_np1, tau_n = abs_x_max[:-1], abs_x_max[1:], np.diff(t_max)
x_n_u, idx_u = np.unique(np.round(x_n, 4), return_inverse=True)
x_np1_u, tau_u = np.zeros_like(x_n_u), np.zeros_like(x_n_u)
for i in range(len(x_n_u)):
    mask = (idx_u == i)
    x_np1_u[i], tau_u[i] = np.mean(x_np1[mask]), np.mean(tau_n[mask])

def f_interp(x): return np.interp(x, x_n_u, x_np1_u)
def tau_interp(x): return np.interp(x, x_n_u, tau_u)

c_crit = x_n_u[np.argmax(x_np1_u)]
f_c = f_interp(c_crit)
ff_c = f_interp(f_c)

N_knead = 50

print("1. Standard Kneading Determinant (Critical Point = c, Start = f(c))")
x_k = np.zeros(N_knead)
eps = np.zeros(N_knead)
T_acc = np.zeros(N_knead)
x_k[0] = f_c
eps[0] = 1.0
T_acc[0] = 0.0

for n in range(1, N_knead):
    x_k[n] = f_interp(x_k[n-1])
    sign = 1.0 if x_k[n-1] < c_crit else -1.0
    eps[n] = eps[n-1] * sign
    T_acc[n] = T_acc[n-1] + tau_interp(x_k[n-1])

def D_orig(s): return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N_knead))
root_orig = root_scalar(D_orig, bracket=[0.5, 2.5], method='brentq').root
print(f" -> Root: {root_orig:.5f}")


print("\n2. Shifted Start (Critical Point = c, Start = f(f(c)))")
x_k2 = np.zeros(N_knead)
eps2 = np.zeros(N_knead)
T_acc2 = np.zeros(N_knead)
x_k2[0] = ff_c
eps2[0] = 1.0
T_acc2[0] = 0.0

for n in range(1, N_knead):
    x_k2[n] = f_interp(x_k2[n-1])
    sign = 1.0 if x_k2[n-1] < c_crit else -1.0
    eps2[n] = eps2[n-1] * sign
    T_acc2[n] = T_acc2[n-1] + tau_interp(x_k2[n-1])

def D_shift_start(s): return sum(eps2[n] * np.exp(-s * T_acc2[n]) for n in range(N_knead))
try:
    root2 = root_scalar(D_shift_start, bracket=[0.0001, 2.5], method='brentq').root
    print(f" -> Root: {root2:.5f}")
except ValueError:
    print(f" -> No root found! D(0.0001)={D_shift_start(0.0001):.2f}, D(2.5)={D_shift_start(2.5):.2f}")


print("\n3. Shifted Start + Shifted Critical Point (Critical Point = f(c), Start = f(f(c)))")
x_k3 = np.zeros(N_knead)
eps3 = np.zeros(N_knead)
T_acc3 = np.zeros(N_knead)
x_k3[0] = ff_c
eps3[0] = 1.0
T_acc3[0] = 0.0

for n in range(1, N_knead):
    x_k3[n] = f_interp(x_k3[n-1])
    # The condition requested: pretending critical point is f(c)
    sign = 1.0 if x_k3[n-1] < f_c else -1.0
    eps3[n] = eps3[n-1] * sign
    T_acc3[n] = T_acc3[n-1] + tau_interp(x_k3[n-1])

def D_shift_all(s): return sum(eps3[n] * np.exp(-s * T_acc3[n]) for n in range(N_knead))
try:
    root3 = root_scalar(D_shift_all, bracket=[0.0001, 2.5], method='brentq').root
    print(f" -> Root: {root3:.5f}")
except ValueError:
    print(f" -> No root found! D(0.0001)={D_shift_all(0.0001):.2f}, D(2.5)={D_shift_all(2.5):.2f}")

