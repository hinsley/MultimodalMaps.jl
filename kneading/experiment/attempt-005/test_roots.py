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

N_knead = 50
x_k = np.zeros(N_knead)
eps = np.zeros(N_knead)
T_acc = np.zeros(N_knead)

x_k[0] = f_interp(c_crit)
eps[0] = 1.0
T_acc[0] = 0.0

for n in range(1, N_knead):
    x_k[n] = f_interp(x_k[n-1])
    sign = 1.0 if x_k[n-1] < c_crit else -1.0
    eps[n] = eps[n-1] * sign
    T_acc[n] = T_acc[n-1] + tau_interp(x_k[n-1])

# Original D(s)
def D_orig(s): return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N_knead))

# Let's try many combinations to see if anything hits ~0.92
def get_root(func):
    try:
        return root_scalar(func, bracket=[0.1, 2.5], method='brentq').root
    except:
        return None

print(f"Original Root: {get_root(D_orig)}")

# What if eps[0] is negative?
def D_neg(s): return -1.0 + sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(1, N_knead))
print(f"Root if eps[0] = -1: {get_root(D_neg)}")

# What if we shift the sequence T_acc, but don't normalize?
def D_shift_T(s): return sum(eps[n] * np.exp(-s * T_acc[n+1]) for n in range(N_knead-1))
print(f"Root if shift T: {get_root(D_shift_T)}")

# What if the user meant: "pretending the critical point is at the IMAGE"
# meaning we just drop the first iteration of the mapping completely. 
# We start with x_k[0] = c_crit.
# T_acc[0] = 0.
x_k_c = np.zeros(N_knead)
eps_c = np.zeros(N_knead)
T_acc_c = np.zeros(N_knead)
x_k_c[0] = c_crit
eps_c[0] = 1.0
T_acc_c[0] = 0.0
for n in range(1, N_knead):
    x_k_c[n] = f_interp(x_k_c[n-1])
    sign = 1.0 if x_k_c[n-1] < c_crit else -1.0
    eps_c[n] = eps_c[n-1] * sign
    T_acc_c[n] = T_acc_c[n-1] + tau_interp(x_k_c[n-1])

def D_start_c(s): return sum(eps_c[n] * np.exp(-s * T_acc_c[n]) for n in range(N_knead))
print(f"Root if starting at c_crit: {get_root(D_start_c)}")

# What if we start at c_crit, but drop the first term?
def D_start_c_drop(s): return sum(eps_c[n] * np.exp(-s * T_acc_c[n]) for n in range(1, N_knead))
print(f"Root if starting at c_crit but dropping eps0: {get_root(D_start_c_drop)}")

