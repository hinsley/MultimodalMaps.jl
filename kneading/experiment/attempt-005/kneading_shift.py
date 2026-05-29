import numpy as np
import matplotlib.pyplot as plt
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

# Let's write out the mathematical series exactly to see if a root physically exists!
N_knead = 25
x_k = np.zeros(N_knead)
eps = np.zeros(N_knead)
T_acc = np.zeros(N_knead)

x_k[0] = f_interp(c_crit)
eps[0] = 1.0
T_acc[0] = 0.0

for n in range(1, N_knead):
    x_k[n] = f_interp(x_k[n-1])
    sign = 1.0 if x_k[n-1] < c_crit else (-1.0 if x_k[n-1] > c_crit else 0.0)
    eps[n] = eps[n-1] * sign
    T_acc[n] = T_acc[n-1] + tau_interp(x_k[n-1])

def D_orig(s): return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N_knead))

# The shifted series evaluates the polynomial from n=1 onward, but algebraically scaled so the first term is 1.0
def D_shifted(s):
    # This is mathematically equivalent to: D_orig(s) - 1.0 (the first term)
    # Then re-normalized by the leading factor eps[1] * exp(-s * T_acc[1])
    # However, if D_orig(s) = 0, then the trailing terms must sum to exactly -1.0.
    # Let's just manually build the exact shifted array to check its physical roots.
    return sum((eps[n] / eps[1]) * np.exp(-s * (T_acc[n] - T_acc[1])) for n in range(1, N_knead))

s_vals = np.linspace(-1, 3, 200)
orig_vals = [D_orig(s) for s in s_vals]
shift_vals = [D_shifted(s) for s in s_vals]

# Check if there are ANY sign crossings in the shifted array!
crossings = np.where(np.diff(np.sign(shift_vals)))[0]
if len(crossings) > 0:
    print(f"Shifted sequence has a root near s = {s_vals[crossings[0]]:.3f}")
else:
    print("Shifted sequence NEVER crosses zero!")

plt.figure(figsize=(8,5))
plt.plot(s_vals, orig_vals, label="Original D(s)")
plt.plot(s_vals, shift_vals, label="Shifted D_shift(s)")
plt.axhline(0, color='k', linestyle='--')
plt.ylim(-5, 5)
plt.legend()
plt.savefig("kneading_shift.png")
