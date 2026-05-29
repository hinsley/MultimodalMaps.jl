import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# 1. LORENZ SYSTEM & VARIATIONAL EQUATIONS
# ==========================================================
sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz(t, state):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def lorenz_var(t, state):
    x, y, z = state[0:3]
    w0, w1, w2 = state[3:6]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    dw0 = sigma * (w1 - w0)
    dw1 = (rho - z) * w0 - w1 - x * w2
    dw2 = y * w0 + x * w1 - beta * w2
    return [dx, dy, dz, dw0, dw1, dw2]

def max_x_event(t, state):
    return state[1] - state[0] # y - x = 0 (maxima of |x|)
max_x_event.direction = 0

# ==========================================================
# 2. EXTRACT 1D RETURN MAP OF |x| MAXIMA
# ==========================================================
print("Extracting 1D Return Map...")
sol_map = solve_ivp(lorenz, (0, 400), [1.0, 1.0, 1.0], method='RK45', 
                    events=[max_x_event], rtol=1e-8, atol=1e-8)

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

# ==========================================================
# 3. KNEADING DETERMINANT CONVERGENCE
# ==========================================================
print("Computing Kneading Convergence...")
N_knead_max = 80
x_k, eps, T_acc = np.zeros(N_knead_max), np.zeros(N_knead_max), np.zeros(N_knead_max)
x_k[0], eps[0], T_acc[0] = f_interp(c_crit), 1.0, 0.0

for n in range(1, N_knead_max):
    x_k[n] = f_interp(x_k[n-1])
    sign = 1.0 if x_k[n-1] < c_crit else (-1.0 if x_k[n-1] > c_crit else 0.0)
    eps[n] = eps[n-1] * sign
    T_acc[n] = T_acc[n-1] + tau_interp(x_k[n-1])

htop_vals = []
T_knead_vals = []

for N in range(2, N_knead_max):
    def D_weighted(s): return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N) if eps[n] != 0)
    
    if D_weighted(0.0) <= 0.0:
        if np.sign(D_weighted(0.5)) != np.sign(D_weighted(2.0)):
            res = root_scalar(D_weighted, bracket=[0.5, 2.0], method='brentq')
            htop_vals.append(res.root)
            T_knead_vals.append(T_acc[N-1])

final_htop = htop_vals[-1]

# ==========================================================
# 4. EXPANSION ENTROPY CONVERGENCE
# ==========================================================
print("Computing Expansion Entropy Convergence...")
num_ics_ee = 200
T_max = 60.0
dt = 1.0
steps = int(T_max / dt)

np.random.seed(42)
states = np.zeros((num_ics_ee, 6))
# Sample inside typical Lorenz bounding box
states[:, 0] = np.random.uniform(-20, 20, num_ics_ee)
states[:, 1] = np.random.uniform(-30, 30, num_ics_ee)
states[:, 2] = np.random.uniform(0, 50, num_ics_ee)
# Start tangent vector as unit vector
v_init = np.random.randn(num_ics_ee, 3)
states[:, 3:6] = v_init / np.linalg.norm(v_init, axis=1, keepdims=True)

log_expansions = np.zeros(num_ics_ee)
EE_vals = []
T_ee_vals = []

for step in range(steps):
    for i in range(num_ics_ee):
        sol_ee = solve_ivp(lorenz_var, (0, dt), states[i], method='RK45', rtol=1e-8, atol=1e-8)
        state_end = sol_ee.y[:, -1]
        
        v = state_end[3:6]
        norm_v = np.linalg.norm(v)
        
        log_expansions[i] += np.log(norm_v)
        state_end[3:6] = v / norm_v
        states[i] = state_end
        
    curr_T = (step + 1) * dt
    max_log = np.max(log_expansions)
    mean_exp = np.mean(np.exp(log_expansions - max_log))
    EE = (max_log + np.log(mean_exp)) / curr_T
    
    EE_vals.append(EE)
    T_ee_vals.append(curr_T)

final_ee = EE_vals[-1]

# ==========================================================
# 5. PLOT RESULTS
# ==========================================================
print("Plotting...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(T_ee_vals, EE_vals, 'b-o', markersize=4, label='Expansion Entropy Ensemble')
ax.plot(T_knead_vals, htop_vals, 'r-s', markersize=4, linewidth=2, label='Weighted Kneading Roots')

ax.axhline(final_htop, color='k', linestyle='--', linewidth=1.5, label=f'Converged Exact Entropy ({final_htop:.4f})')
ax.axhline(final_ee, color='blue', linestyle='--', linewidth=1.5, label=f'Converged Expansion Entropy ({final_ee:.4f})')

ax.set_xlabel('Continuous Elapsed Time (T)', fontsize=12)
ax.set_ylabel('Topological Entropy Estimate', fontsize=12)
ax.set_title('Convergence: Expansion Entropy vs. Weighted Kneading Roots (Lorenz)', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("kneading/experiment/attempt-005/lorenz_convergence.png")
print("Saved lorenz_convergence.png")
