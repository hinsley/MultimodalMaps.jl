import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# 1. LORENZ SYSTEM & POINCARE MAP EXTRACTION
# ==========================================================
sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz(t, state):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def max_x_event(t, state):
    return state[1] - state[0]
max_x_event.direction = 0

print("Extracting 1D Return Map...")
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

# ==========================================================
# 2. COMPUTE KNEADING SEQUENCE
# ==========================================================
N_knead = 200
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

# ==========================================================
# 3. PLOT DETERMINANT WITH TRUNCATIONS
# ==========================================================
print("Plotting Determinant...")

# Expand domain slightly so the root at ~1.13 is clearly visible!
s_vals = np.linspace(0.0, 1.3, 500)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot faint truncated curves (step of 4 to avoid too much noise with N=200)
for N_trunc in range(2, N_knead, 4):
    def D_trunc(s):
        return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N_trunc))
    
    D_trunc_vals = [D_trunc(s) for s in s_vals]
    ax.plot(s_vals, D_trunc_vals, color='darkgreen', alpha=0.2, lw=1.0)

# Plot final N=200 curve
def D_weighted(s): 
    return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N_knead))

D_vals = [D_weighted(s) for s in s_vals]
ax.plot(s_vals, D_vals, 'b-', lw=3.0, zorder=4, label=f'$D(s)$ (N={N_knead})')

ax.axhline(0, color='k', linestyle='--', lw=1.5, zorder=1)

# Plot the root ONLY for N=200 if it falls in the window
crossings = np.where(np.diff(np.sign(D_vals)))[0]
for idx in crossings:
    # Fine-tune the root for accurate plotting
    root_s = root_scalar(D_weighted, bracket=[s_vals[idx], s_vals[idx+1]], method='brentq').root
    ax.plot(root_s, 0, 'ro', markersize=10, zorder=5, label=f'Root $h_{{top}} \\approx {root_s:.4f}$')

ax.set_xlabel('Continuous Entropy ($s$)', fontsize=14)
ax.set_ylabel('Weighted Kneading Determinant $D(s)$', fontsize=14)
ax.set_title(r'Convergence of Weighted Kneading Determinant $D(s)$ (N=200)', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

# Set y-limits
min_D = min(D_vals)
ax.set_ylim(min_D * 1.1, 3)
ax.set_xlim(0.0, 1.3)

plt.tight_layout()
plt.savefig("kneading_determinant_domain.png")
print("Saved kneading_determinant_domain.png")
