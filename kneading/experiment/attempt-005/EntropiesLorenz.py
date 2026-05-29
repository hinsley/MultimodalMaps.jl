import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.ticker as ticker
import warnings

warnings.filterwarnings('ignore')

# ==========================================================
# 1. FAST LEMPEL-ZIV (1976) IMPLEMENTATION
# ==========================================================
def lz76_fast(s):
    n = len(s)
    if n == 0: return 0
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while True:
        if l + k - 1 < n and i + k - 1 < n and s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max: k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l > n: break
                else: i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c

# ==========================================================
# 2. LORENZ SYSTEM & VARIATIONAL EQUATIONS
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
# 3. EXTRACT 1D RETURN MAP OF |x| MAXIMA
# ==========================================================
print("Extracting 1D Return Map...")
sol_map = solve_ivp(lorenz, (0, 1000), [1.0, 1.0, 1.0], method='RK45', 
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
# 4. COMPUTE TOPOLOGICAL ENTROPIES
# ==========================================================
print("Computing Topological Entropies...")
N_knead = 5000

# Calculate the map itinerary from f(c)
curr_x = f_interp(c_crit)
eps_list = []
T_list = []
e = 1.0
T_sum = 0.0
for n in range(1, N_knead + 1):
    # sign of branch of previous point: Left branch (1), Right branch (-1)
    sign = 1.0 if curr_x < c_crit else -1.0
    e *= sign
    eps_list.append(e)
    # T_n are the accumulated times starting from tau(f(c))
    T_sum += tau_interp(curr_x)
    T_list.append(T_sum)
    curr_x = f_interp(curr_x)

# 4a. Exact h_top (Weighted Kneading - Rugh Scalar Determinant)
# For the Lorenz flow, the critical orbit returns are extremely fast, 
# yielding a root that tracks the upper topological supremum.
def D_weighted(s):
    # D(s) = 1 + sum eps_n exp(-s T_n)
    total = sum(eps * np.exp(-s * t) for eps, t in zip(eps_list, T_list))
    return 1.0 + total

H_top_exact = root_scalar(D_weighted, bracket=[0.5, 1.5], method='brentq').root

# 4b. Approx h_top (Suspended Template Entropy)
# This is the standard procedure: h_map / mean_return_time
tau_avg = np.mean(tau_n)
def D_unweighted(t): return 1.0 + sum(eps_list[n-1] * (t**n) for n in range(1, N_knead + 1))
t0 = root_scalar(D_unweighted, bracket=[0.1, 0.9], method='brentq').root
H_top_approx = (-np.log(t0)) / tau_avg

# ==========================================================
# 5. ENSEMBLE 1: KS ENTROPIES (LLE)
# ==========================================================
num_ics_lle = 100
t_max_lle = 2000
dt = 10.0
print(f"Integrating {num_ics_lle} trajectories for LLE distribution...")

lle_vals = []
np.random.seed(42)
for i in range(num_ics_lle):
    if (i+1) % 20 == 0:
        print(f"  Trajectory {i+1}/{num_ics_lle}...")
    idx = np.random.randint(100, sol_map.t.size)
    curr_state = np.append(sol_map.y[:, idx], [1.0, 0.0, 0.0])
    LE_sum = 0.0
    for _ in range(int(t_max_lle / dt)):
        sol = solve_ivp(lorenz_var, (0, dt), curr_state, method='RK45', rtol=1e-5, atol=1e-5)
        curr_state = sol.y[:, -1]
        norm_w = np.linalg.norm(curr_state[3:6])
        LE_sum += np.log(norm_w)
        curr_state[3:6] /= norm_w
    lle_vals.append(LE_sum / t_max_lle)

# Expansion Entropy (EE) is computed as the log average of the finite-time expansion factors
max_le_sum = np.max([val * t_max_lle for val in lle_vals])
mean_exp = np.mean([np.exp(val * t_max_lle - max_le_sum) for val in lle_vals])
EE_val = (max_le_sum + np.log(mean_exp)) / t_max_lle

# ==========================================================
# 6. ENSEMBLE 2: LZ76 COMPLEXITY
# ==========================================================
num_ics_lz = 500
N_iter = 2500
print(f"Simulating {num_ics_lz} trajectories for LZ76 distribution...")

xs_curr = np.linspace(c_crit - 1.5, c_crit + 1.5, num_ics_lz)
for _ in range(20): xs_curr = f_interp(xs_curr) # settle

symbols = np.zeros((num_ics_lz, N_iter), dtype=np.int8)
T_tots = np.zeros(num_ics_lz)
for j in range(N_iter):
    xs_curr = f_interp(xs_curr)
    symbols[:, j] = (xs_curr >= c_crit).astype(np.int8)
    T_tots += tau_interp(xs_curr)

lz_vals = [lz76_fast("".join(symbols[i].astype(str))) * np.log(N_iter) / T_tots[i] for i in range(num_ics_lz)]

# ==========================================================
# 7. FINAL PLOT
# ==========================================================
fig, ax = plt.subplots(figsize=(12, 7))

bins = np.linspace(0.82, 1.18, 60)
c_lz, e_lz = np.histogram(lz_vals, bins=bins)
c_lle, e_lle = np.histogram(lle_vals, bins=bins)

# Scale LLE
scale = num_ics_lz / num_ics_lle
ax.bar(e_lz[:-1], c_lz, width=np.diff(e_lz), align='edge', color='royalblue', edgecolor='black', alpha=0.8, label='LZ76 Complexity')
ax.bar(e_lle[:-1], -c_lle * scale, width=np.diff(e_lle), align='edge', color='darkorange', edgecolor='black', alpha=0.8, label='KS Entropy/LLE')

# Entropy Lines
ax.axvline(H_top_exact, color='red', ls='--', lw=2.5, label=f'Exact $h_{{top}} (Knead) \\approx {H_top_exact:.4f}$')
ax.axvline(H_top_approx, color='purple', ls='-.', lw=2.5, label=f'Approx $h_{{map}}/\\bar{{\\tau}} \\approx {H_top_approx:.4f}$')
ax.axvline(np.mean(lz_vals), color='green', ls=':', lw=3, label=f'Mean LZ76 $\\approx {np.mean(lz_vals):.4f}$')
ax.axvline(np.mean(lle_vals), color='darkred', ls=':', lw=3, label=f'Mean KS $\\approx {np.mean(lle_vals):.4f}$')
ax.axvline(EE_val, color='blue', ls='-', lw=2, label=f'Expansion Entropy $\\approx {EE_val:.4f}$')

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
ax.axhline(0, color='black', lw=1.5)
ax.set_xlabel('Continuous Entropy (nats / time)', fontsize=12)
ax.set_ylabel('Frequency (Trajectories)', fontsize=12)
ax.set_title('The Hierarchy of Chaos: Metric vs. Topological Entropies (Lorenz System)', fontsize=14)
ax.set_ylim(-max(c_lz)*1.1, max(c_lz)*1.1)
ax.legend(loc='upper right', fontsize=10, ncol=2)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("entropies_lorenz.png")
print("Saved entropies_lorenz.png")
