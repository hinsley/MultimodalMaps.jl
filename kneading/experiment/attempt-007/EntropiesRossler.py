import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.ticker as ticker
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# ==========================================================
# 1. FAST LEMPEL-ZIV (1976) IMPLEMENTATION
# ==========================================================
def lz76_fast(s):
    n = len(s)
    if n == 0:
        return 0
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while True:
        if l + k - 1 < n and i + k - 1 < n and s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


# ==========================================================
# 2. ROSSLER SYSTEM & VARIATIONAL EQUATIONS
# ==========================================================
a, b, c_param = 0.2, 0.2, 5.7


def rossler(t, state):
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c_param)]


def rossler_var(t, state):
    x, y, z = state[0:3]
    w0, w1, w2 = state[3:6]

    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c_param)

    # Tangent flow: d/dt w = J(x) w, J = [[0,-1,-1],[1,a,0],[z,0,x-c]]
    dw0 = -w1 - w2
    dw1 = w0 + a * w1
    dw2 = z * w0 + (x - c_param) * w2

    return [dx, dy, dz, dw0, dw1, dw2]


def max_x_event(t, state):
    # Local maxima of x occur when dx/dt = -y - z = 0 with downward crossing.
    return -state[1] - state[2]


max_x_event.direction = -1


# ==========================================================
# 3. RETURN MAP + KNEADING DATA
# ==========================================================
def first_positive_root(func, a=0.01, b=2.0, grid=4000):
    ss = np.linspace(a, b, grid)
    vals = np.array([func(s) for s in ss], dtype=np.float64)

    close = np.where(np.isfinite(vals) & (np.abs(vals) < 1e-12))[0]
    if close.size > 0:
        return float(ss[int(close[0])])

    for i in range(grid - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not np.isfinite(v0) or not np.isfinite(v1):
            continue
        if np.sign(v0) == 0:
            return float(ss[i])
        if np.sign(v0) != np.sign(v1):
            return float(root_scalar(func, bracket=[ss[i], ss[i + 1]], method="brentq").root)

    raise RuntimeError(f"No positive root found in [{a}, {b}]")


print("Extracting 1D Rossler return map...")
sol_map = solve_ivp(
    rossler,
    (0, 8000),
    [1.0, 1.0, 1.0],
    method="RK45",
    events=[max_x_event],
    rtol=1e-10,
    atol=1e-10,
)

x_events_all = sol_map.y_events[0][:, 0]
t_events_all = sol_map.t_events[0]
if x_events_all.size < 600:
    raise RuntimeError("Too few return maxima extracted from Rossler trajectory")

burn = 150
x_events = x_events_all[burn:]
t_events = t_events_all[burn:]

x_n = x_events[:-1]
x_np1 = x_events[1:]
tau_n = np.diff(t_events)

# Denoise via deterministic 1D map averaging on rounded x bins.
x_n_u, idx_u = np.unique(np.round(x_n, 4), return_inverse=True)
x_np1_u = np.zeros_like(x_n_u)
tau_u = np.zeros_like(x_n_u)
for i in range(len(x_n_u)):
    mask = idx_u == i
    x_np1_u[i] = np.mean(x_np1[mask])
    tau_u[i] = np.mean(tau_n[mask])


def f_interp(x):
    return np.interp(x, x_n_u, x_np1_u)


def tau_interp(x):
    return np.interp(x, x_n_u, tau_u)


# Critical point of the map taken at map maximum (unimodal assumption).
c_map = float(x_n_u[np.argmax(x_np1_u)])
f_c = float(f_interp(c_map))
ff_c = float(f_interp(f_c))

# Branch intervals requested by the task:
# left = [f(f(c_map)), c_map], right = [c_map, f(c_map)]
left_lo, left_hi = sorted((ff_c, c_map))
right_lo, right_hi = sorted((c_map, f_c))


# Interval-based branch classifier with a deterministic fallback for rare out-of-range points.
def classify_left(xs):
    arr = np.asarray(xs)
    left = (arr >= left_lo) & (arr <= left_hi)
    right = (arr >= right_lo) & (arr <= right_hi)
    unresolved = ~(left | right)
    if np.any(unresolved):
        left = left.copy()
        left[unresolved] = arr[unresolved] < c_map
    return left, int(np.count_nonzero(unresolved))


left_data, unresolved_data = classify_left(x_n)
right_data = ~left_data
if not np.any(left_data) or not np.any(right_data):
    raise RuntimeError("Failed to form both branch populations from interval definitions")

tau_left = float(np.mean(tau_n[left_data]))
tau_right = float(np.mean(tau_n[right_data]))

print("Computing kneading itineraries...")
N_knead = 5000
curr_x = float(f_interp(c_map))

xs = []
branch_left = []
eps_list = []

e = 1.0
unresolved_knead = 0
for _ in range(N_knead):
    is_left_arr, ucount = classify_left(np.array([curr_x]))
    is_left = bool(is_left_arr[0])
    unresolved_knead += ucount

    sign = 1.0 if is_left else -1.0
    e *= sign

    xs.append(curr_x)
    branch_left.append(is_left)
    eps_list.append(e)

    curr_x = float(f_interp(curr_x))

xs = np.array(xs, dtype=np.float64)
branch_left = np.array(branch_left, dtype=bool)
eps_list = np.array(eps_list, dtype=np.float64)

# Branch-constant times (weighted-system correction).
tau_branch = np.where(branch_left, tau_left, tau_right)
T_branch = np.cumsum(tau_branch)


# ==========================================================
# 4. TOPOLOGICAL ENTROPIES
# ==========================================================
print("Computing topological entropy roots...")


def D_branch_weighted(s):
    return 1.0 + np.sum(eps_list * np.exp(-s * T_branch))


H_top_branch = first_positive_root(D_branch_weighted, a=0.01, b=2.0)


# Unweighted map entropy per iterate, then suspension approximation h_map/tau_mean_branch.
def D_unweighted(t):
    powers = t ** np.arange(1, N_knead + 1)
    return 1.0 + np.sum(eps_list * powers)


t0 = first_positive_root(D_unweighted, a=0.05, b=0.95)
h_map = -np.log(t0)
tau_mean_branch = float(np.mean(tau_branch))
H_top_approx = h_map / tau_mean_branch

print(
    f"  c_map={c_map:.6f}, f(c_map)={f_c:.6f}, f(f(c_map))={ff_c:.6f}\n"
    f"  left=[{left_lo:.6f}, {left_hi:.6f}], right=[{right_lo:.6f}, {right_hi:.6f}]"
)
print(
    f"  tau_left={tau_left:.6f}, tau_right={tau_right:.6f}, "
    f"tau_mean_branch={tau_mean_branch:.6f}, tau_mean_raw={np.mean(tau_n):.6f}"
)
print(f"  unresolved branch assignments (data)   = {unresolved_data}")
print(f"  unresolved branch assignments (knead)  = {unresolved_knead}")
print(f"  h_top (branch-weighted corrected) = {H_top_branch:.6f}")
print(f"  h_map / mean(tau_branch)          = {H_top_approx:.6f}")


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
    if (i + 1) % 20 == 0:
        print(f"  Trajectory {i+1}/{num_ics_lle}...")

    idx = np.random.randint(max(sol_map.t.size // 4, 1), sol_map.t.size)
    curr_state = np.append(sol_map.y[:, idx], [1.0, 0.0, 0.0])
    LE_sum = 0.0

    for _ in range(int(t_max_lle / dt)):
        sol = solve_ivp(rossler_var, (0, dt), curr_state, method="RK45", rtol=1e-5, atol=1e-5)
        curr_state = sol.y[:, -1]
        norm_w = np.linalg.norm(curr_state[3:6])
        LE_sum += np.log(norm_w)
        curr_state[3:6] /= norm_w

    lle_vals.append(LE_sum / t_max_lle)

# Expansion entropy estimator in the same style as attempt-006.
max_le_sum = np.max([val * t_max_lle for val in lle_vals])
mean_exp = np.mean([np.exp(val * t_max_lle - max_le_sum) for val in lle_vals])
EE_val = (max_le_sum + np.log(mean_exp)) / t_max_lle


# ==========================================================
# 6. ENSEMBLE 2: LZ76 COMPLEXITY
# ==========================================================
num_ics_lz = 500
N_iter = 2500
print(f"Simulating {num_ics_lz} trajectories for LZ76 distribution...")

x_lo = max(float(np.min(x_n_u)), left_lo)
x_hi = min(float(np.max(x_n_u)), right_hi)
if x_hi <= x_lo:
    x_lo, x_hi = float(np.min(x_n_u)), float(np.max(x_n_u))

xs_curr = np.linspace(x_lo, x_hi, num_ics_lz)
for _ in range(20):
    xs_curr = f_interp(xs_curr)

symbols = np.zeros((num_ics_lz, N_iter), dtype=np.int8)
T_tots = np.zeros(num_ics_lz)
unresolved_lz = 0

for j in range(N_iter):
    xs_curr = f_interp(xs_curr)
    left_now, ucount = classify_left(xs_curr)
    unresolved_lz += ucount

    symbols[:, j] = (~left_now).astype(np.int8)
    T_tots += np.where(left_now, tau_left, tau_right)

lz_vals = [
    lz76_fast("".join(symbols[i].astype(str))) * np.log(N_iter) / T_tots[i]
    for i in range(num_ics_lz)
]


# ==========================================================
# 7. CONVERGENCE CURVE (BRANCH-WEIGHTED)
# ==========================================================
print("Computing convergence curves...")
N_grid = np.arange(20, 401, 5)
conv_new = []

for N in N_grid:
    eps_N = eps_list[:N]

    T_new_N = np.cumsum(tau_branch[:N])

    def D_new_N(s):
        return 1.0 + np.sum(eps_N * np.exp(-s * T_new_N))

    try:
        conv_new.append(first_positive_root(D_new_N, a=0.01, b=2.0, grid=1500))
    except RuntimeError:
        conv_new.append(np.nan)

conv_new = np.array(conv_new)


# ==========================================================
# 8. FINAL PLOTS
# ==========================================================
outdir = Path(__file__).resolve().parent

print("Rendering plots...")
fig, ax = plt.subplots(figsize=(12, 7))

all_vals = np.array(
    lz_vals + lle_vals + [H_top_branch, H_top_approx, EE_val],
    dtype=np.float64,
)
span = max(np.ptp(all_vals), 0.03)
bins = np.linspace(max(0.0, np.min(all_vals) - 0.12 * span), np.max(all_vals) + 0.12 * span, 70)

c_lz, e_lz = np.histogram(lz_vals, bins=bins)
c_lle, e_lle = np.histogram(lle_vals, bins=bins)

scale = num_ics_lz / num_ics_lle
ax.bar(
    e_lz[:-1],
    c_lz,
    width=np.diff(e_lz),
    align="edge",
    color="royalblue",
    edgecolor="black",
    alpha=0.8,
    label="LZ76 Complexity",
)
ax.bar(
    e_lle[:-1],
    -c_lle * scale,
    width=np.diff(e_lle),
    align="edge",
    color="darkorange",
    edgecolor="black",
    alpha=0.8,
    label="KS Entropy/LLE",
)

ax.axvline(
    H_top_branch,
    color="red",
    ls="--",
    lw=2.5,
    label=f"Flow h_top (branch-weighted) = {H_top_branch:.4f}",
)
ax.axvline(
    H_top_approx,
    color="purple",
    ls="-.",
    lw=2.2,
    label=f"h_map / mean(tau) = {H_top_approx:.4f}",
)
ax.axvline(
    np.mean(lz_vals),
    color="green",
    ls=":",
    lw=3,
    label=f"Mean LZ76 = {np.mean(lz_vals):.4f}",
)
ax.axvline(
    np.mean(lle_vals),
    color="darkred",
    ls=":",
    lw=3,
    label=f"Mean KS/LLE = {np.mean(lle_vals):.4f}",
)
ax.axvline(
    EE_val,
    color="blue",
    ls="-",
    lw=2,
    label=f"Expansion Entropy = {EE_val:.4f}",
)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(abs(x))}"))
ax.axhline(0, color="black", lw=1.5)
ax.set_xlabel("Continuous Entropy (nats / time)", fontsize=12)
ax.set_ylabel("Frequency (Trajectories)", fontsize=12)
ax.set_title("Rössler Entropies: Topological vs. Metric approaches", fontsize=14)
ax.set_ylim(-max(c_lz) * 1.1, max(c_lz) * 1.1)
ax.legend(loc="lower right", fontsize=9, ncol=2)
ax.grid(True, alpha=0.2)
plt.tight_layout()

out_hist = outdir / "entropies_rossler_branch_weighted.png"
plt.savefig(out_hist, dpi=200)
print(f"Saved {out_hist}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(N_grid, conv_new, "r-", lw=2, label="Branch-weighted root")
ax2.axhline(H_top_branch, color="red", ls="--", lw=1)
ax2.set_xlabel("Kneading Truncation N")
ax2.set_ylabel("Root s")
ax2.set_title("Rössler Kneading Root Convergence: Branch-Weighted")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")
plt.tight_layout()

out_conv = outdir / "htop_convergence_branch_vs_pointwise.png"
plt.savefig(out_conv, dpi=200)
print(f"Saved {out_conv}")

print(f"  unresolved branch assignments (LZ loop) = {unresolved_lz}")
print(f"  mean KS/LLE = {np.mean(lle_vals):.6f}")
print(f"  mean LZ76   = {np.mean(lz_vals):.6f}")
print(f"  expansion entropy = {EE_val:.6f}")
