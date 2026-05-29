import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.ticker as ticker
from pathlib import Path
import json
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
# 2. LORENZ SYSTEM & VARIATIONAL EQUATIONS
# ==========================================================
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0


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
    # y - x = 0 marks extrema of x. We later filter to keep Lorenz maxima branch.
    return state[1] - state[0]


max_x_event.direction = 0


# ==========================================================
# 3. RETURN MAP + KNEADING DATA
# ==========================================================
def first_positive_root(func, a=0.01, b=2.0, grid=4000):
    ss = np.linspace(a, b, grid)
    vals = np.array([func(s) for s in ss], dtype=np.float64)

    # Handle exact grid hits.
    close = np.where(np.isfinite(vals) & (np.abs(vals) < 1e-12))[0]
    if close.size > 0:
        return ss[int(close[0])]

    for i in range(grid - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not np.isfinite(v0) or not np.isfinite(v1):
            continue
        if np.sign(v0) == 0:
            return ss[i]
        if np.sign(v0) != np.sign(v1):
            return root_scalar(func, bracket=[ss[i], ss[i + 1]], method="brentq").root
    raise RuntimeError("No positive root found in bracket")


def load_pressure_root_htop():
    outdir = Path(__file__).resolve().parent
    study_json = outdir / "resolution_study_lorenz.json"
    report_json = outdir / "pressure_report_lorenz.json"

    if study_json.exists():
        data = json.loads(study_json.read_text())
        if "reference_h_top" in data:
            return float(data["reference_h_top"]), str(study_json)

    if report_json.exists():
        data = json.loads(report_json.read_text())
        if "h_top_pressure" in data:
            return float(data["h_top_pressure"]), str(report_json)

    raise FileNotFoundError(
        "No pressure-root estimate found. Expected resolution_study_lorenz.json or pressure_report_lorenz.json."
    )


print("Extracting 1D Return Map...")
sol_map = solve_ivp(
    lorenz,
    (0, 1000),
    [1.0, 1.0, 1.0],
    method="RK45",
    events=[max_x_event],
    rtol=1e-8,
    atol=1e-8,
)

y_ev = sol_map.y_events[0]
is_max = y_ev[:, 2] > (rho - 1)
abs_x_max = np.abs(y_ev[is_max, 0])
t_max = sol_map.t_events[0][is_max]

x_n, x_np1, tau_n = abs_x_max[:-1], abs_x_max[1:], np.diff(t_max)

# Denoise via deterministic 1D map averaging on rounded x bins.
x_n_u, idx_u = np.unique(np.round(x_n, 4), return_inverse=True)
x_np1_u, tau_u = np.zeros_like(x_n_u), np.zeros_like(x_n_u)
for i in range(len(x_n_u)):
    mask = idx_u == i
    x_np1_u[i], tau_u[i] = np.mean(x_np1[mask]), np.mean(tau_n[mask])


def f_interp(x):
    return np.interp(x, x_n_u, x_np1_u)


def tau_interp(x):
    return np.interp(x, x_n_u, tau_u)


c_crit = x_n_u[np.argmax(x_np1_u)]
f_c = float(f_interp(c_crit))
ff_c = float(f_interp(f_c))


def closed_interval_mask(values, a, b, tol):
    lo, hi = (a, b) if a <= b else (b, a)
    return (values >= lo - tol) & (values <= hi + tol)


def branch_masks(values, c, f_c_val, ff_c_val):
    values = np.asarray(values, dtype=np.float64)
    scale = max(1.0, abs(c), abs(f_c_val), abs(ff_c_val))
    tol = 1e-9 * scale

    # Branch definitions for flow weighting:
    # left = [f(f(c)), c], right = [c, f(c)].
    left = closed_interval_mask(values, ff_c_val, c, tol)
    right = closed_interval_mask(values, c, f_c_val, tol)

    # Resolve overlap at c robustly and assign any outliers by side of c.
    overlap = left & right
    if np.any(overlap):
        left_side = overlap & (values <= c + tol)
        right_side = overlap & (values > c + tol)
        left = (left & ~overlap) | left_side
        right = (right & ~overlap) | right_side

    unassigned = ~(left | right)
    if np.any(unassigned):
        left = left | (unassigned & (values <= c + tol))
        right = right | (unassigned & (values > c + tol))

    return left, right, tol

print("Computing kneading itineraries...")
N_knead = 5000
curr_x = f_interp(c_crit)

xs = []
branch_left = []
eps_list = []

# For unimodal Lorenz return map: sign(f') is + on left branch, - on right branch.
e = 1.0
for _ in range(N_knead):
    is_left = curr_x < c_crit
    sign = 1.0 if is_left else -1.0
    e *= sign

    xs.append(curr_x)
    branch_left.append(is_left)
    eps_list.append(e)

    curr_x = f_interp(curr_x)

xs = np.array(xs)
branch_left = np.array(branch_left, dtype=bool)
eps_list = np.array(eps_list, dtype=np.float64)

# Branch roof-time constants g_i(s)=exp(-s*tau_i) for weighted kneading.
left_mask_data, right_mask_data, branch_tol = branch_masks(x_n, c_crit, f_c, ff_c)
tau_left = float(np.mean(tau_n[left_mask_data]))
tau_right = float(np.mean(tau_n[right_mask_data]))

# Branch-constant times (flow-weighted correction).
left_mask_xs, _, _ = branch_masks(xs, c_crit, f_c, ff_c)
tau_branch = np.where(left_mask_xs, tau_left, tau_right)
T_branch = np.cumsum(tau_branch)
tau_mean_flow = float(np.mean(np.where(left_mask_data, tau_left, tau_right)))


# ==========================================================
# 4. TOPOLOGICAL ENTROPIES
# ==========================================================
print("Computing topological entropy roots...")


def D_flow_weighted(s):
    return 1.0 + np.sum(eps_list * np.exp(-s * T_branch))


H_top_branch = first_positive_root(D_flow_weighted, a=0.01, b=2.0)

# Unweighted map entropy per iterate, then suspension approximation h_map/tau_avg.
def D_unweighted(t):
    powers = t ** np.arange(1, N_knead + 1)
    return 1.0 + np.sum(eps_list * powers)


t0 = first_positive_root(D_unweighted, a=0.05, b=0.95)
h_map = -np.log(t0)

H_top_pressure, pressure_source = load_pressure_root_htop()

# Saddle-aware mean roof time:
# Lorenz return times have a singular tail from slow passage near the origin saddle.
# Use a long trajectory mean on the same maxima section instead of branch-compressed means.
print("Estimating saddle-aware mean return time from a longer trajectory...")
sol_tau = solve_ivp(
    lorenz,
    (0, 20000),
    [1.0, 1.0, 1.0],
    method="DOP853",
    events=[max_x_event],
    rtol=1e-10,
    atol=1e-10,
)
y_ev_tau = sol_tau.y_events[0]
is_max_tau = y_ev_tau[:, 2] > (rho - 1)
t_max_tau = sol_tau.t_events[0][is_max_tau]

# Burn early transients in event-time space.
event_burn = min(1000, max(0, t_max_tau.size - 2))
t_max_tau = t_max_tau[event_burn:]
tau_saddle = np.diff(t_max_tau)
tau_mean_saddle = float(np.mean(tau_saddle))

H_top_approx = h_map / tau_mean_saddle

print(
    f"  branch intervals: left=[{ff_c:.6f}, {c_crit:.6f}], right=[{c_crit:.6f}, {f_c:.6f}]"
)
print(
    f"  tau_left={tau_left:.6f}, tau_right={tau_right:.6f}, "
    f"tau_mean(flow)={tau_mean_flow:.6f}, tol={branch_tol:.3e}"
)
print(f"  weighted kneading h_top         = {H_top_branch:.6f}")
print(
    f"  saddle-aware tau mean           = {tau_mean_saddle:.6f} "
    f"(max observed {np.max(tau_saddle):.6f})"
)
print(f"  h_map / mean(tau_saddle-aware)  = {H_top_approx:.6f}")
print(f"  pressure-root h_top             = {H_top_pressure:.6f}")
print(f"  pressure-root source            = {pressure_source}")


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

    idx = np.random.randint(100, sol_map.t.size)
    curr_state = np.append(sol_map.y[:, idx], [1.0, 0.0, 0.0])
    LE_sum = 0.0

    for _ in range(int(t_max_lle / dt)):
        sol = solve_ivp(lorenz_var, (0, dt), curr_state, method="RK45", rtol=1e-5, atol=1e-5)
        curr_state = sol.y[:, -1]
        norm_w = np.linalg.norm(curr_state[3:6])
        LE_sum += np.log(norm_w)
        curr_state[3:6] /= norm_w

    lle_vals.append(LE_sum / t_max_lle)

# Expansion entropy estimator.
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
for _ in range(20):
    xs_curr = f_interp(xs_curr)

symbols = np.zeros((num_ics_lz, N_iter), dtype=np.int8)
T_tots = np.zeros(num_ics_lz)

for j in range(N_iter):
    xs_curr = f_interp(xs_curr)
    symbols[:, j] = (xs_curr >= c_crit).astype(np.int8)
    T_tots += np.where(xs_curr < c_crit, tau_left, tau_right)

lz_vals = [
    lz76_fast("".join(symbols[i].astype(str))) * np.log(N_iter) / T_tots[i]
    for i in range(num_ics_lz)
]


# ==========================================================
# 7. CONVERGENCE CURVES (FLOW WEIGHTED)
# ==========================================================
print("Computing convergence curves...")
N_grid = np.arange(20, 401, 5)
conv_flow = []

for N in N_grid:
    eps_N = eps_list[:N]
    T_flow_N = np.cumsum(tau_branch[:N])

    try:
        conv_flow.append(
            first_positive_root(
                lambda s, eps_N=eps_N, T_flow_N=T_flow_N: 1.0
                + np.sum(eps_N * np.exp(-s * T_flow_N)),
                a=0.01,
                b=2.0,
                grid=1500,
            )
        )
    except RuntimeError:
        conv_flow.append(np.nan)

conv_flow = np.array(conv_flow)


# ==========================================================
# 8. FINAL PLOTS
# ==========================================================
outdir = Path(__file__).resolve().parent

print("Rendering plots...")
fig, ax = plt.subplots(figsize=(12, 7))

bins = np.linspace(0.80, 1.20, 70)
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
    lw=2.2,
    label=f"Weighted kneading h_top = {H_top_branch:.4f}",
)
ax.axvline(
    H_top_approx,
    color="purple",
    ls="-.",
    lw=2.2,
    label=f"h_map / mean(tau_saddle-aware) = {H_top_approx:.4f}",
)
ax.axvline(
    H_top_pressure,
    color="teal",
    ls="-",
    lw=2.5,
    label=f"Pressure-root flow h_top = {H_top_pressure:.4f}",
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
    label=f"Mean KS = {np.mean(lle_vals):.4f}",
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
ax.set_title("Lorenz Entropies: Flow-Weighted Kneading", fontsize=14)
ax.set_ylim(-max(c_lz) * 1.1, max(c_lz) * 1.1)
ax.set_xlim(0.85, 1.0)
ax.legend(loc="upper right", fontsize=9, ncol=2)
ax.grid(True, alpha=0.2)
plt.tight_layout()

out_hist = outdir / "entropies_lorenz_branch_weighted.png"
plt.savefig(out_hist, dpi=200)
print(f"Saved {out_hist}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(N_grid, conv_flow, "r-", lw=2, label="Flow-weighted root")
ax2.axhline(H_top_branch, color="red", ls="--", lw=1, label="Flow h_top (branch-weighted)")
ax2.set_xlabel("Kneading Truncation N")
ax2.set_ylabel("Root s")
ax2.set_title("Kneading Root Convergence: Flow-Weighted")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="best")
plt.tight_layout()

out_conv = outdir / "htop_convergence_flow_weighted.png"
plt.savefig(out_conv, dpi=200)
print(f"Saved {out_conv}")
