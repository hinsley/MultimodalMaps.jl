import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

warnings.filterwarnings("ignore")


# ==========================================================
# 1. ROSSLER SYSTEM (a=0.2, b=0.2, c=5.7) + VARIATIONAL FLOW
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

    # Tangent flow d/dt w = J(x)w.
    dw0 = -w1 - w2
    dw1 = w0 + a * w1
    dw2 = z * w0 + (x - c_param) * w2

    return [dx, dy, dz, dw0, dw1, dw2]


def max_x_event(t, state):
    # Local maxima of x occur when dx/dt = -y - z = 0 with downward crossing.
    return -state[1] - state[2]


max_x_event.direction = -1


# ==========================================================
# 2. HELPERS
# ==========================================================
def first_positive_root(func, a0=0.01, b0=2.0, grid=3000):
    ss = np.linspace(a0, b0, grid)
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

    raise RuntimeError(f"No positive root found in [{a0}, {b0}]")


def all_positive_roots(func, a0=0.01, b0=2.0, grid=3000):
    """Return all positive roots detected via sign changes on a dense scan."""
    ss = np.linspace(a0, b0, grid)
    vals = np.array([func(s) for s in ss], dtype=np.float64)

    roots = []

    close = np.where(np.isfinite(vals) & (np.abs(vals) < 1e-12))[0]
    for idx in close:
        roots.append(float(ss[int(idx)]))

    for i in range(grid - 1):
        v0, v1 = vals[i], vals[i + 1]
        if not np.isfinite(v0) or not np.isfinite(v1):
            continue
        if np.sign(v0) == 0:
            continue
        if np.sign(v0) != np.sign(v1):
            try:
                r = float(root_scalar(func, bracket=[ss[i], ss[i + 1]], method="brentq").root)
                roots.append(r)
            except Exception:
                pass

    if not roots:
        return []

    # Deduplicate numerically-close roots.
    roots = sorted(roots)
    dedup = [roots[0]]
    for r in roots[1:]:
        if abs(r - dedup[-1]) > 1e-8:
            dedup.append(r)
    return dedup


# ==========================================================
# 3. RETURN MAP + BRANCH-WEIGHTED KNEADING CONVERGENCE
# ==========================================================
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

# Denoise via deterministic averaging on rounded x bins.
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


c_map = float(x_n_u[np.argmax(x_np1_u)])
f_c = float(f_interp(c_map))
ff_c = float(f_interp(f_c))

# Same branch definitions used in attempt-007/EntropiesRossler.py.
left_lo, left_hi = sorted((ff_c, c_map))
right_lo, right_hi = sorted((c_map, f_c))


def classify_left(xs):
    arr = np.asarray(xs)
    left = (arr >= left_lo) & (arr <= left_hi)
    right = (arr >= right_lo) & (arr <= right_hi)
    unresolved = ~(left | right)
    if np.any(unresolved):
        left = left.copy()
        left[unresolved] = arr[unresolved] < c_map
    return left


left_data = classify_left(x_n)
if not np.any(left_data) or not np.any(~left_data):
    raise RuntimeError("Failed to form both branch populations from interval definitions")

tau_left = float(np.mean(tau_n[left_data]))
tau_right = float(np.mean(tau_n[~left_data]))

print("Computing branch-weighted kneading convergence...")
N_knead_max = 500
curr_x = float(f_interp(c_map))
eps_list = []
tau_branch = []
e = 1.0

for _ in range(N_knead_max):
    is_left = bool(classify_left(np.array([curr_x]))[0])
    sign = 1.0 if is_left else -1.0
    e *= sign

    eps_list.append(e)
    tau_branch.append(tau_left if is_left else tau_right)
    curr_x = float(f_interp(curr_x))

eps = np.array(eps_list, dtype=np.float64)
T_acc = np.cumsum(np.array(tau_branch, dtype=np.float64))

h_knead_vals = []
T_knead_vals = []
last_root = None

for N in range(1, N_knead_max + 1):
    eps_N = eps[:N]
    T_N = T_acc[:N]

    def D_weighted(s):
        return 1.0 + np.sum(eps_N * np.exp(-s * T_N))

    roots = all_positive_roots(D_weighted, a0=0.001, b0=4.0, grid=3000)
    if not roots:
        h_knead_vals.append(np.nan)
        T_knead_vals.append(float(T_N[-1]))
        continue

    # Track the physically-relevant continuous branch.
    # Initialize from the largest positive root (avoids near-zero spurious branch),
    # then pick the root nearest to the previous one.
    if last_root is None:
        root = roots[-1]
    else:
        root = min(roots, key=lambda r: abs(r - last_root))

    last_root = root
    h_knead_vals.append(root)
    T_knead_vals.append(float(T_N[-1]))

if not h_knead_vals:
    raise RuntimeError("No valid weighted kneading convergence points were found")

h_knead_vals = np.array(h_knead_vals, dtype=np.float64)
T_knead_vals = np.array(T_knead_vals, dtype=np.float64)
valid_knead = np.isfinite(h_knead_vals)
if not np.any(valid_knead):
    raise RuntimeError("No valid weighted kneading convergence points were found")
final_h_knead = float(h_knead_vals[valid_knead][-1])


# ==========================================================
# 4. EXPANSION ENTROPY CONVERGENCE (TIME-RESOLVED)
# ==========================================================
print("Computing expansion entropy convergence...")
num_ics_ee = 80
T_max_ee = max(1200.0, float(T_knead_vals[-1]))
dt_ee = 10.0
steps = int(T_max_ee / dt_ee)

np.random.seed(42)
states = np.zeros((num_ics_ee, 6), dtype=np.float64)
start_lo = max(sol_map.t.size // 2, 1)
for i in range(num_ics_ee):
    idx = np.random.randint(start_lo, sol_map.t.size)
    states[i, 0:3] = sol_map.y[:, idx]
    v = np.random.randn(3)
    states[i, 3:6] = v / np.linalg.norm(v)

log_expansions = np.zeros(num_ics_ee, dtype=np.float64)
EE_vals = []
T_ee_vals = []

for step in range(steps):
    for i in range(num_ics_ee):
        sol_ee = solve_ivp(
            rossler_var,
            (0, dt_ee),
            states[i],
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )
        state_end = sol_ee.y[:, -1]

        v = state_end[3:6]
        norm_v = np.linalg.norm(v)
        if norm_v <= 0.0 or not np.isfinite(norm_v):
            norm_v = 1e-16

        log_expansions[i] += np.log(norm_v)
        state_end[3:6] = v / norm_v
        states[i] = state_end

    curr_T = (step + 1) * dt_ee
    max_log = np.max(log_expansions)
    mean_exp = np.mean(np.exp(log_expansions - max_log))
    EE = (max_log + np.log(mean_exp)) / curr_T

    EE_vals.append(float(EE))
    T_ee_vals.append(float(curr_T))

EE_vals = np.array(EE_vals, dtype=np.float64)
T_ee_vals = np.array(T_ee_vals, dtype=np.float64)
final_ee = float(EE_vals[-1])


# ==========================================================
# 5. PLOT: CONTINUOUS TIME CONVERGENCE
# ==========================================================
outdir = Path(__file__).resolve().parent
out_plot = outdir / "rossler_convergence.png"

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    T_ee_vals,
    EE_vals,
    color="blue",
    linewidth=2.0,
    label="Expansion entropy convergence",
)
ax.plot(
    T_knead_vals[valid_knead],
    h_knead_vals[valid_knead],
    color="red",
    linewidth=2.0,
    label="Weighted kneading h_top convergence",
)

ax.axhline(
    final_ee,
    color="blue",
    linestyle="--",
    linewidth=1.5,
    label=f"Expansion final = {final_ee:.6f}",
)
ax.axhline(
    final_h_knead,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Weighted kneading final = {final_h_knead:.6f}",
)

ax.set_xlabel("Continuous elapsed simulation time")
ax.set_ylabel("Topological entropy estimate")
ax.set_title("Rossler convergence: expansion entropy vs weighted kneading")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
plt.tight_layout()

plt.savefig(out_plot, dpi=200)
print(f"Saved {out_plot}")
print(f"Final expansion entropy: {final_ee:.6f}")
print(f"Final weighted kneading topological entropy: {final_h_knead:.6f}")
print(f"Kneading convergence points (finite): {np.count_nonzero(valid_knead)} / {len(h_knead_vals)}")
print(f"Expansion convergence points: {len(EE_vals)}")
