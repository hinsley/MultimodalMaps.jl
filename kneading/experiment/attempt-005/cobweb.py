import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
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

x_n, x_np1 = abs_x_max[:-1], abs_x_max[1:]

# Unique sorting for interpolation
x_n_u, idx_u = np.unique(np.round(x_n, 4), return_inverse=True)
x_np1_u = np.zeros_like(x_n_u)
for i in range(len(x_n_u)):
    mask = (idx_u == i)
    x_np1_u[i] = np.mean(x_np1[mask])

def f_interp(x): return np.interp(x, x_n_u, x_np1_u)

c_crit = x_n_u[np.argmax(x_np1_u)]
print(f"Critical Point c: {c_crit:.4f}")
print(f"f(c): {f_interp(c_crit):.4f}")

# ==========================================================
# 2. GENERATE COBWEB TRAJECTORY
# ==========================================================
N_iterates = 50
orbit_x = np.zeros(N_iterates + 1)
orbit_x[0] = c_crit

for i in range(1, N_knead := N_iterates + 1):
    orbit_x[i] = f_interp(orbit_x[i-1])

# ==========================================================
# 3. PLOT COBWEB DIAGRAM
# ==========================================================
print("Plotting Cobweb Diagram...")
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the map
x_dense = np.linspace(np.min(x_n_u), np.max(x_n_u), 1000)
ax.plot(x_dense, f_interp(x_dense), 'b-', lw=2, label='$x_{n+1} = f(x_n)$')

# Plot y = x line
ax.plot(x_dense, x_dense, 'k--', lw=1.5, label='$y = x$')

# Plot the cobweb lines
cobweb_x = [orbit_x[0]]
cobweb_y = [0]

for i in range(N_iterates):
    # Vertical line to the curve
    cobweb_x.append(orbit_x[i])
    cobweb_y.append(orbit_x[i+1])
    
    # Horizontal line to y=x
    cobweb_x.append(orbit_x[i+1])
    cobweb_y.append(orbit_x[i+1])

ax.plot(cobweb_x, cobweb_y, 'r-', lw=1.0, alpha=0.7, label='Cobweb Trajectory')

# Highlight the critical point and its image
ax.plot(c_crit, f_interp(c_crit), 'go', markersize=8, label=f'Critical Peak ($c$, $f(c)$)')
ax.axvline(c_crit, color='g', linestyle=':', alpha=0.5)

ax.set_xlabel('$x_n$ (Current Return Maxima)', fontsize=14)
ax.set_ylabel('$x_{n+1}$ (Next Return Maxima)', fontsize=14)
ax.set_title('Cobweb Diagram: Lorenz 1D Return Map', fontsize=16)
ax.legend(loc='lower right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(np.min(x_n_u), np.max(x_n_u))
ax.set_ylim(np.min(x_n_u), np.max(x_np1_u) * 1.02)

plt.tight_layout()
plt.savefig("lorenz_cobweb.png")
print("Saved lorenz_cobweb.png")
