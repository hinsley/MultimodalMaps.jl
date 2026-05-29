import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings
import time

warnings.filterwarnings('ignore')

sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz(t, state):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def max_x_event(t, state):
    return state[1] - state[0]
max_x_event.direction = 0

# Let's test a range of maximum integration step sizes
dt_vals = np.logspace(-1, -4, 12)
converged_htops = []
log_dt_vals = []

for dt in dt_vals:
    print(f"Testing dt = {dt:.5f}...")
    start_time = time.time()
    
    # We use a strict base tolerance, and enforce max_step=dt to physically limit the integrator
    sol_map = solve_ivp(lorenz, (0, 400), [1.0, 1.0, 1.0], method='RK45', 
                        events=[max_x_event], rtol=1e-8, atol=1e-8, max_step=dt)

    y_ev = sol_map.y_events[0]
    if len(y_ev) < 100:
        print("  Not enough events.")
        continue
        
    is_max = y_ev[:, 2] > (rho - 1)
    abs_x_max = np.abs(y_ev[is_max, 0])
    t_max = sol_map.t_events[0][is_max]

    x_n, x_np1, tau_n = abs_x_max[:-1], abs_x_max[1:], np.diff(t_max)
    
    # We use round to 4 to group points, matching our original implementation
    x_n_u, idx_u = np.unique(np.round(x_n, 4), return_inverse=True)
    x_np1_u, tau_u = np.zeros_like(x_n_u), np.zeros_like(x_n_u)
    for i in range(len(x_n_u)):
        mask = (idx_u == i)
        x_np1_u[i], tau_u[i] = np.mean(x_np1[mask]), np.mean(tau_n[mask])

    def f_interp(x): return np.interp(x, x_n_u, x_np1_u)
    def tau_interp(x): return np.interp(x, x_n_u, tau_u)
    c_crit = x_n_u[np.argmax(x_np1_u)]

    N_knead_max = 150
    x_k, eps, T_acc = np.zeros(N_knead_max), np.zeros(N_knead_max), np.zeros(N_knead_max)
    x_k[0], eps[0], T_acc[0] = f_interp(c_crit), 1.0, 0.0

    for n in range(1, N_knead_max):
        x_k[n] = f_interp(x_k[n-1])
        sign = 1.0 if x_k[n-1] < c_crit else (-1.0 if x_k[n-1] > c_crit else 0.0)
        eps[n] = eps[n-1] * sign
        T_acc[n] = T_acc[n-1] + tau_interp(x_k[n-1])

    prev_htop = -1.0
    converged = False
    
    for N in range(2, N_knead_max):
        def D_weighted(s): return sum(eps[n] * np.exp(-s * T_acc[n]) for n in range(N) if eps[n] != 0)
        
        if D_weighted(0.0) <= 0.0:
            if np.sign(D_weighted(0.5)) != np.sign(D_weighted(2.5)):
                res = root_scalar(D_weighted, bracket=[0.5, 2.5], method='brentq')
                curr_htop = res.root
                
                # Check for 4 digits stable
                if prev_htop != -1.0 and abs(curr_htop - prev_htop) < 1e-4:
                    converged_htops.append(curr_htop)
                    log_dt_vals.append(np.log10(dt))
                    converged = True
                    print(f"  Converged at N={N} to {curr_htop:.5f} (took {time.time()-start_time:.2f}s)")
                    break
                
                prev_htop = curr_htop
                
    if not converged:
        if prev_htop != -1.0:
            converged_htops.append(prev_htop)
            log_dt_vals.append(np.log10(dt))
            print(f"  Failed to fully converge within N={N_knead_max}, appending last estimate {prev_htop:.5f}")
        else:
            print(f"  Failed to find roots for dt={dt}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(log_dt_vals, converged_htops, 'b-o', markersize=8, linewidth=2)
plt.axhline(1.1306, color='r', linestyle='--', label='1.1306 Reference')
plt.xlabel('log10(dt) (Maximum Integration Step Size)', fontsize=12)
plt.ylabel('Converged h_top Estimate', fontsize=12)
plt.title('Weighted Kneading Entropy Convergence vs. Integration dt', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kneading_dt_convergence.png')
print("Saved kneading_dt_convergence.png")
