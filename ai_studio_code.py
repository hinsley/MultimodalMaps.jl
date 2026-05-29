import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import warnings

# Suppress warnings from polynomial fits and root finders for a clean output
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DEFINE RÖSSLER SYSTEM & POINCARÉ SECTION
# =============================================================================
a, b, c_param = 0.2, 0.2, 5.7

def rossler(t, state):
    x, y, z = state
    return[-y - z, x + a * y, b + z * (x - c_param)]

# Define the Poincaré section at the local maximum of x (where dx/dt = 0)
def max_x_event(t, state):
    return -state[1] - state[2]
max_x_event.direction = -1

# =============================================================================
# 2. EXTRACT RETURNS, ROOF TIMES, AND FIT 1D POLYNOMIALS
# =============================================================================
print("Integrating Rössler ODE to extract Poincaré returns...")
sol = solve_ivp(rossler, (0, 6000),[1.0, 1.0, 1.0], method='RK45', 
                events=[max_x_event], rtol=1e-10, atol=1e-10)

# Discard the first 100 returns to ensure we are settled on the attractor
t_events = sol.t_events[0][100:]
x_events = sol.y_events[0][100:, 0]

x_curr = x_events[:-1]
x_next = x_events[1:]
tau_n  = np.diff(t_events)

# Fit degree 8 polynomials to the 1D map and roof times
deg = 8
f_poly = np.poly1d(np.polyfit(x_curr, x_next, deg))
tau_poly = np.poly1d(np.polyfit(x_curr, tau_n, deg))

# Find the EXACT critical point using the derivative of the polynomial map
deriv = f_poly.deriv()
roots = deriv.roots
real_roots = roots[np.isreal(roots)].real
c_approx = real_roots[(real_roots > np.min(x_curr)) & (real_roots < np.max(x_curr))][0]

def f_prime(x): return deriv(x)
def f_double_prime(x): return deriv.deriv()(x)
res_c = root_scalar(f_prime, x0=c_approx, fprime=f_double_prime, method='newton')
c_exact = res_c.root

# =============================================================================
# 3. CALCULATE CONVERGENCE OF RUGH'S WEIGHTED KNEADING DETERMINANT
# =============================================================================
print("Computing roots of Rugh's weighted kneading determinant...")
N_rugh_max = 50
x_rugh = np.zeros(N_rugh_max)
epsilon = np.zeros(N_rugh_max)
T = np.zeros(N_rugh_max)

x_rugh[0] = f_poly(c_exact)
epsilon[0] = 1.0
T[0] = 0.0

# Generate the sequence of discrete positions, signs, and continuous accumulated time
for n in range(1, N_rugh_max):
    x_rugh[n] = f_poly(x_rugh[n-1])
    if x_rugh[n-1] < c_exact:
        sign = 1.0
    elif x_rugh[n-1] > c_exact:
        sign = -1.0
    else:
        sign = 0.0
    epsilon[n] = epsilon[n-1] * sign
    T[n] = T[n-1] + tau_poly(x_rugh[n-1])

valid_t, valid_s = [],[]

# Find the continuous entropy roots for successive series truncations
for N_trunc in range(1, N_rugh_max):
    def D(s):
        return sum(epsilon[n] * np.exp(-s * T[n]) for n in range(N_trunc) if epsilon[n] != 0)
    
    # A root is physically valid only when the alternating sequence provides a bracket
    if D(0) <= 0:
        res = root_scalar(D, bracket=[0.0, 2.0], method='brentq')
        valid_t.append(T[N_trunc-1])
        valid_s.append(res.root)

# Filter out edge-case bracket zeroes to show the clean mathematical convergence
filtered_t =[t for t, s in zip(valid_t, valid_s) if s > 0.01]
filtered_s = [s for s in valid_s if s > 0.01]

# =============================================================================
# 4. CALCULATE CONVERGENCE OF EXPANSION ENTROPY (ENSEMBLE SIMULATION)
# =============================================================================
print("Integrating 150-point ensemble for Expansion Entropy (this takes ~30 seconds)...")

def rossler_var(t, state):
    x, y, z = state[0:3]
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c_param)
    
    # Full 3x3 Jacobian pushed forward via 3 tangent vectors
    w11, w12, w13 = state[3:6]
    w21, w22, w23 = state[6:9]
    w31, w32, w33 = state[9:12]
    
    dw11, dw12, dw13 = -w21 - w31, -w22 - w32, -w23 - w33
    dw21, dw22, dw23 = w11 + a*w21, w12 + a*w22, w13 + a*w23
    dw31, dw32, dw33 = z*w11 + (x-c_param)*w31, z*w12 + (x-c_param)*w32, z*w13 + (x-c_param)*w33
    
    return[dx, dy, dz, dw11, dw12, dw13, dw21, dw22, dw23, dw31, dw32, dw33]

N_points = 150
np.random.seed(42)
states = np.zeros((N_points, 12))
# Initialize randomly around the bounding box of the attractor
states[:, 0] = np.random.uniform(-10, 15, N_points)
states[:, 1] = np.random.uniform(-15, 10, N_points)
states[:, 2] = np.random.uniform(0, 30, N_points)

# Set the 3x3 deviation matrix block to the Identity matrix
states[:, 3] = 1.0; states[:, 7] = 1.0; states[:, 11] = 1.0

T_max, dt = 120.0, 4.0
steps = int(T_max / dt)

log_expansions = np.zeros(N_points)
EE_values, ee_times = [],[]

for step in range(steps):
    for i in range(N_points):
        sol_ee = solve_ivp(rossler_var, (0, dt), states[i], method='RK45', rtol=1e-5, atol=1e-5)
        states[i] = sol_ee.y[:, -1]
        
        # Calculate maximum singular value (Operator 2-Norm)
        M = states[i, 3:12].reshape((3, 3))
        norm_M = np.linalg.norm(M, ord=2)
        
        # Accumulate log volume expansion and renormalize to prevent overflow
        log_expansions[i] += np.log(norm_M)
        states[i, 3:12] = (M / norm_M).flatten()
        
    curr_T = (step + 1) * dt
    # Stable average over the ensemble utilizing log-sum-exp approach
    max_log = np.max(log_expansions)
    mean_exp = np.mean(np.exp(log_expansions - max_log))
    EE = (max_log + np.log(mean_exp)) / curr_T
    
    EE_values.append(EE)
    ee_times.append(curr_T)

# =============================================================================
# 5. RENDER THE PIXEL-PERFECT PLOT
# =============================================================================
print("Done! Rendering Plot...")

plt.figure(figsize=(10, 6))

# Expansion Entropy Curve
plt.plot(ee_times, EE_values, 'b-o', markersize=4, label='Expansion Entropy Ensemble')

# Weighted Kneading Roots Curve
plt.plot(filtered_t, filtered_s, 'r-s', markersize=4, linewidth=2, label='Weighted Kneading Truncation Roots')

# Converged horizontal line
final_rugh = filtered_s[-1]
plt.axhline(final_rugh, color='k', linestyle='--', linewidth=1.5, alpha=0.8, 
            label=f'Converged Exact Entropy ({final_rugh:.6f})')

# Formatting and aesthetics
plt.xlim(0, 120)
plt.ylim(0.08, 0.16)
plt.xlabel('Continuous Elapsed Time (T)', fontsize=12)
plt.ylabel('Topological Entropy Estimate', fontsize=12)
plt.title('Convergence: Expansion Entropy vs. Weighted Kneading Roots', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Display the final reproduction
plt.show()