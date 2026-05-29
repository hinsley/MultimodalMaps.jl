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

print("Rugh Final Root:", filtered_s[-1])
