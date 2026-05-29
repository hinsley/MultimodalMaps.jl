using DifferentialEquations
using LinearAlgebra
using Base.Threads
using Statistics
using CairoMakie
using PyCall

# To perfectly match Python, we use Python! We just want the EXACT exact array.
np = pyimport("numpy")
scipy_integrate = pyimport("scipy.integrate")

rossler_var_py = py"
def rossler_var(t, state):
    x, y, z = state[0:3]
    dx = -y - z
    dy = x + 0.2 * y
    dz = 0.2 + z * (x - 5.7)
    
    w11, w12, w13 = state[3:6]
    w21, w22, w23 = state[6:9]
    w31, w32, w33 = state[9:12]
    
    dw11, dw12, dw13 = -w21 - w31, -w22 - w32, -w23 - w33
    dw21, dw22, dw23 = w11 + 0.2*w21, w12 + 0.2*w22, w13 + 0.2*w23
    dw31, dw32, dw33 = z*w11 + (x-5.7)*w31, z*w12 + (x-5.7)*w32, z*w13 + (x-5.7)*w33
    
    return [dx, dy, dz, dw11, dw12, dw13, dw21, dw22, dw23, dw31, dw32, dw33]
"

np.random.seed(42)
N_points = 150
states = np.zeros((N_points, 12))
states[:, 1] = np.random.uniform(-10, 15, N_points)
states[:, 2] = np.random.uniform(-15, 10, N_points)
states[:, 3] = np.random.uniform(0, 30, N_points)
states[:, 4] = 1.0; states[:, 8] = 1.0; states[:, 12] = 1.0

T_max, dt = 500.0, 4.0
steps = Int(T_max / dt)

log_expansions = zeros(N_points)
EE_values = Float64[]
ee_times = Float64[]

for step in 1:steps
    for i in 1:N_points
        sol = scipy_integrate.solve_ivp(rossler_var_py, (0, dt), states[i, :], method=\"RK45\", rtol=1e-5, atol=1e-5)
        state_end = sol.y[:, end]
        
        M = reshape(state_end[4:12], 3, 3)'
        norm_M = opnorm(M, 2)
        
        log_expansions[i] += log(norm_M)
        states[i, 4:12] .= vec((M ./ norm_M)')
        states[i, 1:3] .= state_end[1:3]
    end
    
    curr_T = step * dt
    max_log = maximum(log_expansions)
    mean_exp = mean(exp.(log_expansions .- max_log))
    EE = (max_log + log(mean_exp)) / curr_T
    
    push!(EE_values, EE)
    push!(ee_times, curr_T)
end

open("kneading/experiment/attempt-004/pycall_ee.txt", "w") do io
    for v in EE_values
        println(io, v)
    end
end
