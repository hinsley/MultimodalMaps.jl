using DifferentialEquations
using LinearAlgebra
using Base.Threads
using Statistics

function rossler_var!(du, u, p, t)
    x, y, z = u[1], u[2], u[3]
    a, b, c = p
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
    
    w11, w12, w13 = u[4], u[5], u[6]
    w21, w22, w23 = u[7], u[8], u[9]
    w31, w32, w33 = u[10], u[11], u[12]
    
    du[4], du[5], du[6] = -w21 - w31, -w22 - w32, -w23 - w33
    du[7], du[8], du[9] = w11 + a*w21, w12 + a*w22, w13 + a*w23
    du[10], du[11], du[12] = z*w11 + (x-c)*w31, z*w12 + (x-c)*w32, z*w13 + (x-c)*w33
end

p = (0.2, 0.2, 5.7)

N_points = 150
import Random
Random.seed!(42)

states = zeros(12, N_points)
for i in 1:N_points
    states[1, i] = -10.0 + 25.0 * rand()
    states[2, i] = -15.0 + 25.0 * rand()
    states[3, i] = 30.0 * rand()
    states[4:12, i] .= [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
end

T_max = 500.0
dt = 4.0
steps = Int(T_max / dt)

log_expansions = zeros(N_points)
ee_vals = Float64[]

for step in 1:steps
    for i in 1:N_points
        prob = ODEProblem(rossler_var!, states[:, i], (0.0, dt), p)
        sol = solve(prob, DP5(), reltol=1e-5, abstol=1e-5)
        
        state = copy(sol.u[end])
        
        # QR reorthonormalization exactly tracking Hunt & Ott discrete QR metric
        # Python script calculates opnorm directly. Let's do exactly what Python does.
        M = [state[4] state[5] state[6];
             state[7] state[8] state[9];
             state[10] state[11] state[12]]
             
        # norm_M = np.linalg.norm(M, ord=2) -> maximum singular value
        norm_M = opnorm(M, 2)
        log_expansions[i] += log(norm_M)
        
        M_new = M ./ norm_M
        state[4:6] .= M_new[1, :]
        state[7:9] .= M_new[2, :]
        state[10:12] .= M_new[3, :]
        states[:, i] .= state
    end
    
    curr_T = step * dt
    max_log = maximum(log_expansions)
    mean_exp = mean(exp.(log_expansions .- max_log))
    EE = (max_log + log(mean_exp)) / curr_T
    
    push!(ee_vals, EE)
end

println("Julia custom EE at T=500: ", ee_vals[end])
