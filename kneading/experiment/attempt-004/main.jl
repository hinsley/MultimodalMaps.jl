using DifferentialEquations
using LinearAlgebra
using CairoMakie
using Statistics
using Interpolations
using Roots
using Random

println("Integrating 150-point ensemble for Expansion Entropy (Fast Julia port of exact Python logic)...")

# EXACT same integration as Python solve_ivp
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

a, b, c_param = 0.2, 0.2, 5.7
p = (a, b, c_param)

N_points = 150
Random.seed!(42)
states = zeros(12, N_points)
for i in 1:N_points
    states[1, i] = -10.0 + 25.0 * rand()
    states[2, i] = -15.0 + 25.0 * rand()
    states[3, i] = 30.0 * rand()
    states[4:12, i] .= [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
end

T_max, dt = 500.0, 4.0
steps = Int(T_max / dt)

log_expansions = zeros(N_points)
EE_values = Float64[]
ee_times = Float64[]

# Setup a dummy problem so we can use reinit! to make it blisteringly fast (100x faster than python)
base_prob = ODEProblem(rossler_var!, states[:, 1], (0.0, dt), p)
# DP5 is the Julia equivalent of RK45
integrator = init(base_prob, DP5(), reltol=1e-5, abstol=1e-5, save_everystep=false)

for step in 1:steps
    for i in 1:N_points
        reinit!(integrator, states[:, i])
        solve!(integrator)
        
        state = copy(integrator.u)
        
        # M is manually extracted to match Python's row-major reshape
        M = [state[4] state[5] state[6];
             state[7] state[8] state[9];
             state[10] state[11] state[12]]
             
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
    
    push!(EE_values, EE)
    push!(ee_times, curr_T)
end

final_le = EE_values[end]
println("EE final value (T=500): ", final_le)


println("Generating empirical 1D return map from ODE (x maxima)...")

function rossler!(du, u, p, t)
    x, y, z = u
    du[1] = -y - z
    du[2] = x + p[1] * y
    du[3] = p[2] + z * (x - p[3])
end

prob_1d = ODEProblem(rossler!, [1.0, 1.0, 1.0], (0.0, 6000.0), p)
condition(u, t, integrator) = -u[2] - u[3]

crossings = []
function affect!(integrator)
    push!(crossings, (copy(integrator.u), integrator.t))
end
cb = ContinuousCallback(condition, nothing, affect!) # downcrossing -> local maxima of x

sol = solve(prob_1d, Tsit5(), callback=cb, reltol=1e-10, abstol=1e-10)
valid_crossings = crossings[101:end]

xs_n = Float64[]
xs_np1 = Float64[]
taus = Float64[]

for i in 1:length(valid_crossings)-1
    push!(xs_n, valid_crossings[i][1][1])
    push!(xs_np1, valid_crossings[i+1][1][1])
    push!(taus, valid_crossings[i+1][2] - valid_crossings[i][2])
end

perm = sortperm(xs_n)
x_sorted = xs_n[perm]
fx_sorted = xs_np1[perm]
tau_sorted = taus[perm]

x_uniq = Float64[]
fx_uniq = Float64[]
tau_uniq = Float64[]

let
    curr_x = x_sorted[1]
    curr_fx_sum = fx_sorted[1]
    curr_tau_sum = tau_sorted[1]
    count = 1

    for i in 2:length(x_sorted)
        if x_sorted[i] > curr_x + 1e-10
            push!(x_uniq, curr_x)
            push!(fx_uniq, curr_fx_sum / count)
            push!(tau_uniq, curr_tau_sum / count)
            
            curr_x = x_sorted[i]
            curr_fx_sum = fx_sorted[i]
            curr_tau_sum = tau_sorted[i]
            count = 1
        else
            curr_fx_sum += fx_sorted[i]
            curr_tau_sum += tau_sorted[i]
            count += 1
        end
    end
    push!(x_uniq, curr_x)
    push!(fx_uniq, curr_fx_sum / count)
    push!(tau_uniq, curr_tau_sum / count)
end

f_interp = LinearInterpolation(x_uniq, fx_uniq, extrapolation_bc=Line())
tau_interp = LinearInterpolation(x_uniq, tau_uniq, extrapolation_bc=Line())

# It's a unimodal map of maxima
c_idx = argmax(fx_uniq)
c_exact = x_uniq[c_idx]

N_rugh_max = 120
x_rugh = zeros(N_rugh_max)
epsilon = zeros(N_rugh_max)
T_rugh_accum = zeros(N_rugh_max)

x_rugh[1] = f_interp(c_exact)
epsilon[1] = 1.0
T_rugh_accum[1] = 0.0

for n in 2:N_rugh_max
    x_rugh[n] = f_interp(x_rugh[n-1])
    if x_rugh[n-1] < c_exact
        sgn = 1.0
    elseif x_rugh[n-1] > c_exact
        sgn = -1.0
    else
        sgn = 0.0
    end
    epsilon[n] = epsilon[n-1] * sgn
    T_rugh_accum[n] = T_rugh_accum[n-1] + tau_interp(x_rugh[n-1])
end

function compute_user_D(s, N)
    sum_val = 0.0
    for n in 1:N
        if epsilon[n] != 0.0
            sum_val += epsilon[n] * exp(-s * T_rugh_accum[n])
        end
    end
    return sum_val
end

println("Computing htop convergence...")
htop_rugh_vals = Float64[]
T_rugh_vals = Float64[]

for N in 2:N_rugh_max
    D(s) = compute_user_D(s, N)
    
    if D(0.0) <= 0.0
        root_r = find_zero(D, (0.0, 2.0), Bisection())
        if root_r > 0.01
            push!(htop_rugh_vals, root_r)
            push!(T_rugh_vals, T_rugh_accum[N])
        end
    end
end

final_htop = htop_rugh_vals[end]

println("Plotting...")
fig_conv = Figure(size=(1000, 600))
ax_conv = Axis(fig_conv[1,1], xlabel="Continuous Elapsed Time (T)", ylabel="Topological Entropy Estimate", title="Convergence: Expansion Entropy vs. Weighted Kneading Roots")

lines!(ax_conv, ee_times, EE_values, color=:blue, linewidth=1.5, label="Expansion Entropy Ensemble")
scatter!(ax_conv, ee_times, EE_values, color=:blue, markersize=5)

lines!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, linewidth=2, label="Weighted Kneading Truncation Roots")
scatter!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, marker=:rect, markersize=5)

hlines!(ax_conv, [final_htop], color=:black, linestyle=:dash, linewidth=1.5, label="Converged Exact Entropy ($(round(final_htop, digits=6)))")

xlims!(ax_conv, 0, 500)
# Dynamic bounds per user request but maximum cut off at 0.2
all_y = vcat(EE_values, htop_rugh_vals)
ymin = minimum(all_y) * 0.95
ylims!(ax_conv, ymin, 0.2)

axislegend(ax_conv, position=:lt)

save("kneading/experiment/attempt-004/htop_convergence_final.png", fig_conv)
println("Saved htop_convergence_final.png")
