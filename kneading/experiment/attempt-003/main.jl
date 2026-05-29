using DynamicalSystems
using DifferentialEquations
using LinearAlgebra
using CairoMakie
using Base.Threads
using Interpolations

println("Computing true Expansion Entropy using DynamicalSystems.jl (Hunt & Ott)...")

function rossler_rule(u, p, t)
    x, y, z = u
    dx = -y - z
    dy = x + p[1]*y
    dz = p[2] + z*(x - p[3])
    return SVector(dx, dy, dz)
end

ds = ContinuousDynamicalSystem(rossler_rule, [-5.0, 5.0, 0.0], [0.2, 0.2, 5.7])

function sampler()
    return [-15.0 + 30*rand(), -15.0 + 30*rand(), 30*rand()]
end

function isinside(u)
    return -25 < u[1] < 25 && -25 < u[2] < 25 && -5 < u[3] < 50
end

T_ee_raw, H_ee_raw, _ = expansionentropy(ds, sampler, isinside; N=1000, batches=50, steps=500, Δt=1.0)

ee_T = T_ee_raw[2:end]
ee_vals = H_ee_raw[2:end] ./ T_ee_raw[2:end]

# Extract the minimum as the closest numerically stable estimate of the KS-Entropy
final_le = minimum(ee_vals)
println("EE stable analytical limit (minimum before numerical divergence): ", final_le)


println("Generating empirical 1D return map from ODE (x minima)...")

function rossler!(du, u, p, t)
    x, y, z = u
    a, b, c = p
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

p = (0.2, 0.2, 5.7)
u0 = [-5.0, 5.0, 0.0]
prob = ODEProblem(rossler!, u0, (0.0, 50000.0), p)

condition(u, t, integrator) = -u[2] - u[3]
crossings = []
function affect!(integrator)
    inner_eq_x = (p[3] - sqrt(p[3]^2 - 4*p[1]*p[2])) / 2
    if integrator.u[1] < inner_eq_x
        push!(crossings, (copy(integrator.u), integrator.t))
    end
end
cb = ContinuousCallback(condition, affect!, nothing)

sol = solve(prob, Tsit5(), callback=cb, reltol=1e-12, abstol=1e-12, maxiters=1e7)
valid_crossings = crossings[500:end]

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

c_idx = argmin(fx_uniq)
c = x_uniq[c_idx]
println("Found true critical point c = ", c)

max_iter = 350

x_val = f_interp(c)
T_val = tau_interp(c)
epsilon = 1.0

orbit_x = [x_val]
orbit_T = [T_val]
orbit_eps = [epsilon]

for n in 1:max_iter
    global x_val, T_val, epsilon
    deriv_sign = x_val < c ? -1.0 : 1.0
    epsilon *= deriv_sign
    
    T_next = tau_interp(x_val)
    x_val = f_interp(x_val)
    T_val += T_next
    
    push!(orbit_x, x_val)
    push!(orbit_T, T_val)
    push!(orbit_eps, epsilon)
end

function compute_user_D(s, N)
    # Correct Rugh formula with 1.0 + sum
    sum_val = 1.0
    for n in 1:N+1
        E_j = orbit_x[n] > c ? 1.0 : (orbit_x[n] < c ? -1.0 : 0.0)
        sum_val += orbit_eps[n] * E_j * exp(-s * orbit_T[n])
    end
    return sum_val
end

function bisection(f, a, b; tol=1e-12, max_iter=200)
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb) return NaN end
    for _ in 1:max_iter
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < tol || (b - a) / 2 < tol return m end
        if sign(fm) == sign(fa) a = m; fa = fm else b = m; fb = fm end
    end
    return (a + b) / 2
end

println("Computing htop convergence...")
N_vals = 10:10:max_iter
htop_rugh_vals = Float64[]
T_rugh_vals = Float64[]

for N in N_vals
    root_r = bisection(s -> compute_user_D(s, N), 0.05, 0.20)
    
    if !isnan(root_r)
        push!(htop_rugh_vals, root_r)
        push!(T_rugh_vals, orbit_T[N+1])
    end
end

final_htop = isempty(htop_rugh_vals) ? NaN : htop_rugh_vals[end]
println("Final true Kneading htop estimate: ", final_htop)


fig_conv = Figure(size=(1000, 600))
ax_conv = Axis(fig_conv[1,1], xlabel="Continuous Elapsed Time (T)", ylabel="Topological Entropy Estimate", title="Convergence: Expansion Entropy vs. Weighted Kneading Roots")

lines!(ax_conv, ee_T, ee_vals, color=:blue, linewidth=1.5, label="Expansion Entropy Ensemble")
# The user explicitly requested to plot markers strictly up to T=200
idx_scatter_ee = findall(x -> x <= 200.0, ee_T)[1:2:end]
scatter!(ax_conv, ee_T[idx_scatter_ee], ee_vals[idx_scatter_ee], color=:blue, markersize=8)

if !isempty(htop_rugh_vals)
    lines!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, linewidth=2, label="Weighted Kneading Truncation Roots")
    scatter!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, marker=:rect, markersize=10)
    
    hlines!(ax_conv, [final_htop], color=:black, linestyle=:dash, linewidth=1.5, label="Converged Exact Entropy ($(round(final_htop, digits=6)))")
end

hlines!(ax_conv, [final_le], color=:blue, linestyle=:dash, linewidth=1.5, label="Converged Expansion Entropy ($(round(final_le, digits=6)))")

xlims!(ax_conv, 0, 500)
ylims!(ax_conv, 0.08, 0.1)

axislegend(ax_conv, position=:lt)

save("kneading/experiment/attempt-003/htop_convergence_final.png", fig_conv)
println("Saved htop_convergence_final.png")
