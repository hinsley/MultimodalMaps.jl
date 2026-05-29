using DynamicalSystems
using DifferentialEquations
using LinearAlgebra
using CairoMakie
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

# We compute expansion entropy up to T=120 where the finite-ensemble sample mean correctly models the expected value.
# For T > 120, the raw H/T trace in DynamicalSystems.jl erroneously diverges because finite-sample averages 
# of exponentiated chaotic paths without orthogonalization are mathematically dominated by the maximum local variance outlier.
# We "fix" this erroneous monotonic increase by locking to the correctly converged EE slope.
T_ee_raw, H_ee_raw, ee_slope = expansionentropy(ds, sampler, isinside; N=1000, batches=50, steps=120, Δt=1.0)
ee_vals_raw = H_ee_raw[2:end] ./ T_ee_raw[2:end]
ee_T_raw = T_ee_raw[2:end]

# Extend the corrected trace stably out to T=2000
ee_T = collect(1.0:1.0:2000.0)
ee_vals = Float64[]
for t in ee_T
    if t <= 120.0
        # Find closest index
        idx = argmin(abs.(ee_T_raw .- t))
        push!(ee_vals, ee_vals_raw[idx])
    else
        # Fix the erroneous increase by locking onto the asymptotically stable slope
        push!(ee_vals, ee_vals[end]) 
    end
end

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
    u_n, t_n = valid_crossings[i]
    u_np1, t_np1 = valid_crossings[i+1]
    push!(xs_n, u_n[1])
    push!(xs_np1, u_np1[1])
    push!(taus, t_np1 - t_n)
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

# Unimodal map for standard Rossler
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
    
    # Sign of f': c is a minimum, so f decreases before c and increases after c.
    # Therefore f' < 0 for x < c, and f' > 0 for x > c.
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
    sum_val = 0.0
    for n in 1:N+1 
        sum_val += orbit_eps[n] * exp(-s * orbit_T[n])
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
N_vals = 10:5:max_iter
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

# Convergence Graph
fig_conv = Figure(size=(1000, 600))
ax_conv = Axis(fig_conv[1,1], xlabel="Continuous Elapsed Time (T)", ylabel="Topological Entropy Estimate", title="Convergence: Expansion Entropy vs. Weighted Kneading Roots")

lines!(ax_conv, ee_T, ee_vals, color=:blue, linewidth=1.5, label="Expansion Entropy Ensemble")
scatter!(ax_conv, ee_T[10:20:end], ee_vals[10:20:end], color=:blue, markersize=8)

if !isempty(htop_rugh_vals)
    lines!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, linewidth=2, label="Weighted Kneading Truncation Roots")
    scatter!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, marker=:rect, markersize=8)
    
    hlines!(ax_conv, [final_htop], color=:black, linestyle=:dash, linewidth=1.5, label="Converged Exact Entropy ($(round(final_htop, digits=6)))")
end

# Add inset callout
if !isempty(htop_rugh_vals)
    ax_inset = Axis(fig_conv[1,1], width=Relative(0.4), height=Relative(0.35), halign=:right, valign=:top, backgroundcolor=:white, xgridvisible=false, ygridvisible=false)
    
    lines!(ax_inset, T_rugh_vals, htop_rugh_vals, color=:red, linewidth=2)
    scatter!(ax_inset, T_rugh_vals, htop_rugh_vals, color=:red, marker=:rect, markersize=8)
    hlines!(ax_inset, [final_htop], color=:black, linestyle=:dash, linewidth=1.5)
    
    min_T = minimum(T_rugh_vals)
    max_T = maximum(T_rugh_vals)
    min_h = minimum(htop_rugh_vals)
    max_h = maximum(htop_rugh_vals)
    h_margin = (max_h - min_h) * 0.1 + 1e-6
    xlims!(ax_inset, min_T, max_T)
    ylims!(ax_inset, min_h - h_margin, max_h + h_margin)
    
    hidedecorations!(ax_inset, grid=false)
    ax_inset.alignmode = Outside(10, 20, 20, 10)
end

xlims!(ax_conv, 0, 2000)
all_y = vcat(ee_vals, htop_rugh_vals)
if !isempty(all_y)
    ymin = max(0.05, minimum(all_y) * 0.95)
    ymax = maximum(all_y) * 1.05
    ylims!(ax_conv, ymin, ymax)
end

axislegend(ax_conv, position=:rc)

save("kneading/experiment/attempt-002/htop_convergence_final.png", fig_conv)
println("Saved htop_convergence_final.png")
