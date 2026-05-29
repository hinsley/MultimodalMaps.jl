using DifferentialEquations
using LinearAlgebra
using CairoMakie
using Interpolations

function rossler!(du, u, p, t)
    x, y, z = u
    a, b, c = p
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b * x + z * (x - c)
end

function get_rossler_le()
    function jac!(du, u, p, t)
        x, y, z = u[1], u[2], u[3]
        a, b, c = p
        du[1] = -y - z
        du[2] = x + a * y
        du[3] = b * x + z * (x - c)
        
        J = [0.0 -1.0 -1.0; 1.0 a 0.0; b+z 0.0 x-c]
        
        for i in 1:3
            v = [u[3+i], u[6+i], u[9+i]]
            Jv = J * v
            du[3+i] = Jv[1]
            du[6+i] = Jv[2]
            du[9+i] = Jv[3]
        end
    end
    
    p = (0.355, 0.3, 5.5)
    u0_full = [-7.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    T_step = 1.0
    n_steps = 5000
    u_curr = copy(u0_full)
    
    le_sum = 0.0
    T_total = 0.0

    le_vals = Float64[]
    T_vals = Float64[]

    for i in 1:n_steps
        prob = ODEProblem(jac!, u_curr, (0.0, T_step), p)
        sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-9, maxiters=1e7)
        u_curr = copy(sol.u[end])
        
        Phi = [u_curr[4] u_curr[7] u_curr[10];
               u_curr[5] u_curr[8] u_curr[11];
               u_curr[6] u_curr[9] u_curr[12]]
               
        Q, R = qr(Phi)
        le_sum += log(abs(R[1,1]))
        T_total += T_step
        
        if i % 10 == 0
            push!(le_vals, le_sum / T_total)
            push!(T_vals, T_total)
        end
        
        u_curr[4:6] = Q[:,1]
        u_curr[7:9] = Q[:,2]
        u_curr[10:12] = Q[:,3]
    end

    return T_vals, le_vals, (le_sum / T_total)
end

println("Computing true Lyapunov Exponent...")
T_le, le_vals, final_le = get_rossler_le()
println("Final LE: ", final_le)

println("Generating empirical 1D return map from ODE (x minima)...")
p = (0.355, 0.3, 5.5)
u0 = [-5.0, 5.0, 0.0]
prob = ODEProblem(rossler!, u0, (0.0, 50000.0), p)

# Up crossing of x' = -y - z -> means -y - z goes from negative to positive.
# This corresponds to x'' > 0 -> local minima of x.
condition(u, t, integrator) = -u[2] - u[3]
crossings = []
function affect!(integrator)
    # The minima of x on Rossler are negative, so this filter is safe.
    if integrator.u[1] < 0
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

# Deduplicate points for linear interpolation
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

# Find legitimate critical points (turning points)
diffs = sign.(diff(fx_uniq))
all_cps = Float64[]
for i in 1:length(diffs)-1
    if diffs[i] != diffs[i+1] && diffs[i] != 0
        push!(all_cps, x_uniq[i+1])
    end
end
cps = [all_cps[1]]
for i in 2:length(all_cps)
    if abs(all_cps[i] - cps[end]) > 0.1
        push!(cps, all_cps[i])
    end
end
println("Found true critical points on attractor: ", cps)

k = length(cps)
max_iter = 100
orbits = []

# Generate orbits using the empirical map
for i in 1:k
    c_i = cps[i]
    x0_val = f_interp(c_i)
    
    orbit_x = [x0_val]
    orbit_T = [0.0]
    orbit_eps = [1.0]
    
    curr_x_val = x0_val
    curr_T_val = 0.0
    curr_eps_val = 1.0
    
    for n in 1:max_iter
        dx = 1e-6
        fp = f_interp(curr_x_val + dx)
        fm = f_interp(curr_x_val - dx)
        deriv_sign = sign(fp - fm)
        
        curr_eps_val *= deriv_sign
        
        nx = f_interp(curr_x_val)
        nt = tau_interp(curr_x_val)
        
        curr_x_val = nx
        curr_T_val += nt
        
        push!(orbit_x, curr_x_val)
        push!(orbit_T, curr_T_val)
        push!(orbit_eps, curr_eps_val)
    end
    
    push!(orbits, (x=orbit_x, T=orbit_T, eps=orbit_eps))
end

function E_step(x, c_j)
    if x > c_j
        return 1.0
    elseif x < c_j
        return -1.0
    else
        return 0.0
    end
end

# Compute the accurate Rugh Kneading Determinant R_{ij}(s)
function det_R(s, N)
    M = zeros(Float64, k, k)
    for i in 1:k
        dx = 1e-6
        fp = f_interp(cps[i] + dx)
        deriv_c_plus = sign(fp - f_interp(cps[i]))
        
        for j in 1:k
            c_j = cps[j]
            sum_val = (i == j) ? 1.0 : 0.0
            
            for n in 0:N
                x_n = orbits[i].x[n+1]
                T_n = orbits[i].T[n+1]
                
                # In Rugh formulation, time starts accumulating with tau(c_i)
                T_n_rugh = T_n + tau_interp(cps[i])
                
                # eps_n is multiplied by s(c_i^+)
                eps_n_rugh = orbits[i].eps[n+1] * deriv_c_plus
                
                # E_step exactly substitutes 2 * sigma
                sum_val += eps_n_rugh * E_step(x_n, c_j) * exp(-s * T_n_rugh)
            end
            M[i, j] = sum_val
        end
    end
    return det(M)
end

function bisection(f, a, b; tol=1e-12, max_iter=200)
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb)
        return NaN
    end
    for _ in 1:max_iter
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) < tol || (b - a) / 2 < tol
            return m
        end
        if sign(fm) == sign(fa)
            a = m
            fa = fm
        else
            b = m
            fb = fm
        end
    end
    return (a + b) / 2
end

println("Computing htop convergence...")
N_vals = 10:max_iter
htop_rugh_vals = Float64[]
T_rugh_vals = Float64[]

for N in N_vals
    D_r(s) = det_R(s, N)
    
    # Bracket strictly away from s=0 to avoid the trivial root
    ss = range(0.02, 0.20, length=500)
    
    ds_r = [D_r(s) for s in ss]
    root_r = NaN
    for i in 1:length(ss)-1
        if sign(ds_r[i]) != sign(ds_r[i+1]) && sign(ds_r[i]) != 0
            root_r = bisection(D_r, ss[i], ss[i+1])
            break
        end
    end
    
    if !isnan(root_r)
        push!(htop_rugh_vals, root_r)
        push!(T_rugh_vals, orbits[1].T[N+1])
    end
end

final_htop = isempty(htop_rugh_vals) ? NaN : htop_rugh_vals[end]
println("Final true htop estimate: ", final_htop)

# Graphing exactly as requested
fig_conv = Figure(size=(800, 600))
ax_conv = Axis(fig_conv[1,1], xlabel="Continuous Time (T)", ylabel="Estimated h_top", title="Topological Entropy Convergence")

# Plot expansion entropy
lines!(ax_conv, T_le, le_vals, color=:blue, linewidth=1, label="Lyapunov Exponent / EE = $(round(final_le, digits=5))")
hlines!(ax_conv, [final_le], color=:blue, linestyle=:dash, linewidth=1)

# Plot kneading entropy
if !isempty(htop_rugh_vals)
    lines!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, linewidth=1, label="Kneading h_top = $(round(final_htop, digits=5))")
    scatter!(ax_conv, T_rugh_vals, htop_rugh_vals, color=:red, markersize=4)
    hlines!(ax_conv, [final_htop], color=:red, linestyle=:dash, linewidth=1)
end

axislegend(ax_conv, position=:rc)

# Add inset
if !isempty(htop_rugh_vals)
    ax_inset = Axis(fig_conv[1,1], width=Relative(0.4), height=Relative(0.35), halign=:right, valign=:bottom, backgroundcolor=:white, xgridvisible=false, ygridvisible=false)
    
    lines!(ax_inset, T_rugh_vals, htop_rugh_vals, color=:red, linewidth=1)
    scatter!(ax_inset, T_rugh_vals, htop_rugh_vals, color=:red, markersize=4)
    hlines!(ax_inset, [final_htop], color=:red, linestyle=:dash, linewidth=1)
    
    # Set limits to zoom in on the Kneading data
    min_T = minimum(T_rugh_vals)
    max_T = maximum(T_rugh_vals)
    min_h = minimum(htop_rugh_vals)
    max_h = maximum(htop_rugh_vals)
    h_margin = (max_h - min_h) * 0.1 + 1e-6
    xlims!(ax_inset, min_T, max_T)
    ylims!(ax_inset, min_h - h_margin, max_h + h_margin)
    
    # Hide axis decorations for inset
    hidedecorations!(ax_inset, grid=false)
    
    # Translate it slightly up and left so it doesn't touch the borders
    ax_inset.alignmode = Outside(10, 20, 20, 10)
end

save("kneading/experiment/attempt-001/htop_convergence_final.png", fig_conv)
println("Saved htop_convergence_final.png")
