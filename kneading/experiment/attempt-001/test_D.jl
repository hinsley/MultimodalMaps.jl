using DifferentialEquations
using CairoMakie

# Rossler System
function rossler!(du, u, p, t)
    x, y, z = u
    a, b, c = p
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b * x + z * (x - c)
end

function get_return(x0; a=0.355, b=0.3, c=5.5)
    p = (a, b, c)
    u0 = [x0, 0.0, 0.0]
    prob = ODEProblem(rossler!, u0, (0.0, 50.0), p)
    
    condition(u, t, integrator) = u[2] + u[3]
    function real_affect!(integrator)
        if integrator.u[1] < 0 && integrator.t > 1.0
            terminate!(integrator)
        end
    end
    cb = ContinuousCallback(condition, real_affect!)
    
    sol = solve(prob, Tsit5(), callback=cb, reltol=1e-11, abstol=1e-11)
    return sol.u[end][1], sol.t[end]
end

c = -3.5571142284569137
max_iter = 100
orbit_x = Float64[]
orbit_T = Float64[]
orbit_eps = Float64[]

curr_x, curr_t = get_return(c)
curr_T = curr_t
curr_eps = 1.0

push!(orbit_x, curr_x)
push!(orbit_T, curr_T)
push!(orbit_eps, curr_eps)

for n in 1:max_iter
    global curr_x, curr_T, curr_eps
    
    dx = 1e-6
    f_plus, _ = get_return(curr_x + dx)
    f_minus, _ = get_return(curr_x - dx)
    deriv_sign = sign(f_plus - f_minus)
    
    curr_eps *= deriv_sign
    
    next_x, next_t = get_return(curr_x)
    curr_x = next_x
    curr_T += next_t
    
    push!(orbit_x, curr_x)
    push!(orbit_T, curr_T)
    push!(orbit_eps, curr_eps)
end

function D(s, N=100)
    sum(orbit_eps[n+1] * exp(-s * orbit_T[n+1]) for n in 0:N)
end

ss = range(0.0, 0.5, length=500)
ds = [D(s) for s in ss]

for (s, d) in zip(ss, ds)
    println("s = $s, D(s) = $d")
end
