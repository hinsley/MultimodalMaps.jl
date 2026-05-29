using DifferentialEquations
using CairoMakie

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
    
    # We want to cross y + z = 0, which is x' = 0
    # Since we start at y=0, z=0, we are at y+z=0 at t=0
    condition(u, t, integrator) = u[2] + u[3]
    
    affect!(integrator) = terminate!(integrator)
    
    # Only terminate if x < 0 and t > 1.0
    function real_affect!(integrator)
        if integrator.u[1] < 0 && integrator.t > 1.0
            terminate!(integrator)
        end
    end
    
    cb = ContinuousCallback(condition, real_affect!)
    
    sol = solve(prob, Tsit5(), callback=cb, reltol=1e-9, abstol=1e-9)
    
    return sol.u[end][1], sol.t[end]
end

xs = range(-7.1, 0.0, length=200)
rets = [get_return(x) for x in xs]
x_next = first.(rets)
t_next = last.(rets)

fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, xs, x_next, markersize=2)
save("test_return.png", fig)

println("Test ran successfully")
