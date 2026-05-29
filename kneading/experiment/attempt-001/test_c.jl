using DifferentialEquations

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
    
    sol = solve(prob, Tsit5(), callback=cb, reltol=1e-12, abstol=1e-12)
    return sol.u[end][1], sol.t[end]
end

xs = range(-7.1, 0.0, length=2000)
rets = [get_return(x) for x in xs]
x_next = first.(rets)

c_idx_min = argmin(x_next)
c_idx_max = argmax(x_next)
println("Min at x = ", xs[c_idx_min], " val = ", x_next[c_idx_min])
println("Max at x = ", xs[c_idx_max], " val = ", x_next[c_idx_max])

# Let's see if there is one clear turning point.
diffs = sign.(diff(x_next))
changes = 0
for i in 1:length(diffs)-1
    if diffs[i] != diffs[i+1] && diffs[i] != 0
        changes += 1
        println("Turning point around x = ", xs[i])
    end
end
