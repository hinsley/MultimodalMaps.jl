# Rossler event debug visualization
# Shows where threshold crossings and revolution events occur in state space.
#
# Run from repo root:
#   julia scans/rossler/rossler_event_debug.jl
#
# Optional environment overrides:
#   A, B, C, Z_THRESH, T_END, T_TRANSIENT, SAVE_DT, DT, SEED, MAX_SYMBOLS, ADAPTIVE, MAX_STATE, WAIT, U0_X, U0_Y, U0_Z

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DifferentialEquations
using GLMakie
using Printf

GLMakie.activate!()
using Random

# ============================================================
# Parameters
# ============================================================
const A = parse(Float64, get(ENV, "A", "0.5"))
const B = parse(Float64, get(ENV, "B", "0.2"))
const C = parse(Float64, get(ENV, "C", "5.0"))
const Z_THRESH = parse(Float64, get(ENV, "Z_THRESH", "1.0"))

const T_END = parse(Float64, get(ENV, "T_END", "200"))
const T_TRANSIENT = parse(Float64, get(ENV, "T_TRANSIENT", "10"))
const SAVE_DT = parse(Float64, get(ENV, "SAVE_DT", "0.01"))
const DT = parse(Float64, get(ENV, "DT", "0.005"))
const ADAPTIVE = parse(Bool, get(ENV, "ADAPTIVE", "false"))

const SEED = parse(Int, get(ENV, "SEED", "1"))
const MAX_SYMBOLS = parse(Int, get(ENV, "MAX_SYMBOLS", "64"))
const MAX_STATE = parse(Float64, get(ENV, "MAX_STATE", "1000"))
const WAIT = parse(Bool, get(ENV, "WAIT", "true"))

# ============================================================
# Rossler system (origin equilibrium)
# ============================================================
function rossler!(du, u, p, t)
    x, y, z = u
    a, b, c = p
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b * x + z * (x - c)
end

function initial_condition()
    if haskey(ENV, "U0_X") && haskey(ENV, "U0_Y") && haskey(ENV, "U0_Z")
        return [
            parse(Float64, ENV["U0_X"]),
            parse(Float64, ENV["U0_Y"]),
            parse(Float64, ENV["U0_Z"]),
        ]
    end
    rng = MersenneTwister(SEED)
    return [
        2.0 * rand(rng) - 1.0,
        2.0 * rand(rng) - 1.0,
        2.0 * rand(rng) - 1.0,
    ]
end

# ============================================================
# Event capture (matches sweep logic)
# ============================================================
spike_points = Point3f[]
revolution_points = Point3f[]
symbols = Int[]

function z_upcross_condition(u, t, integrator)
    if t < T_TRANSIENT
        return false
    end
    z_prev = integrator.uprev[3]
    z_curr = u[3]
    return (z_prev <= Z_THRESH && z_curr > Z_THRESH)
end

function spike_affect!(integrator)
    if integrator.t < T_TRANSIENT
        return
    end
    u_prev = integrator.uprev
    u = integrator.u
    z_prev = u_prev[3]
    z_curr = u[3]
    denom = z_curr - z_prev
    θ = denom == 0.0 ? 0.0 : (Z_THRESH - z_prev) / denom
    if θ < 0.0
        θ = 0.0
    elseif θ > 1.0
        θ = 1.0
    end

    x_cross = u_prev[1] + θ * (u[1] - u_prev[1])
    y_cross = u_prev[2] + θ * (u[2] - u_prev[2])
    push!(spike_points, Point3f(x_cross, y_cross, Z_THRESH))
    if length(symbols) < MAX_SYMBOLS
        push!(symbols, 1)
    end
end

const spike_cb = DiscreteCallback(
    z_upcross_condition,
    spike_affect!;
    save_positions=(false, false)
)

function y_downcross_condition(u, t, integrator)
    if t < T_TRANSIENT
        return false
    end
    y_prev = integrator.uprev[2]
    y_curr = u[2]
    return (y_prev >= 0.0 && y_curr < 0.0)
end

function revolution_affect!(integrator)
    if integrator.t < T_TRANSIENT
        return
    end
    u_prev = integrator.uprev
    u = integrator.u

    y_prev = u_prev[2]
    y_curr = u[2]
    denom = y_prev - y_curr
    θ = denom == 0.0 ? 0.0 : y_prev / denom
    if θ < 0.0
        θ = 0.0
    elseif θ > 1.0
        θ = 1.0
    end

    x_cross = u_prev[1] + θ * (u[1] - u_prev[1])
    if x_cross >= 0.0
        return
    end
    z_cross = u_prev[3] + θ * (u[3] - u_prev[3])

    push!(revolution_points, Point3f(x_cross, 0.0, z_cross))
    if length(symbols) < MAX_SYMBOLS
        push!(symbols, 0)
    end
end

const revolution_cb = DiscreteCallback(
    y_downcross_condition,
    revolution_affect!;
    save_positions=(false, false)
)

function blowup_condition(u, t, integrator)
    return (abs(u[1]) > MAX_STATE || abs(u[2]) > MAX_STATE || abs(u[3]) > MAX_STATE)
end

function blowup_affect!(integrator)
    terminate!(integrator)
end

const blowup_cb = DiscreteCallback(
    blowup_condition,
    blowup_affect!;
    save_positions=(false, false)
)

const cb = CallbackSet(spike_cb, revolution_cb, blowup_cb)

# ============================================================
# Integrate and plot
# ============================================================
u0 = initial_condition()
p = (A, B, C)

println("Rossler event debug:")
println("  a=$A, b=$B, c=$C")
println("  z threshold: $Z_THRESH")
println("  tspan: (0, $T_END), transient=$T_TRANSIENT, save_dt=$SAVE_DT, dt=$DT")
println("  max state: $MAX_STATE")
println("  u0 = $u0")
println()

prob = ODEProblem(rossler!, u0, (0.0, T_END), p)
sol = solve(
    prob,
    Tsit5();
    callback=cb,
    saveat=SAVE_DT,
    dt=DT,
    adaptive=ADAPTIVE,
    unstable_check=(t, u, p, dt) -> false,
    maxiters=10_000_000,
    save_everystep=false
)

println("Spike upcrossings: $(length(spike_points))")
println("Revolutions: $(length(revolution_points))")
println("Recorded symbols: $(length(symbols)) (max $MAX_SYMBOLS)")

function lz_complexity_string(s::AbstractString)
    n = lastindex(s)
    n == 0 && return 0
    c = 0
    i = 1
    while i <= n
        k = 1
        while i + k - 1 <= n
            prefix_end = i + k - 2
            if prefix_end < 1
                break
            end
            if occursin(SubString(s, i, i + k - 1), SubString(s, 1, prefix_end))
                k += 1
            else
                break
            end
        end
        c += 1
        i += k
    end
    return c
end

if !isempty(symbols)
    bits = join(string.(symbols))
    ones = count(==(1), symbols)
    transitions = sum(symbols[1:end-1] .!= symbols[2:end])
    lz_c = lz_complexity_string(bits)
    lz_norm = lz_c * log2(length(symbols)) / length(symbols)
    println("Symbols bits: $bits")
    println("Ones: $ones, transitions: $transitions")
    println("LZ c: $lz_c, normalized: $lz_norm")
end

xs = [u[1] for u in sol.u]
ys = [u[2] for u in sol.u]
zs = [u[3] for u in sol.u]

fig = Figure(size=(1100, 800))
ax3 = Axis3(fig[1, 1], xlabel="x", ylabel="y", zlabel="z", title="Rossler trajectory with events")
lines!(ax3, xs, ys, zs, color=:gray)
scatter!(ax3, spike_points, color=:red, markersize=6, label="spike (z upcross)")
scatter!(ax3, revolution_points, color=:deepskyblue, markersize=6, label="revolution (y downcross, x<0)")
axislegend(ax3; position=:rb)

ax2 = Axis(fig[2, 1], xlabel="x", ylabel="y", aspect=DataAspect(), title="xy projection")
lines!(ax2, xs, ys, color=:gray)
scatter!(ax2, [p[1] for p in spike_points], [p[2] for p in spike_points], color=:red, markersize=6)
scatter!(ax2, [p[1] for p in revolution_points], [p[2] for p in revolution_points], color=:deepskyblue, markersize=6)

output_path = joinpath(
    @__DIR__,
    @sprintf("rossler_event_debug_a%.3f_c%.3f.png", A, C)
)
save(output_path, fig)
println("Saved debug plot to: $output_path")

screen = display(fig)
if !isinteractive() && WAIT
    wait(screen)
end
