# Shimizu-Morioka parameter sweep on one unstable-manifold branch from the origin.
#
# System (B = 0):
#   x' = y
#   y' = x - lambda*y - x*z
#   z' = -alpha*(z - x^2)
#
# Symbolic events:
#   - Record x=0 crossings as symbol 0
#   - Record y=0 crossings as symbol 1
#   - Encode sequence as a1/2 + a2/4 + a3/8 + ...
#
# Sweep plane:
#   alpha  in [0.15, 0.65]
#   lambda in [0.45, 0.9]
#
# Output:
#   scans/SM_xy_roots_heatmap.png
#
# Run from repo root:
#   julia scans/SM_xy_roots.jl
#
# Optional environment overrides:
#   N_ALPHA, N_LAMBDA, T_END, DT, MAX_SYMBOLS, EPS0, ABSTOL, RELTOL, MAX_ITERS

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Random
using StaticArrays
using Statistics
using Base.Threads

# ============================================================
# Configuration
# ============================================================
const B = 0.0

const N_ALPHA = parse(Int, get(ENV, "N_ALPHA", "220"))
const N_LAMBDA = parse(Int, get(ENV, "N_LAMBDA", "220"))

const ALPHA_MIN = parse(Float64, get(ENV, "ALPHA_MIN", "0.15"))
const ALPHA_MAX = parse(Float64, get(ENV, "ALPHA_MAX", "0.65"))
const LAMBDA_MIN = parse(Float64, get(ENV, "LAMBDA_MIN", "0.45"))
const LAMBDA_MAX = parse(Float64, get(ENV, "LAMBDA_MAX", "0.9"))

const T_END = parse(Float64, get(ENV, "T_END", "200.0"))
const DT = parse(Float64, get(ENV, "DT", "0.1"))
const MAX_SYMBOLS = parse(Int, get(ENV, "MAX_SYMBOLS", "80"))
const EPS0 = parse(Float64, get(ENV, "EPS0", "1e-7"))
const ABSTOL = parse(Float64, get(ENV, "ABSTOL", "1e-10"))
const RELTOL = parse(Float64, get(ENV, "RELTOL", "1e-10"))
const MAX_ITERS = parse(Int, get(ENV, "MAX_ITERS", "10000000"))
const COLOR_SEED = parse(Int, get(ENV, "COLOR_SEED", "1"))
const COLOR_LEVELS = parse(Int, get(ENV, "COLOR_LEVELS", "1024"))

@assert N_ALPHA >= 2 "N_ALPHA must be >= 2"
@assert N_LAMBDA >= 2 "N_LAMBDA must be >= 2"
@assert T_END > 0 "T_END must be > 0"
@assert DT > 0 "DT must be > 0"
@assert MAX_SYMBOLS >= 1 "MAX_SYMBOLS must be >= 1"
@assert EPS0 > 0 "EPS0 must be > 0"

# ============================================================
# Shimizu-Morioka dynamics (B fixed at 0)
# ============================================================
@inline function shimizu_morioka(u, p, t)
    alpha, lambda = p
    x, y, z = u
    return @SVector [
        y,
        x - lambda * y - x * z - B * x^3,
        -alpha * (z - x^2)
    ]
end

@inline function unstable_side_initial_condition(lambda::Float64; eps0::Float64=EPS0)
    # Linearization at origin:
    # [0  1  0;
    #  1 -lambda 0;
    #  0  0 -alpha]
    # Unstable eigenvalue in x-y block:
    mu = 0.5 * (-lambda + sqrt(lambda^2 + 4.0))
    v = @SVector [1.0, mu, 0.0]
    return eps0 * (v / norm(v))
end

function integrate_symbols(alpha::Float64, lambda::Float64)
    p = @SVector [alpha, lambda]
    u0 = unstable_side_initial_condition(lambda)
    tspan = (0.0, T_END)

    symbols = UInt8[]
    times = Float64[]
    sizehint!(symbols, MAX_SYMBOLS)
    sizehint!(times, MAX_SYMBOLS)

    function append_symbol!(bit::UInt8, integrator)
        if integrator.t <= 0.0
            return
        end
        if length(symbols) >= MAX_SYMBOLS
            terminate!(integrator)
            return
        end
        push!(symbols, bit)
        push!(times, integrator.t)
        if length(symbols) >= MAX_SYMBOLS
            terminate!(integrator)
        end
    end

    x_condition(u, t, integrator) = u[1]
    y_condition(u, t, integrator) = u[2]

    x_affect!(integrator) = append_symbol!(0x00, integrator)
    y_affect!(integrator) = append_symbol!(0x01, integrator)

    x_cb = ContinuousCallback(
        x_condition,
        x_affect!;
        rootfind=true,
        save_positions=(false, false)
    )
    y_cb = ContinuousCallback(
        y_condition,
        y_affect!;
        rootfind=true,
        save_positions=(false, false)
    )
    cb = CallbackSet(x_cb, y_cb)

    prob = ODEProblem{false}(shimizu_morioka, u0, tspan, p)
    solve(
        prob,
        Vern9();
        callback=cb,
        abstol=ABSTOL,
        reltol=RELTOL,
        adaptive=true,
        dt=DT,
        dtmax=DT,
        maxiters=MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false
    )

    return symbols, times
end

@inline function encode_binary(symbols::AbstractVector{UInt8})
    value = 0.0
    weight = 0.5
    @inbounds for bit in symbols
        if bit == 0x01
            value += weight
        end
        weight *= 0.5
    end
    return value
end

# ============================================================
# Parameter sweep
# ============================================================
alphas = collect(range(ALPHA_MIN, ALPHA_MAX, length=N_ALPHA))
lambdas = collect(range(LAMBDA_MIN, LAMBDA_MAX, length=N_LAMBDA))

# Matrix layout for Makie heatmap:
# x-axis first (alpha), y-axis second (lambda).
encoded = Matrix{Float64}(undef, N_ALPHA, N_LAMBDA)
event_counts = Matrix{Int}(undef, N_ALPHA, N_LAMBDA)
failures = Base.Threads.Atomic{Int}(0)

println("Running Shimizu-Morioka xy-zero sweep:")
println("  alpha range:  [$ALPHA_MIN, $ALPHA_MAX] with $N_ALPHA points")
println("  lambda range: [$LAMBDA_MIN, $LAMBDA_MAX] with $N_LAMBDA points")
println("  B (fixed): $B")
println("  t_end: $T_END, dt(max): $DT")
println("  max symbols: $MAX_SYMBOLS")
println("  threads: $(Threads.nthreads())")
println()

Threads.@threads for j in eachindex(lambdas)
    lambda = lambdas[j]
    for i in eachindex(alphas)
        alpha = alphas[i]
        try
            symbols, _ = integrate_symbols(alpha, lambda)
            encoded[i, j] = encode_binary(symbols)
            event_counts[i, j] = length(symbols)
        catch err
            encoded[i, j] = NaN
            event_counts[i, j] = 0
            Base.Threads.atomic_add!(failures, 1)
            @warn "Failed at alpha=$alpha lambda=$lambda" exception=(err, catch_backtrace())
        end
    end
end

valid_vals = vec(encoded[.!isnan.(encoded)])
if !isempty(valid_vals)
    println("Encoded value range: [$(minimum(valid_vals)), $(maximum(valid_vals))], mean=$(mean(valid_vals))")
end
println("Event count range: [$(minimum(event_counts)), $(maximum(event_counts))], mean=$(mean(event_counts))")
println("Failed integrations: $(failures[])")

# Print one representative sequence at center of the scan.
mid_alpha = alphas[cld(N_ALPHA, 2)]
mid_lambda = lambdas[cld(N_LAMBDA, 2)]
mid_symbols, mid_times = integrate_symbols(mid_alpha, mid_lambda)
mid_bits = join(Int.(mid_symbols), "")
println()
println("Center sample:")
println("  alpha=$mid_alpha lambda=$mid_lambda")
println("  events recorded: $(length(mid_symbols))")
println("  sequence (0=x-cross, 1=y-cross): $mid_bits")
if !isempty(mid_times)
    println("  first event time: $(mid_times[1]), last event time: $(mid_times[end])")
end

# ============================================================
# Plot heatmap
# ============================================================
fig = Figure(size=(960, 760))
ax = Axis(
    fig[1, 1];
    xlabel="alpha",
    ylabel="lambda",
    title="Shimizu-Morioka (B=0): xy zero-crossing binary value"
)

hm = heatmap!(
    ax,
    alphas,
    lambdas,
    encoded;
    colormap=let
        rng = MersenneTwister(COLOR_SEED)
        [RGBf(rand(rng), rand(rng), rand(rng)) for _ in 1:max(16, COLOR_LEVELS)]
    end
)

output_path = joinpath(@__DIR__, "SM_xy_roots_heatmap.png")
save(output_path, fig)
println("Saved heatmap to: $output_path")
