# Shimizu-Morioka GPU kneading sweep
# Uses DiffEqGPU with Metal backend for Apple M-series GPUs

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DiffEqGPU
using OrdinaryDiffEq
using StaticArrays
using Metal
using CairoMakie
using Colors
using SciMLBase

# ============================================================
# Sweep Resolution Configuration
# ============================================================
const N_ALPHA = 2000   # Number of points in alpha dimension
const N_LAMBDA = 2000  # Number of points in lambda dimension

# ============================================================
# Parameter Ranges
# ============================================================
const ALPHA_MIN = 0.15f0
const ALPHA_MAX = 0.65f0
const LAMBDA_MIN = 0.5f0
const LAMBDA_MAX = 0.95f0
const B = 0.0f0

# ============================================================
# Integration and Kneading Configuration
# ============================================================
const MAX_SYMBOLS = 10f0
const MAX_SYMBOLS_INT = 10
const EPSILON = 1f-4
const X_MIN = 1f-3
const DT = 0.005f0
const TSPAN = (0.0f0, 2_000.0f0)

# ============================================================
# Shimizu-Morioka System (augmented with kneading state)
# dx/dt = y
# dy/dt = x - λy - xz - Bx³
# dz/dt = -αz + αx²
# Extra state: (sequence_code, symbol_count)
# ============================================================
function shimizu_morioka_kneading(u, p, t)
    x, y, z = u[1], u[2], u[3]
    α, λ, _ = p
    dx = y
    dy = x - λ * y - x * z - B * x^3
    dz = -α * z + α * x^2
    return SVector{5,Float32}(dx, dy, dz, 0.0f0, 0.0f0)
end

# ============================================================
# Unstable manifold at the origin
# ============================================================
function unstable_direction(λ::Float32)
    μ = (sqrt(λ * λ + 4.0f0) - λ) * 0.5f0
    invn = inv(sqrt(1.0f0 + μ * μ))
    return SVector{3,Float32}(invn, μ * invn, 0.0f0), μ
end

function initial_condition(α::Float32, λ::Float32)
    dir, μ = unstable_direction(λ)
    x0 = EPSILON * dir[1]
    y0 = EPSILON * dir[2]
    b = α / (2.0f0 * μ + α)
    z0 = b * x0 * x0
    return SVector{5,Float32}(
        x0,
        y0,
        z0,
        0.0f0,
        0.0f0
    )
end

# ============================================================
# Callback: record symbols at |x(t)| local maxima
# Symbol = 1 if x > 0, 0 if x < 0 (at the |x| peak)
# ============================================================
function y_sign_change_condition(u, t, integrator)
    y_prev = integrator.uprev[2]
    y_curr = u[2]
    return (y_prev <= 0.0f0 && y_curr > 0.0f0) ||
           (y_prev >= 0.0f0 && y_curr < 0.0f0)
end

function record_symbol!(integrator)
    u_prev = integrator.uprev
    u = integrator.u
    seq_code = u[4]
    count = u[5]

    if count >= MAX_SYMBOLS
        return
    end

    y_prev = u_prev[2]
    y_curr = u[2]
    denom = y_prev - y_curr
    θ = denom == 0.0f0 ? 0.0f0 : y_prev / denom
    if θ < 0.0f0
        θ = 0.0f0
    elseif θ > 1.0f0
        θ = 1.0f0
    end

    x_cross = u_prev[1] + θ * (u[1] - u_prev[1])
    z_cross = u_prev[3] + θ * (u[3] - u_prev[3])
    if abs(x_cross) <= X_MIN
        return
    end

    ydot = x_cross * (1.0f0 - z_cross) - B * x_cross^3
    is_max = (x_cross > 0.0f0 && ydot < 0.0f0) || (x_cross < 0.0f0 && ydot > 0.0f0)
    if is_max
        symbol = x_cross > 0.0f0 ? 1.0f0 : 0.0f0
        seq_code = seq_code * 2.0f0 + symbol
        count += 1.0f0
        integrator.u = SVector{5,Float32}(u[1], u[2], u[3], seq_code, count)
    end
end

const kneading_cb = DiscreteCallback(
    y_sign_change_condition,
    record_symbol!;
    save_positions=(false, false)
)

# ============================================================
# Generate Parameter Grid
# ============================================================
alphas = range(ALPHA_MIN, ALPHA_MAX, length=N_ALPHA)
lambdas = range(LAMBDA_MIN, LAMBDA_MAX, length=N_LAMBDA)

params = [
    @SVector [Float32(α), Float32(λ), B]
    for α in alphas, λ in lambdas
]
u0s = [initial_condition(Float32(α), Float32(λ)) for α in alphas, λ in lambdas]

params_flat = vec(params)
u0s_flat = vec(u0s)
trajectories = length(params_flat)

println("Running Shimizu-Morioka kneading sweep:")
println("  Alpha range: [$ALPHA_MIN, $ALPHA_MAX] with $N_ALPHA points")
println("  Lambda range: [$LAMBDA_MIN, $LAMBDA_MAX] with $N_LAMBDA points")
println("  Total trajectories: $trajectories")
println("  Time span: $TSPAN")
println("  Max symbols: $(MAX_SYMBOLS_INT)")
println()

# ============================================================
# Set Up Ensemble Problem
# ============================================================
prob = ODEProblem{false}(shimizu_morioka_kneading, u0s_flat[1], TSPAN, params_flat[1])
prob_func = (prob, i, repeat) -> remake(prob, u0=u0s_flat[i], p=params_flat[i])
ensemble_prob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)
backend = Metal.MetalBackend()

# ============================================================
# Run GPU Sweep (adaptive)
# ============================================================
println("Compiling and running GPU sweep...")
saveat = Float32[TSPAN[2]]
@time Metal.@sync sol = solve(
    ensemble_prob,
    GPUTsit5(),
    EnsembleGPUKernel(backend);
    trajectories=trajectories,
    adaptive=true,
    dt=DT,
    abstol=1f-7,
    reltol=1f-7,
    callback=kneading_cb,
    merge_callbacks=true,
    saveat=saveat,
    save_everystep=false,
    save_start=false
)

# ============================================================
# Decode sequences and build color map
# ============================================================
function sequence_bits(code::Int, count::Int)
    if count == 0
        return Int[]
    end
    bits = Vector{Int}(undef, count)
    c = code
    for i in count:-1:1
        bits[i] = c & 1
        c >>= 1
    end
    return bits
end

sequence_keys = Vector{Tuple{Int,Int}}(undef, trajectories)
for i in 1:trajectories
    u_final = sol.u[i].u[end]
    seq_code = Int(round(u_final[4]))
    seq_count = Int(round(u_final[5]))
    sequence_keys[i] = (seq_count, seq_code)
end

unique_keys = unique(sequence_keys)
sort!(unique_keys, by=key -> (key[1], key[2]))

key_to_index = Dict{Tuple{Int,Int},Int}()
for (idx, key) in enumerate(unique_keys)
    key_to_index[key] = idx
end

sequence_indices = [key_to_index[key] for key in sequence_keys]
sequence_grid = reshape(sequence_indices, N_ALPHA, N_LAMBDA)

println("Unique kneading sequences: $(length(unique_keys))")
for (idx, key) in enumerate(unique_keys)
    count, code = key
    seq_str = count == 0 ? "none" : join(sequence_bits(code, count))
    println("  $(lpad(idx, 3)) -> $seq_str")
end

# ============================================================
# Plot heatmap
# ============================================================
println("Generating heatmap...")
fig = Figure(size=(900, 700))
ax = Axis(fig[1, 1], xlabel="α", ylabel="λ", title="Kneading sequences (|x| maxima)")

num_sequences = length(unique_keys)
colormap_count = max(2, num_sequences)
colormap = distinguishable_colors(colormap_count)

hm = heatmap!(ax,
    collect(alphas),
    collect(lambdas),
    sequence_grid;
    colormap=colormap,
    colorrange=(1, colormap_count)
)

Colorbar(fig[1, 2], hm, label="Sequence index")

output_path = joinpath(@__DIR__, "sm_kneading_heatmap.png")
save(output_path, fig)
println("Saved heatmap to: $output_path")
