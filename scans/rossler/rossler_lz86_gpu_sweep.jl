# Rössler GPU parameter sweep: normalized Lempel-Ziv (LZ86) complexity
#
# Parameter axes:
#   - x axis: c ∈ [2, 7]
#   - y axis: a ∈ [0.3, 0.55]
#
# Rössler variant with equilibrium at the origin:
#   x' = -y - z
#   y' =  x + a y
#   z' =  b x + z (x - c)
#
# Symbolic dynamics:
#   - Define a "revolution" as an upward crossing of y=0 with x>0 (positive x-axis in xy-plane).
#   - Track whether z exceeded a small threshold since the previous revolution.
#     Emit symbol "1" if yes, otherwise "0".
#
# Output:
#   Heatmap of normalized LZ86 complexity of the binary sequence at each (c, a) grid point.
#
# Run from repo root:
#   julia scans/rossler/rossler_lz86_gpu_sweep.jl
#
# Optional environment overrides:
#   N_C, N_A, T_END, T_TRANSIENT, DT, Z_THRESH, B, MAX_SYMBOLS, SEED

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CairoMakie
using DiffEqGPU
using Metal
using OrdinaryDiffEq
using Random
using SciMLBase
using StaticArrays

# ============================================================
# Sweep configuration
# ============================================================
const N_C = parse(Int, get(ENV, "N_C", "400"))
const N_A = parse(Int, get(ENV, "N_A", "400"))

const C_MIN = 2.0f0
const C_MAX = 7.0f0
const A_MIN = 0.3f0
const A_MAX = 0.55f0

# System parameter b (fixed during sweep)
const B = parse(Float32, get(ENV, "B", "0.2"))

# Integration and symbol configuration
const T_END = parse(Float32, get(ENV, "T_END", "300"))
const T_TRANSIENT = parse(Float32, get(ENV, "T_TRANSIENT", "10"))
const DT = parse(Float32, get(ENV, "DT", "0.01"))
const Z_THRESH = parse(Float32, get(ENV, "Z_THRESH", "0.001"))
const MAX_SYMBOLS = parse(Int, get(ENV, "MAX_SYMBOLS", "100"))

const CHUNK_BITS = 20
const N_CHUNKS = cld(MAX_SYMBOLS, CHUNK_BITS)
const STATE_DIM = 5 + N_CHUNKS # (x,y,z, z_seen_high, count, chunks...)

const IDX_X = 1
const IDX_Y = 2
const IDX_Z = 3
const IDX_ZFLAG = 4
const IDX_COUNT = 5
const IDX_CHUNK0 = 6

const MAX_SYMBOLS_F = Float32(MAX_SYMBOLS)
const CHUNK_BITS_F = Float32(CHUNK_BITS)

if N_CHUNKS > 5
    error("MAX_SYMBOLS too large for current chunk packing; set MAX_SYMBOLS <= 100.")
end

@inline function pack_state(x, y, z, zflag, count, c1, c2, c3, c4, c5)
    if N_CHUNKS == 1
        return SVector{STATE_DIM,Float32}(x, y, z, zflag, count, c1)
    elseif N_CHUNKS == 2
        return SVector{STATE_DIM,Float32}(x, y, z, zflag, count, c1, c2)
    elseif N_CHUNKS == 3
        return SVector{STATE_DIM,Float32}(x, y, z, zflag, count, c1, c2, c3)
    elseif N_CHUNKS == 4
        return SVector{STATE_DIM,Float32}(x, y, z, zflag, count, c1, c2, c3, c4)
    else
        return SVector{STATE_DIM,Float32}(x, y, z, zflag, count, c1, c2, c3, c4, c5)
    end
end

# ============================================================
# Rössler system (augmented state for symbolic dynamics)
# ============================================================
function rossler_augmented(u, p, t)
    x, y, z = u[IDX_X], u[IDX_Y], u[IDX_Z]
    a, b, c = p

    dx = -y - z
    dy = x + a * y
    dz = b * x + z * (x - c)

    return pack_state(dx, dy, dz, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

# ============================================================
# Initial conditions (random per trajectory)
# ============================================================
function random_initial_condition(rng::AbstractRNG)
    x0 = 2.0f0 * rand(rng, Float32) - 1.0f0
    y0 = 2.0f0 * rand(rng, Float32) - 1.0f0
    z0 = 2.0f0 * rand(rng, Float32) - 1.0f0
    return pack_state(x0, y0, z0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
end

# ============================================================
# Callbacks
# ============================================================

# Detect upward crossings of y=0 (candidate "revolution" events).
function y_upcross_condition(u, t, integrator)
    y_prev = integrator.uprev[IDX_Y]
    y_curr = u[IDX_Y]
    return (y_prev <= 0.0f0 && y_curr > 0.0f0)
end

function record_symbol!(integrator)
    if integrator.t < T_TRANSIENT
        return
    end

    u_prev = integrator.uprev
    u = integrator.u

    y_prev = u_prev[IDX_Y]
    y_curr = u[IDX_Y]
    denom = y_prev - y_curr
    θ = denom == 0.0f0 ? 0.0f0 : y_prev / denom
    if θ < 0.0f0
        θ = 0.0f0
    elseif θ > 1.0f0
        θ = 1.0f0
    end

    x_cross = u_prev[IDX_X] + θ * (u[IDX_X] - u_prev[IDX_X])
    if x_cross <= 0.0f0
        return
    end
    z_cross = u_prev[IDX_Z] + θ * (u[IDX_Z] - u_prev[IDX_Z])

    z_seen_high = u[IDX_ZFLAG]
    count = u[IDX_COUNT]

    # Emit symbol and reset z_seen_high regardless of whether we still record.
    symbol = (z_seen_high > 0.5f0 || z_cross > Z_THRESH) ? 1.0f0 : 0.0f0
    z_seen_high = 0.0f0

    c1 = N_CHUNKS >= 1 ? u[IDX_CHUNK0 + 0] : 0.0f0
    c2 = N_CHUNKS >= 2 ? u[IDX_CHUNK0 + 1] : 0.0f0
    c3 = N_CHUNKS >= 3 ? u[IDX_CHUNK0 + 2] : 0.0f0
    c4 = N_CHUNKS >= 4 ? u[IDX_CHUNK0 + 3] : 0.0f0
    c5 = N_CHUNKS >= 5 ? u[IDX_CHUNK0 + 4] : 0.0f0
    if count < MAX_SYMBOLS_F
        if count < CHUNK_BITS_F
            c1 = c1 * 2.0f0 + symbol
        elseif count < 2.0f0 * CHUNK_BITS_F
            c2 = c2 * 2.0f0 + symbol
        elseif count < 3.0f0 * CHUNK_BITS_F
            c3 = c3 * 2.0f0 + symbol
        elseif count < 4.0f0 * CHUNK_BITS_F
            c4 = c4 * 2.0f0 + symbol
        else
            c5 = c5 * 2.0f0 + symbol
        end
        count += 1.0f0
    end

    integrator.u = pack_state(
        u[IDX_X],
        u[IDX_Y],
        u[IDX_Z],
        z_seen_high,
        count,
        c1,
        c2,
        c3,
        c4,
        c5
    )
end

const revolution_cb = DiscreteCallback(
    y_upcross_condition,
    record_symbol!;
    save_positions=(false, false)
)

# Detect upward threshold crossings of z to set the "z_seen_high" flag.
function z_thresh_condition(u, t, integrator)
    if t < T_TRANSIENT
        return false
    end
    z_prev = integrator.uprev[IDX_Z]
    z_curr = u[IDX_Z]
    return (z_prev <= Z_THRESH && z_curr > Z_THRESH)
end

function mark_z_high!(integrator)
    u = integrator.u
    c1 = N_CHUNKS >= 1 ? u[IDX_CHUNK0 + 0] : 0.0f0
    c2 = N_CHUNKS >= 2 ? u[IDX_CHUNK0 + 1] : 0.0f0
    c3 = N_CHUNKS >= 3 ? u[IDX_CHUNK0 + 2] : 0.0f0
    c4 = N_CHUNKS >= 4 ? u[IDX_CHUNK0 + 3] : 0.0f0
    c5 = N_CHUNKS >= 5 ? u[IDX_CHUNK0 + 4] : 0.0f0

    integrator.u = pack_state(
        u[IDX_X],
        u[IDX_Y],
        u[IDX_Z],
        1.0f0,
        u[IDX_COUNT],
        c1,
        c2,
        c3,
        c4,
        c5
    )
end

const z_thresh_cb = DiscreteCallback(
    z_thresh_condition,
    mark_z_high!;
    save_positions=(false, false)
)

const cb = CallbackSet(revolution_cb, z_thresh_cb)

# ============================================================
# Lempel-Ziv (LZ76 / "LZ86") complexity on packed chunks
# ============================================================
@inline function _chunk_len(count::Int, chunk_idx::Int)
    last = count % CHUNK_BITS
    last = last == 0 ? CHUNK_BITS : last
    last_chunk = cld(count, CHUNK_BITS)
    return chunk_idx < last_chunk ? CHUNK_BITS : last
end

@inline function _bit_at(chunks::NTuple{N_CHUNKS,Int}, count::Int, pos::Int)
    chunk_idx = (pos - 1) ÷ CHUNK_BITS + 1
    within = (pos - 1) % CHUNK_BITS
    len = _chunk_len(count, chunk_idx)
    shift = (len - 1 - within)
    return (chunks[chunk_idx] >> shift) & 1
end

function lz_complexity_lz76(chunks::NTuple{N_CHUNKS,Int}, n::Int)
    n <= 0 && return 0
    n == 1 && return 1

    i = 1
    k = 1
    l = 1
    c = 1
    k_max = 1

    while true
        if _bit_at(chunks, n, i + k - 1) == _bit_at(chunks, n, l + k - 1)
            k += 1
            if l + k > n
                c += 1
                break
            end
        else
            if k > k_max
                k_max = k
            end
            i += 1
            if i == l
                c += 1
                l += k_max
                if l > n
                    break
                end
                i = 1
                k = 1
                k_max = 1
            else
                k = 1
            end
        end
    end

    return c
end

function normalized_lz86_complexity(chunks::NTuple{N_CHUNKS,Int}, n::Int)
    n <= 1 && return 0.0f0
    c = lz_complexity_lz76(chunks, n)
    return Float32(c * log2(float(n)) / n)
end

# ============================================================
# Run sweep
# ============================================================
cs = range(C_MIN, C_MAX, length=N_C)
as = range(A_MIN, A_MAX, length=N_A)

seed = parse(Int, get(ENV, "SEED", "1"))
rng = MersenneTwister(seed)

params = [
    @SVector [Float32(a), Float32(B), Float32(c)]
    for c in cs, a in as
]
u0s = [random_initial_condition(rng) for _ in 1:length(params)]

params_flat = vec(params)
u0s_flat = vec(reshape(u0s, size(params)))
trajectories = length(params_flat)

println("Running Rössler LZ86 sweep:")
println("  c range: [$C_MIN, $C_MAX] with $N_C points")
println("  a range: [$A_MIN, $A_MAX] with $N_A points")
println("  b (fixed): $B")
println("  Total trajectories: $trajectories")
println("  Time span: (0, $T_END), transient: $T_TRANSIENT")
println("  z threshold: $Z_THRESH")
println("  max symbols: $MAX_SYMBOLS (packed into $N_CHUNKS chunks)")
println("  seed: $seed")
println()

prob = ODEProblem{false}(rossler_augmented, u0s_flat[1], (0.0f0, T_END), params_flat[1])
prob_func = (prob, i, repeat) -> remake(prob, u0=u0s_flat[i], p=params_flat[i])
ensemble_prob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)
backend = Metal.MetalBackend()

println("Compiling and running GPU sweep...")
saveat = Float32[T_END]
Metal.@sync sol = solve(
    ensemble_prob,
    GPUTsit5(),
    EnsembleGPUKernel(backend);
    trajectories=trajectories,
    adaptive=true,
    dt=DT,
    abstol=1f-6,
    reltol=1f-6,
    callback=cb,
    merge_callbacks=true,
    saveat=saveat,
    save_everystep=false,
    save_start=false
)

println("Computing normalized LZ86 complexity on CPU...")
complexities = Vector{Float32}(undef, trajectories)
Threads.@threads for i in 1:trajectories
    u_final = sol.u[i].u[end]
    n = Int(round(u_final[IDX_COUNT]))
    n = min(n, MAX_SYMBOLS)

    chunks = ntuple(j -> Int(round(u_final[IDX_CHUNK0 + j - 1])), N_CHUNKS)
    complexities[i] = normalized_lz86_complexity(chunks, n)
end

complexity_grid = reshape(complexities, N_C, N_A)

println("Generating heatmap...")
fig = Figure(size=(900, 700))
ax = Axis(
    fig[1, 1];
    xlabel="c",
    ylabel="a",
    title="Rössler (origin equilibrium): normalized LZ86 complexity"
)

hm = heatmap!(ax, collect(cs), collect(as), complexity_grid; colormap=:viridis)
Colorbar(fig[1, 2], hm, label="C_LZ (normalized)")

output_path = joinpath(@__DIR__, "rossler_lz86_complexity_heatmap.png")
save(output_path, fig)
println("Saved heatmap to: $output_path")
