# Rössler CPU parameter sweep: power-series value (x=1) of the binary sequence
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
#   - Revolutions are y=0 downward crossings at x<0 (negative x-axis, 180° shift).
#   - Spikes are z upcrossings above the threshold.
#   - Encode spikes as "1" and revolutions as "0" in one combined sequence.
#
# Output:
#   Heatmap of power-series values (x=1) of the binary sequence, mapped to random colors.
#
# Run from repo root:
#   julia scans/rossler/rossler_lz86_cpu_sweep.jl
#
# Optional environment overrides:
#   N_C, N_A, T_END, T_TRANSIENT, DT, Z_THRESH, B, MAX_SYMBOLS, SEED, ADAPTIVE, MAX_STATE, COLOR_SEED

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using CairoMakie
using OrdinaryDiffEq
using Random
using SciMLBase
using StaticArrays
using Statistics

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
const T_END = parse(Float32, get(ENV, "T_END", "2000"))
const T_TRANSIENT = parse(Float32, get(ENV, "T_TRANSIENT", "10"))
const DT = parse(Float32, get(ENV, "DT", "0.005"))
const Z_THRESH = parse(Float32, get(ENV, "Z_THRESH", "1.0"))
const MAX_SYMBOLS = parse(Int, get(ENV, "MAX_SYMBOLS", "64"))
const ADAPTIVE = parse(Bool, get(ENV, "ADAPTIVE", "false"))
const MAX_STATE = parse(Float32, get(ENV, "MAX_STATE", "1000"))

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

# Detect downward crossings of y=0 (revolution events at 180°).
function y_downcross_condition(u, t, integrator)
    y_prev = integrator.uprev[IDX_Y]
    y_curr = u[IDX_Y]
    return (y_prev >= 0.0f0 && y_curr < 0.0f0)
end

function append_symbol!(integrator, symbol::Float32)
    u = integrator.u
    count = u[IDX_COUNT]
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
        if count >= MAX_SYMBOLS_F
            terminate!(integrator)
        end
    end

    integrator.u = pack_state(
        u[IDX_X],
        u[IDX_Y],
        u[IDX_Z],
        u[IDX_ZFLAG],
        count,
        c1,
        c2,
        c3,
        c4,
        c5
    )
end

function record_revolution!(integrator)
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
    if x_cross >= 0.0f0
        return
    end

    append_symbol!(integrator, 0.0f0)
end

const revolution_cb = DiscreteCallback(
    y_downcross_condition,
    record_revolution!;
    save_positions=(false, false)
)

# Detect upward threshold crossings of z (spikes).
function z_upcross_condition(u, t, integrator)
    if t < T_TRANSIENT
        return false
    end
    z_prev = integrator.uprev[IDX_Z]
    z_curr = u[IDX_Z]
    return (z_prev <= Z_THRESH && z_curr > Z_THRESH)
end

function record_spike!(integrator)
    append_symbol!(integrator, 1.0f0)
end

const spike_cb = DiscreteCallback(
    z_upcross_condition,
    record_spike!;
    save_positions=(false, false)
)

function blowup_condition(u, t, integrator)
    return (abs(u[IDX_X]) > MAX_STATE ||
            abs(u[IDX_Y]) > MAX_STATE ||
            abs(u[IDX_Z]) > MAX_STATE)
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

function sequence_bits(chunks::NTuple{N_CHUNKS,Int}, n::Int)
    n <= 0 && return ""
    bits = Vector{Char}(undef, n)
    for i in 1:n
        bits[i] = _bit_at(chunks, n, i) == 1 ? '1' : '0'
    end
    return String(bits)
end

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

println("Running Rössler power-series CPU sweep:")
println("  c range: [$C_MIN, $C_MAX] with $N_C points")
println("  a range: [$A_MIN, $A_MAX] with $N_A points")
println("  b (fixed): $B")
println("  Total trajectories: $trajectories")
println("  Time span: (0, $T_END), transient: $T_TRANSIENT")
println("  z threshold: $Z_THRESH")
println("  max symbols: $MAX_SYMBOLS (packed into $N_CHUNKS chunks)")
println("  seed: $seed")
println("  threads: $(Threads.nthreads())")
println("  adaptive: $ADAPTIVE")
println("  max state: $MAX_STATE")
println()

prob = ODEProblem{false}(rossler_augmented, u0s_flat[1], (0.0f0, T_END), params_flat[1])
prob_func = (prob, i, repeat) -> remake(prob, u0=u0s_flat[i], p=params_flat[i])
ensemble_prob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)

println("Running CPU sweep with threads...")
saveat = Float32[T_END]
sol = solve(
    ensemble_prob,
    Tsit5(),
    EnsembleThreads();
    trajectories=trajectories,
    adaptive=ADAPTIVE,
    dt=DT,
    abstol=1f-6,
    reltol=1f-6,
    callback=cb,
    merge_callbacks=true,
    saveat=saveat,
    save_everystep=false,
    save_start=false
)

println("Computing power-series values on CPU...")
series_values = Vector{Int}(undef, trajectories)
ones_counts = Vector{Int}(undef, trajectories)
counts = Vector{Int}(undef, trajectories)
transitions = Vector{Int}(undef, trajectories)
Threads.@threads for i in 1:trajectories
    u_final = sol.u[i].u[end]
    n = Int(round(u_final[IDX_COUNT]))
    n = min(n, MAX_SYMBOLS)

    chunks = ntuple(j -> Int(round(u_final[IDX_CHUNK0 + j - 1])), N_CHUNKS)
    series_values[i] = 0
    counts[i] = n
    ones = 0
    for j in 1:N_CHUNKS
        ones += count_ones(chunks[j])
    end
    ones_counts[i] = ones
    series_values[i] = ones
    if n <= 1
        transitions[i] = 0
    else
        prev = _bit_at(chunks, n, 1)
        tcount = 0
        for pos in 2:n
            b = _bit_at(chunks, n, pos)
            if b != prev
                tcount += 1
                prev = b
            end
        end
        transitions[i] = tcount
    end
end

println("Symbol counts: min=$(minimum(counts)) max=$(maximum(counts)) mean=$(mean(counts))")
println("Ones counts: min=$(minimum(ones_counts)) max=$(maximum(ones_counts)) mean=$(mean(ones_counts))")
println("Transitions: min=$(minimum(transitions)) max=$(maximum(transitions)) mean=$(mean(transitions))")

debug_samples = parse(Int, get(ENV, "DEBUG_SAMPLES", "0"))
if debug_samples > 0
    println("Sample sequences:")
    for i in 1:min(debug_samples, trajectories)
        u_final = sol.u[i].u[end]
        n = min(Int(round(u_final[IDX_COUNT])), MAX_SYMBOLS)
        chunks = ntuple(j -> Int(round(u_final[IDX_CHUNK0 + j - 1])), N_CHUNKS)
        println("  [$i] n=$n ones=$(ones_counts[i]) bits=$(sequence_bits(chunks, n))")
    end
end

if trajectories > 0
    idx_max = argmax(transitions)
    idx_min = argmin(transitions)
    for (label, idx) in [("max transitions", idx_max), ("min transitions", idx_min)]
        u_final = sol.u[idx].u[end]
        n = min(Int(round(u_final[IDX_COUNT])), MAX_SYMBOLS)
        chunks = ntuple(j -> Int(round(u_final[IDX_CHUNK0 + j - 1])), N_CHUNKS)
        println("Sequence with $label:")
        println("  idx=$idx n=$n ones=$(ones_counts[idx]) transitions=$(transitions[idx]) bits=$(sequence_bits(chunks, n))")
    end
end

unique_vals = sort(unique(series_values))
color_seed = parse(Int, get(ENV, "COLOR_SEED", "1"))
rng = MersenneTwister(color_seed)
colormap = [RGBf0(rand(rng), rand(rng), rand(rng)) for _ in 1:length(unique_vals)]
val_to_index = Dict(val => idx for (idx, val) in enumerate(unique_vals))
index_grid = reshape([val_to_index[v] for v in series_values], N_C, N_A)

println("Power-series values: min=$(minimum(series_values)) max=$(maximum(series_values)) unique=$(length(unique_vals))")
println("Color seed: $color_seed")

println("Generating heatmap...")
fig = Figure(size=(900, 700))
ax = Axis(
    fig[1, 1];
    xlabel="c",
    ylabel="a",
    title="Rössler (origin equilibrium): power-series value (x=1) (CPU sweep)"
)

hm = heatmap!(
    ax,
    collect(cs),
    collect(as),
    index_grid;
    colormap=colormap,
    colorrange=(1, length(unique_vals))
)
Colorbar(fig[1, 2], hm, label="Power-series value (sum of bits)")

output_path = joinpath(@__DIR__, "rossler_power_series_heatmap_cpu.png")
save(output_path, fig)
println("Saved heatmap to: $output_path")
