using Pkg

const ATTEMPT33_ROOT = @__DIR__
const REPO_ROOT_033 = normpath(joinpath(ATTEMPT33_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_033)

using DifferentialEquations
using LinearAlgebra
using Printf
using StaticArrays

const ATTEMPT033_N_ALPHA = parse(Int, get(ENV, "ATTEMPT033_N_ALPHA", "500"))
const ATTEMPT033_N_LAMBDA = parse(Int, get(ENV, "ATTEMPT033_N_LAMBDA", "500"))
const ATTEMPT033_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT033_ALPHA_MIN", "0.0"))
const ATTEMPT033_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT033_ALPHA_MAX", "0.7"))
const ATTEMPT033_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT033_LAMBDA_MIN", "0.2"))
const ATTEMPT033_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT033_LAMBDA_MAX", "1.6"))
const ATTEMPT033_MAX_EVENT_ITERATES = parse(Int, get(ENV, "ATTEMPT033_MAX_EVENT_ITERATES", "16"))

const ATTEMPT033_T_END = parse(Float64, get(ENV, "ATTEMPT033_T_END", "200.0"))
const ATTEMPT033_DT = parse(Float64, get(ENV, "ATTEMPT033_DT", "0.02"))
const ATTEMPT033_ABSTOL = parse(Float64, get(ENV, "ATTEMPT033_ABSTOL", "1e-9"))
const ATTEMPT033_RELTOL = parse(Float64, get(ENV, "ATTEMPT033_RELTOL", "1e-9"))
const ATTEMPT033_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT033_MAX_ITERS", "40000000"))
const ATTEMPT033_EPS0 = parse(Float64, get(ENV, "ATTEMPT033_EPS0", "1e-7"))
const ATTEMPT033_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT033_MAX_STATE", "200.0"))
const ATTEMPT033_TANGENT_EPS = parse(Float64, get(ENV, "ATTEMPT033_TANGENT_EPS", "1e-12"))
const ATTEMPT033_MIN_EVENT_TIME = parse(Float64, get(ENV, "ATTEMPT033_MIN_EVENT_TIME", "0.5"))
const ATTEMPT033_MIN_SECTION_X = parse(Float64, get(ENV, "ATTEMPT033_MIN_SECTION_X", "1e-5"))
const ATTEMPT033_EVENT_Z_THRESHOLD = parse(Float64, get(ENV, "ATTEMPT033_EVENT_Z_THRESHOLD", "1.0"))

const ATTEMPT033_SAMPLE_T_END = parse(Float64, get(ENV, "ATTEMPT033_SAMPLE_T_END", "5000.0"))
const ATTEMPT033_SAMPLE_DT = parse(Float64, get(ENV, "ATTEMPT033_SAMPLE_DT", "0.02"))
const ATTEMPT033_SAMPLE_ABSTOL = parse(Float64, get(ENV, "ATTEMPT033_SAMPLE_ABSTOL", "1e-10"))
const ATTEMPT033_SAMPLE_RELTOL = parse(Float64, get(ENV, "ATTEMPT033_SAMPLE_RELTOL", "1e-10"))
const ATTEMPT033_SAMPLE_MAXITERS = parse(Int, get(ENV, "ATTEMPT033_SAMPLE_MAXITERS", "80000000"))
const ATTEMPT033_MAX_SAMPLE_EVENTS = parse(Int, get(ENV, "ATTEMPT033_MAX_SAMPLE_EVENTS", "3000"))
const ATTEMPT033_TRANSIENT_EVENTS = parse(Int, get(ENV, "ATTEMPT033_TRANSIENT_EVENTS", "150"))
const ATTEMPT033_RESEED_PERIOD = parse(Int, get(ENV, "ATTEMPT033_RESEED_PERIOD", "100"))

const ATTEMPT033_LOCAL_X_WINDOW = parse(Float64, get(ENV, "ATTEMPT033_LOCAL_X_WINDOW", "0.08"))
const ATTEMPT033_LOCAL_GRID_POINTS = parse(Int, get(ENV, "ATTEMPT033_LOCAL_GRID_POINTS", "9"))
const ATTEMPT033_FULL_X_WINDOW = parse(Float64, get(ENV, "ATTEMPT033_FULL_X_WINDOW", "0.8"))
const ATTEMPT033_FULL_X_GRID_POINTS = parse(Int, get(ENV, "ATTEMPT033_FULL_X_GRID_POINTS", "81"))
const ATTEMPT033_FULL_RESEED_Z_TRIES = parse(Int, get(ENV, "ATTEMPT033_FULL_RESEED_Z_TRIES", "4"))
const ATTEMPT033_GOLDEN_TOL = parse(Float64, get(ENV, "ATTEMPT033_GOLDEN_TOL", "1e-6"))
const ATTEMPT033_GOLDEN_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT033_GOLDEN_MAX_ITERS", "24"))
const ATTEMPT033_CRITICAL_SLOPE_EPS = parse(Float64, get(ENV, "ATTEMPT033_CRITICAL_SLOPE_EPS", "1e-4"))
const ATTEMPT033_SEED_Z_MIN = parse(Float64, get(ENV, "ATTEMPT033_SEED_Z_MIN", "-1.5"))
const ATTEMPT033_SEED_Z_MAX = parse(Float64, get(ENV, "ATTEMPT033_SEED_Z_MAX", "0.0"))
const ATTEMPT033_SEED_Z_SAMPLES = parse(Int, get(ENV, "ATTEMPT033_SEED_Z_SAMPLES", "61"))
const ATTEMPT033_RESEED_MIN_X_HALF_WIDTH = parse(Float64, get(ENV, "ATTEMPT033_RESEED_MIN_X_HALF_WIDTH", "0.05"))

const ALPHAS_033 = collect(range(ATTEMPT033_ALPHA_MIN, ATTEMPT033_ALPHA_MAX, length=ATTEMPT033_N_ALPHA))
const LAMBDAS_033 = collect(range(ATTEMPT033_LAMBDA_MIN, ATTEMPT033_LAMBDA_MAX, length=ATTEMPT033_N_LAMBDA))
const TANGENT_INITIAL_033 = @SVector [0.0, 0.0, 1.0]
const INVALID_STATE_033 = SVector{3, Float64}(NaN, NaN, NaN)

struct CriticalSeed33
    state::SVector{3, Float64}
    kind::Symbol
end

struct BranchPoint33
    source_index::Int
    event_time::Float64
    x::Float64
    z::Float64
    next_r::Float64
end

struct SeedFamilyPoint33
    z0::Float64
    current_time::Float64
    current_state::SVector{3, Float64}
    next_r::Float64
end

struct SMCriticalContourResult33
    alpha::Float64
    lambda::Float64
    critical_state::SVector{3, Float64}
    critical_kind::String
    critical_source::String
    absxmax_count::Int
    absxmax_dot_values::Vector{Float64}
    status::String
end

@inline function shimizu_morioka_vector_033(u::SVector{3, Float64}, p::SVector{2, Float64})
    alpha, lambda = p
    x, y, z = u
    return @SVector [y, x - lambda * y - x * z, -alpha * (z - x * x)]
end

@inline function shimizu_morioka!(du, u, p, t)
    alpha, lambda = p
    x, y, z = u
    du[1] = y
    du[2] = x - lambda * y - x * z
    du[3] = -alpha * (z - x * x)
    return nothing
end

@inline function shimizu_morioka_tangent!(du, u, p, t)
    alpha, lambda = p

    x = u[1]
    y = u[2]
    z = u[3]
    vx = u[4]
    vy = u[5]
    vz = u[6]

    du[1] = y
    du[2] = x - lambda * y - x * z
    du[3] = -alpha * (z - x * x)

    du[4] = vy
    du[5] = (1.0 - z) * vx - lambda * vy - x * vz
    du[6] = 2.0 * alpha * x * vx - alpha * vz
    return nothing
end

@inline function unstable_side_initial_condition_033(lambda::Float64; eps0::Float64=ATTEMPT033_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / norm(vec))
end

@inline absxmax_proxy_033(u::SVector{3, Float64}) = u[1] * u[2]
@inline absx_vertex_value_033(u::SVector{3, Float64}) = u[1] * u[1]
@inline signed_absx_component_033(u::SVector{3, Float64}, v::SVector{3, Float64}) = (u[1] >= 0.0 ? 1.0 : -1.0) * v[1]

function quadratic_vertex_time_033(
    t0::Float64,
    z0::Float64,
    t1::Float64,
    z1::Float64,
    t2::Float64,
    z2::Float64,
)
    denom = (t0 - t1) * (t0 - t2) * (t1 - t2)
    iszero(denom) && return nothing
    a = (t2 * (z1 - z0) + t1 * (z0 - z2) + t0 * (z2 - z1)) / denom
    b = (t2^2 * (z0 - z1) + t1^2 * (z2 - z0) + t0^2 * (z1 - z2)) / denom
    abs(a) <= 1e-14 && return nothing
    t_vertex = -b / (2a)
    return min(t0, t2) <= t_vertex <= max(t0, t2) ? t_vertex : nothing
end

@inline function normalize_tangent_033(v::SVector{3, Float64})
    tangent_norm = norm(v)
    tangent_norm > ATTEMPT033_TANGENT_EPS || return nothing
    return v / tangent_norm
end

@inline function orthonormalize_against_flow_033(
    v::SVector{3, Float64},
    u::SVector{3, Float64},
    p::SVector{2, Float64},
)
    all(isfinite, v) || return nothing
    flow = shimizu_morioka_vector_033(u, p)
    all(isfinite, flow) || return nothing

    flow_norm_sq = dot(flow, flow)
    if flow_norm_sq > ATTEMPT033_TANGENT_EPS^2
        v = v - (dot(v, flow) / flow_norm_sq) * flow
    end
    return normalize_tangent_033(v)
end

@inline crossing_theta_033(prev_value::Float64, curr_value::Float64) =
    iszero(curr_value - prev_value) ? 0.5 : clamp(-prev_value / (curr_value - prev_value), 0.0, 1.0)

@inline function interpolate_state_033(
    u_prev::SVector{3, Float64},
    u_curr::SVector{3, Float64},
    theta::Float64,
)
    return (1.0 - theta) * u_prev + theta * u_curr
end

@inline function interpolate_tangent_033(
    v_prev::SVector{3, Float64},
    v_curr::SVector{3, Float64},
    theta::Float64,
)
    return (1.0 - theta) * v_prev + theta * v_curr
end

function collect_absxmax_events_033(alpha::Float64, lambda::Float64)
    p = [alpha, lambda]
    times = Float64[]
    states = SVector{3, Float64}[]
    sizehint!(times, ATTEMPT033_MAX_SAMPLE_EVENTS)
    sizehint!(states, ATTEMPT033_MAX_SAMPLE_EVENTS)

    function condition(u, t, integrator)
        return t <= ATTEMPT033_MIN_EVENT_TIME ? 1.0 : u[2]
    end

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        if abs(x) > ATTEMPT033_MIN_SECTION_X && z > ATTEMPT033_EVENT_Z_THRESHOLD
            push!(times, integrator.t)
            push!(states, SVector{3, Float64}(integrator.u))
            if length(times) >= ATTEMPT033_MAX_SAMPLE_EVENTS
                terminate!(integrator)
            end
        end
    end

    cb = ContinuousCallback(condition, affect!; rootfind=true, save_positions=(false, false))
    prob = ODEProblem(
        shimizu_morioka!,
        collect(unstable_side_initial_condition_033(lambda)),
        (0.0, ATTEMPT033_SAMPLE_T_END),
        p,
    )
    solve(
        prob,
        Tsit5();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT033_SAMPLE_DT,
        dtmax=ATTEMPT033_SAMPLE_DT,
        abstol=ATTEMPT033_SAMPLE_ABSTOL,
        reltol=ATTEMPT033_SAMPLE_RELTOL,
        maxiters=ATTEMPT033_SAMPLE_MAXITERS,
        save_start=false,
        save_end=false,
        save_everystep=false,
        verbose=false,
    )

    return times, states
end

function collect_first_two_absxmax_hits_033(alpha::Float64, lambda::Float64, orbit_u0::SVector{3, Float64})
    p = [alpha, lambda]
    times = Float64[]
    states = SVector{3, Float64}[]
    sizehint!(times, 2)
    sizehint!(states, 2)
    armed = Ref(false)

    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        if abs(x) > ATTEMPT033_MIN_SECTION_X && z > ATTEMPT033_EVENT_Z_THRESHOLD
            push!(times, integrator.t)
            push!(states, SVector{3, Float64}(integrator.u))
            if length(times) >= 2
                terminate!(integrator)
            end
        end
    end

    arm_cb = PresetTimeCallback([ATTEMPT033_MIN_EVENT_TIME], integrator -> begin
        armed[] = true
    end)
    event_cb = ContinuousCallback(condition, affect!; rootfind=true, save_positions=(false, false))
    cb = CallbackSet(arm_cb, event_cb)

    prob = ODEProblem(shimizu_morioka!, collect(orbit_u0), (0.0, ATTEMPT033_SAMPLE_T_END), p)
    solve(
        prob,
        Tsit5();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT033_SAMPLE_DT,
        dtmax=ATTEMPT033_SAMPLE_DT,
        abstol=ATTEMPT033_SAMPLE_ABSTOL,
        reltol=ATTEMPT033_SAMPLE_RELTOL,
        maxiters=ATTEMPT033_SAMPLE_MAXITERS,
        save_start=false,
        save_end=false,
        save_everystep=false,
        verbose=false,
    )

    return times, states
end

function build_seed_family_033(alpha::Float64, lambda::Float64)
    family = SeedFamilyPoint33[]
    sizehint!(family, ATTEMPT033_SEED_Z_SAMPLES)
    base = unstable_side_initial_condition_033(lambda)
    for z0 in range(ATTEMPT033_SEED_Z_MAX, ATTEMPT033_SEED_Z_MIN, length=ATTEMPT033_SEED_Z_SAMPLES)
        orbit_u0 = base + SVector{3, Float64}(0.0, 0.0, z0)
        times, states = collect_first_two_absxmax_hits_033(alpha, lambda, orbit_u0)
        length(states) >= 2 || continue
        current_state = states[1]
        current_state[1] > ATTEMPT033_MIN_SECTION_X || continue
        push!(family, SeedFamilyPoint33(z0, times[1], current_state, states[2][1]^2))
    end
    return family
end

function build_positive_branch_033(times::Vector{Float64}, states::Vector{SVector{3, Float64}})
    length(states) >= ATTEMPT033_TRANSIENT_EVENTS + 3 || return BranchPoint33[]
    branch = BranchPoint33[]
    sizehint!(branch, length(states))
    for idx in (ATTEMPT033_TRANSIENT_EVENTS + 1):(length(states) - 1)
        current = states[idx]
        next_state = states[idx + 1]
        current[1] > ATTEMPT033_MIN_SECTION_X || continue
        push!(branch, BranchPoint33(idx, times[idx], current[1], current[3], next_state[1]^2))
    end
    sort!(branch, by=point -> point.x)
    return branch
end

function first_smooth_critical_candidate_033(branch::Vector{BranchPoint33})
    length(branch) >= 3 || return nothing
    for idx in 2:(length(branch) - 1)
        left = branch[idx - 1].next_r
        mid = branch[idx].next_r
        right = branch[idx + 1].next_r
        rise = mid - left
        fall = right - mid
        if rise > ATTEMPT033_CRITICAL_SLOPE_EPS && fall < -ATTEMPT033_CRITICAL_SLOPE_EPS
            return idx, :maximum
        elseif rise < -ATTEMPT033_CRITICAL_SLOPE_EPS && fall > ATTEMPT033_CRITICAL_SLOPE_EPS
            return idx, :minimum
        end
    end
    return nothing
end

function first_smooth_seed_candidate_033(family::Vector{SeedFamilyPoint33})
    length(family) >= 3 || return nothing
    for idx in 2:(length(family) - 1)
        left = family[idx - 1].next_r
        mid = family[idx].next_r
        right = family[idx + 1].next_r
        rise = mid - left
        fall = right - mid
        if rise > ATTEMPT033_CRITICAL_SLOPE_EPS && fall < -ATTEMPT033_CRITICAL_SLOPE_EPS
            return idx, :maximum
        elseif rise < -ATTEMPT033_CRITICAL_SLOPE_EPS && fall > ATTEMPT033_CRITICAL_SLOPE_EPS
            return idx, :minimum
        end
    end
    return nothing
end

function next_absx_return_event_033(alpha::Float64, lambda::Float64, state::SVector{3, Float64})
    all(isfinite, state) || return nothing
    maximum(abs, state) <= ATTEMPT033_MAX_STATE || return nothing

    p = [alpha, lambda]
    hit_state = Ref(INVALID_STATE_033)
    hit_time = Ref(NaN)
    armed = Ref(false)

    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        if abs(x) > ATTEMPT033_MIN_SECTION_X && z > ATTEMPT033_EVENT_Z_THRESHOLD
            hit_state[] = SVector{3, Float64}(integrator.u)
            hit_time[] = integrator.t
            terminate!(integrator)
        end
    end

    arm_cb = PresetTimeCallback([ATTEMPT033_MIN_EVENT_TIME], integrator -> begin
        armed[] = true
    end)
    event_cb = ContinuousCallback(condition, affect!; rootfind=true, save_positions=(false, false))
    cb = CallbackSet(arm_cb, event_cb)

    prob = ODEProblem(shimizu_morioka!, collect(state), (0.0, ATTEMPT033_T_END), p)
    sol = solve(
        prob,
        Tsit5();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT033_DT,
        dtmax=ATTEMPT033_DT,
        abstol=ATTEMPT033_ABSTOL,
        reltol=ATTEMPT033_RELTOL,
        maxiters=ATTEMPT033_MAX_ITERS,
        save_start=false,
        save_end=false,
        save_everystep=false,
        verbose=false,
    )

    if sol.retcode in (ReturnCode.Success, ReturnCode.Terminated) && isfinite(hit_time[])
        return hit_time[], hit_state[]
    end
    return nothing
end

@inline objective_transform_033(value::Float64, kind::Symbol) = kind === :minimum ? value : -value

function objective_value_033(alpha::Float64, lambda::Float64, x::Float64, z_fixed::Float64, kind::Symbol)
    if !isfinite(x) || !isfinite(z_fixed) || x <= ATTEMPT033_MIN_SECTION_X
        return Inf, NaN
    end
    hit = next_absx_return_event_033(alpha, lambda, SVector{3, Float64}(x, 0.0, z_fixed))
    isnothing(hit) && return Inf, NaN
    _, hit_state = hit
    next_r = hit_state[1]^2
    return objective_transform_033(next_r, kind), next_r
end

function golden_section_refine_033(
    alpha::Float64,
    lambda::Float64,
    z_fixed::Float64,
    a::Float64,
    b::Float64,
    kind::Symbol,
)
    left = max(ATTEMPT033_MIN_SECTION_X, min(a, b))
    right = max(a, b)
    right > left || return nothing

    golden_ratio = (sqrt(5.0) - 1.0) / 2.0
    c = right - golden_ratio * (right - left)
    d = left + golden_ratio * (right - left)
    fc, rc = objective_value_033(alpha, lambda, c, z_fixed, kind)
    fd, rd = objective_value_033(alpha, lambda, d, z_fixed, kind)

    best_x = isfinite(fc) && (!isfinite(fd) || fc <= fd) ? c : d
    best_value = isfinite(fc) && (!isfinite(fd) || fc <= fd) ? rc : rd
    isfinite(best_value) || return nothing

    for _ in 1:ATTEMPT033_GOLDEN_MAX_ITERS
        abs(right - left) <= ATTEMPT033_GOLDEN_TOL && break
        if fc <= fd
            right = d
            d = c
            fd = fc
            rd = rc
            c = right - golden_ratio * (right - left)
            fc, rc = objective_value_033(alpha, lambda, c, z_fixed, kind)
        else
            left = c
            c = d
            fc = fd
            rc = rd
            d = left + golden_ratio * (right - left)
            fd, rd = objective_value_033(alpha, lambda, d, z_fixed, kind)
        end
        if isfinite(fc) && fc <= objective_transform_033(best_value, kind)
            best_x = c
            best_value = rc
        end
        if isfinite(fd) && fd <= objective_transform_033(best_value, kind)
            best_x = d
            best_value = rd
        end
    end

    return isfinite(best_value) ? (best_x, best_value) : nothing
end

function local_refinement_bracket_033(
    alpha::Float64,
    lambda::Float64,
    seed::CriticalSeed33,
)
    center_x = seed.state[1]
    z_fixed = seed.state[3]
    left = max(ATTEMPT033_MIN_SECTION_X, center_x - ATTEMPT033_LOCAL_X_WINDOW)
    right = center_x + ATTEMPT033_LOCAL_X_WINDOW
    right > left || return nothing

    xs = collect(range(left, right, length=ATTEMPT033_LOCAL_GRID_POINTS))
    values = fill(NaN, length(xs))
    for idx in eachindex(xs)
        _, raw_value = objective_value_033(alpha, lambda, xs[idx], z_fixed, seed.kind)
        values[idx] = raw_value
    end

    candidates = Int[]
    for idx in 2:(length(xs) - 1)
        all(isfinite, (values[idx - 1], values[idx], values[idx + 1])) || continue
        if seed.kind === :minimum
            values[idx] <= values[idx - 1] && values[idx] <= values[idx + 1] && push!(candidates, idx)
        else
            values[idx] >= values[idx - 1] && values[idx] >= values[idx + 1] && push!(candidates, idx)
        end
    end
    isempty(candidates) && return nothing

    best_idx = candidates[1]
    best_distance = abs(xs[best_idx] - center_x)
    for idx in candidates[2:end]
        distance = abs(xs[idx] - center_x)
        if distance < best_distance
            best_idx = idx
            best_distance = distance
        end
    end
    return xs[best_idx - 1], xs[best_idx + 1]
end

function full_refinement_bracket_033(alpha::Float64, lambda::Float64, z_fixed::Float64, center_x::Float64)
    left = max(ATTEMPT033_MIN_SECTION_X, center_x - ATTEMPT033_FULL_X_WINDOW)
    right = center_x + ATTEMPT033_FULL_X_WINDOW
    right > left || return nothing

    xs = collect(range(left, right, length=ATTEMPT033_FULL_X_GRID_POINTS))
    values = fill(NaN, length(xs))
    for idx in eachindex(xs)
        _, raw_value = objective_value_033(alpha, lambda, xs[idx], z_fixed, :minimum)
        values[idx] = raw_value
    end

    for idx in 2:(length(xs) - 1)
        all(isfinite, (values[idx - 1], values[idx], values[idx + 1])) || continue
        rise = values[idx] - values[idx - 1]
        fall = values[idx + 1] - values[idx]
        if rise > ATTEMPT033_CRITICAL_SLOPE_EPS && fall < -ATTEMPT033_CRITICAL_SLOPE_EPS
            return xs[idx - 1], xs[idx + 1], :maximum
        elseif rise < -ATTEMPT033_CRITICAL_SLOPE_EPS && fall > ATTEMPT033_CRITICAL_SLOPE_EPS
            return xs[idx - 1], xs[idx + 1], :minimum
        end
    end
    return nothing
end

function full_reseed_critical_seed_033(alpha::Float64, lambda::Float64)
    family = build_seed_family_033(alpha, lambda)
    isempty(family) && return nothing
    family_sorted = sort!(copy(family), by = point -> point.current_state[1])
    for point in Iterators.take(family_sorted, ATTEMPT033_FULL_RESEED_Z_TRIES)
        z_fixed = point.current_state[3]
        bracket = full_refinement_bracket_033(alpha, lambda, z_fixed, point.current_state[1])
        isnothing(bracket) && continue
        bracket_left, bracket_right, kind = bracket
        if bracket_right - bracket_left < 2.0 * ATTEMPT033_RESEED_MIN_X_HALF_WIDTH
            x_mid = point.current_state[1]
            bracket_left = max(ATTEMPT033_MIN_SECTION_X, x_mid - ATTEMPT033_RESEED_MIN_X_HALF_WIDTH)
            bracket_right = x_mid + ATTEMPT033_RESEED_MIN_X_HALF_WIDTH
        end
        refined = golden_section_refine_033(alpha, lambda, z_fixed, bracket_left, bracket_right, kind)
        isnothing(refined) && continue
        x_refined, _ = refined
        return CriticalSeed33(SVector{3, Float64}(x_refined, 0.0, z_fixed), kind), "full"
    end
    return nothing
end

function local_refine_critical_seed_033(alpha::Float64, lambda::Float64, seed::CriticalSeed33)
    bracket = local_refinement_bracket_033(alpha, lambda, seed)
    isnothing(bracket) && return nothing
    left, right = bracket
    refined = golden_section_refine_033(alpha, lambda, seed.state[3], left, right, seed.kind)
    isnothing(refined) && return nothing
    x_refined, _ = refined
    return CriticalSeed33(SVector{3, Float64}(x_refined, 0.0, seed.state[3]), seed.kind), "continued"
end

function critical_seed_for_point_033(
    alpha::Float64,
    lambda::Float64,
    row_idx::Int,
    previous_seed::Union{Nothing, CriticalSeed33},
)
    periodic_reseed = row_idx == 1 || mod(row_idx - 1, ATTEMPT033_RESEED_PERIOD) == 0
    if periodic_reseed || isnothing(previous_seed)
        return full_reseed_critical_seed_033(alpha, lambda)
    end

    local_seed = local_refine_critical_seed_033(alpha, lambda, previous_seed)
    if !isnothing(local_seed)
        return local_seed
    end

    full = full_reseed_critical_seed_033(alpha, lambda)
    if !isnothing(full)
        seed, _ = full
        return seed, "fallback_full"
    end

    return previous_seed, "carried"
end

function scan_absx_orbit_from_state_033(alpha::Float64, lambda::Float64, orbit_u0::SVector{3, Float64})
    p = @SVector [alpha, lambda]
    tangent_u0 = orthonormalize_against_flow_033(TANGENT_INITIAL_033, orbit_u0, p)
    if isnothing(tangent_u0)
        return 0, Float64[], "bad_tangent"
    end

    u0 = [orbit_u0[1], orbit_u0[2], orbit_u0[3], tangent_u0[1], tangent_u0[2], tangent_u0[3]]
    prob = ODEProblem(shimizu_morioka_tangent!, u0, (0.0, ATTEMPT033_T_END), collect(p))
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT033_DT,
        dtmax=ATTEMPT033_DT,
        abstol=ATTEMPT033_ABSTOL,
        reltol=ATTEMPT033_RELTOL,
        maxiters=ATTEMPT033_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    absxmax_dot_values = Float64[]
    sizehint!(absxmax_dot_values, ATTEMPT033_MAX_EVENT_ITERATES)
    status = "ok"

    u_prev = orbit_u0
    v_prev = tangent_u0
    t_prev = integ.t
    absxmax_prev = absxmax_proxy_033(u_prev)
    have_prevprev = false
    u_prevprev = u_prev
    t_prevprev = t_prev

    while integ.t < ATTEMPT033_T_END && length(absxmax_dot_values) < ATTEMPT033_MAX_EVENT_ITERATES
        step!(integ)

        u_curr = SVector{3, Float64}(integ.u[1:3])
        v_curr_raw = SVector{3, Float64}(integ.u[4:6])
        v_curr = orthonormalize_against_flow_033(v_curr_raw, u_curr, p)
        if isnothing(v_curr)
            status = "bad_tangent"
            break
        end
        integ.u[4] = v_curr[1]
        integ.u[5] = v_curr[2]
        integ.u[6] = v_curr[3]

        t_curr = integ.t
        if !all(isfinite, u_curr) || !all(isfinite, v_curr)
            status = "nonfinite"
            break
        end
        if maximum(abs, u_curr) > ATTEMPT033_MAX_STATE
            status = "blowup"
            break
        end

        absxmax_curr = absxmax_proxy_033(u_curr)
        if t_curr >= ATTEMPT033_MIN_EVENT_TIME && absxmax_prev >= 0.0 && absxmax_curr < 0.0
            theta = crossing_theta_033(absxmax_prev, absxmax_curr)
            if have_prevprev
                t_vertex = quadratic_vertex_time_033(
                    t_prevprev,
                    absx_vertex_value_033(u_prevprev),
                    t_prev,
                    absx_vertex_value_033(u_prev),
                    t_curr,
                    absx_vertex_value_033(u_curr),
                )
                if !isnothing(t_vertex)
                    theta = clamp((t_vertex - t_prev) / (t_curr - t_prev), 0.0, 1.0)
                end
            end

            u_hit = interpolate_state_033(u_prev, u_curr, theta)
            v_hit = orthonormalize_against_flow_033(interpolate_tangent_033(v_prev, v_curr, theta), u_hit, p)
            if isnothing(v_hit)
                status = "bad_tangent"
                break
            end
            push!(absxmax_dot_values, signed_absx_component_033(u_hit, v_hit))
        end

        have_prevprev = true
        u_prevprev = u_prev
        t_prevprev = t_prev
        u_prev = u_curr
        v_prev = v_curr
        t_prev = t_curr
        absxmax_prev = absxmax_curr
    end

    if status == "ok" && length(absxmax_dot_values) < ATTEMPT033_MAX_EVENT_ITERATES
        status = "short"
    end

    return length(absxmax_dot_values), absxmax_dot_values, status
end

function scan_parameter_point_033(
    alpha::Float64,
    lambda::Float64,
    row_idx::Int,
    previous_seed::Union{Nothing, CriticalSeed33},
)
    seed_info = critical_seed_for_point_033(alpha, lambda, row_idx, previous_seed)
    if isnothing(seed_info)
        return SMCriticalContourResult33(
            alpha,
            lambda,
            INVALID_STATE_033,
            "",
            "failed",
            0,
            Float64[],
            "critical_fail",
        ), previous_seed
    end

    seed, source = seed_info
    count, dot_values, status = scan_absx_orbit_from_state_033(alpha, lambda, seed.state)
    result = SMCriticalContourResult33(
        alpha,
        lambda,
        seed.state,
        String(seed.kind),
        source,
        count,
        dot_values,
        status,
    )
    return result, seed
end

join_float_list_033(values::Vector{Float64}) = join((@sprintf("%.9f", v) for v in values), ",")

function join_state_033(value::SVector{3, Float64})
    if !all(isfinite, value)
        return ""
    end
    return @sprintf("%.9f,%.9f,%.9f", value[1], value[2], value[3])
end

function parse_float_list_033(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end

function parse_state_033(field::AbstractString)
    isempty(field) && return INVALID_STATE_033
    coords = split(field, ',')
    length(coords) == 3 || error("Malformed state triple: $(field)")
    return SVector{3, Float64}(parse.(Float64, coords))
end
