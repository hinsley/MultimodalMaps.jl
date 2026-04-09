using Pkg

const ATTEMPT33_ROOT = @__DIR__
const REPO_ROOT_033 = normpath(joinpath(ATTEMPT33_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_033)

using DifferentialEquations
using LinearAlgebra
using Printf
using SciMLSensitivity
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
const ATTEMPT033_RESEED_PERIOD = parse(Int, get(ENV, "ATTEMPT033_RESEED_PERIOD", "500"))

const ATTEMPT033_LOCAL_X_WINDOW = parse(Float64, get(ENV, "ATTEMPT033_LOCAL_X_WINDOW", "0.12"))
const ATTEMPT033_LOCAL_GRID_POINTS = parse(Int, get(ENV, "ATTEMPT033_LOCAL_GRID_POINTS", "13"))
const ATTEMPT033_FULL_X_WINDOW = parse(Float64, get(ENV, "ATTEMPT033_FULL_X_WINDOW", "0.8"))
const ATTEMPT033_FULL_X_GRID_POINTS = parse(Int, get(ENV, "ATTEMPT033_FULL_X_GRID_POINTS", "81"))
const ATTEMPT033_FULL_RESEED_Z_TRIES = parse(Int, get(ENV, "ATTEMPT033_FULL_RESEED_Z_TRIES", "12"))
const ATTEMPT033_GOLDEN_TOL = parse(Float64, get(ENV, "ATTEMPT033_GOLDEN_TOL", "1e-6"))
const ATTEMPT033_GOLDEN_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT033_GOLDEN_MAX_ITERS", "24"))
const ATTEMPT033_CRITICAL_SLOPE_EPS = parse(Float64, get(ENV, "ATTEMPT033_CRITICAL_SLOPE_EPS", "1e-4"))
const ATTEMPT033_SEED_Z_MIN = parse(Float64, get(ENV, "ATTEMPT033_SEED_Z_MIN", "-1.5"))
const ATTEMPT033_SEED_Z_MAX = parse(Float64, get(ENV, "ATTEMPT033_SEED_Z_MAX", "0.0"))
const ATTEMPT033_SEED_Z_SAMPLES = parse(Int, get(ENV, "ATTEMPT033_SEED_Z_SAMPLES", "61"))
const ATTEMPT033_RESEED_MIN_X_HALF_WIDTH = parse(Float64, get(ENV, "ATTEMPT033_RESEED_MIN_X_HALF_WIDTH", "0.05"))
const ATTEMPT033_ANGLE_DELTA_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_DELTA_TOL", "1e-10"))
const ATTEMPT033_ANGLE_SIGMA_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_SIGMA_TOL", "1e-12"))
const ATTEMPT033_ANGLE_EVENT_DENOM_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_EVENT_DENOM_TOL", "1e-12"))
const ATTEMPT033_ANGLE_NEWTON_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT033_ANGLE_NEWTON_MAX_ITERS", "6"))
const ATTEMPT033_ANGLE_GRAD_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_GRAD_TOL", "1e-6"))
const ATTEMPT033_ANGLE_STEP_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_STEP_TOL", "1e-8"))
const ATTEMPT033_ANGLE_MIN_DAMPING = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_MIN_DAMPING", "0.03125"))
const ATTEMPT033_ANGLE_HESS_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_HESS_TOL", "1e-8"))
const ATTEMPT033_ANGLE_SCAN_POINTS = parse(Int, get(ENV, "ATTEMPT033_ANGLE_SCAN_POINTS", "41"))
const ATTEMPT033_ANGLE_MAX_RAW_STEP = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_MAX_RAW_STEP", "0.02"))

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
    critical_theta::Float64
    critical_theta_dx::Float64
    critical_theta_dxx::Float64
    absxmax_count::Int
    absxmax_dot_values::Vector{Float64}
    status::String
end

struct ReturnSectionJacobian33
    current_state::SVector{3, Float64}
    event_state::SVector{3, Float64}
    event_time::Float64
    return_dx::SVector{3, Float64}
    return_dz::SVector{3, Float64}
    return_dxx::SVector{3, Float64}
    return_dxz::SVector{3, Float64}
    return_dxxx::SVector{3, Float64}
    return_dxxz::SVector{3, Float64}
end

struct AngleEval33
    x::Float64
    z_fixed::Float64
    theta::Float64
    theta_dx::Float64
    theta_dxx::Float64
    current_state::SVector{3, Float64}
    event_state::SVector{3, Float64}
    jacobian::SMatrix{2, 2, Float64, 4}
end

struct AngleNewtonTrace33
    iter::Int
    x::Float64
    theta::Float64
    theta_dx::Float64
    theta_dxx::Float64
    step::Float64
    damping::Float64
    accepted::Bool
end

@inline function shimizu_morioka_vector_033(u::SVector{3, TU}, p::SVector{2, TP}) where {TU<:Real, TP<:Real}
    T = promote_type(TU, TP)
    x = T(u[1])
    y = T(u[2])
    z = T(u[3])
    alpha = T(p[1])
    lambda = T(p[2])
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

@inline function shimizu_morioka_jacobian_033(u::SVector{3, Float64}, p::SVector{2, Float64})
    x = u[1]
    z = u[3]
    alpha = p[1]
    lambda = p[2]
    return @SMatrix [
        0.0 1.0 0.0
        (1.0 - z) (-lambda) (-x)
        (2.0 * alpha * x) 0.0 (-alpha)
    ]
end

@inline function shimizu_morioka_bilinear_033(v::SVector{3, Float64}, w::SVector{3, Float64}, alpha::Float64)
    return @SVector [
        0.0,
        -(v[1] * w[3] + v[3] * w[1]),
        2.0 * alpha * v[1] * w[1],
    ]
end

function shimizu_morioka_angle_taylor!(du, w, p, t)
    alpha = p[1]
    lambda = p[2]
    u = @SVector [w[1], w[2], w[3]]
    pvec = @SVector [alpha, lambda]
    flow = shimizu_morioka_vector_033(u, pvec)
    J = shimizu_morioka_jacobian_033(u, pvec)

    Ux = @SVector [w[4], w[5], w[6]]
    Uz = @SVector [w[7], w[8], w[9]]
    Uxx = @SVector [w[10], w[11], w[12]]
    Uxz = @SVector [w[13], w[14], w[15]]
    Uxxx = @SVector [w[16], w[17], w[18]]
    Uxxz = @SVector [w[19], w[20], w[21]]

    dUx = J * Ux
    dUz = J * Uz
    dUxx = J * Uxx + shimizu_morioka_bilinear_033(Ux, Ux, alpha)
    dUxz = J * Uxz + shimizu_morioka_bilinear_033(Ux, Uz, alpha)
    dUxxx = J * Uxxx + 3.0 * shimizu_morioka_bilinear_033(Uxx, Ux, alpha)
    dUxxz = J * Uxxz + shimizu_morioka_bilinear_033(Uxx, Uz, alpha) + 2.0 * shimizu_morioka_bilinear_033(Uxz, Ux, alpha)

    du[1] = flow[1]
    du[2] = flow[2]
    du[3] = flow[3]
    du[4] = dUx[1]
    du[5] = dUx[2]
    du[6] = dUx[3]
    du[7] = dUz[1]
    du[8] = dUz[2]
    du[9] = dUz[3]
    du[10] = dUxx[1]
    du[11] = dUxx[2]
    du[12] = dUxx[3]
    du[13] = dUxz[1]
    du[14] = dUxz[2]
    du[15] = dUxz[3]
    du[16] = dUxxx[1]
    du[17] = dUxxx[2]
    du[18] = dUxxx[3]
    du[19] = dUxxz[1]
    du[20] = dUxxz[2]
    du[21] = dUxxz[3]
    return nothing
end

function next_absx_return_linearization_033(
    alpha::Float64,
    lambda::Float64,
    x0::Float64,
    z0::Float64,
)
    x0 > ATTEMPT033_MIN_SECTION_X || error("Section x must stay positive for angle continuation")

    w0 = zeros(21)
    w0[1] = x0
    w0[3] = z0
    w0[4] = 1.0
    w0[9] = 1.0

    hit = Ref(false)
    armed = Ref(false)
    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        if abs(x) > ATTEMPT033_MIN_SECTION_X && z > ATTEMPT033_EVENT_Z_THRESHOLD
            hit[] = true
            terminate!(integrator)
        end
    end

    arm_cb = PresetTimeCallback([ATTEMPT033_MIN_EVENT_TIME], integrator -> begin
        armed[] = true
    end)
    event_cb = ContinuousCallback(condition, affect!; rootfind=true, save_positions=(false, false))
    cb = CallbackSet(arm_cb, event_cb)
    prob = ODEProblem(shimizu_morioka_angle_taylor!, w0, (0.0, ATTEMPT033_T_END), [alpha, lambda])
    sol = solve(
        prob,
        Vern9();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT033_DT,
        dtmax=ATTEMPT033_DT,
        abstol=ATTEMPT033_ABSTOL,
        reltol=ATTEMPT033_RELTOL,
        maxiters=ATTEMPT033_MAX_ITERS,
        save_start=false,
        save_end=true,
        save_everystep=false,
        verbose=false,
    )

    hit[] || error("Exact angle taylor solve did not hit the next |x|-maximum")

    w_final = sol.u[end]
    event_state = @SVector [w_final[1], w_final[2], w_final[3]]
    maximum(abs, event_state) <= ATTEMPT033_MAX_STATE || error("State blew up during angle linearization solve")

    pvec = @SVector [alpha, lambda]
    flow = shimizu_morioka_vector_033(event_state, pvec)
    J = shimizu_morioka_jacobian_033(event_state, pvec)
    denom = flow[2]
    abs(denom) > ATTEMPT033_ANGLE_EVENT_DENOM_TOL || error("Event-time denominator nearly singular during angle linearization")

    raw_x = @SVector [w_final[4], w_final[5], w_final[6]]
    raw_z = @SVector [w_final[7], w_final[8], w_final[9]]
    raw_xx = @SVector [w_final[10], w_final[11], w_final[12]]
    raw_xz = @SVector [w_final[13], w_final[14], w_final[15]]
    raw_xxx = @SVector [w_final[16], w_final[17], w_final[18]]
    raw_xxz = @SVector [w_final[19], w_final[20], w_final[21]]

    flow_x = J * raw_x
    flow_z = J * raw_z
    flow_xx = J * raw_xx + shimizu_morioka_bilinear_033(raw_x, raw_x, alpha)
    flow_xz = J * raw_xz + shimizu_morioka_bilinear_033(raw_x, raw_z, alpha)
    flow_tt = J * flow
    flow_ttt = J * flow_tt + shimizu_morioka_bilinear_033(flow, flow, alpha)
    flow_ttx = J * flow_x + shimizu_morioka_bilinear_033(flow, raw_x, alpha)
    flow_ttz = J * flow_z + shimizu_morioka_bilinear_033(flow, raw_z, alpha)

    event_time_x = -raw_x[2] / denom
    event_time_z = -raw_z[2] / denom
    return_dx = raw_x + flow * event_time_x
    return_dz = raw_z + flow * event_time_z

    event_time_xx = -(raw_xx[2] + 2.0 * flow_x[2] * event_time_x + flow_tt[2] * event_time_x^2) / denom
    event_time_xz = -(raw_xz[2] + flow_x[2] * event_time_z + flow_z[2] * event_time_x + flow_tt[2] * event_time_x * event_time_z) / denom
    return_dxx = raw_xx + 2.0 * flow_x * event_time_x + flow_tt * event_time_x^2 + flow * event_time_xx
    return_dxz = raw_xz + flow_x * event_time_z + flow_z * event_time_x + flow_tt * event_time_x * event_time_z + flow * event_time_xz

    event_time_xxx = -(
        raw_xxx[2] +
        3.0 * flow_xx[2] * event_time_x +
        3.0 * flow_x[2] * event_time_xx +
        3.0 * flow_ttx[2] * event_time_x^2 +
        3.0 * flow_tt[2] * event_time_x * event_time_xx +
        flow_ttt[2] * event_time_x^3
    ) / denom
    return_dxxx = raw_xxx +
        3.0 * flow_xx * event_time_x +
        3.0 * flow_x * event_time_xx +
        3.0 * flow_ttx * event_time_x^2 +
        3.0 * flow_tt * event_time_x * event_time_xx +
        flow_ttt * event_time_x^3 +
        flow * event_time_xxx

    event_time_xxz = -(
        raw_xxz[2] +
        flow_xx[2] * event_time_z +
        2.0 * flow_xz[2] * event_time_x +
        2.0 * flow_x[2] * event_time_xz +
        flow_z[2] * event_time_xx +
        2.0 * flow_ttx[2] * event_time_x * event_time_z +
        flow_ttz[2] * event_time_x^2 +
        flow_tt[2] * (event_time_xx * event_time_z + 2.0 * event_time_xz * event_time_x) +
        flow_ttt[2] * event_time_x^2 * event_time_z
    ) / denom
    return_dxxz = raw_xxz +
        flow_xx * event_time_z +
        2.0 * flow_xz * event_time_x +
        2.0 * flow_x * event_time_xz +
        flow_z * event_time_xx +
        2.0 * flow_ttx * event_time_x * event_time_z +
        flow_ttz * event_time_x^2 +
        flow_tt * (event_time_xx * event_time_z + 2.0 * event_time_xz * event_time_x) +
        flow_ttt * event_time_x^2 * event_time_z +
        flow * event_time_xxz

    return ReturnSectionJacobian33(
        SVector{3, Float64}(x0, 0.0, z0),
        event_state,
        sol.t[end],
        return_dx,
        return_dz,
        return_dxx,
        return_dxz,
        return_dxxx,
        return_dxxz,
    )
end

@inline function section_return_matrix_033(linearization::ReturnSectionJacobian33)
    return @SMatrix [
        linearization.return_dx[1] linearization.return_dz[1]
        linearization.return_dx[3] linearization.return_dz[3]
    ]
end

@inline function section_return_matrix_x_033(linearization::ReturnSectionJacobian33)
    return @SMatrix [
        linearization.return_dxx[1] linearization.return_dxz[1]
        linearization.return_dxx[3] linearization.return_dxz[3]
    ]
end

@inline function section_return_matrix_xx_033(linearization::ReturnSectionJacobian33)
    return @SMatrix [
        linearization.return_dxxx[1] linearization.return_dxxz[1]
        linearization.return_dxxx[3] linearization.return_dxxz[3]
    ]
end

function eigendirection_angle_033(
    A::SMatrix{2, 2, Float64, 4},
    A_x::SMatrix{2, 2, Float64, 4},
    A_xx::SMatrix{2, 2, Float64, 4},
)
    a = A[1, 1]
    b = A[1, 2]
    c = A[2, 1]
    d = A[2, 2]
    a_x = A_x[1, 1]
    b_x = A_x[1, 2]
    c_x = A_x[2, 1]
    d_x = A_x[2, 2]
    a_xx = A_xx[1, 1]
    b_xx = A_xx[1, 2]
    c_xx = A_xx[2, 1]
    d_xx = A_xx[2, 2]

    delta = (a - d)^2 + 4.0 * b * c
    delta > ATTEMPT033_ANGLE_DELTA_TOL || error("Return-map eigendirections are not real/distinct enough")

    m = b - c
    sigma = m^2
    sigma > ATTEMPT033_ANGLE_SIGMA_TOL || error("Return-map eigendirection-angle denominator nearly singular")

    theta = atan(sqrt(delta), abs(m))
    q = delta / sigma
    q > ATTEMPT033_ANGLE_DELTA_TOL || error("Return-map angle quotient lost positivity")

    delta_x = 2.0 * (a - d) * (a_x - d_x) + 4.0 * (b_x * c + b * c_x)
    delta_xx = 2.0 * (a_x - d_x)^2 + 2.0 * (a - d) * (a_xx - d_xx) + 4.0 * (b_xx * c + 2.0 * b_x * c_x + b * c_xx)
    m_x = b_x - c_x
    m_xx = b_xx - c_xx
    sigma_x = 2.0 * m * m_x
    sigma_xx = 2.0 * m_x^2 + 2.0 * m * m_xx

    q_x = delta_x / sigma - delta * sigma_x / sigma^2
    q_xx = delta_xx / sigma - 2.0 * delta_x * sigma_x / sigma^2 - delta * sigma_xx / sigma^2 + 2.0 * delta * sigma_x^2 / sigma^3

    u = sqrt(q)
    u_x = q_x / (2.0 * u)
    u_xx = q_xx / (2.0 * u) - q_x^2 / (4.0 * u^3)

    theta_x = u_x / (1.0 + q)
    theta_xx = u_xx / (1.0 + q) - u_x * q_x / (1.0 + q)^2
    return theta, theta_x, theta_xx
end

function evaluate_angle_033(alpha::Float64, lambda::Float64, x::Float64, z_fixed::Float64)
    linearization = next_absx_return_linearization_033(alpha, lambda, x, z_fixed)
    A = section_return_matrix_033(linearization)
    A_x = section_return_matrix_x_033(linearization)
    A_xx = section_return_matrix_xx_033(linearization)
    theta, theta_dx, theta_dxx = eigendirection_angle_033(A, A_x, A_xx)
    return AngleEval33(
        x,
        z_fixed,
        theta,
        theta_dx,
        theta_dxx,
        linearization.current_state,
        linearization.event_state,
        A,
    )
end

function scan_angle_minimum_033(
    alpha::Float64,
    lambda::Float64,
    z_fixed::Float64,
    center_x::Float64,
    half_width::Float64;
    grid_points::Int=ATTEMPT033_ANGLE_SCAN_POINTS,
)
    left = max(ATTEMPT033_MIN_SECTION_X, center_x - half_width)
    right = center_x + half_width
    right > left || return nothing

    xs = collect(range(left, right, length=grid_points))
    evals = Vector{Union{Nothing, AngleEval33}}(undef, length(xs))
    thetas = fill(NaN, length(xs))
    for idx in eachindex(xs)
        try
            eval = evaluate_angle_033(alpha, lambda, xs[idx], z_fixed)
            evals[idx] = eval
            thetas[idx] = eval.theta
        catch
            evals[idx] = nothing
        end
    end

    stationaries = Int[]
    for idx in eachindex(xs)
        eval = evals[idx]
        isnothing(eval) && continue
        eval.theta_dxx > 0.0 || continue
        push!(stationaries, idx)
    end
    if !isempty(stationaries)
        best_idx = stationaries[1]
        best_eval = evals[best_idx]::AngleEval33
        best_abs_dx = abs(best_eval.theta_dx)
        best_distance = abs(xs[best_idx] - center_x)
        for idx in stationaries[2:end]
            eval = evals[idx]::AngleEval33
            abs_dx = abs(eval.theta_dx)
            distance = abs(xs[idx] - center_x)
            if abs_dx < best_abs_dx || (abs_dx == best_abs_dx && distance < best_distance)
                best_idx = idx
                best_abs_dx = abs_dx
                best_distance = distance
            end
        end
        return xs[best_idx], evals[best_idx]
    end

    candidates = Int[]
    for idx in 2:(length(xs) - 1)
        all(isfinite, (thetas[idx - 1], thetas[idx], thetas[idx + 1])) || continue
        if thetas[idx] <= thetas[idx - 1] && thetas[idx] <= thetas[idx + 1]
            push!(candidates, idx)
        end
    end

    if isempty(candidates)
        best_idx = 0
        best_theta = Inf
        best_distance = Inf
        for idx in eachindex(xs)
            isfinite(thetas[idx]) || continue
            theta = thetas[idx]
            distance = abs(xs[idx] - center_x)
            if theta < best_theta || (theta == best_theta && distance < best_distance)
                best_idx = idx
                best_theta = theta
                best_distance = distance
            end
        end
        best_idx == 0 && return nothing
        return xs[best_idx], evals[best_idx]
    end

    best_idx = candidates[1]
    best_distance = abs(xs[best_idx] - center_x)
    best_theta = thetas[best_idx]
    for idx in candidates[2:end]
        distance = abs(xs[idx] - center_x)
        theta = thetas[idx]
        if distance < best_distance || (distance == best_distance && theta < best_theta)
            best_idx = idx
            best_distance = distance
            best_theta = theta
        end
    end

    return xs[best_idx], evals[best_idx]
end

function run_damped_newton_angle_033(alpha::Float64, lambda::Float64, x0::Float64, z_fixed::Float64)
    trace = AngleNewtonTrace33[]
    current_x = x0

    for iter in 1:ATTEMPT033_ANGLE_NEWTON_MAX_ITERS
        evaluation = evaluate_angle_033(alpha, lambda, current_x, z_fixed)
        gradient = evaluation.theta_dx
        hessian = evaluation.theta_dxx

        if abs(gradient) <= ATTEMPT033_ANGLE_GRAD_TOL
            hessian > 0.0 || error("Angle Newton landed on a non-minimum at x=$(current_x)")
            push!(trace, AngleNewtonTrace33(iter, current_x, evaluation.theta, gradient, hessian, 0.0, 1.0, true))
            return trace, evaluation
        end

        abs(hessian) > ATTEMPT033_ANGLE_HESS_TOL || error("Angle Newton second derivative nearly singular at x=$(current_x)")
        raw_step = clamp(-gradient / hessian, -ATTEMPT033_ANGLE_MAX_RAW_STEP, ATTEMPT033_ANGLE_MAX_RAW_STEP)
        damping = 1.0
        accepted = false
        candidate_x = current_x

        while damping >= ATTEMPT033_ANGLE_MIN_DAMPING
            candidate_x = current_x + damping * raw_step
            if candidate_x <= ATTEMPT033_MIN_SECTION_X
                damping *= 0.5
                continue
            end

            try
                candidate_eval = evaluate_angle_033(alpha, lambda, candidate_x, z_fixed)
                if abs(candidate_eval.theta_dx) < abs(gradient) && candidate_eval.theta_dxx > 0.0
                    accepted = true
                    break
                end
            catch
            end
            damping *= 0.5
        end

        push!(trace, AngleNewtonTrace33(iter, current_x, evaluation.theta, gradient, hessian, raw_step, damping, accepted))
        accepted || error("Damped angle Newton failed to find an acceptable step from x=$(current_x)")

        current_x = candidate_x
        if abs(damping * raw_step) <= ATTEMPT033_ANGLE_STEP_TOL
            final_eval = evaluate_angle_033(alpha, lambda, current_x, z_fixed)
            return trace, final_eval
        end
    end

    final_eval = evaluate_angle_033(alpha, lambda, current_x, z_fixed)
    return trace, final_eval
end

function refine_angle_seed_033(
    alpha::Float64,
    lambda::Float64,
    x0::Float64,
    z_fixed::Float64;
    half_width::Float64=ATTEMPT033_LOCAL_X_WINDOW,
)
    try
        _, eval = run_damped_newton_angle_033(alpha, lambda, x0, z_fixed)
        return eval, "angle_newton_exact"
    catch
    end

    scanned = scan_angle_minimum_033(alpha, lambda, z_fixed, x0, half_width)
    isnothing(scanned) && return nothing
    scanned_x, scanned_eval = scanned

    try
        _, eval = run_damped_newton_angle_033(alpha, lambda, scanned_x, z_fixed)
        return eval, "angle_scan_newton_exact"
    catch
        return scanned_eval, "angle_scan_fallback_exact"
    end
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

function full_refinement_bracket_033(
    alpha::Float64,
    lambda::Float64,
    z_fixed::Float64,
    center_x::Float64;
    kind_hint::Union{Nothing, Symbol}=nothing,
)
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
            kind_hint === :minimum && continue
            return xs[idx - 1], xs[idx + 1], :maximum
        elseif rise < -ATTEMPT033_CRITICAL_SLOPE_EPS && fall > ATTEMPT033_CRITICAL_SLOPE_EPS
            kind_hint === :maximum && continue
            return xs[idx - 1], xs[idx + 1], :minimum
        end
    end
    return nothing
end

@inline function seed_distance_033(a::SVector{3, Float64}, b::SVector{3, Float64})
    return hypot(a[1] - b[1], a[3] - b[3])
end

function full_reseed_critical_seed_033(
    alpha::Float64,
    lambda::Float64,
    previous_seed::Union{Nothing, CriticalSeed33}=nothing,
)
    times, states = collect_absxmax_events_033(alpha, lambda)
    branch = build_positive_branch_033(times, states)
    critical = first_smooth_critical_candidate_033(branch)
    isnothing(critical) && return nothing

    idx, kind = critical
    center = branch[idx]
    bracket_left = branch[idx - 1].x
    bracket_right = branch[idx + 1].x
    center_x = center.x
    z_fixed = center.z

    if bracket_right - bracket_left < 2.0 * ATTEMPT033_RESEED_MIN_X_HALF_WIDTH
        bracket_left = max(ATTEMPT033_MIN_SECTION_X, center_x - ATTEMPT033_RESEED_MIN_X_HALF_WIDTH)
        bracket_right = center_x + ATTEMPT033_RESEED_MIN_X_HALF_WIDTH
    end

    refined = golden_section_refine_033(alpha, lambda, z_fixed, bracket_left, bracket_right, kind)
    isnothing(refined) && return nothing
    x_refined, _ = refined
    seed = CriticalSeed33(SVector{3, Float64}(x_refined, 0.0, z_fixed), kind)

    angle_refined = refine_angle_seed_033(alpha, lambda, x_refined, z_fixed; half_width=ATTEMPT033_RESEED_MIN_X_HALF_WIDTH)
    isnothing(angle_refined) && return seed, "full_attractor_map_golden_fallback", nothing
    angle_eval, angle_source = angle_refined
    return CriticalSeed33(SVector{3, Float64}(angle_eval.x, 0.0, z_fixed), kind), "full_attractor_map_$(angle_source)", angle_eval
end

function local_refine_critical_seed_033(alpha::Float64, lambda::Float64, seed::CriticalSeed33)
    angle_refined = refine_angle_seed_033(alpha, lambda, seed.state[1], seed.state[3]; half_width=ATTEMPT033_LOCAL_X_WINDOW)
    isnothing(angle_refined) && return nothing
    angle_eval, angle_source = angle_refined
    return CriticalSeed33(SVector{3, Float64}(angle_eval.x, 0.0, seed.state[3]), seed.kind), "continued_$(angle_source)", angle_eval
end

function critical_seed_for_point_033(
    alpha::Float64,
    lambda::Float64,
    row_idx::Int,
    previous_seed::Union{Nothing, CriticalSeed33},
)
    periodic_reseed = row_idx == 1 || mod(row_idx - 1, ATTEMPT033_RESEED_PERIOD) == 0
    if isnothing(previous_seed)
        return full_reseed_critical_seed_033(alpha, lambda, previous_seed)
    end

    if periodic_reseed
        full = full_reseed_critical_seed_033(alpha, lambda, previous_seed)
        if !isnothing(full)
            return full
        end
    end

    local_seed = local_refine_critical_seed_033(alpha, lambda, previous_seed)
    if !isnothing(local_seed)
        return local_seed
    end

    full = full_reseed_critical_seed_033(alpha, lambda, previous_seed)
    if !isnothing(full)
        seed, _, angle_eval = full
        return seed, "fallback_full_angle", angle_eval
    end

    return previous_seed, "carried", nothing
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
            NaN,
            NaN,
            NaN,
            0,
            Float64[],
            "critical_fail",
        ), previous_seed
    end

    seed, source, angle_eval = seed_info
    count, dot_values, status = scan_absx_orbit_from_state_033(alpha, lambda, seed.state)
    result = SMCriticalContourResult33(
        alpha,
        lambda,
        seed.state,
        String(seed.kind),
        source,
        isnothing(angle_eval) ? NaN : angle_eval.theta,
        isnothing(angle_eval) ? NaN : angle_eval.theta_dx,
        isnothing(angle_eval) ? NaN : angle_eval.theta_dxx,
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
