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
const ATTEMPT033_ANGLE_FD_ABS_STEP = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_FD_ABS_STEP", "5e-4"))
const ATTEMPT033_ANGLE_FD_REL_STEP = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_FD_REL_STEP", "1e-3"))
const ATTEMPT033_ANGLE_FD_MAX_SHRINKS = parse(Int, get(ENV, "ATTEMPT033_ANGLE_FD_MAX_SHRINKS", "6"))
const ATTEMPT033_ANGLE_DELTA_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_DELTA_TOL", "1e-10"))
const ATTEMPT033_ANGLE_SECANT_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT033_ANGLE_SECANT_MAX_ITERS", "8"))
const ATTEMPT033_ANGLE_GRAD_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_GRAD_TOL", "1e-6"))
const ATTEMPT033_ANGLE_STEP_TOL = parse(Float64, get(ENV, "ATTEMPT033_ANGLE_STEP_TOL", "1e-8"))
const ATTEMPT033_ANGLE_SCAN_POINTS = parse(Int, get(ENV, "ATTEMPT033_ANGLE_SCAN_POINTS", "17"))

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

struct AngleSecantTrace33
    iter::Int
    x_left::Float64
    x_right::Float64
    x_next::Float64
    theta::Float64
    theta_dx::Float64
    bracketed::Bool
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

function next_absx_return_linearization_033(
    alpha::Float64,
    lambda::Float64,
    x0::Float64,
    z0::Float64,
)
    x0 > ATTEMPT033_MIN_SECTION_X || error("Section x must stay positive for angle continuation")

    function transformed_flow!(du, w, p, t)
        α = p[1]
        λ = p[2]
        base_x = p[3]
        base_z = p[4]
        u = @SVector [w[1] + base_x, w[2], w[3] + base_z]
        flow = shimizu_morioka_vector_033(u, @SVector [α, λ])
        du[1] = flow[1]
        du[2] = flow[2]
        du[3] = flow[3]
        return nothing
    end

    p = [alpha, lambda, x0, z0]
    hit = Ref(false)
    armed = Ref(false)
    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        x = integrator.u[1] + integrator.p[3]
        z = integrator.u[3] + integrator.p[4]
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
    prob = ODEForwardSensitivityProblem(
        transformed_flow!,
        zeros(3),
        (0.0, ATTEMPT033_T_END),
        p,
        ForwardSensitivity(),
    )
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

    hit[] || error("SciMLSensitivity solve did not hit the next |x|-maximum")

    w_final, sensitivity_matrix = extract_local_sensitivities(sol, length(sol.u), Val(true))
    event_state = @SVector [w_final[1] + x0, w_final[2], w_final[3] + z0]
    maximum(abs, event_state) <= ATTEMPT033_MAX_STATE || error("State blew up during angle linearization solve")

    flow = shimizu_morioka_vector_033(event_state, @SVector [alpha, lambda])
    denom = flow[2]
    abs(denom) > 1e-12 || error("Event-time denominator nearly singular during angle linearization")

    raw_x = @SVector [
        sensitivity_matrix[1, 3] + 1.0,
        sensitivity_matrix[2, 3],
        sensitivity_matrix[3, 3],
    ]
    raw_z = @SVector [
        sensitivity_matrix[1, 4],
        sensitivity_matrix[2, 4],
        sensitivity_matrix[3, 4] + 1.0,
    ]

    event_time_x = -raw_x[2] / denom
    event_time_z = -raw_z[2] / denom
    return_dx = raw_x + flow * event_time_x
    return_dz = raw_z + flow * event_time_z

    return ReturnSectionJacobian33(
        SVector{3, Float64}(x0, 0.0, z0),
        event_state,
        sol.t[end],
        return_dx,
        return_dz,
    )
end

@inline function section_return_matrix_033(linearization::ReturnSectionJacobian33)
    return @SMatrix [
        linearization.return_dx[1] linearization.return_dz[1]
        linearization.return_dx[3] linearization.return_dz[3]
    ]
end

function eigendirection_angle_033(A::SMatrix{2, 2, Float64, 4})
    a = A[1, 1]
    b = A[1, 2]
    c = A[2, 1]
    d = A[2, 2]
    delta = (a - d)^2 + 4.0 * b * c
    delta > ATTEMPT033_ANGLE_DELTA_TOL || error("Return-map eigendirections are not real/distinct enough")
    return atan(sqrt(delta), abs(b - c))
end

function angle_value_033(alpha::Float64, lambda::Float64, x::Float64, z_fixed::Float64)
    linearization = next_absx_return_linearization_033(alpha, lambda, x, z_fixed)
    A = section_return_matrix_033(linearization)
    theta = eigendirection_angle_033(A)
    return theta, linearization, A
end

@inline function angle_fd_step_033(x::Float64)
    return max(ATTEMPT033_ANGLE_FD_ABS_STEP, ATTEMPT033_ANGLE_FD_REL_STEP * max(abs(x), 1.0))
end

function evaluate_angle_033(alpha::Float64, lambda::Float64, x::Float64, z_fixed::Float64)
    theta_0, linearization_0, A_0 = angle_value_033(alpha, lambda, x, z_fixed)
    h = angle_fd_step_033(x)
    for _ in 1:ATTEMPT033_ANGLE_FD_MAX_SHRINKS
        lower = x - h
        if lower <= ATTEMPT033_MIN_SECTION_X
            h *= 0.5
            continue
        end

        try
            theta_m, _, _ = angle_value_033(alpha, lambda, x - h, z_fixed)
            theta_p, _, _ = angle_value_033(alpha, lambda, x + h, z_fixed)
            theta_dx = (theta_p - theta_m) / (2.0 * h)
            return AngleEval33(
                x,
                z_fixed,
                theta_0,
                theta_dx,
                NaN,
                linearization_0.current_state,
                linearization_0.event_state,
                A_0,
            )
        catch
            h *= 0.5
        end
    end

    error("Angle derivative stencil failed at x=$(x), z=$(z_fixed)")
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

    best_idx = 0
    best_abs_dx = Inf
    best_distance = Inf
    for idx in eachindex(xs)
        eval = evals[idx]
        isnothing(eval) && continue
        abs_dx = abs(eval.theta_dx)
        distance = abs(xs[idx] - center_x)
        if abs_dx < best_abs_dx || (abs_dx == best_abs_dx && distance < best_distance)
            best_idx = idx
            best_abs_dx = abs_dx
            best_distance = distance
        end
    end
    best_idx == 0 && return nothing

    sign_change_pairs = Tuple{Int, Int}[]
    for idx in 1:(length(xs) - 1)
        eval_left = evals[idx]
        eval_right = evals[idx + 1]
        if isnothing(eval_left) || isnothing(eval_right)
            continue
        end
        phi_left = eval_left.theta_dx
        phi_right = eval_right.theta_dx
        if phi_left == 0.0 || phi_right == 0.0 || signbit(phi_left) != signbit(phi_right)
            push!(sign_change_pairs, (idx, idx + 1))
        end
    end

    if !isempty(sign_change_pairs)
        best_pair = sign_change_pairs[1]
        best_pair_distance = abs(0.5 * (xs[best_pair[1]] + xs[best_pair[2]]) - center_x)
        for pair in sign_change_pairs[2:end]
            distance = abs(0.5 * (xs[pair[1]] + xs[pair[2]]) - center_x)
            if distance < best_pair_distance
                best_pair = pair
                best_pair_distance = distance
            end
        end
        return best_pair, xs, evals
    end

    neighbor_idx = 0
    for offset in 1:(length(xs) - 1)
        for candidate in (best_idx - offset, best_idx + offset)
            if 1 <= candidate <= length(xs) && !isnothing(evals[candidate])
                neighbor_idx = candidate
                break
            end
        end
        neighbor_idx != 0 && break
    end
    neighbor_idx == 0 && return nothing
    return (min(best_idx, neighbor_idx), max(best_idx, neighbor_idx)), xs, evals
end

function run_secant_angle_033(
    alpha::Float64,
    lambda::Float64,
    x0::Float64,
    z_fixed::Float64;
    half_width::Float64=ATTEMPT033_LOCAL_X_WINDOW,
)
    scanned = scan_angle_minimum_033(alpha, lambda, z_fixed, x0, half_width)
    isnothing(scanned) && return nothing
    (idx_left, idx_right), xs, evals = scanned

    left_eval = evals[idx_left]::AngleEval33
    right_eval = evals[idx_right]::AngleEval33
    left_x = xs[idx_left]
    right_x = xs[idx_right]
    bracketed = signbit(left_eval.theta_dx) != signbit(right_eval.theta_dx)
    trace = AngleSecantTrace33[]

    best_eval = abs(left_eval.theta_dx) <= abs(right_eval.theta_dx) ? left_eval : right_eval
    left_bound = max(ATTEMPT033_MIN_SECTION_X, x0 - half_width)
    right_bound = x0 + half_width

    for iter in 1:ATTEMPT033_ANGLE_SECANT_MAX_ITERS
        if abs(left_eval.theta_dx) < abs(best_eval.theta_dx)
            best_eval = left_eval
        end
        if abs(right_eval.theta_dx) < abs(best_eval.theta_dx)
            best_eval = right_eval
        end
        abs(best_eval.theta_dx) <= ATTEMPT033_ANGLE_GRAD_TOL && return trace, best_eval

        denom = right_eval.theta_dx - left_eval.theta_dx
        if abs(denom) <= 1e-14
            x_next = bracketed ? 0.5 * (left_x + right_x) : 0.5 * (left_x + right_x)
        else
            x_next = right_x - right_eval.theta_dx * (right_x - left_x) / denom
            if bracketed && !(min(left_x, right_x) < x_next < max(left_x, right_x))
                x_next = 0.5 * (left_x + right_x)
            end
        end
        x_next = clamp(x_next, left_bound, right_bound)

        push!(trace, AngleSecantTrace33(iter, left_x, right_x, x_next, best_eval.theta, best_eval.theta_dx, bracketed))
        abs(x_next - right_x) <= ATTEMPT033_ANGLE_STEP_TOL && return trace, best_eval

        next_eval = try
            evaluate_angle_033(alpha, lambda, x_next, z_fixed)
        catch
            return trace, best_eval
        end
        if abs(next_eval.theta_dx) < abs(best_eval.theta_dx)
            best_eval = next_eval
        end

        if bracketed
            if next_eval.theta_dx == 0.0
                return trace, next_eval
            elseif signbit(left_eval.theta_dx) != signbit(next_eval.theta_dx)
                right_x = x_next
                right_eval = next_eval
            else
                left_x = x_next
                left_eval = next_eval
            end
        else
            left_x, left_eval = right_x, right_eval
            right_x, right_eval = x_next, next_eval
            bracketed = signbit(left_eval.theta_dx) != signbit(right_eval.theta_dx)
        end
    end

    return trace, best_eval
end

function refine_angle_seed_033(
    alpha::Float64,
    lambda::Float64,
    x0::Float64,
    z_fixed::Float64;
    half_width::Float64=ATTEMPT033_LOCAL_X_WINDOW,
)
    secant = run_secant_angle_033(alpha, lambda, x0, z_fixed; half_width=half_width)
    isnothing(secant) && return nothing
    _, eval = secant
    return eval, "angle_secant"
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
