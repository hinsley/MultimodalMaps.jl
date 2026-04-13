using Pkg

const ATTEMPT24_ROOT = @__DIR__
const REPO_ROOT_024 = normpath(joinpath(ATTEMPT24_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_024)

using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Printf
using StaticArrays

const ATTEMPT024_N_ALPHA = parse(Int, get(ENV, "ATTEMPT024_N_ALPHA", "1200"))
const ATTEMPT024_N_LAMBDA = parse(Int, get(ENV, "ATTEMPT024_N_LAMBDA", "1200"))
const ATTEMPT024_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT024_ALPHA_MIN", "0.0"))
const ATTEMPT024_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT024_ALPHA_MAX", "0.7"))
const ATTEMPT024_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT024_LAMBDA_MIN", "0.2"))
const ATTEMPT024_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT024_LAMBDA_MAX", "1.6"))
const ATTEMPT024_MAX_EVENT_ITERATES = parse(Int, get(ENV, "ATTEMPT024_MAX_EVENT_ITERATES", get(ENV, "ATTEMPT024_MAX_ZMAX", "8")))
const ATTEMPT024_T_END = parse(Float64, get(ENV, "ATTEMPT024_T_END", "200.0"))
const ATTEMPT024_DT = parse(Float64, get(ENV, "ATTEMPT024_DT", "0.02"))
const ATTEMPT024_ABSTOL = parse(Float64, get(ENV, "ATTEMPT024_ABSTOL", "1e-9"))
const ATTEMPT024_RELTOL = parse(Float64, get(ENV, "ATTEMPT024_RELTOL", "1e-9"))
const ATTEMPT024_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT024_MAX_ITERS", "40000000"))
const ATTEMPT024_EPS0 = parse(Float64, get(ENV, "ATTEMPT024_EPS0", "1e-7"))
const ATTEMPT024_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT024_MAX_STATE", "200.0"))
const ATTEMPT024_TANGENT_EPS = parse(Float64, get(ENV, "ATTEMPT024_TANGENT_EPS", "1e-12"))
const ATTEMPT024_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT024_LINEWIDTH", "0.35"))
const ATTEMPT024_FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT024_FIG_WIDTH", "1200"))
const ATTEMPT024_FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT024_FIG_HEIGHT", "1200"))
const ATTEMPT024_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT024_PX_PER_UNIT", "1.0"))
const ATTEMPT024_COLOR_SEED = parse(Int, get(ENV, "ATTEMPT024_COLOR_SEED", "21"))

const ALPHAS_024 = collect(range(ATTEMPT024_ALPHA_MIN, ATTEMPT024_ALPHA_MAX, length=ATTEMPT024_N_ALPHA))
const LAMBDAS_024 = collect(range(ATTEMPT024_LAMBDA_MIN, ATTEMPT024_LAMBDA_MAX, length=ATTEMPT024_N_LAMBDA))
const TANGENT_INITIAL_024 = @SVector [0.0, 0.0, 1.0]

struct SMDualEventResult24
    alpha::Float64
    lambda::Float64
    zmax_count::Int
    zmax_dot_values::Vector{Float64}
    zmax_states::Vector{SVector{3, Float64}}
    absxmax_count::Int
    absxmax_dot_values::Vector{Float64}
    absxmax_states::Vector{SVector{3, Float64}}
    status::String
end

@inline function shimizu_morioka_vector(u::SVector{3, Float64}, p::SVector{2, Float64})
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

@inline function unstable_side_initial_condition(lambda::Float64; eps0::Float64=ATTEMPT024_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / norm(vec))
end

@inline zmax_proxy_024(u::SVector{3, Float64}) = u[3] - u[1] * u[1]
@inline absxmax_proxy_024(u::SVector{3, Float64}) = u[1] * u[2]
@inline absx_vertex_value_024(u::SVector{3, Float64}) = u[1] * u[1]
@inline signed_absx_component_024(u::SVector{3, Float64}, v::SVector{3, Float64}) = (u[1] >= 0.0 ? 1.0 : -1.0) * v[1]

function quadratic_vertex_time(
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

@inline function normalize_tangent(v::SVector{3, Float64})
    tangent_norm = norm(v)
    tangent_norm > ATTEMPT024_TANGENT_EPS || return nothing
    return v / tangent_norm
end

@inline function orthonormalize_against_flow_024(
    v::SVector{3, Float64},
    u::SVector{3, Float64},
    p::SVector{2, Float64},
)
    all(isfinite, v) || return nothing
    flow = shimizu_morioka_vector(u, p)
    all(isfinite, flow) || return nothing

    flow_norm_sq = dot(flow, flow)
    if flow_norm_sq > ATTEMPT024_TANGENT_EPS^2
        v = v - (dot(v, flow) / flow_norm_sq) * flow
    end
    return normalize_tangent(v)
end

@inline crossing_theta_024(prev_value::Float64, curr_value::Float64) =
    iszero(curr_value - prev_value) ? 0.5 : clamp(-prev_value / (curr_value - prev_value), 0.0, 1.0)

@inline function interpolate_state_024(
    u_prev::SVector{3, Float64},
    u_curr::SVector{3, Float64},
    theta::Float64,
)
    return (1.0 - theta) * u_prev + theta * u_curr
end

@inline function interpolate_tangent_024(
    v_prev::SVector{3, Float64},
    v_curr::SVector{3, Float64},
    theta::Float64,
)
    return (1.0 - theta) * v_prev + theta * v_curr
end

function scan_orbit(alpha::Float64, lambda::Float64)::SMDualEventResult24
    p = @SVector [alpha, lambda]
    orbit_u0 = unstable_side_initial_condition(lambda)
    tangent_u0 = orthonormalize_against_flow_024(TANGENT_INITIAL_024, orbit_u0, p)
    if isnothing(tangent_u0)
        return SMDualEventResult24(
            alpha,
            lambda,
            0,
            Float64[],
            SVector{3, Float64}[],
            0,
            Float64[],
            SVector{3, Float64}[],
            "bad_tangent",
        )
    end

    u0 = [orbit_u0[1], orbit_u0[2], orbit_u0[3], tangent_u0[1], tangent_u0[2], tangent_u0[3]]

    prob = ODEProblem(shimizu_morioka_tangent!, u0, (0.0, ATTEMPT024_T_END), collect(p))
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT024_DT,
        dtmax=ATTEMPT024_DT,
        abstol=ATTEMPT024_ABSTOL,
        reltol=ATTEMPT024_RELTOL,
        maxiters=ATTEMPT024_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    zmax_dot_values = Float64[]
    zmax_states = SVector{3, Float64}[]
    absxmax_dot_values = Float64[]
    absxmax_states = SVector{3, Float64}[]
    sizehint!(zmax_dot_values, ATTEMPT024_MAX_EVENT_ITERATES)
    sizehint!(zmax_states, ATTEMPT024_MAX_EVENT_ITERATES)
    sizehint!(absxmax_dot_values, ATTEMPT024_MAX_EVENT_ITERATES)
    sizehint!(absxmax_states, ATTEMPT024_MAX_EVENT_ITERATES)
    status = "ok"

    u_prev = orbit_u0
    v_prev = tangent_u0
    t_prev = integ.t
    zmax_prev = zmax_proxy_024(u_prev)
    absxmax_prev = absxmax_proxy_024(u_prev)
    have_prevprev = false
    u_prevprev = u_prev
    t_prevprev = t_prev

    while integ.t < ATTEMPT024_T_END &&
        (length(zmax_dot_values) < ATTEMPT024_MAX_EVENT_ITERATES ||
         length(absxmax_dot_values) < ATTEMPT024_MAX_EVENT_ITERATES)
        step!(integ)

        u_curr = SVector{3, Float64}(integ.u[1:3])
        v_curr_raw = SVector{3, Float64}(integ.u[4:6])
        v_curr = orthonormalize_against_flow_024(v_curr_raw, u_curr, p)
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
        if maximum(abs, u_curr) > ATTEMPT024_MAX_STATE
            status = "blowup"
            break
        end

        zmax_curr = zmax_proxy_024(u_curr)
        if length(zmax_dot_values) < ATTEMPT024_MAX_EVENT_ITERATES && zmax_prev <= 0.0 && zmax_curr > 0.0
            theta = crossing_theta_024(zmax_prev, zmax_curr)
            if have_prevprev
                t_vertex = quadratic_vertex_time(t_prevprev, u_prevprev[3], t_prev, u_prev[3], t_curr, u_curr[3])
                if !isnothing(t_vertex)
                    theta = clamp((t_vertex - t_prev) / (t_curr - t_prev), 0.0, 1.0)
                end
            end

            u_hit = interpolate_state_024(u_prev, u_curr, theta)
            v_hit = orthonormalize_against_flow_024(interpolate_tangent_024(v_prev, v_curr, theta), u_hit, p)
            if isnothing(v_hit)
                status = "bad_tangent"
                break
            end
            push!(zmax_dot_values, v_hit[3])
            push!(zmax_states, u_hit)
        end

        absxmax_curr = absxmax_proxy_024(u_curr)
        if length(absxmax_dot_values) < ATTEMPT024_MAX_EVENT_ITERATES && absxmax_prev >= 0.0 && absxmax_curr < 0.0
            theta = crossing_theta_024(absxmax_prev, absxmax_curr)
            if have_prevprev
                t_vertex = quadratic_vertex_time(
                    t_prevprev,
                    absx_vertex_value_024(u_prevprev),
                    t_prev,
                    absx_vertex_value_024(u_prev),
                    t_curr,
                    absx_vertex_value_024(u_curr),
                )
                if !isnothing(t_vertex)
                    theta = clamp((t_vertex - t_prev) / (t_curr - t_prev), 0.0, 1.0)
                end
            end

            u_hit = interpolate_state_024(u_prev, u_curr, theta)
            v_hit = orthonormalize_against_flow_024(interpolate_tangent_024(v_prev, v_curr, theta), u_hit, p)
            if isnothing(v_hit)
                status = "bad_tangent"
                break
            end
            push!(absxmax_dot_values, signed_absx_component_024(u_hit, v_hit))
            push!(absxmax_states, u_hit)
        end

        have_prevprev = true
        u_prevprev = u_prev
        t_prevprev = t_prev
        u_prev = u_curr
        v_prev = v_curr
        t_prev = t_curr
        zmax_prev = zmax_curr
        absxmax_prev = absxmax_curr
    end

    if status == "ok" &&
        (length(zmax_dot_values) < ATTEMPT024_MAX_EVENT_ITERATES ||
         length(absxmax_dot_values) < ATTEMPT024_MAX_EVENT_ITERATES)
        status = "short"
    end

    return SMDualEventResult24(
        alpha,
        lambda,
        length(zmax_dot_values),
        zmax_dot_values,
        zmax_states,
        length(absxmax_dot_values),
        absxmax_dot_values,
        absxmax_states,
        status,
    )
end

join_float_list(values::Vector{Float64}) = join((@sprintf("%.9f", v) for v in values), ",")
join_state_list(values::Vector{SVector{3, Float64}}) =
    join((@sprintf("%.9f,%.9f,%.9f", value[1], value[2], value[3]) for value in values), ";")

function parse_float_list(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end

function parse_state_list(field::AbstractString)
    isempty(field) && return SVector{3, Float64}[]
    states = SVector{3, Float64}[]
    sizehint!(states, count(==(';'), field) + 1)

    for state_field in split(field, ';')
        coords = split(state_field, ',')
        length(coords) == 3 || error("Malformed state triple: $(state_field)")
        push!(states, SVector{3, Float64}(parse.(Float64, coords)))
    end

    return states
end
