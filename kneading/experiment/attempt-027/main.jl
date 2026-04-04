using Pkg

const ATTEMPT25_ROOT = @__DIR__
const REPO_ROOT_025 = normpath(joinpath(ATTEMPT25_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025)

using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Printf
using StaticArrays

const ATTEMPT025_N_ALPHA = parse(Int, get(ENV, "ATTEMPT025_N_ALPHA", "2000"))
const ATTEMPT025_N_LAMBDA = parse(Int, get(ENV, "ATTEMPT025_N_LAMBDA", "2000"))
const ATTEMPT025_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT025_ALPHA_MIN", "0.0"))
const ATTEMPT025_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT025_ALPHA_MAX", "0.7"))
const ATTEMPT025_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT025_LAMBDA_MIN", "0.2"))
const ATTEMPT025_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT025_LAMBDA_MAX", "1.6"))
const ATTEMPT025_MAX_EVENT_ITERATES = parse(Int, get(ENV, "ATTEMPT025_MAX_EVENT_ITERATES", "16"))
const ATTEMPT025_PLOT_ITERATE_CAP = min(
    ATTEMPT025_MAX_EVENT_ITERATES,
    parse(Int, get(ENV, "ATTEMPT025_PLOT_ITERATE_CAP", "8")),
)
const ATTEMPT025_T_END = parse(Float64, get(ENV, "ATTEMPT025_T_END", "200.0"))
const ATTEMPT025_DT = parse(Float64, get(ENV, "ATTEMPT025_DT", "0.02"))
const ATTEMPT025_ABSTOL = parse(Float64, get(ENV, "ATTEMPT025_ABSTOL", "1e-9"))
const ATTEMPT025_RELTOL = parse(Float64, get(ENV, "ATTEMPT025_RELTOL", "1e-9"))
const ATTEMPT025_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT025_MAX_ITERS", "40000000"))
const ATTEMPT025_EPS0 = parse(Float64, get(ENV, "ATTEMPT025_EPS0", "1e-7"))
const ATTEMPT025_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT025_MAX_STATE", "200.0"))
const ATTEMPT025_TANGENT_EPS = parse(Float64, get(ENV, "ATTEMPT025_TANGENT_EPS", "1e-12"))
const ATTEMPT025_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT025_LINEWIDTH", "0.35"))
const ATTEMPT025_FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT025_FIG_WIDTH", "2000"))
const ATTEMPT025_FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT025_FIG_HEIGHT", "2000"))
const ATTEMPT025_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT025_PX_PER_UNIT", "4.0"))
const ATTEMPT025_COLOR_SEED = parse(Int, get(ENV, "ATTEMPT025_COLOR_SEED", "21"))

const ALPHAS_025 = collect(range(ATTEMPT025_ALPHA_MIN, ATTEMPT025_ALPHA_MAX, length=ATTEMPT025_N_ALPHA))
const LAMBDAS_025 = collect(range(ATTEMPT025_LAMBDA_MIN, ATTEMPT025_LAMBDA_MAX, length=ATTEMPT025_N_LAMBDA))
const TANGENT_INITIAL_025 = @SVector [0.0, 0.0, 1.0]
const INVALID_STATE_025 = SVector{3, Float64}(NaN, NaN, NaN)

struct SMAbsXResult25
    alpha::Float64
    lambda::Float64
    absxmax_count::Int
    absxmax_dot_values::Vector{Float64}
    absxmax_return_times::Vector{Float64}
    absxmax_states::Vector{SVector{3, Float64}}
    status::String
end

@inline function shimizu_morioka_vector_025(u::SVector{3, Float64}, p::SVector{2, Float64})
    alpha, lambda = p
    x, y, z = u
    return @SVector [y, x - lambda * y - x * z, -alpha * (z - x * x)]
end

@inline function shimizu_morioka_tangent_025!(du, u, p, t)
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

@inline function unstable_side_initial_condition_025(lambda::Float64; eps0::Float64=ATTEMPT025_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / norm(vec))
end

@inline absxmax_proxy_025(u::SVector{3, Float64}) = u[1] * u[2]
@inline absx_vertex_value_025(u::SVector{3, Float64}) = u[1] * u[1]
@inline signed_absx_component_025(u::SVector{3, Float64}, v::SVector{3, Float64}) = (u[1] >= 0.0 ? 1.0 : -1.0) * v[1]

function quadratic_vertex_time_025(
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

@inline function normalize_tangent_025(v::SVector{3, Float64})
    tangent_norm = norm(v)
    tangent_norm > ATTEMPT025_TANGENT_EPS || return nothing
    return v / tangent_norm
end

@inline function orthonormalize_against_flow_025(
    v::SVector{3, Float64},
    u::SVector{3, Float64},
    p::SVector{2, Float64},
)
    all(isfinite, v) || return nothing
    flow = shimizu_morioka_vector_025(u, p)
    all(isfinite, flow) || return nothing

    flow_norm_sq = dot(flow, flow)
    if flow_norm_sq > ATTEMPT025_TANGENT_EPS^2
        v = v - (dot(v, flow) / flow_norm_sq) * flow
    end
    return normalize_tangent_025(v)
end

@inline crossing_theta_025(prev_value::Float64, curr_value::Float64) =
    iszero(curr_value - prev_value) ? 0.5 : clamp(-prev_value / (curr_value - prev_value), 0.0, 1.0)

@inline function interpolate_state_025(
    u_prev::SVector{3, Float64},
    u_curr::SVector{3, Float64},
    theta::Float64,
)
    return (1.0 - theta) * u_prev + theta * u_curr
end

@inline function interpolate_tangent_025(
    v_prev::SVector{3, Float64},
    v_curr::SVector{3, Float64},
    theta::Float64,
)
    return (1.0 - theta) * v_prev + theta * v_curr
end

function scan_orbit_025(alpha::Float64, lambda::Float64)::SMAbsXResult25
    p = @SVector [alpha, lambda]
    orbit_u0 = unstable_side_initial_condition_025(lambda)
    tangent_u0 = orthonormalize_against_flow_025(TANGENT_INITIAL_025, orbit_u0, p)
    if isnothing(tangent_u0)
        return SMAbsXResult25(
            alpha,
            lambda,
            0,
            Float64[],
            Float64[],
            SVector{3, Float64}[],
            "bad_tangent",
        )
    end

    u0 = [orbit_u0[1], orbit_u0[2], orbit_u0[3], tangent_u0[1], tangent_u0[2], tangent_u0[3]]
    prob = ODEProblem(shimizu_morioka_tangent_025!, u0, (0.0, ATTEMPT025_T_END), collect(p))
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT025_DT,
        dtmax=ATTEMPT025_DT,
        abstol=ATTEMPT025_ABSTOL,
        reltol=ATTEMPT025_RELTOL,
        maxiters=ATTEMPT025_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    absxmax_dot_values = Float64[]
    absxmax_return_times = Float64[]
    absxmax_states = SVector{3, Float64}[]
    sizehint!(absxmax_dot_values, ATTEMPT025_MAX_EVENT_ITERATES)
    sizehint!(absxmax_return_times, ATTEMPT025_MAX_EVENT_ITERATES)
    sizehint!(absxmax_states, ATTEMPT025_MAX_EVENT_ITERATES)
    status = "ok"

    u_prev = orbit_u0
    v_prev = tangent_u0
    t_prev = integ.t
    absxmax_prev = absxmax_proxy_025(u_prev)
    have_prevprev = false
    u_prevprev = u_prev
    t_prevprev = t_prev

    while integ.t < ATTEMPT025_T_END && length(absxmax_dot_values) < ATTEMPT025_MAX_EVENT_ITERATES
        step!(integ)

        u_curr = SVector{3, Float64}(integ.u[1:3])
        v_curr_raw = SVector{3, Float64}(integ.u[4:6])
        v_curr = orthonormalize_against_flow_025(v_curr_raw, u_curr, p)
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
        if maximum(abs, u_curr) > ATTEMPT025_MAX_STATE
            status = "blowup"
            break
        end

        absxmax_curr = absxmax_proxy_025(u_curr)
        if absxmax_prev >= 0.0 && absxmax_curr < 0.0
            theta = crossing_theta_025(absxmax_prev, absxmax_curr)
            if have_prevprev
                t_vertex = quadratic_vertex_time_025(
                    t_prevprev,
                    absx_vertex_value_025(u_prevprev),
                    t_prev,
                    absx_vertex_value_025(u_prev),
                    t_curr,
                    absx_vertex_value_025(u_curr),
                )
                if !isnothing(t_vertex)
                    theta = clamp((t_vertex - t_prev) / (t_curr - t_prev), 0.0, 1.0)
                end
            end

            u_hit = interpolate_state_025(u_prev, u_curr, theta)
            v_hit = orthonormalize_against_flow_025(interpolate_tangent_025(v_prev, v_curr, theta), u_hit, p)
            if isnothing(v_hit)
                status = "bad_tangent"
                break
            end

            t_hit = t_prev + theta * (t_curr - t_prev)
            push!(absxmax_dot_values, signed_absx_component_025(u_hit, v_hit))
            push!(absxmax_return_times, t_hit)
            push!(absxmax_states, u_hit)
        end

        have_prevprev = true
        u_prevprev = u_prev
        t_prevprev = t_prev
        u_prev = u_curr
        v_prev = v_curr
        t_prev = t_curr
        absxmax_prev = absxmax_curr
    end

    if status == "ok" && length(absxmax_dot_values) < ATTEMPT025_MAX_EVENT_ITERATES
        status = "short"
    end

    return SMAbsXResult25(
        alpha,
        lambda,
        length(absxmax_dot_values),
        absxmax_dot_values,
        absxmax_return_times,
        absxmax_states,
        status,
    )
end

join_float_list_025(values::Vector{Float64}) = join((@sprintf("%.9f", v) for v in values), ",")
join_state_list_025(values::Vector{SVector{3, Float64}}) =
    join((@sprintf("%.9f,%.9f,%.9f", value[1], value[2], value[3]) for value in values), ";")

function parse_float_list_025(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end

function parse_state_list_025(field::AbstractString)
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
