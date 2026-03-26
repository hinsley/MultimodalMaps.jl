using Pkg

const ATTEMPT21_ROOT = @__DIR__
const REPO_ROOT_021 = normpath(joinpath(ATTEMPT21_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_021)

using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Printf
using StaticArrays

const ATTEMPT021_N_ALPHA = parse(Int, get(ENV, "ATTEMPT021_N_ALPHA", "1000"))
const ATTEMPT021_N_LAMBDA = parse(Int, get(ENV, "ATTEMPT021_N_LAMBDA", "1000"))
const ATTEMPT021_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT021_ALPHA_MIN", "0.0"))
const ATTEMPT021_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT021_ALPHA_MAX", "0.7"))
const ATTEMPT021_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT021_LAMBDA_MIN", "0.2"))
const ATTEMPT021_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT021_LAMBDA_MAX", "1.6"))
const ATTEMPT021_MAX_ZMAX = parse(Int, get(ENV, "ATTEMPT021_MAX_ZMAX", "8"))
const ATTEMPT021_T_END = parse(Float64, get(ENV, "ATTEMPT021_T_END", "350.0"))
const ATTEMPT021_DT = parse(Float64, get(ENV, "ATTEMPT021_DT", "0.02"))
const ATTEMPT021_ABSTOL = parse(Float64, get(ENV, "ATTEMPT021_ABSTOL", "1e-9"))
const ATTEMPT021_RELTOL = parse(Float64, get(ENV, "ATTEMPT021_RELTOL", "1e-9"))
const ATTEMPT021_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT021_MAX_ITERS", "20000000"))
const ATTEMPT021_EPS0 = parse(Float64, get(ENV, "ATTEMPT021_EPS0", "1e-7"))
const ATTEMPT021_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT021_MAX_STATE", "200.0"))
const ATTEMPT021_NEAR_SADDLE_SPEED = parse(Float64, get(ENV, "ATTEMPT021_NEAR_SADDLE_SPEED", "0.003"))
const ATTEMPT021_NEAR_SADDLE_RADIUS = parse(Float64, get(ENV, "ATTEMPT021_NEAR_SADDLE_RADIUS", "0.08"))
const ATTEMPT021_RED_EPS = parse(Float64, get(ENV, "ATTEMPT021_RED_EPS", "1e-6"))
const ATTEMPT021_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT021_LINEWIDTH", "0.9"))
const ATTEMPT021_FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT021_FIG_WIDTH", "1000"))
const ATTEMPT021_FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT021_FIG_HEIGHT", "1000"))
const ATTEMPT021_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT021_PX_PER_UNIT", "2.0"))

const ALPHAS_021 = collect(range(ATTEMPT021_ALPHA_MIN, ATTEMPT021_ALPHA_MAX, length=ATTEMPT021_N_ALPHA))
const LAMBDAS_021 = collect(range(ATTEMPT021_LAMBDA_MIN, ATTEMPT021_LAMBDA_MAX, length=ATTEMPT021_N_LAMBDA))

@assert ATTEMPT021_N_ALPHA >= 2
@assert ATTEMPT021_N_LAMBDA >= 2
@assert ATTEMPT021_MAX_ZMAX >= 3

struct SMScanResult21
    alpha::Float64
    lambda::Float64
    maxima_count::Int
    blue_mask::UInt16
    red_mask::UInt16
    zmax_values::Vector{Float64}
    return_times::Vector{Float64}
    status::String
end

@inline function shimizu_morioka!(du, u, p, t)
    alpha, lambda = p
    x, y, z = u
    du[1] = y
    du[2] = x - lambda * y - x * z
    du[3] = -alpha * (z - x * x)
    return nothing
end

@inline function unstable_side_initial_condition(lambda::Float64; eps0::Float64=ATTEMPT021_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / norm(vec))
end

@inline flow_speed(u::SVector{3, Float64}, p::SVector{2, Float64}) = begin
    alpha, lambda = p
    x, y, z = u
    dx = y
    dy = x - lambda * y - x * z
    dz = -alpha * (z - x * x)
    sqrt(dx * dx + dy * dy + dz * dz)
end

@inline zprime_proxy(u::SVector{3, Float64}) = u[3] - u[1] * u[1]

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

    a =
        (t2 * (z1 - z0) + t1 * (z0 - z2) + t0 * (z2 - z1)) / denom
    b =
        (t2^2 * (z0 - z1) + t1^2 * (z2 - z0) + t0^2 * (z1 - z2)) / denom

    abs(a) <= 1e-14 && return nothing
    t_vertex = -b / (2a)
    return min(t0, t2) <= t_vertex <= max(t0, t2) ? t_vertex : nothing
end

function quadratic_interp(
    t::Float64,
    t0::Float64,
    z0::Float64,
    t1::Float64,
    z1::Float64,
    t2::Float64,
    z2::Float64,
)
    l0 = ((t - t1) * (t - t2)) / ((t0 - t1) * (t0 - t2))
    l1 = ((t - t0) * (t - t2)) / ((t1 - t0) * (t1 - t2))
    l2 = ((t - t0) * (t - t1)) / ((t2 - t0) * (t2 - t1))
    return z0 * l0 + z1 * l1 + z2 * l2
end

@inline function pack_result(
    alpha::Float64,
    lambda::Float64,
    zmax_values::Vector{Float64},
    return_times::Vector{Float64},
    blue_mask::UInt16,
    red_mask::UInt16,
    status::String,
)
    return SMScanResult21(alpha, lambda, length(zmax_values), blue_mask, red_mask, zmax_values, return_times, status)
end

function scan_orbit(alpha::Float64, lambda::Float64)::SMScanResult21
    p = @SVector [alpha, lambda]
    u0 = unstable_side_initial_condition(lambda)
    prob = ODEProblem(shimizu_morioka!, collect(u0), (0.0, ATTEMPT021_T_END), collect(p))
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT021_DT,
        dtmax=ATTEMPT021_DT,
        abstol=ATTEMPT021_ABSTOL,
        reltol=ATTEMPT021_RELTOL,
        maxiters=ATTEMPT021_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    zmax_values = Float64[]
    return_times = Float64[]
    sizehint!(zmax_values, ATTEMPT021_MAX_ZMAX)
    sizehint!(return_times, ATTEMPT021_MAX_ZMAX)

    blue_mask = UInt16(0)
    red_mask = UInt16(0)

    u_prev = SVector{3, Float64}(integ.u)
    t_prev = integ.t
    g_prev = zprime_proxy(u_prev)
    seg_min_speed = flow_speed(u_prev, p)
    seg_min_radius = norm(u_prev)
    status = "ok"
    have_prevprev = false
    u_prevprev = u_prev
    t_prevprev = t_prev

    while integ.t < ATTEMPT021_T_END && length(zmax_values) < ATTEMPT021_MAX_ZMAX
        step!(integ)
        u_curr = SVector{3, Float64}(integ.u)
        t_curr = integ.t

        if !all(isfinite, u_curr)
            status = "nonfinite"
            break
        end
        if maximum(abs, u_curr) > ATTEMPT021_MAX_STATE
            status = "blowup"
            break
        end

        speed = flow_speed(u_curr, p)
        radius = norm(u_curr)
        if speed < seg_min_speed
            seg_min_speed = speed
            seg_min_radius = radius
        end

        g_curr = zprime_proxy(u_curr)
        if g_prev <= 0.0 && g_curr > 0.0
            θ = iszero(g_curr - g_prev) ? 0.5 : clamp(-g_prev / (g_curr - g_prev), 0.0, 1.0)
            t_hit = t_prev + θ * (t_curr - t_prev)
            z_hit = u_prev[3] + θ * (u_curr[3] - u_prev[3])
            if have_prevprev
                t_vertex = quadratic_vertex_time(t_prevprev, u_prevprev[3], t_prev, u_prev[3], t_curr, u_curr[3])
                if !isnothing(t_vertex)
                    t_hit = t_vertex
                    z_hit = quadratic_interp(t_vertex, t_prevprev, u_prevprev[3], t_prev, u_prev[3], t_curr, u_curr[3])
                end
            end
            push!(return_times, t_hit)
            push!(zmax_values, z_hit)

            seg_idx = length(zmax_values)
            if seg_min_speed <= ATTEMPT021_NEAR_SADDLE_SPEED && seg_min_radius <= ATTEMPT021_NEAR_SADDLE_RADIUS
                blue_mask |= UInt16(1) << (seg_idx - 1)
            end

            if length(zmax_values) >= 3
                a = zmax_values[end - 2]
                b = zmax_values[end - 1]
                c = zmax_values[end]
                rise = b - a
                fall = c - b
                if rise * fall <= 0.0 && max(abs(rise), abs(fall)) > ATTEMPT021_RED_EPS
                    red_mask |= UInt16(1) << (length(zmax_values) - 2)
                end
            end

            seg_min_speed = speed
            seg_min_radius = radius
        end

        have_prevprev = true
        u_prevprev = u_prev
        t_prevprev = t_prev
        u_prev = u_curr
        t_prev = t_curr
        g_prev = g_curr
    end

    if status == "ok" && length(zmax_values) < ATTEMPT021_MAX_ZMAX
        status = "short"
    end

    return pack_result(alpha, lambda, zmax_values, return_times, blue_mask, red_mask, status)
end

join_float_list(values::Vector{Float64}) = join((@sprintf("%.9f", v) for v in values), ",")

function parse_float_list(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end
