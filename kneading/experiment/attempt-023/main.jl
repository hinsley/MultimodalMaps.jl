using Pkg

const ATTEMPT23_ROOT = @__DIR__
const REPO_ROOT_023 = normpath(joinpath(ATTEMPT23_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_023)

using CairoMakie
using DifferentialEquations
using LinearAlgebra
using Printf
using StaticArrays

const ATTEMPT023_N_ALPHA = parse(Int, get(ENV, "ATTEMPT023_N_ALPHA", "5000"))
const ATTEMPT023_N_LAMBDA = parse(Int, get(ENV, "ATTEMPT023_N_LAMBDA", "5000"))
const ATTEMPT023_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT023_ALPHA_MIN", "0.0"))
const ATTEMPT023_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT023_ALPHA_MAX", "0.7"))
const ATTEMPT023_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT023_LAMBDA_MIN", "0.2"))
const ATTEMPT023_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT023_LAMBDA_MAX", "1.6"))
const ATTEMPT023_MAX_ZMAX = parse(Int, get(ENV, "ATTEMPT023_MAX_ZMAX", "16"))
const ATTEMPT023_T_END = parse(Float64, get(ENV, "ATTEMPT023_T_END", "700.0"))
const ATTEMPT023_DT = parse(Float64, get(ENV, "ATTEMPT023_DT", "0.02"))
const ATTEMPT023_ABSTOL = parse(Float64, get(ENV, "ATTEMPT023_ABSTOL", "1e-9"))
const ATTEMPT023_RELTOL = parse(Float64, get(ENV, "ATTEMPT023_RELTOL", "1e-9"))
const ATTEMPT023_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT023_MAX_ITERS", "40000000"))
const ATTEMPT023_EPS0 = parse(Float64, get(ENV, "ATTEMPT023_EPS0", "1e-7"))
const ATTEMPT023_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT023_MAX_STATE", "200.0"))
const ATTEMPT023_RED_EPS = parse(Float64, get(ENV, "ATTEMPT023_RED_EPS", "1e-6"))
const ATTEMPT023_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT023_LINEWIDTH", "0.9"))
const ATTEMPT023_FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT023_FIG_WIDTH", "1000"))
const ATTEMPT023_FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT023_FIG_HEIGHT", "1000"))
const ATTEMPT023_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT023_PX_PER_UNIT", "4.0"))
const ATTEMPT023_COLOR_SEED = parse(Int, get(ENV, "ATTEMPT023_COLOR_SEED", "21"))

const ALPHAS_023 = collect(range(ATTEMPT023_ALPHA_MIN, ATTEMPT023_ALPHA_MAX, length=ATTEMPT023_N_ALPHA))
const LAMBDAS_023 = collect(range(ATTEMPT023_LAMBDA_MIN, ATTEMPT023_LAMBDA_MAX, length=ATTEMPT023_N_LAMBDA))

struct SMCriticalityResult23
    alpha::Float64
    lambda::Float64
    maxima_count::Int
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

@inline function unstable_side_initial_condition(lambda::Float64; eps0::Float64=ATTEMPT023_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / norm(vec))
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
    a = (t2 * (z1 - z0) + t1 * (z0 - z2) + t0 * (z2 - z1)) / denom
    b = (t2^2 * (z0 - z1) + t1^2 * (z2 - z0) + t0^2 * (z1 - z2)) / denom
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

function scan_orbit(alpha::Float64, lambda::Float64)::SMCriticalityResult23
    p = @SVector [alpha, lambda]
    u0 = unstable_side_initial_condition(lambda)
    prob = ODEProblem(shimizu_morioka!, collect(u0), (0.0, ATTEMPT023_T_END), collect(p))
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT023_DT,
        dtmax=ATTEMPT023_DT,
        abstol=ATTEMPT023_ABSTOL,
        reltol=ATTEMPT023_RELTOL,
        maxiters=ATTEMPT023_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    zmax_values = Float64[]
    return_times = Float64[]
    sizehint!(zmax_values, ATTEMPT023_MAX_ZMAX)
    sizehint!(return_times, ATTEMPT023_MAX_ZMAX)
    red_mask = UInt16(0)
    status = "ok"

    u_prev = SVector{3, Float64}(integ.u)
    t_prev = integ.t
    g_prev = zprime_proxy(u_prev)
    have_prevprev = false
    u_prevprev = u_prev
    t_prevprev = t_prev

    while integ.t < ATTEMPT023_T_END && length(zmax_values) < ATTEMPT023_MAX_ZMAX
        step!(integ)
        u_curr = SVector{3, Float64}(integ.u)
        t_curr = integ.t

        if !all(isfinite, u_curr)
            status = "nonfinite"
            break
        end
        if maximum(abs, u_curr) > ATTEMPT023_MAX_STATE
            status = "blowup"
            break
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

            if length(zmax_values) >= 3
                a = zmax_values[end - 2]
                b = zmax_values[end - 1]
                c = zmax_values[end]
                rise = b - a
                fall = c - b
                if rise * fall <= 0.0 && max(abs(rise), abs(fall)) > ATTEMPT023_RED_EPS
                    red_mask |= UInt16(1) << (length(zmax_values) - 2)
                end
            end
        end

        have_prevprev = true
        u_prevprev = u_prev
        t_prevprev = t_prev
        u_prev = u_curr
        t_prev = t_curr
        g_prev = g_curr
    end

    if status == "ok" && length(zmax_values) < ATTEMPT023_MAX_ZMAX
        status = "short"
    end

    return SMCriticalityResult23(alpha, lambda, length(zmax_values), red_mask, zmax_values, return_times, status)
end

join_float_list(values::Vector{Float64}) = join((@sprintf("%.9f", v) for v in values), ",")

function parse_float_list(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end
