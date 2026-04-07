using Pkg

const ATTEMPT028_ROOT = @__DIR__
const REPO_ROOT_028 = normpath(joinpath(ATTEMPT028_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_028)

using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using Printf
using SciMLSensitivity
using StaticArrays

const ATTEMPT028_ALPHA = parse(Float64, get(ENV, "ATTEMPT028_ALPHA", "0.4"))
const ATTEMPT028_LAMBDA = parse(Float64, get(ENV, "ATTEMPT028_LAMBDA", "0.7"))
const ATTEMPT028_B = 0.0

const ATTEMPT028_SAMPLE_T_END = parse(Float64, get(ENV, "ATTEMPT028_SAMPLE_T_END", "20000.0"))
const ATTEMPT028_SAMPLE_DT = parse(Float64, get(ENV, "ATTEMPT028_SAMPLE_DT", "0.02"))
const ATTEMPT028_SAMPLE_ABSTOL = parse(Float64, get(ENV, "ATTEMPT028_SAMPLE_ABSTOL", "1e-11"))
const ATTEMPT028_SAMPLE_RELTOL = parse(Float64, get(ENV, "ATTEMPT028_SAMPLE_RELTOL", "1e-11"))
const ATTEMPT028_SAMPLE_MAXITERS = parse(Int, get(ENV, "ATTEMPT028_SAMPLE_MAXITERS", "80000000"))
const ATTEMPT028_MAX_SAMPLE_EVENTS = parse(Int, get(ENV, "ATTEMPT028_MAX_SAMPLE_EVENTS", "20000"))
const ATTEMPT028_TRANSIENT_EVENTS = parse(Int, get(ENV, "ATTEMPT028_TRANSIENT_EVENTS", "1000"))

const ATTEMPT028_EVENT_T_END = parse(Float64, get(ENV, "ATTEMPT028_EVENT_T_END", "20.0"))
const ATTEMPT028_EVENT_DT = parse(Float64, get(ENV, "ATTEMPT028_EVENT_DT", "0.01"))
const ATTEMPT028_EVENT_ABSTOL = parse(Float64, get(ENV, "ATTEMPT028_EVENT_ABSTOL", "1e-12"))
const ATTEMPT028_EVENT_RELTOL = parse(Float64, get(ENV, "ATTEMPT028_EVENT_RELTOL", "1e-12"))
const ATTEMPT028_EVENT_MAXITERS = parse(Int, get(ENV, "ATTEMPT028_EVENT_MAXITERS", "20000000"))
const ATTEMPT028_MIN_EVENT_TIME = parse(Float64, get(ENV, "ATTEMPT028_MIN_EVENT_TIME", "0.5"))
const ATTEMPT028_EPS0 = parse(Float64, get(ENV, "ATTEMPT028_EPS0", "1e-7"))
const ATTEMPT028_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT028_MAX_STATE", "200.0"))
const ATTEMPT028_MIN_SECTION_X = parse(Float64, get(ENV, "ATTEMPT028_MIN_SECTION_X", "1e-5"))
const ATTEMPT028_TARGET_NEXT_SIGN = parse(Int, get(ENV, "ATTEMPT028_TARGET_NEXT_SIGN", "-1"))
const ATTEMPT028_TARGET_EXTREMUM_RAW = lowercase(get(ENV, "ATTEMPT028_TARGET_EXTREMUM", "minimum"))

const ATTEMPT028_FIT_WINDOW = parse(Int, get(ENV, "ATTEMPT028_FIT_WINDOW", "15"))
const ATTEMPT028_FIT_DEGREE = parse(Int, get(ENV, "ATTEMPT028_FIT_DEGREE", "4"))

const ATTEMPT028_NEWTON_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT028_NEWTON_MAX_ITERS", "8"))
const ATTEMPT028_NEWTON_GRAD_TOL = parse(Float64, get(ENV, "ATTEMPT028_NEWTON_GRAD_TOL", "1e-10"))
const ATTEMPT028_NEWTON_STEP_TOL = parse(Float64, get(ENV, "ATTEMPT028_NEWTON_STEP_TOL", "1e-12"))
const ATTEMPT028_NEWTON_MIN_DAMPING = parse(Float64, get(ENV, "ATTEMPT028_NEWTON_MIN_DAMPING", "0.015625"))

const ATTEMPT028_EVENTS_PATH = joinpath(
    ATTEMPT028_ROOT,
    "alpha0p4_lambda0p7_B0_absxmax_events.tsv",
)
const ATTEMPT028_BRANCH_PATH = joinpath(
    ATTEMPT028_ROOT,
    "alpha0p4_lambda0p7_B0_positive_absx_nextneg_branch.tsv",
)
const ATTEMPT028_TRACE_PATH = joinpath(
    ATTEMPT028_ROOT,
    "alpha0p4_lambda0p7_B0_hook_newton_trace.tsv",
)
const ATTEMPT028_SUMMARY_PATH = joinpath(
    ATTEMPT028_ROOT,
    "alpha0p4_lambda0p7_B0_hook_newton_summary.md",
)

@inline function target_is_minimum_028()
    if ATTEMPT028_TARGET_EXTREMUM_RAW in ("minimum", "min")
        return true
    elseif ATTEMPT028_TARGET_EXTREMUM_RAW in ("maximum", "max")
        return false
    end
    error("ATTEMPT028_TARGET_EXTREMUM must be one of: minimum, min, maximum, max")
end

@inline target_extremum_label_028() = target_is_minimum_028() ? "minimum" : "maximum"
@inline target_curvature_sign_028() = target_is_minimum_028() ? 1.0 : -1.0

struct BranchPoint028
    source_index::Int
    event_time::Float64
    x::Float64
    z::Float64
    s::Float64
    next_r::Float64
    next_x::Float64
end

struct LocalPolyFit028
    shift::Float64
    coeffs::Vector{Float64}
    xmin::Float64
    xmax::Float64
end

struct NaturalSpline028
    xs::Vector{Float64}
    ys::Vector{Float64}
    second_derivatives::Vector{Float64}
end

struct ReturnMapEval028
    s::Float64
    value::Float64
    first_derivative::Float64
    second_derivative::Float64
    event_time::Float64
    event_state::SVector{3, Float64}
    current_state::SVector{3, Float64}
    current_state_d1::SVector{3, Float64}
    current_state_d2::SVector{3, Float64}
    event_time_d1::Float64
    event_time_d2::Float64
    sciml_first_derivative::Float64
    first_derivative_mismatch::Float64
    fit_xmin::Float64
    fit_xmax::Float64
end

@inline dual_strip_028(x::Real) = x
@inline dual_strip_028(x::ForwardDiff.Dual) = ForwardDiff.value(x)

struct NewtonTrace028
    iter::Int
    s::Float64
    value::Float64
    first_derivative::Float64
    second_derivative::Float64
    step::Float64
    damping::Float64
    accepted::Bool
    sciml_first_derivative::Float64
    first_derivative_mismatch::Float64
end

@inline function shimizu_morioka_vector_028(u::SVector{3, TU}, alpha::TA, lambda::TL) where {TU<:Real, TA<:Real, TL<:Real}
    T = promote_type(TU, TA, TL)
    x = T(u[1])
    y = T(u[2])
    z = T(u[3])
    α = T(alpha)
    λ = T(lambda)
    return @SVector [
        y,
        x - λ * y - x * z - T(ATTEMPT028_B) * x^3,
        -α * (z - x * x),
    ]
end

@inline function shimizu_morioka_jacobian_028(u::SVector{3, Float64}, alpha::Float64, lambda::Float64)
    x, _, z = u
    return @SMatrix [
        0.0 1.0 0.0
        1.0 - z - 3.0 * ATTEMPT028_B * x * x -lambda -x
        2.0 * alpha * x 0.0 -alpha
    ]
end

@inline function shimizu_morioka_state_028!(du, u, p, t)
    alpha = p[1]
    lambda = p[2]
    x = u[1]
    y = u[2]
    z = u[3]
    du[1] = y
    du[2] = x - lambda * y - x * z - ATTEMPT028_B * x^3
    du[3] = -alpha * (z - x * x)
    return nothing
end

@inline function unstable_side_initial_condition_028(lambda::Float64; eps0::Float64=ATTEMPT028_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / norm(vec))
end

function collect_absxmax_events_028(alpha::Float64, lambda::Float64)
    u0 = unstable_side_initial_condition_028(lambda)
    p = [alpha, lambda]
    times = Float64[]
    states = SVector{3, Float64}[]
    sizehint!(times, ATTEMPT028_MAX_SAMPLE_EVENTS)
    sizehint!(states, ATTEMPT028_MAX_SAMPLE_EVENTS)

    function condition(u, t, integrator)
        return t <= ATTEMPT028_MIN_EVENT_TIME ? 1.0 : u[2]
    end

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        if abs(x) > ATTEMPT028_MIN_SECTION_X && z > 1.0
            push!(times, integrator.t)
            push!(states, SVector{3, Float64}(integrator.u))
            if length(times) >= ATTEMPT028_MAX_SAMPLE_EVENTS
                terminate!(integrator)
            end
        end
    end

    cb = ContinuousCallback(
        condition,
        affect!;
        rootfind=true,
        save_positions=(false, false),
    )
    prob = ODEProblem(
        shimizu_morioka_state_028!,
        collect(u0),
        (0.0, ATTEMPT028_SAMPLE_T_END),
        p,
    )
    solve(
        prob,
        Vern9();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT028_SAMPLE_DT,
        dtmax=ATTEMPT028_SAMPLE_DT,
        abstol=ATTEMPT028_SAMPLE_ABSTOL,
        reltol=ATTEMPT028_SAMPLE_RELTOL,
        maxiters=ATTEMPT028_SAMPLE_MAXITERS,
        save_start=false,
        save_end=false,
        save_everystep=false,
    )

    return times, states
end

function write_events_tsv_028(path::String, times::Vector{Float64}, states::Vector{SVector{3, Float64}})
    open(path, "w") do io
        println(io, "event_index\ttime\tx\ty\tz")
        for i in eachindex(times)
            state = states[i]
            @printf(io, "%d\t%.12f\t%.12f\t%.12f\t%.12f\n", i, times[i], state[1], state[2], state[3])
        end
    end
end

function build_positive_branch_028(times::Vector{Float64}, states::Vector{SVector{3, Float64}})
    branch = BranchPoint028[]
    sizehint!(branch, length(states))
    for i in (ATTEMPT028_TRANSIENT_EVENTS + 1):(length(states) - 1)
        current = states[i]
        next_state = states[i + 1]
        if current[1] > ATTEMPT028_MIN_SECTION_X
            s = current[1]^2
            next_r = next_state[1]^2
            push!(branch, BranchPoint028(i, times[i], current[1], current[3], s, next_r, next_state[1]))
        end
    end
    sort!(branch, by=point -> point.s)
    return branch
end

function filter_branch_by_next_sign_028(branch::Vector{BranchPoint028}, target_sign::Int)
    target_sign in (-1, 1) || error("ATTEMPT028_TARGET_NEXT_SIGN must be ±1")
    if target_sign < 0
        filtered = [point for point in branch if point.next_x < -ATTEMPT028_MIN_SECTION_X]
    else
        filtered = [point for point in branch if point.next_x > ATTEMPT028_MIN_SECTION_X]
    end
    sort!(filtered, by=point -> point.s)
    return filtered
end

function write_branch_tsv_028(path::String, branch::Vector{BranchPoint028})
    open(path, "w") do io
        println(io, "sorted_index\tsource_event_index\tevent_time\tx\tz\ts\tnext_r\tnext_x")
        for (i, point) in enumerate(branch)
            @printf(
                io,
                "%d\t%d\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\n",
                i,
                point.source_index,
                point.event_time,
                point.x,
                point.z,
                point.s,
                point.next_r,
                point.next_x,
            )
        end
    end
end

function build_natural_spline_028(xs::Vector{Float64}, ys::Vector{Float64})
    length(xs) == length(ys) || error("Spline x/y lengths must match")
    length(xs) >= 3 || error("Need at least 3 points for a spline")
    all(diff(xs) .> 0.0) || error("Spline abscissae must be strictly increasing")

    n = length(xs)
    lower = zeros(Float64, n - 1)
    diag = zeros(Float64, n)
    upper = zeros(Float64, n - 1)
    rhs = zeros(Float64, n)

    diag[1] = 1.0
    diag[end] = 1.0
    for i in 2:(n - 1)
        h_prev = xs[i] - xs[i - 1]
        h_next = xs[i + 1] - xs[i]
        lower[i - 1] = h_prev
        diag[i] = 2.0 * (h_prev + h_next)
        upper[i] = h_next
        rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h_next - (ys[i] - ys[i - 1]) / h_prev)
    end

    second_derivatives = Tridiagonal(lower, diag, upper) \ rhs
    return NaturalSpline028(xs, ys, second_derivatives)
end

function spline_interval_index_028(spline::NaturalSpline028, x::Real)
    xv = dual_strip_028(x)
    xv <= spline.xs[1] && return 1
    xv >= spline.xs[end] && return length(spline.xs) - 1
    idx = searchsortedlast(spline.xs, xv)
    return clamp(idx, 1, length(spline.xs) - 1)
end

function spline_eval_028(spline::NaturalSpline028, x::T) where {T<:Real}
    i = spline_interval_index_028(spline, x)
    x0 = spline.xs[i]
    x1 = spline.xs[i + 1]
    y0 = spline.ys[i]
    y1 = spline.ys[i + 1]
    m0 = spline.second_derivatives[i]
    m1 = spline.second_derivatives[i + 1]
    h = x1 - x0
    a = (x1 - x) / h
    b = (x - x0) / h
    return a * y0 + b * y1 + ((a^3 - a) * m0 + (b^3 - b) * m1) * h^2 / 6.0
end

function spline_derivative_028(spline::NaturalSpline028, x::T) where {T<:Real}
    i = spline_interval_index_028(spline, x)
    x0 = spline.xs[i]
    x1 = spline.xs[i + 1]
    y0 = spline.ys[i]
    y1 = spline.ys[i + 1]
    m0 = spline.second_derivatives[i]
    m1 = spline.second_derivatives[i + 1]
    h = x1 - x0
    a = (x1 - x) / h
    b = (x - x0) / h
    return (y1 - y0) / h + ((-3.0 * a^2 + 1.0) * m0 + (3.0 * b^2 - 1.0) * m1) * h / 6.0
end

function spline_second_derivative_028(spline::NaturalSpline028, x::T) where {T<:Real}
    i = spline_interval_index_028(spline, x)
    x0 = spline.xs[i]
    x1 = spline.xs[i + 1]
    h = x1 - x0
    a = (x1 - x) / h
    b = (x - x0) / h
    return a * spline.second_derivatives[i] + b * spline.second_derivatives[i + 1]
end

function fit_shifted_polynomial_028(xs::Vector{Float64}, ys::Vector{Float64}, x0::Float64)
    degree = min(ATTEMPT028_FIT_DEGREE, length(xs) - 1)
    degree >= 2 || error("Need at least 3 points for local polynomial fit")
    A = Matrix{Float64}(undef, length(xs), degree + 1)
    for i in eachindex(xs)
        ξ = xs[i] - x0
        A[i, 1] = 1.0
        for j in 1:degree
            A[i, j + 1] = ξ^j
        end
    end
    coeffs = A \ ys
    return LocalPolyFit028(x0, coeffs, minimum(xs), maximum(xs))
end

function nearest_window_fit_028(branch_s::Vector{Float64}, branch_y::Vector{Float64}, x0::Float64)
    order = sortperm(abs.(branch_s .- x0))
    count = min(ATTEMPT028_FIT_WINDOW, length(order))
    idx = sort(order[1:count])
    xs = branch_s[idx]
    ys = branch_y[idx]
    return fit_shifted_polynomial_028(xs, ys, x0)
end

@inline function polyval_shifted_028(fit::LocalPolyFit028, x::T) where {T<:Real}
    ξ = x - fit.shift
    value = zero(T)
    for coeff in reverse(fit.coeffs)
        value = value * ξ + coeff
    end
    return value
end

@inline function polyder1_shifted_028(fit::LocalPolyFit028, x::T) where {T<:Real}
    ξ = x - fit.shift
    value = zero(T)
    for k in reverse(2:length(fit.coeffs))
        value = value * ξ + (k - 1) * fit.coeffs[k]
    end
    return value
end

@inline function polyder2_shifted_028(fit::LocalPolyFit028, x::T) where {T<:Real}
    length(fit.coeffs) <= 2 && return zero(T)
    ξ = x - fit.shift
    value = zero(T)
    for k in reverse(3:length(fit.coeffs))
        value = value * ξ + (k - 1) * (k - 2) * fit.coeffs[k]
    end
    return value
end

function initial_guess_from_branch_028(branch::Vector{BranchPoint028})
    branch_s = [point.s for point in branch]
    branch_r = [point.next_r for point in branch]
    candidates = Int[]
    for i in 2:(length(branch) - 1)
        if target_is_minimum_028()
            if branch_r[i] <= branch_r[i - 1] && branch_r[i] <= branch_r[i + 1]
                push!(candidates, i)
            end
        elseif branch_r[i] >= branch_r[i - 1] && branch_r[i] >= branch_r[i + 1]
            push!(candidates, i)
        end
    end
    if isempty(candidates)
        center_index = target_is_minimum_028() ? argmin(branch_r) : argmax(branch_r)
    elseif target_is_minimum_028()
        center_index = first(candidates)
    else
        center_index = first(candidates)
    end
    s0 = branch_s[center_index]
    lo = max(1, center_index - 1)
    hi = min(length(branch), center_index + 1)
    if hi - lo + 1 >= 3
        quad_fit = fit_shifted_polynomial_028(branch_s[lo:hi], branch_r[lo:hi], s0)
        curvature_ok = target_is_minimum_028() ? quad_fit.coeffs[3] > 0.0 : quad_fit.coeffs[3] < 0.0
        if length(quad_fit.coeffs) >= 3 && curvature_ok
            vertex = s0 - quad_fit.coeffs[2] / (2.0 * quad_fit.coeffs[3])
            if quad_fit.xmin <= vertex <= quad_fit.xmax
                s0 = vertex
            end
        end
    end
    return s0, center_index
end

function section_curve_data_028(s::Float64, spline::NaturalSpline028)
    s > 0.0 || error("Section parameter s must stay positive")
    spline.xs[1] <= s <= spline.xs[end] || error("Section parameter outside sampled branch domain")
    x = sqrt(s)
    x > ATTEMPT028_MIN_SECTION_X || error("Section parameter too close to x = 0")
    current_state = @SVector [x, 0.0, spline_eval_028(spline, s)]
    current_d1 = @SVector [0.5 / x, 0.0, spline_derivative_028(spline, s)]
    current_d2 = @SVector [-0.25 / (x^3), 0.0, spline_second_derivative_028(spline, s)]
    return current_state, current_d1, current_d2
end

function shimizu_morioka_augmented_028!(du, u, p, t)
    alpha, lambda = p

    x = u[1]
    y = u[2]
    z = u[3]
    sx = u[4]
    sy = u[5]
    sz = u[6]
    qx = u[7]
    qy = u[8]
    qz = u[9]

    du[1] = y
    du[2] = x - lambda * y - x * z - ATTEMPT028_B * x^3
    du[3] = -alpha * (z - x * x)

    du[4] = sy
    du[5] = (1.0 - z - 3.0 * ATTEMPT028_B * x * x) * sx - lambda * sy - x * sz
    du[6] = 2.0 * alpha * x * sx - alpha * sz

    du[7] = qy
    du[8] = (1.0 - z - 3.0 * ATTEMPT028_B * x * x) * qx - lambda * qy - x * qz -
            (6.0 * ATTEMPT028_B * x * sx * sx + 2.0 * sx * sz)
    du[9] = 2.0 * alpha * (sx * sx + x * qx) - alpha * qz
    return nothing
end

function solve_next_absx_event_manual_028(
    alpha::Float64,
    lambda::Float64,
    current_state::SVector{3, Float64},
    current_d1::SVector{3, Float64},
    current_d2::SVector{3, Float64},
    target_next_sign::Int,
)
    u0 = [
        current_state[1],
        current_state[2],
        current_state[3],
        current_d1[1],
        current_d1[2],
        current_d1[3],
        current_d2[1],
        current_d2[2],
        current_d2[3],
    ]
    p = [alpha, lambda]
    ATTEMPT028_MIN_EVENT_TIME < ATTEMPT028_EVENT_T_END || error("ATTEMPT028_MIN_EVENT_TIME must stay below ATTEMPT028_EVENT_T_END")
    hit = Ref(false)
    armed = Ref(false)

    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        sign_ok = target_next_sign < 0 ? x < -ATTEMPT028_MIN_SECTION_X : x > ATTEMPT028_MIN_SECTION_X
        if sign_ok && z > 1.0
            hit[] = true
            terminate!(integrator)
        end
    end

    arm_cb = PresetTimeCallback([ATTEMPT028_MIN_EVENT_TIME], integrator -> begin
        armed[] = true
    end)
    cont_cb = ContinuousCallback(
        condition,
        affect!;
        rootfind=true,
        save_positions=(false, false),
    )
    cb = CallbackSet(arm_cb, cont_cb)

    prob = ODEProblem(
        shimizu_morioka_augmented_028!,
        u0,
        (0.0, ATTEMPT028_EVENT_T_END),
        p,
    )
    sol = solve(
        prob,
        Vern9();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT028_EVENT_DT,
        dtmax=ATTEMPT028_EVENT_DT,
        abstol=ATTEMPT028_EVENT_ABSTOL,
        reltol=ATTEMPT028_EVENT_RELTOL,
        maxiters=ATTEMPT028_EVENT_MAXITERS,
        save_start=false,
        save_end=true,
        save_everystep=false,
    )

    hit[] || error("No negative-branch |x|-maximum found before ATTEMPT028_EVENT_T_END")

    u_end = sol.u[end]
    event_state = SVector{3, Float64}(u_end[1:3])
    S1 = SVector{3, Float64}(u_end[4:6])
    S2 = SVector{3, Float64}(u_end[7:9])
    event_time = sol.t[end]
    maximum(abs, event_state) <= ATTEMPT028_MAX_STATE || error("State blew up during event solve")
    return event_state, S1, S2, event_time
end

function solve_next_absx_event_sciml_028(
    alpha::Float64,
    lambda::Float64,
    s::Float64,
    spline::NaturalSpline028,
    target_next_sign::Int,
)
    function current_state_from_parameter(param_s::T) where {T<:Real}
        x = sqrt(param_s)
        return @SVector [x, zero(T), spline_eval_028(spline, param_s)]
    end

    function current_state_derivative(param_s::T) where {T<:Real}
        x = sqrt(param_s)
        return @SVector [T(0.5) / x, zero(T), spline_derivative_028(spline, param_s)]
    end

    function transformed_flow!(du, w, p, t)
        α = p[1]
        λ = p[2]
        param_s = p[3]
        current_state = current_state_from_parameter(param_s)
        u = @SVector [w[1] + current_state[1], w[2], w[3] + current_state[3]]
        flow = shimizu_morioka_vector_028(u, α, λ)
        du[1] = flow[1]
        du[2] = flow[2]
        du[3] = flow[3]
        return nothing
    end

    p = [alpha, lambda, s]
    ATTEMPT028_MIN_EVENT_TIME < ATTEMPT028_EVENT_T_END || error("ATTEMPT028_MIN_EVENT_TIME must stay below ATTEMPT028_EVENT_T_END")
    hit = Ref(false)
    armed = Ref(false)

    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        current_state = current_state_from_parameter(integrator.p[3])
        x = integrator.u[1] + current_state[1]
        z = integrator.u[3] + current_state[3]
        sign_ok = target_next_sign < 0 ? x < -ATTEMPT028_MIN_SECTION_X : x > ATTEMPT028_MIN_SECTION_X
        if sign_ok && z > 1.0
            hit[] = true
            terminate!(integrator)
        end
    end

    arm_cb = PresetTimeCallback([ATTEMPT028_MIN_EVENT_TIME], integrator -> begin
        armed[] = true
    end)
    cont_cb = ContinuousCallback(
        condition,
        affect!;
        rootfind=true,
        save_positions=(false, false),
    )
    cb = CallbackSet(arm_cb, cont_cb)
    prob = ODEForwardSensitivityProblem(
        transformed_flow!,
        zeros(3),
        (0.0, ATTEMPT028_EVENT_T_END),
        p,
        ForwardSensitivity(),
    )
    sol = solve(
        prob,
        Vern9();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT028_EVENT_DT,
        dtmax=ATTEMPT028_EVENT_DT,
        abstol=ATTEMPT028_EVENT_ABSTOL,
        reltol=ATTEMPT028_EVENT_RELTOL,
        maxiters=ATTEMPT028_EVENT_MAXITERS,
        save_start=false,
        save_end=true,
        save_everystep=false,
    )

    hit[] || error("SciMLSensitivity solve did not hit the next |x|-maximum")

    w_final, sensitivity_matrix = extract_local_sensitivities(sol, length(sol.u), Val(true))
    current_state = current_state_from_parameter(s)
    current_state_d1 = current_state_derivative(s)
    physical_state = @SVector [
        w_final[1] + current_state[1],
        w_final[2],
        w_final[3] + current_state[3],
    ]
    physical_d1 = @SVector [
        sensitivity_matrix[1, 3] + current_state_d1[1],
        sensitivity_matrix[2, 3],
        sensitivity_matrix[3, 3] + current_state_d1[3],
    ]
    return physical_state, physical_d1
end

function evaluate_return_map_028(alpha::Float64, lambda::Float64, s::Float64, spline::NaturalSpline028, target_next_sign::Int)
    current_state, current_d1, current_d2 = section_curve_data_028(s, spline)
    event_state, S1, S2, event_time = solve_next_absx_event_manual_028(
        alpha,
        lambda,
        current_state,
        current_d1,
        current_d2,
        target_next_sign,
    )

    flow = shimizu_morioka_vector_028(event_state, alpha, lambda)
    jacobian = shimizu_morioka_jacobian_028(event_state, alpha, lambda)
    JS1 = jacobian * S1
    Jf = jacobian * flow

    denom = flow[2]
    abs(denom) > 1e-12 || error("Event-time derivative denominator nearly singular")

    event_time_d1 = -S1[2] / denom
    event_time_d2 = -(S2[2] + 2.0 * JS1[2] * event_time_d1 + Jf[2] * event_time_d1^2) / denom

    event_state_d1 = S1 + flow * event_time_d1
    event_state_d2 = S2 + 2.0 * JS1 * event_time_d1 + Jf * event_time_d1^2 + flow * event_time_d2

    value = event_state[1]^2
    first_derivative = 2.0 * event_state[1] * event_state_d1[1]
    second_derivative = 2.0 * event_state_d1[1]^2 + 2.0 * event_state[1] * event_state_d2[1]

    sciml_state, sciml_state_d1 = solve_next_absx_event_sciml_028(alpha, lambda, s, spline, target_next_sign)
    sciml_flow = shimizu_morioka_vector_028(sciml_state, alpha, lambda)
    sciml_event_time_d1 = -sciml_state_d1[2] / sciml_flow[2]
    sciml_event_state_d1 = sciml_state_d1 + sciml_flow * sciml_event_time_d1
    sciml_first_derivative = 2.0 * sciml_state[1] * sciml_event_state_d1[1]
    first_derivative_mismatch = abs(first_derivative - sciml_first_derivative)

    return ReturnMapEval028(
        s,
        value,
        first_derivative,
        second_derivative,
        event_time,
        event_state,
        current_state,
        current_d1,
        current_d2,
        event_time_d1,
        event_time_d2,
        sciml_first_derivative,
        first_derivative_mismatch,
        spline.xs[1],
        spline.xs[end],
    )
end

function run_damped_newton_028(alpha::Float64, lambda::Float64, s0::Float64, spline::NaturalSpline028, target_next_sign::Int)
    trace = NewtonTrace028[]
    current_s = s0
    domain_min = spline.xs[1]
    domain_max = spline.xs[end]

    for iter in 1:ATTEMPT028_NEWTON_MAX_ITERS
        evaluation = evaluate_return_map_028(alpha, lambda, current_s, spline, target_next_sign)
        gradient = evaluation.first_derivative
        hessian = evaluation.second_derivative

        if abs(gradient) <= ATTEMPT028_NEWTON_GRAD_TOL
            target_curvature_sign_028() * hessian > 0.0 || error("Newton landed on the wrong extremum type at s=$(current_s)")
            push!(
                trace,
                NewtonTrace028(
                    iter,
                    current_s,
                    evaluation.value,
                    gradient,
                    hessian,
                    0.0,
                    1.0,
                    true,
                    evaluation.sciml_first_derivative,
                    evaluation.first_derivative_mismatch,
                ),
            )
            return trace, evaluation
        end

        abs(hessian) > 1e-12 || error("Second derivative nearly singular during Newton refinement")
        raw_step = -gradient / hessian
        damping = 1.0
        accepted = false
        candidate_s = current_s

        while damping >= ATTEMPT028_NEWTON_MIN_DAMPING
            candidate_s = current_s + damping * raw_step
            if !(domain_min < candidate_s < domain_max)
                damping *= 0.5
                continue
            end

            try
                candidate_eval = evaluate_return_map_028(alpha, lambda, candidate_s, spline, target_next_sign)
                if abs(candidate_eval.first_derivative) < abs(gradient) &&
                   target_curvature_sign_028() * candidate_eval.second_derivative > 0.0 &&
                   candidate_s >= candidate_eval.fit_xmin &&
                   candidate_s <= candidate_eval.fit_xmax
                    accepted = true
                    break
                end
            catch
            end
            damping *= 0.5
        end

        push!(
            trace,
            NewtonTrace028(
                iter,
                current_s,
                evaluation.value,
                gradient,
                hessian,
                raw_step,
                damping,
                accepted,
                evaluation.sciml_first_derivative,
                evaluation.first_derivative_mismatch,
            ),
        )

        accepted || error("Damped Newton failed to find an acceptable step from s=$(current_s)")

        current_s = candidate_s

        if abs(damping * raw_step) <= ATTEMPT028_NEWTON_STEP_TOL
            final_eval = evaluate_return_map_028(alpha, lambda, current_s, spline, target_next_sign)
            return trace, final_eval
        end
    end

    final_eval = evaluate_return_map_028(alpha, lambda, current_s, spline, target_next_sign)
    return trace, final_eval
end

function write_trace_tsv_028(path::String, trace::Vector{NewtonTrace028}, final_eval::ReturnMapEval028)
    open(path, "w") do io
        println(io, "iter\ts\tmap_value\tfirst_derivative\tsecond_derivative\traw_step\tdamping\taccepted\tsciml_first_derivative\tfirst_derivative_mismatch")
        for row in trace
            @printf(
                io,
                "%d\t%.12f\t%.12f\t%.12e\t%.12e\t%.12e\t%.6f\t%s\t%.12e\t%.12e\n",
                row.iter,
                row.s,
                row.value,
                row.first_derivative,
                row.second_derivative,
                row.step,
                row.damping,
                row.accepted ? "true" : "false",
                row.sciml_first_derivative,
                row.first_derivative_mismatch,
            )
        end
        println(io)
        println(io, "# final")
        @printf(io, "# s=%.12f\n", final_eval.s)
        @printf(io, "# map_value=%.12f\n", final_eval.value)
        @printf(io, "# first_derivative=%.12e\n", final_eval.first_derivative)
        @printf(io, "# second_derivative=%.12e\n", final_eval.second_derivative)
        @printf(io, "# event_time=%.12f\n", final_eval.event_time)
        @printf(io, "# event_state=(%.12f, %.12f, %.12f)\n", final_eval.event_state[1], final_eval.event_state[2], final_eval.event_state[3])
        @printf(io, "# current_state=(%.12f, %.12f, %.12f)\n", final_eval.current_state[1], final_eval.current_state[2], final_eval.current_state[3])
        @printf(io, "# event_time_d1=%.12e\n", final_eval.event_time_d1)
        @printf(io, "# event_time_d2=%.12e\n", final_eval.event_time_d2)
        @printf(io, "# sciml_first_derivative=%.12e\n", final_eval.sciml_first_derivative)
        @printf(io, "# first_derivative_mismatch=%.12e\n", final_eval.first_derivative_mismatch)
    end
end

function write_summary_md_028(
    path::String,
    branch::Vector{BranchPoint028},
    initial_guess::Float64,
    final_eval::ReturnMapEval028,
    trace::Vector{NewtonTrace028},
)
    branch_s = [point.s for point in branch]
    branch_r = [point.next_r for point in branch]
    ratios = Float64[]
    for i in 1:(length(trace) - 1)
        current_grad = abs(trace[i].first_derivative)
        next_grad = abs(trace[i + 1].first_derivative)
        if current_grad > 0.0
            push!(ratios, next_grad / current_grad^2)
        end
    end

    open(path, "w") do io
        println(io, "# Attempt 028 Hook Newton Test")
        println(io)
        println(io, "Fixed parameters:")
        println(io)
        @printf(io, "- `alpha = %.6f`\n", ATTEMPT028_ALPHA)
        @printf(io, "- `lambda = %.6f`\n", ATTEMPT028_LAMBDA)
        @printf(io, "- `B = %.1f`\n", ATTEMPT028_B)
        println(io)
        println(io, "Method summary:")
        println(io)
        println(io, "- Collected a long `|x|`-maxima orbit from one unstable-manifold branch of the origin.")
        println(io, "- Kept positive-branch maxima and filtered to the `next_x < 0` hook subbranch before building the sampled map `s_n = x_n^2`, `F(s_n) = x_{n+1}^2`.")
        @printf(io, "- Chose an initial hook guess from the sampled branch near the first discrete local %s of `F` along the filtered subbranch.\n", target_extremum_label_028())
        println(io, "- Parameterized the local `y = 0` section curve as `(x, y, z) = (sqrt(s), 0, z(s))` via an exact natural cubic spline through the sampled subbranch.")
        println(io, "- Used event-defined derivatives with return-time shift for the next `|x|`-maximum.")
        println(io, "- Cross-checked the first derivative against `SciMLSensitivity.ODEForwardSensitivityProblem`.")
        println(io)
        println(io, "Sample sizes:")
        println(io)
        @printf(io, "- Collected `|x|`-maxima: `%d`\n", ATTEMPT028_MAX_SAMPLE_EVENTS)
        @printf(io, "- Filtered `next_x < 0` subbranch points after transient cut: `%d`\n", length(branch))
        @printf(io, "- Filtered subbranch `s` range: `[%.9f, %.9f]`\n", minimum(branch_s), maximum(branch_s))
        @printf(io, "- Filtered subbranch `F(s)` range: `[%.9f, %.9f]`\n", minimum(branch_r), maximum(branch_r))
        println(io)
        println(io, "Newton result:")
        println(io)
        @printf(io, "- Target extremum type: `%s`\n", target_extremum_label_028())
        @printf(io, "- Initial guess `s0 = %.12f`\n", initial_guess)
        @printf(io, "- Final `s* = %.12f`\n", final_eval.s)
        @printf(io, "- Final `F(s*) = %.12f`\n", final_eval.value)
        @printf(io, "- Final `F'(s*) = %.12e`\n", final_eval.first_derivative)
        @printf(io, "- Final `F''(s*) = %.12e`\n", final_eval.second_derivative)
        @printf(io, "- Final next-event time `T(s*) = %.12f`\n", final_eval.event_time)
        @printf(io, "- Final SciMLSensitivity first-derivative mismatch `|F'_manual - F'_SciML| = %.12e`\n", final_eval.first_derivative_mismatch)
        println(io)
        println(io, "Quadratic-convergence diagnostic:")
        println(io)
        if isempty(ratios)
            println(io, "- Not enough accepted Newton steps to estimate `|F'_{k+1}| / |F'_k|^2`.")
        else
            @printf(io, "- Ratios `|F'_{k+1}| / |F'_k|^2`: `%s`\n", join((@sprintf("%.6e", r) for r in ratios), ", "))
        end
        println(io)
        println(io, "Outputs:")
        println(io)
        println(io, "- Sampled maxima TSV: `$(basename(ATTEMPT028_EVENTS_PATH))`")
        println(io, "- Positive branch TSV: `$(basename(ATTEMPT028_BRANCH_PATH))`")
        println(io, "- Newton trace TSV: `$(basename(ATTEMPT028_TRACE_PATH))`")
    end
end

function main()
    println("Attempt 028 hook Newton test")
    println("  alpha  = $(ATTEMPT028_ALPHA)")
    println("  lambda = $(ATTEMPT028_LAMBDA)")
    println("  B      = $(ATTEMPT028_B)")
    println("  target extremum = $(target_extremum_label_028())")
    println("  sample events target = $(ATTEMPT028_MAX_SAMPLE_EVENTS)")
    println()

    times, states = collect_absxmax_events_028(ATTEMPT028_ALPHA, ATTEMPT028_LAMBDA)
    length(times) >= ATTEMPT028_TRANSIENT_EVENTS + 50 || error("Not enough sampled |x|-maxima for attempt-028")
    write_events_tsv_028(ATTEMPT028_EVENTS_PATH, times, states)

    branch = build_positive_branch_028(times, states)
    branch = filter_branch_by_next_sign_028(branch, ATTEMPT028_TARGET_NEXT_SIGN)
    length(branch) >= ATTEMPT028_FIT_WINDOW + 4 || error("Not enough filtered branch points for local hook fit")
    write_branch_tsv_028(ATTEMPT028_BRANCH_PATH, branch)
    spline = build_natural_spline_028([point.s for point in branch], [point.z for point in branch])

    initial_guess, guess_index = initial_guess_from_branch_028(branch)
    println("Initial hook guess from sampled branch:")
    @printf("  sorted branch index = %d\n", guess_index)
    @printf("  s0 = %.12f\n", initial_guess)
    println()

    trace, final_eval = run_damped_newton_028(ATTEMPT028_ALPHA, ATTEMPT028_LAMBDA, initial_guess, spline, ATTEMPT028_TARGET_NEXT_SIGN)
    write_trace_tsv_028(ATTEMPT028_TRACE_PATH, trace, final_eval)
    write_summary_md_028(ATTEMPT028_SUMMARY_PATH, branch, initial_guess, final_eval, trace)

    println("Final Newton state:")
    @printf("  s*          = %.12f\n", final_eval.s)
    @printf("  F(s*)       = %.12f\n", final_eval.value)
    @printf("  F'(s*)      = %.12e\n", final_eval.first_derivative)
    @printf("  F''(s*)     = %.12e\n", final_eval.second_derivative)
    @printf("  event time  = %.12f\n", final_eval.event_time)
    @printf("  SciML mismatch = %.12e\n", final_eval.first_derivative_mismatch)
    println()
    println("Wrote:")
    println("  $(ATTEMPT028_EVENTS_PATH)")
    println("  $(ATTEMPT028_BRANCH_PATH)")
    println("  $(ATTEMPT028_TRACE_PATH)")
    println("  $(ATTEMPT028_SUMMARY_PATH)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
