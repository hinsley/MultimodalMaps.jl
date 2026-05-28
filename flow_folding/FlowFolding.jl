module FlowFolding

using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using Statistics

export ExtremumKind,
       StateMaximum,
       StateMinimum,
       CriticalKind,
       CriticalMaximum,
       CriticalMinimum,
       FlowFoldingProblem,
       FlowEvent,
       TangentEvent,
       ReturnMapSample,
       CriticalPointEstimate,
       SaddleFocusSeedRay,
       SeededEvent,
       SeededCriticalResult,
       ContinuationPoint,
       out_of_place,
       rhs,
       extremum_event_value,
       extremum_second_derivative,
       accepts_extremum,
       collect_extrema,
       collect_extrema_rk4,
       collect_tangent_extrema,
       collect_tangent_extrema_rk4,
       tangent_sign_sequence,
       return_map_sample,
       estimate_critical_points,
       saddle_focus_seed_ray,
       seed_state,
       seed_tangent,
       collect_seeded_extrema,
       criticality_residual,
       find_seeded_critical_points,
       continue_seeded_critical_point,
       binary_sequence_value,
       least_period,
       normalized_periodic_word,
       detect_critical_points

@enum ExtremumKind StateMaximum StateMinimum
@enum CriticalKind CriticalMaximum CriticalMinimum

struct FlowFoldingProblem{F,P,J}
    f::F
    p::P
    jacobian::J
    variable_index::Int
    extremum::ExtremumKind
    dimension::Union{Nothing,Int}
end

function FlowFoldingProblem(
    f,
    p;
    variable_index::Integer,
    extremum::ExtremumKind,
    dimension::Union{Nothing,Integer}=nothing,
    jacobian=nothing,
)
    variable_index >= 1 || throw(ArgumentError("variable_index must be positive"))
    dim = isnothing(dimension) ? nothing : Int(dimension)
    if !isnothing(dim) && variable_index > dim
        throw(ArgumentError("variable_index cannot exceed dimension"))
    end
    return FlowFoldingProblem{typeof(f),typeof(p),typeof(jacobian)}(
        f,
        p,
        jacobian,
        Int(variable_index),
        extremum,
        dim,
    )
end

function out_of_place(f!, dimension::Integer)
    dimension >= 1 || throw(ArgumentError("dimension must be positive"))
    return function (u, p, t)
        du = similar(u, length(u))
        f!(du, u, p, t)
        return du
    end
end

function rhs(problem::FlowFoldingProblem, u, t=0.0)
    values = problem.f(u, problem.p, t)
    length(values) >= problem.variable_index || throw(BoundsError(values, problem.variable_index))
    if !isnothing(problem.dimension) && length(values) != problem.dimension
        throw(DimensionMismatch("vector field returned $(length(values)) entries, expected $(problem.dimension)"))
    end
    return values
end

function _rhs_vector(problem::FlowFoldingProblem, u, t=0.0)
    return collect(rhs(problem, u, t))
end

_float_vector(values) = Float64.(collect(values))

function _jacobian(problem::FlowFoldingProblem, u, t=0.0)
    if isnothing(problem.jacobian)
        return ForwardDiff.jacobian(z -> _rhs_vector(problem, z, t), u)
    end
    return problem.jacobian(u, problem.p, t)
end

function _rhs!(du, u, problem::FlowFoldingProblem, t)
    values = rhs(problem, u, t)
    @inbounds for i in eachindex(du)
        du[i] = values[i]
    end
    return nothing
end

function extremum_event_value(problem::FlowFoldingProblem, u, t=0.0)
    return rhs(problem, u, t)[problem.variable_index]
end

function extremum_second_derivative(problem::FlowFoldingProblem, u, t=0.0)
    h(z) = problem.f(z, problem.p, t)[problem.variable_index]
    return dot(ForwardDiff.gradient(h, u), rhs(problem, u, t))
end

function accepts_extremum(problem::FlowFoldingProblem, u, t=0.0; event_atol=0.0)
    abs(extremum_event_value(problem, u, t)) <= event_atol || return false
    second_derivative = extremum_second_derivative(problem, u, t)
    if problem.extremum == StateMaximum
        return second_derivative < 0
    elseif problem.extremum == StateMinimum
        return second_derivative > 0
    end
    error("unsupported extremum kind $(problem.extremum)")
end

struct FlowEvent
    t::Float64
    u::Vector{Float64}
    value::Float64
    second_derivative::Float64
end

struct TangentEvent
    t::Float64
    u::Vector{Float64}
    value::Float64
    second_derivative::Float64
    tangent_component::Float64
    sign::Int
end

struct ReturnMapSample
    events::Vector{FlowEvent}
    x::Vector{Float64}
    y::Vector{Float64}
end

struct CriticalPointEstimate
    x::Float64
    y::Float64
    kind::CriticalKind
    curvature::Float64
    sorted_index::Int
end

function _accepts_second_derivative(problem::FlowFoldingProblem, second_derivative)
    if problem.extremum == StateMaximum
        return second_derivative < 0
    elseif problem.extremum == StateMinimum
        return second_derivative > 0
    end
    return false
end

function _blowup_callback(max_state)
    isfinite(max_state) || return nothing

    condition(u, t, integrator) = any(abs(x) > max_state for x in u)
    affect!(integrator) = terminate!(integrator)
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

_noop_affect!(integrator) = nothing

function _directional_extremum_callback(problem::FlowFoldingProblem, condition, affect!)
    if problem.extremum == StateMinimum
        return ContinuousCallback(condition, affect!, _noop_affect!; save_positions=(false, false))
    elseif problem.extremum == StateMaximum
        return ContinuousCallback(condition, _noop_affect!, affect!; save_positions=(false, false))
    end

    error("unsupported extremum kind $(problem.extremum)")
end

function _event_callback(problem, events, transient_time, max_events)
    condition(u, t, integrator) = extremum_event_value(problem, u, t)

    function record!(integrator)
        integrator.t >= transient_time || return
        u_event = collect(Float64, integrator.u)
        second_derivative = extremum_second_derivative(problem, u_event, integrator.t)
        _accepts_second_derivative(problem, second_derivative) || return

        push!(
            events,
            FlowEvent(
                Float64(integrator.t),
                u_event,
                Float64(u_event[problem.variable_index]),
                Float64(second_derivative),
            ),
        )
        if length(events) >= max_events
            terminate!(integrator)
        end
    end

    return _directional_extremum_callback(problem, condition, record!)
end

function collect_extrema(
    problem::FlowFoldingProblem,
    u0;
    tspan=(0.0, 500.0),
    transient_time=0.0,
    max_events::Integer=200,
    alg=Tsit5(),
    abstol=1e-9,
    reltol=1e-9,
    maxiters=10_000_000,
    max_state=Inf,
)
    max_events > 0 || throw(ArgumentError("max_events must be positive"))
    events = FlowEvent[]
    callbacks = Any[_event_callback(problem, events, transient_time, Int(max_events))]
    blowup_cb = _blowup_callback(max_state)
    if !isnothing(blowup_cb)
        push!(callbacks, blowup_cb)
    end

    prob = ODEProblem((du, u, p, t) -> _rhs!(du, u, problem, t), collect(Float64, u0), tspan)
    solve(
        prob,
        alg;
        callback=CallbackSet(callbacks...),
        save_everystep=false,
        save_start=false,
        save_end=false,
        abstol=abstol,
        reltol=reltol,
        maxiters=maxiters,
    )
    return events
end

function _rk4_step(problem::FlowFoldingProblem, u::Vector{Float64}, t::Float64, dt::Float64)
    k1 = _rhs_vector(problem, u, t)
    k2 = _rhs_vector(problem, u .+ 0.5 * dt .* k1, t + 0.5 * dt)
    k3 = _rhs_vector(problem, u .+ 0.5 * dt .* k2, t + 0.5 * dt)
    k4 = _rhs_vector(problem, u .+ dt .* k3, t + dt)
    return u .+ (dt / 6.0) .* (k1 .+ 2.0 .* k2 .+ 2.0 .* k3 .+ k4)
end

function _crosses_extremum(problem::FlowFoldingProblem, h0, h1)
    if problem.extremum == StateMaximum
        return h0 > 0 && h1 <= 0
    elseif problem.extremum == StateMinimum
        return h0 < 0 && h1 >= 0
    end
    return false
end

function collect_extrema_rk4(
    problem::FlowFoldingProblem,
    u0;
    dt=0.02,
    t_end=500.0,
    transient_events::Integer=0,
    max_events::Integer=200,
    max_state=Inf,
)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    t_end > 0 || throw(ArgumentError("t_end must be positive"))
    transient_events >= 0 || throw(ArgumentError("transient_events cannot be negative"))
    max_events > 0 || throw(ArgumentError("max_events must be positive"))

    u = collect(Float64, u0)
    t = 0.0
    h_prev = Float64(extremum_event_value(problem, u, t))
    accepted_seen = 0
    events = FlowEvent[]

    while t < t_end && length(events) < max_events
        u_prev = copy(u)
        t_prev = t
        u = _rk4_step(problem, u, t, dt)
        t += dt

        if any(!isfinite, u) || any(abs(x) > max_state for x in u)
            break
        end

        h_curr = Float64(extremum_event_value(problem, u, t))
        if _crosses_extremum(problem, h_prev, h_curr)
            denom = h_prev - h_curr
            θ = denom == 0 ? 0.0 : clamp(h_prev / denom, 0.0, 1.0)
            u_event = u_prev .+ θ .* (u .- u_prev)
            t_event = t_prev + θ * dt
            second_derivative = extremum_second_derivative(problem, u_event, t_event)

            if _accepts_second_derivative(problem, second_derivative)
                accepted_seen += 1
                if accepted_seen > transient_events
                    push!(
                        events,
                        FlowEvent(
                            t_event,
                            u_event,
                            Float64(u_event[problem.variable_index]),
                            Float64(second_derivative),
                        ),
                    )
                end
            end
        end

        h_prev = h_curr
    end

    return events
end

function return_map_sample(events::Vector{FlowEvent})
    length(events) >= 2 || return ReturnMapSample(events, Float64[], Float64[])
    x = [event.value for event in events[1:end-1]]
    y = [event.value for event in events[2:end]]
    return ReturnMapSample(events, x, y)
end

function _default_basis(n::Int)
    return [[Float64(i == j) for i in 1:n] for j in 1:n]
end

function _projected_unit_tangent(
    problem::FlowFoldingProblem,
    state::AbstractVector,
    tangent::AbstractVector,
    t::Float64;
    basis=_default_basis(length(state)),
)
    flow = _rhs_vector(problem, state, t)
    flow_norm2 = dot(flow, flow)
    v = collect(Float64, tangent)
    if isfinite(flow_norm2) && flow_norm2 > 1e-24
        v .-= (dot(v, flow) / flow_norm2) .* flow
    end
    v_norm = norm(v)
    if isfinite(v_norm) && v_norm > 1e-12
        return v ./ v_norm
    end

    for candidate in basis
        v = collect(Float64, candidate)
        if isfinite(flow_norm2) && flow_norm2 > 1e-24
            v .-= (dot(v, flow) / flow_norm2) .* flow
        end
        v_norm = norm(v)
        if isfinite(v_norm) && v_norm > 1e-12
            return v ./ v_norm
        end
    end

    throw(ArgumentError("could not construct a nonzero tangent transverse to the flow"))
end

function _normalize_augmented_tangent!(y, problem::FlowFoldingProblem, t, n; basis)
    tangent = _projected_unit_tangent(problem, @view(y[1:n]), @view(y[(n + 1):(2n)]), Float64(t); basis=basis)
    @inbounds for i in 1:n
        y[n + i] = tangent[i]
    end
    return nothing
end

function _tangent_augmented_rhs!(dy, y, problem::FlowFoldingProblem, t, n)
    u = collect(@view y[1:n])
    v = collect(@view y[(n + 1):(2n)])
    fvec = _rhs_vector(problem, u, t)
    J = _jacobian(problem, u, t)
    dy[1:n] .= fvec
    dy[(n + 1):(2n)] .= J * v
    return nothing
end

function collect_tangent_extrema(
    problem::FlowFoldingProblem,
    u0,
    tangent0;
    observable_index::Integer=problem.variable_index,
    tspan=(0.0, 500.0),
    transient_time=0.0,
    max_events::Integer=64,
    alg=Tsit5(),
    abstol=1e-9,
    reltol=1e-9,
    maxiters=10_000_000,
    max_state=Inf,
    reorthonormalize=true,
    basis=nothing,
)
    max_events > 0 || throw(ArgumentError("max_events must be positive"))
    n = length(u0)
    1 <= observable_index <= n || throw(ArgumentError("observable_index out of range"))
    local_basis = isnothing(basis) ? _default_basis(n) : basis
    tangent = _projected_unit_tangent(problem, u0, tangent0, 0.0; basis=local_basis)
    y0 = vcat(collect(Float64, u0), tangent)
    events = TangentEvent[]

    condition(y, t, integrator) = extremum_event_value(problem, @view(y[1:n]), t)

    function record!(integrator)
        integrator.t >= transient_time || return
        if reorthonormalize
            _normalize_augmented_tangent!(integrator.u, problem, integrator.t, n; basis=local_basis)
        end
        u_event = collect(Float64, @view integrator.u[1:n])
        second_derivative = extremum_second_derivative(problem, u_event, integrator.t)
        _accepts_second_derivative(problem, second_derivative) || return

        component = Float64(integrator.u[n + observable_index])
        push!(
            events,
            TangentEvent(
                Float64(integrator.t),
                u_event,
                Float64(u_event[problem.variable_index]),
                Float64(second_derivative),
                component,
                component > 0 ? 1 : (component < 0 ? -1 : 0),
            ),
        )
        if length(events) >= max_events
            terminate!(integrator)
        end
        return nothing
    end

    event_cb = _directional_extremum_callback(problem, condition, record!)

    function reorth_condition(u, t, integrator)
        return reorthonormalize
    end
    function reorth_affect!(integrator)
        _normalize_augmented_tangent!(integrator.u, problem, integrator.t, n; basis=local_basis)
        return nothing
    end

    callbacks = Any[event_cb]
    if reorthonormalize
        push!(callbacks, DiscreteCallback(reorth_condition, reorth_affect!; save_positions=(false, false)))
    end
    blowup_cb = _blowup_callback(max_state)
    if !isnothing(blowup_cb)
        push!(callbacks, blowup_cb)
    end

    prob = ODEProblem((dy, y, p, t) -> _tangent_augmented_rhs!(dy, y, problem, t, n), y0, tspan)
    solve(
        prob,
        alg;
        callback=CallbackSet(callbacks...),
        save_everystep=false,
        save_start=false,
        save_end=false,
        abstol=abstol,
        reltol=reltol,
        maxiters=maxiters,
    )

    return events
end

function tangent_sign_sequence(args...; kwargs...)
    return [event.sign for event in collect_tangent_extrema(args...; kwargs...)]
end

function _rk4_tangent_step(
    problem::FlowFoldingProblem,
    u::Vector{Float64},
    v::Vector{Float64},
    t::Float64,
    dt::Float64,
)
    function augmented_derivative(state, tangent, time)
        fvec = _float_vector(_rhs_vector(problem, state, time))
        J = _jacobian(problem, state, time)
        return fvec, Float64.(J * tangent)
    end

    k1u, k1v = augmented_derivative(u, v, t)
    k2u, k2v = augmented_derivative(u .+ 0.5 * dt .* k1u, v .+ 0.5 * dt .* k1v, t + 0.5 * dt)
    k3u, k3v = augmented_derivative(u .+ 0.5 * dt .* k2u, v .+ 0.5 * dt .* k2v, t + 0.5 * dt)
    k4u, k4v = augmented_derivative(u .+ dt .* k3u, v .+ dt .* k3v, t + dt)

    u_next = u .+ (dt / 6.0) .* (k1u .+ 2.0 .* k2u .+ 2.0 .* k3u .+ k4u)
    v_next = v .+ (dt / 6.0) .* (k1v .+ 2.0 .* k2v .+ 2.0 .* k3v .+ k4v)
    return u_next, v_next
end

function collect_tangent_extrema_rk4(
    problem::FlowFoldingProblem,
    u0,
    tangent0;
    observable_index::Integer=problem.variable_index,
    dt=0.02,
    t_end=500.0,
    transient_events::Integer=0,
    max_events::Integer=64,
    max_state=Inf,
    reorthonormalize=true,
    basis=nothing,
)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    t_end > 0 || throw(ArgumentError("t_end must be positive"))
    transient_events >= 0 || throw(ArgumentError("transient_events cannot be negative"))
    max_events > 0 || throw(ArgumentError("max_events must be positive"))
    n = length(u0)
    1 <= observable_index <= n || throw(ArgumentError("observable_index out of range"))

    local_basis = isnothing(basis) ? _default_basis(n) : basis
    u = collect(Float64, u0)
    v = _projected_unit_tangent(problem, u, tangent0, 0.0; basis=local_basis)
    t = 0.0
    h_prev = Float64(extremum_event_value(problem, u, t))
    accepted_seen = 0
    events = TangentEvent[]

    while t < t_end && length(events) < max_events
        u_prev = copy(u)
        v_prev = copy(v)
        t_prev = t
        u, v = _rk4_tangent_step(problem, u, v, t, dt)
        t += dt

        if any(!isfinite, u) || any(!isfinite, v) || any(abs(x) > max_state for x in u)
            break
        end

        if reorthonormalize
            v = _projected_unit_tangent(problem, u, v, t; basis=local_basis)
        end

        h_curr = Float64(extremum_event_value(problem, u, t))
        if _crosses_extremum(problem, h_prev, h_curr)
            denom = h_prev - h_curr
            θ = denom == 0 ? 0.0 : clamp(h_prev / denom, 0.0, 1.0)
            u_event = u_prev .+ θ .* (u .- u_prev)
            v_event = v_prev .+ θ .* (v .- v_prev)
            t_event = t_prev + θ * dt
            if reorthonormalize
                v_event = _projected_unit_tangent(problem, u_event, v_event, t_event; basis=local_basis)
            end
            second_derivative = extremum_second_derivative(problem, u_event, t_event)

            if _accepts_second_derivative(problem, second_derivative)
                accepted_seen += 1
                if accepted_seen > transient_events
                    component = Float64(v_event[observable_index])
                    push!(
                        events,
                        TangentEvent(
                            t_event,
                            u_event,
                            Float64(u_event[problem.variable_index]),
                            Float64(second_derivative),
                            component,
                            component > 0 ? 1 : (component < 0 ? -1 : 0),
                        ),
                    )
                end
            end
        end

        h_prev = h_curr
    end

    return events
end

function _moving_average(values::Vector{Float64}, radius::Int)
    radius <= 0 && return copy(values)
    smoothed = similar(values)
    for i in eachindex(values)
        lo = max(firstindex(values), i - radius)
        hi = min(lastindex(values), i + radius)
        smoothed[i] = mean(@view values[lo:hi])
    end
    return smoothed
end

function _quadratic_vertex(x1, y1, x2, y2, x3, y3)
    A = [x1^2 x1 1.0; x2^2 x2 1.0; x3^2 x3 1.0]
    a, b, c = A \ [y1, y2, y3]
    abs(a) <= eps(Float64) && return nothing
    xv = -b / (2a)
    yv = a * xv^2 + b * xv + c
    return xv, yv, 2a
end

function estimate_critical_points(
    sample::ReturnMapSample;
    smoothing_radius::Integer=2,
    min_dx=1e-8,
)
    n = length(sample.x)
    n >= 5 || return CriticalPointEstimate[]

    order = sortperm(sample.x)
    xs = sample.x[order]
    ys = sample.y[order]
    ys_smooth = _moving_average(ys, Int(smoothing_radius))
    estimates = CriticalPointEstimate[]

    for i in 2:(n - 1)
        xs[i + 1] - xs[i - 1] > min_dx || continue
        left_slope = (ys_smooth[i] - ys_smooth[i - 1]) / max(xs[i] - xs[i - 1], min_dx)
        right_slope = (ys_smooth[i + 1] - ys_smooth[i]) / max(xs[i + 1] - xs[i], min_dx)
        (left_slope == 0 || right_slope == 0 || sign(left_slope) == sign(right_slope)) && continue

        vertex = _quadratic_vertex(
            xs[i - 1],
            ys_smooth[i - 1],
            xs[i],
            ys_smooth[i],
            xs[i + 1],
            ys_smooth[i + 1],
        )
        isnothing(vertex) && continue
        xv, yv, curvature = vertex
        min(xs[i - 1], xs[i + 1]) <= xv <= max(xs[i - 1], xs[i + 1]) || continue

        kind = curvature < 0 ? CriticalMaximum : CriticalMinimum
        push!(estimates, CriticalPointEstimate(Float64(xv), Float64(yv), kind, Float64(curvature), i))
    end

    return estimates
end

function detect_critical_points(problem::FlowFoldingProblem, u0; kwargs...)
    events = collect_extrema(problem, u0; kwargs...)
    sample = return_map_sample(events)
    return estimate_critical_points(sample), sample
end

struct SaddleFocusSeedRay
    equilibrium::Vector{Float64}
    direction::Vector{Float64}
    eigenvalue::ComplexF64
    eigenbasis::Matrix{Float64}
    radius_scale::Float64
end

function saddle_focus_seed_ray(
    problem::FlowFoldingProblem,
    equilibrium;
    radius_scale=1.0,
    imag_atol=1e-10,
)
    ueq = collect(Float64, equilibrium)
    J = _jacobian(problem, ueq, 0.0)
    eig = eigen(J)
    candidates = [
        i for i in eachindex(eig.values)
        if real(eig.values[i]) > 0 && imag(eig.values[i]) > imag_atol
    ]
    isempty(candidates) && throw(ArgumentError("no unstable complex eigenpair found at equilibrium"))

    idx = candidates[argmin([real(eig.values[i]) for i in candidates])]
    λ = ComplexF64(eig.values[idx])
    v = eig.vectors[:, idx]
    P = hcat(real.(v), imag.(v))
    Dh = ForwardDiff.gradient(u -> problem.f(u, problem.p, 0.0)[problem.variable_index], ueq)
    row = vec(transpose(Dh) * P)
    norm(row) > 0 || throw(ArgumentError("event tangent hyperplane is degenerate on leading eigenspace"))

    coeffs = [-row[2], row[1]]
    w = P * coeffs
    w ./= norm(w)
    accel = dot(Dh, J * w)
    accel_sign = accel == 0 ? 1.0 : sign(accel)
    σ = problem.extremum == StateMaximum ? -1.0 : 1.0
    direction = σ * accel_sign .* w
    direction ./= norm(direction)

    return SaddleFocusSeedRay(ueq, collect(direction), λ, P, Float64(radius_scale))
end

seed_state(ray::SaddleFocusSeedRay, rho) = ray.equilibrium .+ (ray.radius_scale * exp(rho)) .* ray.direction
seed_tangent(ray::SaddleFocusSeedRay, rho) = (ray.radius_scale * exp(rho)) .* ray.direction

struct SeededEvent
    t::Float64
    u::Vector{Float64}
    value::Float64
    derivative::Float64
    event_denominator::Float64
end

struct SeededCriticalResult
    rho::Float64
    residual::Float64
    event_index::Int
    x_current::Float64
    x_next::Float64
    dx_current::Float64
    dx_next::Float64
    event_denominator::Float64
    events::Vector{SeededEvent}
    valid::Bool
    message::String
end

function _event_corrected_derivative(problem::FlowFoldingProblem, u, v, t; denom_atol=0.0)
    fvec = _rhs_vector(problem, u, t)
    Dh = ForwardDiff.gradient(z -> problem.f(z, problem.p, t)[problem.variable_index], u)
    denom = dot(Dh, fvec)
    if !isfinite(denom) || abs(denom) <= denom_atol
        return NaN, denom
    end
    v_event = v .- (dot(Dh, v) / denom) .* fvec
    return v_event[problem.variable_index], denom
end

function collect_seeded_extrema(
    problem::FlowFoldingProblem,
    ray::SaddleFocusSeedRay,
    rho;
    max_events::Integer,
    tspan=(0.0, 500.0),
    launch_guard_time=nothing,
    denom_atol=1e-10,
    alg=Tsit5(),
    abstol=1e-9,
    reltol=1e-9,
    maxiters=10_000_000,
    max_state=Inf,
)
    max_events > 0 || throw(ArgumentError("max_events must be positive"))

    u0 = seed_state(ray, rho)
    v0 = seed_tangent(ray, rho)
    n = length(u0)
    y0 = vcat(u0, v0)
    guard = isnothing(launch_guard_time) ? (pi / max(abs(imag(ray.eigenvalue)), eps(Float64))) : Float64(launch_guard_time)
    events = SeededEvent[]

    function augmented!(dy, y, p, t)
        u = collect(@view y[1:n])
        v = collect(@view y[(n + 1):(2n)])
        fvec = _rhs_vector(problem, u, t)
        J = _jacobian(problem, u, t)
        dy[1:n] .= fvec
        dy[(n + 1):(2n)] .= J * v
        return nothing
    end

    condition(y, t, integrator) = extremum_event_value(problem, @view(y[1:n]), t)

    function affect!(integrator)
        integrator.t >= guard || return
        y = integrator.u
        u = collect(Float64, @view y[1:n])
        v = collect(Float64, @view y[(n + 1):(2n)])
        second_derivative = extremum_second_derivative(problem, u, integrator.t)
        _accepts_second_derivative(problem, second_derivative) || return

        derivative, denom = _event_corrected_derivative(problem, u, v, integrator.t; denom_atol=denom_atol)
        isfinite(denom) && abs(denom) > denom_atol || return
        push!(
            events,
            SeededEvent(
                Float64(integrator.t),
                u,
                Float64(u[problem.variable_index]),
                Float64(derivative),
                Float64(denom),
            ),
        )
        if length(events) >= max_events
            terminate!(integrator)
        end
    end

    event_cb = _directional_extremum_callback(problem, condition, affect!)
    callbacks = Any[event_cb]
    blowup_cb = _blowup_callback(max_state)
    if !isnothing(blowup_cb)
        push!(callbacks, blowup_cb)
    end

    prob = ODEProblem(augmented!, y0, tspan)
    solve(
        prob,
        alg;
        callback=CallbackSet(callbacks...),
        save_everystep=false,
        save_start=false,
        save_end=false,
        abstol=abstol,
        reltol=reltol,
        maxiters=maxiters,
    )
    return events
end

function criticality_residual(
    problem::FlowFoldingProblem,
    ray::SaddleFocusSeedRay,
    rho;
    event_index::Integer,
    denominator_atol=1e-9,
    kwargs...,
)
    event_index >= 1 || throw(ArgumentError("event_index must be positive"))
    events = collect_seeded_extrema(problem, ray, rho; max_events=Int(event_index) + 1, kwargs...)
    if length(events) < event_index + 1
        return SeededCriticalResult(
            Float64(rho),
            NaN,
            Int(event_index),
            NaN,
            NaN,
            NaN,
            NaN,
            NaN,
            events,
            false,
            "not enough accepted extrema",
        )
    end

    current = events[event_index]
    next = events[event_index + 1]
    if !isfinite(current.derivative) || abs(current.derivative) <= denominator_atol
        return SeededCriticalResult(
            Float64(rho),
            NaN,
            Int(event_index),
            current.value,
            next.value,
            current.derivative,
            next.derivative,
            current.event_denominator,
            events,
            false,
            "unsafe return-coordinate derivative denominator",
        )
    end

    residual = next.derivative / current.derivative
    return SeededCriticalResult(
        Float64(rho),
        Float64(residual),
        Int(event_index),
        current.value,
        next.value,
        current.derivative,
        next.derivative,
        current.event_denominator,
        events,
        isfinite(residual),
        "ok",
    )
end

function find_seeded_critical_points(
    problem::FlowFoldingProblem,
    ray::SaddleFocusSeedRay;
    event_index::Integer,
    rho_range=(-12.0, -2.0),
    samples::Integer=41,
    root_tol=1e-6,
    max_bisection_iterations::Integer=40,
    kwargs...,
)
    samples >= 2 || throw(ArgumentError("samples must be at least 2"))
    rhos = collect(range(Float64(rho_range[1]), Float64(rho_range[2]), length=Int(samples)))
    results = [
        criticality_residual(problem, ray, rho; event_index=event_index, kwargs...)
        for rho in rhos
    ]

    roots = SeededCriticalResult[]
    for i in 1:(length(rhos) - 1)
        left = results[i]
        right = results[i + 1]
        left.valid && right.valid || continue
        sign(left.residual) == sign(right.residual) && continue

        lo = rhos[i]
        hi = rhos[i + 1]
        flo = left.residual
        best = left
        for _ in 1:max_bisection_iterations
            mid = (lo + hi) / 2
            fmid = criticality_residual(problem, ray, mid; event_index=event_index, kwargs...)
            fmid.valid || break
            best = fmid
            abs(fmid.residual) <= root_tol && break
            if sign(flo) == sign(fmid.residual)
                lo = mid
                flo = fmid.residual
            else
                hi = mid
            end
        end
        push!(roots, best)
    end
    return roots, results
end

struct ContinuationPoint{M}
    parameter::M
    rho::Float64
    residual::Float64
    x_current::Float64
    x_next::Float64
    valid::Bool
    iterations::Int
    message::String
end

function continue_seeded_critical_point(
    problem_ray_builder,
    parameters;
    initial_rho,
    event_index::Integer,
    tolerance=1e-6,
    max_iterations::Integer=10,
    finite_difference_step=1e-4,
    max_newton_step=0.5,
    kwargs...,
)
    points = Any[]
    rho = Float64(initial_rho)

    for parameter in parameters
        problem, ray = problem_ray_builder(parameter)
        result = criticality_residual(problem, ray, rho; event_index=event_index, kwargs...)
        iterations = 0
        message = result.message

        while result.valid && abs(result.residual) > tolerance && iterations < max_iterations
            iterations += 1
            rp = criticality_residual(problem, ray, rho + finite_difference_step; event_index=event_index, kwargs...)
            rm = criticality_residual(problem, ray, rho - finite_difference_step; event_index=event_index, kwargs...)
            if !(rp.valid && rm.valid)
                message = "finite-difference residual failed"
                break
            end
            slope = (rp.residual - rm.residual) / (2 * finite_difference_step)
            if !isfinite(slope) || abs(slope) <= eps(Float64)
                message = "Newton slope vanished"
                break
            end
            step = clamp(result.residual / slope, -max_newton_step, max_newton_step)
            rho -= step
            result = criticality_residual(problem, ray, rho; event_index=event_index, kwargs...)
            message = result.message
        end

        valid = result.valid && abs(result.residual) <= tolerance
        push!(
            points,
            ContinuationPoint(
                parameter,
                rho,
                result.residual,
                result.x_current,
                result.x_next,
                valid,
                iterations,
                valid ? "ok" : message,
            ),
        )
    end

    return points
end

function binary_sequence_value(bits; start_index=1, end_index=length(bits))
    start_index >= 1 || throw(ArgumentError("start_index must be positive"))
    end_index <= length(bits) || throw(ArgumentError("end_index cannot exceed sequence length"))
    value = 0.0
    for i in start_index:end_index
        value += bits[i] / 2.0^(end_index + 1 - i)
    end
    return value
end

function least_period(bits::AbstractVector{<:Integer})
    n = length(bits)
    n == 0 && return 0
    for p in 1:n
        n % p == 0 || continue
        ok = true
        for i in 1:n
            if bits[i] != bits[((i - 1) % p) + 1]
                ok = false
                break
            end
        end
        ok && return p
    end
    return n
end

function normalized_periodic_word(bits::AbstractVector{<:Integer})
    p = least_period(bits)
    p == 0 && return Int[]
    word = Int.(bits[1:p])
    rotations = [vcat(word[i:end], word[1:(i - 1)]) for i in eachindex(word)]
    return minimum(rotations)
end

end
