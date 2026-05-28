module FlowFolding

using ForwardDiff
using LinearAlgebra

export ExtremumKind,
       StateMaximum,
       StateMinimum,
       FlowFoldingProblem,
       extremum_event_value,
       extremum_second_derivative,
       accepts_extremum,
       detect_critical_points

@enum ExtremumKind StateMaximum StateMinimum

"""
    FlowFoldingProblem(f, p; variable_index, extremum)

Minimal problem description for extremum-based return-map folding diagnostics.

`f` is currently expected to be an out-of-place vector field with signature
`f(u, p, t)`. The selected state variable is `u[variable_index]`; its extrema
occur on the event surface `f(u, p, t)[variable_index] == 0`.
"""
struct FlowFoldingProblem{F,P}
    f::F
    p::P
    variable_index::Int
    extremum::ExtremumKind
end

function FlowFoldingProblem(f, p; variable_index::Integer, extremum::ExtremumKind)
    variable_index >= 1 || throw(ArgumentError("variable_index must be positive"))
    return FlowFoldingProblem{typeof(f),typeof(p)}(f, p, Int(variable_index), extremum)
end

function _rhs(problem::FlowFoldingProblem, u, t)
    values = problem.f(u, problem.p, t)
    length(values) >= problem.variable_index || throw(BoundsError(values, problem.variable_index))
    return values
end

"""
    extremum_event_value(problem, u, t=0.0)

Return the event function h(u, p) = xdot for the selected state variable x.
"""
function extremum_event_value(problem::FlowFoldingProblem, u, t=0.0)
    return _rhs(problem, u, t)[problem.variable_index]
end

"""
    extremum_second_derivative(problem, u, t=0.0)

Return d/dt h = Dh(u, p) * f(u, p), used to distinguish maxima from minima.
"""
function extremum_second_derivative(problem::FlowFoldingProblem, u, t=0.0)
    h(z) = problem.f(z, problem.p, t)[problem.variable_index]
    return dot(ForwardDiff.gradient(h, u), _rhs(problem, u, t))
end

"""
    accepts_extremum(problem, u, t=0.0; event_atol=0.0)

Check the max/min section convention:

- `h == 0` within `event_atol`
- `d/dt h < 0` for `StateMaximum`
- `d/dt h > 0` for `StateMinimum`

This intentionally applies no coordinate-sign filter.
"""
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

"""
    detect_critical_points(problem; kwargs...)

Placeholder for the full flow-folding detector.

The eventual implementation should integrate trajectories to accepted extrema,
build a one-dimensional extremum return coordinate on the attractor, and solve
the event-corrected criticality residual. This scaffold deliberately does not
build that solver yet.
"""
function detect_critical_points(::FlowFoldingProblem; kwargs...)
    throw(ErrorException("flow_folding critical-point detection is not implemented yet"))
end

end
