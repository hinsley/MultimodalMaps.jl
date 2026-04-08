using Pkg

const ATTEMPT31_ROOT = @__DIR__
const REPO_ROOT_031 = normpath(joinpath(ATTEMPT31_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_031)

using DifferentialEquations
using Printf
using SciMLSensitivity
using StaticArrays

const ATTEMPT031_N_ALPHA = parse(Int, get(ENV, "ATTEMPT031_N_ALPHA", "500"))
const ATTEMPT031_N_LAMBDA = parse(Int, get(ENV, "ATTEMPT031_N_LAMBDA", "500"))
const ATTEMPT031_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT031_ALPHA_MIN", "0.0"))
const ATTEMPT031_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT031_ALPHA_MAX", "0.7"))
const ATTEMPT031_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT031_LAMBDA_MIN", "0.2"))
const ATTEMPT031_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT031_LAMBDA_MAX", "1.6"))
const ATTEMPT031_MAX_EVENT_ITERATES = parse(Int, get(ENV, "ATTEMPT031_MAX_EVENT_ITERATES", "16"))
const ATTEMPT031_PLOT_ITERATE_CAP = min(
    ATTEMPT031_MAX_EVENT_ITERATES,
    parse(Int, get(ENV, "ATTEMPT031_PLOT_ITERATE_CAP", "7")),
)
const ATTEMPT031_T_END = parse(Float64, get(ENV, "ATTEMPT031_T_END", "200.0"))
const ATTEMPT031_DT = parse(Float64, get(ENV, "ATTEMPT031_DT", "0.02"))
const ATTEMPT031_ABSTOL = parse(Float64, get(ENV, "ATTEMPT031_ABSTOL", "1e-9"))
const ATTEMPT031_RELTOL = parse(Float64, get(ENV, "ATTEMPT031_RELTOL", "1e-9"))
const ATTEMPT031_MAX_ITERS = parse(Int, get(ENV, "ATTEMPT031_MAX_ITERS", "40000000"))
const ATTEMPT031_EPS0 = parse(Float64, get(ENV, "ATTEMPT031_EPS0", "1e-7"))
const ATTEMPT031_MAX_STATE = parse(Float64, get(ENV, "ATTEMPT031_MAX_STATE", "200.0"))
const ATTEMPT031_MIN_SECTION_X = parse(Float64, get(ENV, "ATTEMPT031_MIN_SECTION_X", "1e-5"))
const ATTEMPT031_DENOM_TOL = parse(Float64, get(ENV, "ATTEMPT031_DENOM_TOL", "1e-12"))
const ATTEMPT031_EVENT_BISECTION_STEPS = parse(Int, get(ENV, "ATTEMPT031_EVENT_BISECTION_STEPS", "56"))
const ATTEMPT031_EVENT_TIME_TOL = parse(Float64, get(ENV, "ATTEMPT031_EVENT_TIME_TOL", "1e-12"))
const ATTEMPT031_FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT031_FIG_WIDTH", "2000"))
const ATTEMPT031_FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT031_FIG_HEIGHT", "2000"))
const ATTEMPT031_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT031_PX_PER_UNIT", "4.0"))
const ATTEMPT031_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT031_LINEWIDTH", "0.35"))

const ALPHAS_031 = collect(range(ATTEMPT031_ALPHA_MIN, ATTEMPT031_ALPHA_MAX, length=ATTEMPT031_N_ALPHA))
const LAMBDAS_031 = collect(range(ATTEMPT031_LAMBDA_MIN, ATTEMPT031_LAMBDA_MAX, length=ATTEMPT031_N_LAMBDA))

struct SMSensitivityResult31
    alpha::Float64
    lambda::Float64
    absxmax_count::Int
    absxmax_sensitivity_values::Vector{Float64}
    absxmax_return_times::Vector{Float64}
    absxmax_states::Vector{SVector{3, Float64}}
    status::String
end

@inline function shimizu_morioka_vector_031(
    u::SVector{3, T1},
    alpha::T2,
    lambda::T3,
) where {T1<:Real, T2<:Real, T3<:Real}
    P = promote_type(T1, T2, T3)
    x = P(u[1])
    y = P(u[2])
    z = P(u[3])
    α = P(alpha)
    λ = P(lambda)
    return SVector{3, P}(y, x - λ * y - x * z, -α * (z - x * x))
end

@inline function unstable_side_initial_condition_031(lambda::Float64; eps0::Float64=ATTEMPT031_EPS0)
    mu = 0.5 * (-lambda + sqrt(lambda * lambda + 4.0))
    vec = @SVector [1.0, mu, 0.0]
    return eps0 * (vec / sqrt(sum(abs2, vec)))
end

@inline section_event_value_031(u::SVector{3, Float64}) = u[2]
@inline absx_value_031(u::SVector{3, Float64}) = u[1] * u[1]

function base_problem_031(alpha::Float64, lambda::Float64, u0::SVector{3, Float64})
    function base_flow!(du, u, p, t)
        flow = shimizu_morioka_vector_031(SVector{3, Float64}(u), alpha, lambda)
        du[1] = flow[1]
        du[2] = flow[2]
        du[3] = flow[3]
        return nothing
    end

    return ODEProblem(base_flow!, collect(u0), (0.0, ATTEMPT031_T_END))
end

function transformed_problem_031(alpha::Float64, lambda::Float64, base_state::SVector{3, Float64})
    function transformed_flow!(du, w, p, t)
        xshift = p[1]
        u = @SVector [w[1] + base_state[1] + xshift, w[2] + base_state[2], w[3] + base_state[3]]
        flow = shimizu_morioka_vector_031(u, alpha, lambda)
        du[1] = flow[1]
        du[2] = flow[2]
        du[3] = flow[3]
        return nothing
    end

    return ODEForwardSensitivityProblem(
        transformed_flow!,
        zeros(3),
        (0.0, ATTEMPT031_T_END),
        [0.0],
        ForwardSensitivity(),
    )
end

@inline function physical_state_and_sensitivity_031(
    augmented_state::AbstractVector{<:Real},
    base_state::SVector{3, Float64},
)
    physical_state = @SVector [
        Float64(augmented_state[1]) + base_state[1],
        Float64(augmented_state[2]) + base_state[2],
        Float64(augmented_state[3]) + base_state[3],
    ]
    physical_sensitivity = @SVector [
        Float64(augmented_state[4]) + 1.0,
        Float64(augmented_state[5]),
        Float64(augmented_state[6]),
    ]
    return physical_state, physical_sensitivity
end

@inline function augmented_state_at_031(integ, t::Float64)
    return SVector{6, Float64}(integ(t))
end

@inline function base_state_at_031(integ, t::Float64)
    return SVector{3, Float64}(integ(t))
end

function refine_section_root_base_031(
    integ,
    left_t::Float64,
    right_t::Float64,
    left_y::Float64,
    right_y::Float64,
)
    if abs(left_y) <= ATTEMPT031_DENOM_TOL
        state = base_state_at_031(integ, left_t)
        return left_t, state
    elseif abs(right_y) <= ATTEMPT031_DENOM_TOL
        state = base_state_at_031(integ, right_t)
        return right_t, state
    end

    signbit(left_y) == signbit(right_y) && error("Base section root refinement requires a sign change.")

    left = left_t
    right = right_t
    f_left = left_y

    for _ in 1:ATTEMPT031_EVENT_BISECTION_STEPS
        (right - left) <= ATTEMPT031_EVENT_TIME_TOL && break
        mid = 0.5 * (left + right)
        state = base_state_at_031(integ, mid)
        f_mid = section_event_value_031(state)
        if abs(f_mid) <= ATTEMPT031_DENOM_TOL
            return mid, state
        elseif signbit(f_mid) == signbit(f_left)
            left = mid
            f_left = f_mid
        else
            right = mid
        end
    end

    t_hit = 0.5 * (left + right)
    return t_hit, base_state_at_031(integ, t_hit)
end

function refine_section_root_031(
    integ,
    base_state::SVector{3, Float64},
    left_t::Float64,
    right_t::Float64,
    left_y::Float64,
    right_y::Float64,
)
    if abs(left_y) <= ATTEMPT031_DENOM_TOL
        augmented_state = augmented_state_at_031(integ, left_t)
        return left_t, augmented_state
    elseif abs(right_y) <= ATTEMPT031_DENOM_TOL
        augmented_state = augmented_state_at_031(integ, right_t)
        return right_t, augmented_state
    end

    signbit(left_y) == signbit(right_y) && error("Section root refinement requires a sign change.")

    left = left_t
    right = right_t
    f_left = left_y
    f_right = right_y

    for _ in 1:ATTEMPT031_EVENT_BISECTION_STEPS
        (right - left) <= ATTEMPT031_EVENT_TIME_TOL && break
        mid = 0.5 * (left + right)
        physical_state, _ = physical_state_and_sensitivity_031(augmented_state_at_031(integ, mid), base_state)
        f_mid = section_event_value_031(physical_state)
        if abs(f_mid) <= ATTEMPT031_DENOM_TOL
            return mid, augmented_state_at_031(integ, mid)
        elseif signbit(f_mid) == signbit(f_left)
            left = mid
            f_left = f_mid
        else
            right = mid
            f_right = f_mid
        end
    end

    t_hit = 0.5 * (left + right)
    return t_hit, augmented_state_at_031(integ, t_hit)
end

function find_first_absxmax_state_031(alpha::Float64, lambda::Float64)
    u0 = unstable_side_initial_condition_031(lambda)
    prob = base_problem_031(alpha, lambda, u0)
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT031_DT,
        dtmax=ATTEMPT031_DT,
        abstol=ATTEMPT031_ABSTOL,
        reltol=ATTEMPT031_RELTOL,
        maxiters=ATTEMPT031_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    prev_state = u0
    prev_t = integ.t
    prev_y = section_event_value_031(prev_state)

    while integ.t < ATTEMPT031_T_END
        step!(integ)

        curr_state = SVector{3, Float64}(integ.u)
        curr_t = integ.t
        curr_y = section_event_value_031(curr_state)

        if !all(isfinite, curr_state)
            return nothing, NaN, "nonfinite"
        end
        if maximum(abs, curr_state) > ATTEMPT031_MAX_STATE
            return nothing, NaN, "blowup"
        end

        crossed_section =
            (prev_y > 0.0 && curr_y <= 0.0) ||
            (prev_y < 0.0 && curr_y >= 0.0)

        if crossed_section
            t_hit, event_state = refine_section_root_base_031(integ, prev_t, curr_t, prev_y, curr_y)
            if abs(event_state[1]) > ATTEMPT031_MIN_SECTION_X && event_state[3] > 1.0
                return event_state, t_hit, "ok"
            end
        end

        prev_state = curr_state
        prev_t = curr_t
        prev_y = curr_y
    end

    return nothing, NaN, "no_first_absxmax"
end

function event_sensitivity_x2_031(
    event_state::SVector{3, Float64},
    raw_sensitivity::SVector{3, Float64},
    alpha::Float64,
    lambda::Float64,
)
    flow = shimizu_morioka_vector_031(event_state, alpha, lambda)
    denom = flow[2]
    abs(denom) > ATTEMPT031_DENOM_TOL || error("Event-time sensitivity denominator nearly singular.")
    event_time_d1 = -raw_sensitivity[2] / denom
    event_state_d1 = raw_sensitivity + flow * event_time_d1
    return 2.0 * event_state[1] * event_state_d1[1]
end

function next_absxmax_sensitivity_031(
    alpha::Float64,
    lambda::Float64,
    base_state::SVector{3, Float64},
)
    prob = transformed_problem_031(alpha, lambda, base_state)
    integ = init(
        prob,
        Tsit5();
        adaptive=true,
        dt=ATTEMPT031_DT,
        dtmax=ATTEMPT031_DT,
        abstol=ATTEMPT031_ABSTOL,
        reltol=ATTEMPT031_RELTOL,
        maxiters=ATTEMPT031_MAX_ITERS,
        save_everystep=false,
        save_start=false,
        save_end=false,
    )

    prev_augmented = SVector{6, Float64}(integ.u)
    prev_state, _ = physical_state_and_sensitivity_031(prev_augmented, base_state)
    prev_t = integ.t
    prev_y = section_event_value_031(prev_state)

    while integ.t < ATTEMPT031_T_END
        step!(integ)

        curr_augmented = SVector{6, Float64}(integ.u)
        curr_state, _ = physical_state_and_sensitivity_031(curr_augmented, base_state)
        curr_t = integ.t
        curr_y = section_event_value_031(curr_state)

        if !all(isfinite, curr_state) || !all(isfinite, curr_augmented)
            return nothing, NaN, NaN, "nonfinite"
        end
        if maximum(abs, curr_state) > ATTEMPT031_MAX_STATE
            return nothing, NaN, NaN, "blowup"
        end

        crossed_section =
            (prev_y > 0.0 && curr_y <= 0.0) ||
            (prev_y < 0.0 && curr_y >= 0.0)

        if crossed_section
            t_hit, augmented_hit = refine_section_root_031(integ, base_state, prev_t, curr_t, prev_y, curr_y)
            event_state, raw_sensitivity = physical_state_and_sensitivity_031(augmented_hit, base_state)

            if abs(event_state[1]) > ATTEMPT031_MIN_SECTION_X && event_state[3] > 1.0
                try
                    sensitivity_value = event_sensitivity_x2_031(event_state, raw_sensitivity, alpha, lambda)
                    return event_state, t_hit, sensitivity_value, "ok"
                catch error
                    if error isa ErrorException && occursin("denominator nearly singular", sprint(showerror, error))
                        return nothing, NaN, NaN, "near_tangency"
                    end
                    rethrow(error)
                end
            end
        end

        prev_augmented = curr_augmented
        prev_state = curr_state
        prev_t = curr_t
        prev_y = curr_y
    end

    return nothing, NaN, NaN, "no_next_absxmax"
end

function scan_orbit_031(alpha::Float64, lambda::Float64)::SMSensitivityResult31
    first_event_state, _, prelude_status = find_first_absxmax_state_031(alpha, lambda)
    if isnothing(first_event_state)
        return SMSensitivityResult31(
            alpha,
            lambda,
            0,
            Float64[],
            Float64[],
            SVector{3, Float64}[],
            prelude_status,
        )
    end

    absxmax_sensitivity_values = Float64[]
    absxmax_return_times = Float64[]
    absxmax_states = SVector{3, Float64}[]
    sizehint!(absxmax_sensitivity_values, ATTEMPT031_MAX_EVENT_ITERATES)
    sizehint!(absxmax_return_times, ATTEMPT031_MAX_EVENT_ITERATES)
    sizehint!(absxmax_states, ATTEMPT031_MAX_EVENT_ITERATES)

    status = "ok"
    current_state = first_event_state

    while length(absxmax_sensitivity_values) < ATTEMPT031_MAX_EVENT_ITERATES
        next_state, dt_return, sensitivity_value, next_status =
            next_absxmax_sensitivity_031(alpha, lambda, current_state)

        if isnothing(next_state)
            if next_status == "no_next_absxmax"
                status = "short"
            else
                status = next_status
            end
            break
        end

        push!(absxmax_sensitivity_values, sensitivity_value)
        push!(absxmax_return_times, dt_return)
        push!(absxmax_states, next_state)
        current_state = next_state
    end

    if status == "ok" && length(absxmax_sensitivity_values) < ATTEMPT031_MAX_EVENT_ITERATES
        status = "short"
    end

    return SMSensitivityResult31(
        alpha,
        lambda,
        length(absxmax_sensitivity_values),
        absxmax_sensitivity_values,
        absxmax_return_times,
        absxmax_states,
        status,
    )
end

join_float_list_031(values::Vector{Float64}) = join((@sprintf("%.12e", v) for v in values), ",")
join_state_list_031(values::Vector{SVector{3, Float64}}) =
    join((@sprintf("%.12e,%.12e,%.12e", value[1], value[2], value[3]) for value in values), ";")

function parse_float_list_031(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end

function parse_state_list_031(field::AbstractString)
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

result_usable_031(result::SMSensitivityResult31) = result.status == "ok" || result.status == "short"
