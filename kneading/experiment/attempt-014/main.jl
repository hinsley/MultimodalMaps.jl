using Pkg

const ATTEMPT14_ROOT = @__DIR__
const ATTEMPT09_ROOT = normpath(joinpath(ATTEMPT14_ROOT, "..", "attempt-009"))
const REPO_ROOT_014 = normpath(joinpath(ATTEMPT14_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_014)

include(joinpath(ATTEMPT09_ROOT, "main.jl"))

using Printf

env_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

function env_optional_float(name::String)
    return haskey(ENV, name) ? parse(Float64, ENV[name]) : nothing
end

const ATTEMPT14_MAX_SEQ_LENGTH = parse(Int, get(ENV, "ATTEMPT014_MAX_SEQ_LENGTH", "7"))
const SOLVER_014 = Tsit5()

mutable struct SSCSStateMachine014
    scs::Vector{Int}
    count::Int
    last_symbol::EventSymbol
    last2_symbol::EventSymbol
    V_sd::Float64
end

function make_state_machine(V_sd::Float64)
    scs = Int[]
    sizehint!(scs, ATTEMPT14_MAX_SEQ_LENGTH)
    return SSCSStateMachine014(scs, 0, Void, Void, V_sd)
end

function make_affect_sscs!(state_machine::SSCSStateMachine014)
    function affect_sscs!(integrator, idx)
        if idx == 1
            current_V = integrator.u[6]
            current_algorithmic_event = current_V > state_machine.V_sd ? Vplus : Vminus

            if current_algorithmic_event == Vminus
                if current_V > V_RETURN_THRESHOLD || state_machine.last2_symbol != Vplus
                    push!(state_machine.scs, state_machine.count)
                else
                    push!(state_machine.scs, -state_machine.count)
                end
                state_machine.count = 0
            else
                state_machine.count += 1
            end

            state_machine.last2_symbol = state_machine.last_symbol
            state_machine.last_symbol = current_algorithmic_event

            if length(state_machine.scs) >= ATTEMPT14_MAX_SEQ_LENGTH
                terminate!(integrator)
            end
            if current_algorithmic_event == Vplus && state_machine.count > MAX_SPIKE_COUNT
                terminate!(integrator)
            end
        elseif idx == 2
            state_machine.last2_symbol = state_machine.last_symbol
            state_machine.last_symbol = I
        end
    end

    return affect_sscs!
end

function compute_sscs(p, u0::SVector{6, Float64}, V_sd::Float64; abstol::Float64, reltol::Float64)::Vector{Int}
    state_machine = make_state_machine(V_sd)
    callback = VectorContinuousCallback(
        sscs_condition!,
        make_affect_sscs!(state_machine),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(Plant.melibeNew, u0, SSCS_ODE_TSPAN, p)
    solve(prob, SOLVER_014; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)
    return state_machine.scs
end

zigzag_encode(n::Int)::BigInt = n >= 0 ? 2 * BigInt(n) : -2 * BigInt(n) - 1

function cantor_pair(a::BigInt, b::BigInt)::BigInt
    s = a + b
    return (s * (s + 1)) ÷ 2 + b
end

function encode_sequence(seq::Vector{Int})::BigInt
    code = BigInt(0)
    for value in reverse(seq)
        code = cantor_pair(zigzag_encode(value), code) + 1
    end
    return code
end

function sampled_ticks(values::Vector{Float64}, label_fmt::String)
    tick_count = min(6, length(values))
    tick_indices = unique(round.(Int, range(1, length(values), length=tick_count)))
    fmt = Printf.Format(label_fmt)
    ticks = values[tick_indices]
    tick_labels = [Printf.format(fmt, tick) for tick in ticks]
    return ticks, tick_labels
end

function fixed_ticks(values::Vector{Float64}, label_fmt::String, step::Union{Nothing, Float64})
    if isnothing(step)
        return sampled_ticks(values, label_fmt)
    end

    ticks = collect(first(values):step:last(values))
    if isempty(ticks) || last(ticks) < last(values) - 1.0e-9
        push!(ticks, last(values))
    end
    fmt = Printf.Format(label_fmt)
    tick_labels = [Printf.format(fmt, tick) for tick in ticks]
    return ticks, tick_labels
end
