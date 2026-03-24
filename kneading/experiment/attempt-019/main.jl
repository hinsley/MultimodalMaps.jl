using Pkg

const ATTEMPT19_ROOT = @__DIR__
const ATTEMPT17_ROOT = normpath(joinpath(ATTEMPT19_ROOT, "..", "attempt-017"))
const REPO_ROOT_019 = normpath(joinpath(ATTEMPT19_ROOT, "..", "..", ".."))

function alias_env!(dst::String, src::String, default::String)
    if !haskey(ENV, dst)
        ENV[dst] = get(ENV, src, default)
    end
end

function alias_optional_env!(dst::String, src::String)
    if !haskey(ENV, dst) && haskey(ENV, src)
        ENV[dst] = ENV[src]
    end
end

alias_env!("ATTEMPT017_NX", "ATTEMPT019_NX", "500")
alias_env!("ATTEMPT017_NY", "ATTEMPT019_NY", "500")
alias_env!("ATTEMPT017_DELTA_X_MIN", "ATTEMPT019_DELTA_X_MIN", "-3.2")
alias_env!("ATTEMPT017_DELTA_X_MAX", "ATTEMPT019_DELTA_X_MAX", "0.2")
alias_env!("ATTEMPT017_DELTA_CA_MIN", "ATTEMPT019_DELTA_CA_MIN", "-50.0")
alias_env!("ATTEMPT017_DELTA_CA_MAX", "ATTEMPT019_DELTA_CA_MAX", "-10.0")
alias_env!("ATTEMPT017_MAX_SEQ_LENGTH", "ATTEMPT019_MAX_ORDINAL_LENGTH", "5")
alias_env!("ATTEMPT017_MAP_RESOLUTION", "ATTEMPT019_MAP_RESOLUTION", "40")
alias_optional_env!("ATTEMPT017_DELTA_X_TICK_STEP", "ATTEMPT019_DELTA_X_TICK_STEP")
alias_optional_env!("ATTEMPT017_DELTA_CA_TICK_STEP", "ATTEMPT019_DELTA_CA_TICK_STEP")
alias_optional_env!("ATTEMPT011_T0_LOCAL_REFINE_STEPS", "ATTEMPT019_T0_LOCAL_REFINE_STEPS")
alias_optional_env!("ATTEMPT011_T0_CA_WINDOW", "ATTEMPT019_T0_CA_WINDOW")
alias_optional_env!("ATTEMPT011_T0_CA_TOL", "ATTEMPT019_T0_CA_TOL")

Pkg.activate(REPO_ROOT_019)

include(joinpath(ATTEMPT17_ROOT, "main.jl"))

const ATTEMPT19_ORDINAL_LENGTH = parse(Int, get(ENV, "ATTEMPT019_MAX_ORDINAL_LENGTH", "5"))
const ATTEMPT19_MINIMA_TMAX = parse(Float64, get(ENV, "ATTEMPT019_MINIMA_TMAX", string(SSCS_ODE_TSPAN[2])))
const ATTEMPT19_MINIMA_TSPAN = (0.0, ATTEMPT19_MINIMA_TMAX)
const ATTEMPT19_EVENT_EPS = parse(Float64, get(ENV, "ATTEMPT019_EVENT_EPS", "1.0e-6"))
const ATTEMPT19_MINIMA_ABSTOL = parse(Float64, get(ENV, "ATTEMPT019_MINIMA_ABSTOL", "3.0e-6"))
const ATTEMPT19_MINIMA_RELTOL = parse(Float64, get(ENV, "ATTEMPT019_MINIMA_RELTOL", "3.0e-6"))

function make_ca_minima_callback(
    x_eq_SF::Float64,
    minima::Vector{Float64},
    max_events::Int,
)
    function condition(u, t, integrator)
        if t <= ATTEMPT19_EVENT_EPS || u[1] > x_eq_SF
            return 1.0
        end
        return Plant.melibeNew(u, integrator.p, integrator.t)[5]
    end

    function affect!(integrator)
        push!(minima, Float64(integrator.u[5]))
        if length(minima) >= max_events
            terminate!(integrator)
        end
    end

    return ContinuousCallback(condition, affect!, affect_neg! = nothing, save_positions=(false, true))
end

function collect_T_ca_minima(
    p,
    T0::SVector{6, Float64},
    x_eq_SF::Float64;
    count::Int=ATTEMPT19_ORDINAL_LENGTH,
    abstol::Float64=ATTEMPT19_MINIMA_ABSTOL,
    reltol::Float64=ATTEMPT19_MINIMA_RELTOL,
)::Vector{Float64}
    minima = Float64[]
    sizehint!(minima, count)

    callback = make_ca_minima_callback(x_eq_SF, minima, count)
    prob = ODEProblem(Plant.melibeNew, T0, ATTEMPT19_MINIMA_TSPAN, p)
    solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)

    if isempty(minima)
        error("Recorded no Ca minima.")
    end

    return minima
end

function ordinal_pattern(values::Vector{Float64})::Vector{Int}
    ordering = sortperm(eachindex(values); by=i -> (values[i], i))
    return Int[ordering...]
end

encode_ordinal_pattern(values::Vector{Float64}) = encode_sequence(ordinal_pattern(values))
