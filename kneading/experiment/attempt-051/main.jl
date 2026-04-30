using Pkg

const ATTEMPT51_ROOT = @__DIR__
const ATTEMPT11_ROOT = normpath(joinpath(ATTEMPT51_ROOT, "..", "attempt-011"))
const REPO_ROOT_051 = normpath(joinpath(ATTEMPT51_ROOT, "..", "..", ".."))

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

alias_env!("ATTEMPT010_NX", "ATTEMPT051_NX", "1000")
alias_env!("ATTEMPT010_NY", "ATTEMPT051_NY", "1000")
alias_env!("ATTEMPT010_DELTA_X_MIN", "ATTEMPT051_DELTA_X_MIN", "-2.0")
alias_env!("ATTEMPT010_DELTA_X_MAX", "ATTEMPT051_DELTA_X_MAX", "2.0")
alias_env!("ATTEMPT010_DELTA_CA_MIN", "ATTEMPT051_DELTA_CA_MIN", "-60.0")
alias_env!("ATTEMPT010_DELTA_CA_MAX", "ATTEMPT051_DELTA_CA_MAX", "20.0")
alias_env!("ATTEMPT010_MAX_SEQ_LENGTH", "ATTEMPT051_MAX_SEQ_LENGTH", "12")
alias_optional_env!("ATTEMPT010_DELTA_X_TICK_STEP", "ATTEMPT051_DELTA_X_TICK_STEP")
alias_optional_env!("ATTEMPT010_DELTA_CA_TICK_STEP", "ATTEMPT051_DELTA_CA_TICK_STEP")
alias_optional_env!("ATTEMPT011_T0_LOCAL_REFINE_STEPS", "ATTEMPT051_T0_LOCAL_REFINE_STEPS")
alias_optional_env!("ATTEMPT011_T0_CA_WINDOW", "ATTEMPT051_T0_CA_WINDOW")
alias_optional_env!("ATTEMPT011_T0_CA_TOL", "ATTEMPT051_T0_CA_TOL")

Pkg.activate(REPO_ROOT_051)

include(joinpath(ATTEMPT11_ROOT, "main.jl"))

const ATTEMPT51_MAP_RESOLUTION = parse(Int, get(ENV, "ATTEMPT051_MAP_RESOLUTION", "40"))
const ATTEMPT51_SSCS_TMAX = parse(Float64, get(ENV, "ATTEMPT051_SSCS_TMAX", "1.0e5"))
const ATTEMPT51_SSCS_ODE_TSPAN = (0.0, ATTEMPT51_SSCS_TMAX)
const ATTEMPT51_G_H = parse(Float64, get(ENV, "ATTEMPT051_G_H", "0.0"))
const ATTEMPT51_TAU_Y = parse(Float64, get(ENV, "ATTEMPT051_TAU_Y", "2.0e4"))

function build_params(delta_x::Float64, delta_ca::Float64)::SVector{18, Float64}
    p = copy(BASE_PARAMS)
    p[4] = ATTEMPT51_G_H
    p[15] = ATTEMPT51_TAU_Y
    p[17] = delta_x
    p[18] = delta_ca
    return SVector{18, Float64}(p)
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
    prob = ODEProblem(Plant.melibeNew, u0, ATTEMPT51_SSCS_ODE_TSPAN, p)
    solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)
    return state_machine.scs
end

function initialize_T_Ca0(p, x_eq_SF::Float64, gamma_sd_minus0::SVector{6, Float64})::SVector{6, Float64}
    callback = make_ca_min_callback(x_eq_SF)

    prob = ODEProblem(Plant.melibeNew, gamma_sd_minus0, TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
    gamma_sd_minus_endpoint = sol.u[end]
    gamma_sd_minus_ca_min = Float64(gamma_sd_minus_endpoint[5])
    gamma_sd_minus_ca_min_V = Float64(find_zero(
        V -> EquilibriaSubset.Ca_null_Ca(p, V) - gamma_sd_minus_ca_min,
        Float64(gamma_sd_minus_endpoint[6]),
    ))

    V_eq_SF = find_equilibria(p)[2]
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=ATTEMPT51_MAP_RESOLUTION))
    u0s = SVector{6, Float64}[
        SVector{6, Float64}((
            Plant.xinf(p, V) - 1.0e-4,
            Plant.yinf(V),
            Plant.ninf(V),
            Plant.hinf(V),
            EquilibriaSubset.Ca_null_Ca(p, V),
            V,
        )) for V in Vs
    ]

    return_ca_mins = Float64[]
    sizehint!(return_ca_mins, length(u0s))
    first_max_index = nothing
    for (idx, u0) in enumerate(u0s)
        local_prob = ODEProblem(Plant.melibeNew, u0, TSPAN, p)
        local_sol = solve(local_prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
        push!(return_ca_mins, Float64(local_sol.u[end][5]))

        if idx >= 3
            mid = idx - 1
            if return_ca_mins[mid] > return_ca_mins[mid - 1] && return_ca_mins[mid] > return_ca_mins[mid + 1]
                first_max_index = mid
                break
            end
        end
    end

    if isnothing(first_max_index)
        error("Could not locate the first local maximum of the reduced return map.")
    end

    reference_u0 = u0s[first_max_index]
    T_Ca0_guess = reference_u0[5]

    a = T_Ca0_guess - 3.0e-3
    b = T_Ca0_guess + 3.0e-3
    golden_ratio = (sqrt(5) - 1) / 2
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    fc = return_voltage_at_ca_min(p, c, reference_u0[1], callback)
    fd = return_voltage_at_ca_min(p, d, reference_u0[1], callback)
    while abs(b - a) > 1.0e-8
        if fc > fd
            b = d
            d = c
            fd = fc
            c = b - golden_ratio * (b - a)
            fc = return_voltage_at_ca_min(p, c, reference_u0[1], callback)
        else
            a = c
            c = d
            fc = fd
            d = a + golden_ratio * (b - a)
            fd = return_voltage_at_ca_min(p, d, reference_u0[1], callback)
        end
    end

    T_Ca0 = (a + b) / 2
    return SVector{6, Float64}((
        reference_u0[1],
        reference_u0[2],
        reference_u0[3],
        reference_u0[4],
        T_Ca0,
        reference_u0[6],
    ))
end
