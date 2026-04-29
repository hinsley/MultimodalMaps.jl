using Pkg

const ATTEMPT50_ROOT = @__DIR__
const ATTEMPT11_ROOT = normpath(joinpath(ATTEMPT50_ROOT, "..", "attempt-011"))
const REPO_ROOT_050 = normpath(joinpath(ATTEMPT50_ROOT, "..", "..", ".."))

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

alias_env!("ATTEMPT010_NX", "ATTEMPT050_NX", "1000")
alias_env!("ATTEMPT010_NY", "ATTEMPT050_NY", "1000")
alias_env!("ATTEMPT010_DELTA_X_MIN", "ATTEMPT050_DELTA_X_MIN", "-1.5")
alias_env!("ATTEMPT010_DELTA_X_MAX", "ATTEMPT050_DELTA_X_MAX", "-0.5")
alias_env!("ATTEMPT010_DELTA_CA_MIN", "ATTEMPT050_DELTA_CA_MIN", "-45.0")
alias_env!("ATTEMPT010_DELTA_CA_MAX", "ATTEMPT050_DELTA_CA_MAX", "-20.0")
alias_env!("ATTEMPT010_MAX_SEQ_LENGTH", "ATTEMPT050_MAX_SEQ_LENGTH", "12")
alias_optional_env!("ATTEMPT010_DELTA_X_TICK_STEP", "ATTEMPT050_DELTA_X_TICK_STEP")
alias_optional_env!("ATTEMPT010_DELTA_CA_TICK_STEP", "ATTEMPT050_DELTA_CA_TICK_STEP")
alias_optional_env!("ATTEMPT011_T0_LOCAL_REFINE_STEPS", "ATTEMPT050_T0_LOCAL_REFINE_STEPS")
alias_optional_env!("ATTEMPT011_T0_CA_WINDOW", "ATTEMPT050_T0_CA_WINDOW")
alias_optional_env!("ATTEMPT011_T0_CA_TOL", "ATTEMPT050_T0_CA_TOL")

Pkg.activate(REPO_ROOT_050)

include(joinpath(ATTEMPT11_ROOT, "main.jl"))

const ATTEMPT50_MAP_RESOLUTION = parse(Int, get(ENV, "ATTEMPT050_MAP_RESOLUTION", "40"))
const ATTEMPT50_SSCS_TMAX = parse(Float64, get(ENV, "ATTEMPT050_SSCS_TMAX", "1.0e5"))
const ATTEMPT50_SSCS_ODE_TSPAN = (0.0, ATTEMPT50_SSCS_TMAX)

function compute_sscs(p, u0::SVector{6, Float64}, V_sd::Float64; abstol::Float64, reltol::Float64)::Vector{Int}
    state_machine = make_state_machine(V_sd)
    callback = VectorContinuousCallback(
        sscs_condition!,
        make_affect_sscs!(state_machine),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(Plant.melibeNew, u0, ATTEMPT50_SSCS_ODE_TSPAN, p)
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
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=ATTEMPT50_MAP_RESOLUTION))
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
