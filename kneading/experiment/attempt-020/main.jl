using Pkg

const ATTEMPT20_ROOT = @__DIR__
const ATTEMPT17_ROOT = normpath(joinpath(ATTEMPT20_ROOT, "..", "attempt-017"))
const REPO_ROOT_020 = normpath(joinpath(ATTEMPT20_ROOT, "..", "..", ".."))

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

alias_env!("ATTEMPT017_NX", "ATTEMPT020_NX", "1200")
alias_env!("ATTEMPT017_NY", "ATTEMPT020_NY", "1200")
alias_env!("ATTEMPT017_DELTA_X_MIN", "ATTEMPT020_DELTA_X_MIN", "-3.2")
alias_env!("ATTEMPT017_DELTA_X_MAX", "ATTEMPT020_DELTA_X_MAX", "0.2")
alias_env!("ATTEMPT017_DELTA_CA_MIN", "ATTEMPT020_DELTA_CA_MIN", "-50.0")
alias_env!("ATTEMPT017_DELTA_CA_MAX", "ATTEMPT020_DELTA_CA_MAX", "-10.0")
alias_env!("ATTEMPT017_MAX_SEQ_LENGTH", "ATTEMPT020_MAX_SEQ_LENGTH", "10")
alias_env!("ATTEMPT017_MAP_RESOLUTION", "ATTEMPT020_MAP_RESOLUTION", "40")
alias_optional_env!("ATTEMPT017_DELTA_X_TICK_STEP", "ATTEMPT020_DELTA_X_TICK_STEP")
alias_optional_env!("ATTEMPT017_DELTA_CA_TICK_STEP", "ATTEMPT020_DELTA_CA_TICK_STEP")
alias_optional_env!("ATTEMPT011_T0_LOCAL_REFINE_STEPS", "ATTEMPT020_T0_LOCAL_REFINE_STEPS")
alias_optional_env!("ATTEMPT011_T0_CA_WINDOW", "ATTEMPT020_T0_CA_WINDOW")
alias_optional_env!("ATTEMPT011_T0_CA_TOL", "ATTEMPT020_T0_CA_TOL")

Pkg.activate(REPO_ROOT_020)

include(joinpath(ATTEMPT17_ROOT, "main.jl"))

const ATTEMPT20_MAP_RESOLUTION = parse(Int, get(ENV, "ATTEMPT020_MAP_RESOLUTION", get(ENV, "ATTEMPT017_MAP_RESOLUTION", "40")))

struct TContourScanResult20
    delta_x::Float64
    delta_ca::Float64
    T_scs::Vector{Int}
    T0_V::Float64
    T0_Ca::Float64
    T0_method::String
    T_encoding::Union{BigInt, Nothing}
    error_message::Union{Nothing, String}
end

function initialize_T_Ca0(
    p,
    x_eq_SF::Float64,
    gamma_sd_minus0::SVector{6, Float64},
)::SVector{6, Float64}
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
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=ATTEMPT20_MAP_RESOLUTION))
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

function finalize_T_point(
    delta_x::Float64,
    delta_ca::Float64,
    p,
    saddle_data,
    T0::SVector{6, Float64},
    T0_method::String,
)::TContourScanResult20
    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    if isempty(T_scs) || abs(T_scs[1]) != 1
        error("T SSCS does not start with ±1: $(T_scs)")
    end

    return TContourScanResult20(
        delta_x,
        delta_ca,
        T_scs,
        Float64(T0[6]),
        Float64(T0[5]),
        T0_method,
        encode_sequence(T_scs),
        nothing,
    )
end

function run_T_point(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::TContourScanResult20
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)

    if !isnothing(candidate_seed)
        try
            T0, iterations = initialize_T_Ca0_from_seed(
                p,
                saddle_data.x_eq_SF,
                saddle_data.gamma_sd_minus0,
                candidate_seed,
            )
            method = "continued:$(iterations)"
            return finalize_T_point(delta_x, delta_ca, p, saddle_data, T0, method)
        catch
            # Fall through to the full initializer.
        end
    end

    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    return finalize_T_point(delta_x, delta_ca, p, saddle_data, T0, "full")
end

function run_T_point_safe(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::TContourScanResult20
    try
        return run_T_point(delta_x, delta_ca, candidate_seed)
    catch err
        return TContourScanResult20(delta_x, delta_ca, Int[], NaN, NaN, "", nothing, sprint(showerror, err))
    end
end

make_candidate_seed(previous_successful::Union{Nothing, TContourScanResult20}) =
    isnothing(previous_successful) ? nothing : T0ContinuationSeed(previous_successful.T0_V, previous_successful.T0_Ca)
