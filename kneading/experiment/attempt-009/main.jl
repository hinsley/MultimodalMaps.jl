using Pkg

const ATTEMPT_ROOT = @__DIR__
const REPO_ROOT = normpath(joinpath(ATTEMPT_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT)

using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using Printf
using Roots
using StaticArrays

include(joinpath(ATTEMPT_ROOT, "vendor", "Plant.jl"))
include(joinpath(ATTEMPT_ROOT, "vendor", "equilibria_subset.jl"))
include(joinpath(ATTEMPT_ROOT, "vendor", "symbolics_subset.jl"))
include(joinpath(REPO_ROOT, "kneading", "power_series.jl"))
include(joinpath(REPO_ROOT, "kneading", "smallest_root.jl"))

using .Plant
using .EquilibriaSubset
using .SymbolicsSubset

const BASE_PARAMS = Float64.(Plant.default_params[1:15])
const DELTA_X = -1.0
const DELTA_CAS = collect(range(-36.0, -30.0, length=12))
const TSPAN = (0.0, 2.0e4)
const MAP_RESOLUTION = 20
const TRANSIENT_TIME = 1.0e3
const MAX_SEQ_LENGTH = 20
const MAX_SPIKE_COUNT = 20
const SSCS_ODE_TSPAN = (0.0, 5.0e4)
const KNEADING_DETERMINANT_PRUNE_WINDOW = 10
const MAX_PERIODICITY_PRUNE = 4
const V_RETURN_THRESHOLD = -46.0

@enum EventSymbol begin
    Void
    I
    Vplus
    Vminus
end

struct SweepResult
    delta_ca::Float64
    lz_complexity::Float64
    htop::Float64
    T_scs::Vector{Int}
    gamma_scs::Vector{Int}
    T_kneading::Vector{Int}
    gamma_kneading::Vector{Int}
end

function build_params(delta_x::Float64, delta_ca::Float64)::SVector{17, Float64}
    return SVector{17, Float64}([BASE_PARAMS..., delta_x, delta_ca])
end

function find_equilibria(p)
    return find_zeros(v -> EquilibriaSubset.Ca_difference(p, v), Plant.xinfinv(p, 0.99e0), Plant.xinfinv(p, 0.01e0))
end

function compute_gamma_sd_minus0(p)
    V_eqs = find_equilibria(p)
    if length(V_eqs) < 3
        error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
    end

    V_eq_SF = V_eqs[2]
    x_eq_SF = Plant.xinf(p, V_eq_SF)

    V_eq_SD = V_eqs[3]
    Ca_eq_SD = EquilibriaSubset.Ca_null_Ca(p, V_eq_SD)
    x_eq_SD = Plant.xinf(p, V_eq_SD)
    SD_eq = @SVector [x_eq_SD, 0.0, Plant.ninf(V_eq_SD), Plant.hinf(V_eq_SD), Ca_eq_SD, V_eq_SD]

    jac = ForwardDiff.jacobian(u -> Plant.melibeNew(u, p, 0), SD_eq)
    vals, vecs = eigen(jac)
    _, unstable_idx = findmax(real.(vals))
    gamma_sd_minus0 = SVector{6}(SD_eq .- 0.001 .* real.(vecs)[:, unstable_idx])

    return (; x_eq_SF, V_eq_SD, gamma_sd_minus0)
end

function make_ca_min_callback(x_eq_SF)
    function condition(u, t, integrator)
        if t < TRANSIENT_TIME || u[1] > x_eq_SF
            return 1.0
        end
        return Plant.melibeNew(u, integrator.p, integrator.t)[5]
    end

    function affect!(integrator)
        terminate!(integrator)
    end

    return ContinuousCallback(condition, affect!, affect_neg! = nothing)
end

function return_voltage_at_ca_min(p, Ca0, x0, x_eq_SF)
    callback = make_ca_min_callback(x_eq_SF)
    u0 = EquilibriaSubset.dune(p, x0, Ca0)
    prob = ODEProblem(Plant.melibeNew!, u0, TSPAN, p)
    sol = solve(prob, Tsit5(), callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
    return sol.u[end][6]
end

function find_first_local_maximum(values::Vector{Float64})
    for i in 2:(length(values) - 1)
        if values[i] > values[i - 1] && values[i] > values[i + 1]
            return i
        end
    end
    return nothing
end

function initialize_T_Ca0(p, x_eq_SF, gamma_sd_minus0)
    callback = make_ca_min_callback(x_eq_SF)

    prob = ODEProblem(Plant.melibeNew, gamma_sd_minus0, TSPAN, p)
    sol = solve(prob, Tsit5(), callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=true)

    gamma_sd_minus_ca_min = sol.u[end][5]
    gamma_sd_minus_ca_min_V = find_zero(
        V -> EquilibriaSubset.Ca_null_Ca(p, V) - gamma_sd_minus_ca_min,
        sol.u[end][6],
    )

    V_eqs = find_equilibria(p)
    V_eq_SF = V_eqs[2]
    Vs = range(V_eq_SF, gamma_sd_minus_ca_min_V, length=MAP_RESOLUTION)
    u0s = [
        SVector{6, Float64}([
            Plant.xinf(p, V) - 1.0e-4,
            0.0,
            Plant.ninf(V),
            Plant.hinf(V),
            EquilibriaSubset.Ca_null_Ca(p, V),
            V,
        ])
        for V in Vs
    ]
    Ca0s = [u0[5] for u0 in u0s]

    return_ca_mins = Float64[]
    for u0 in u0s
        local_prob = ODEProblem(Plant.melibeNew, u0, TSPAN, p)
        local_sol = solve(local_prob, Tsit5(), callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
        push!(return_ca_mins, local_sol.u[end][5])
    end

    first_max_index = find_first_local_maximum(return_ca_mins)
    if isnothing(first_max_index)
        error("Could not locate the first local maximum of the reduced return map.")
    end

    T_Ca0_guess = Ca0s[first_max_index]
    reference_u0 = u0s[first_max_index]

    a = T_Ca0_guess - 3.0e-3
    b = T_Ca0_guess + 3.0e-3
    golden_ratio = (sqrt(5) - 1) / 2
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    while abs(b - a) > 1.0e-8
        fc = return_voltage_at_ca_min(p, c, reference_u0[1], x_eq_SF)
        fd = return_voltage_at_ca_min(p, d, reference_u0[1], x_eq_SF)
        if fc > fd
            b = d
        else
            a = c
        end
        c = b - golden_ratio * (b - a)
        d = a + golden_ratio * (b - a)
    end

    T_Ca0 = (a + b) / 2
    return SVector{6}([reference_u0[1:4]..., T_Ca0, reference_u0[6]])
end

function make_state_machine(V_sd)
    return Dict{Symbol, Any}(
        :scs => Int[],
        :count => 0,
        :last_symbol => Void,
        :last2_symbol => Void,
        :V_sd => V_sd,
    )
end

function sscs_condition!(out, u, t, integrator)
    if t < TRANSIENT_TIME
        out[1] = 1.0
        out[2] = 1.0
        return
    end

    Vdot_val = Plant.dV(integrator.p, u[1], u[2], u[3], u[4], u[5], u[6])
    out[1] = -Vdot_val

    Vddot_val = Plant.numerical_derivative(
        (params_nd, h_nd, hdot_nd, n_nd, ndot_nd, x_nd, xdot_nd, Ca_nd, Cadot_nd, V_nd, Vdot_selector) -> Vdot_selector,
        u,
        integrator.p,
        1.0e-4,
    )
    out[2] = -Vddot_val
end

function make_affect_sscs!(state_machine)
    function affect_sscs!(integrator, idx)
        if idx == 1
            current_V = integrator.u[6]
            current_algorithmic_event = current_V > state_machine[:V_sd] ? Vplus : Vminus

            if current_algorithmic_event == Vminus
                if current_V > V_RETURN_THRESHOLD || state_machine[:last2_symbol] != Vplus
                    push!(state_machine[:scs], state_machine[:count])
                else
                    push!(state_machine[:scs], -state_machine[:count])
                end
                state_machine[:count] = 0
            else
                state_machine[:count] += 1
            end

            state_machine[:last2_symbol] = state_machine[:last_symbol]
            state_machine[:last_symbol] = current_algorithmic_event

            if length(state_machine[:scs]) >= MAX_SEQ_LENGTH
                terminate!(integrator)
            end
            if current_algorithmic_event == Vplus && state_machine[:count] > MAX_SPIKE_COUNT
                terminate!(integrator)
            end
        elseif idx == 2
            state_machine[:last2_symbol] = state_machine[:last_symbol]
            state_machine[:last_symbol] = I
        end
    end

    return affect_sscs!
end

function compute_sscs(p, u0, V_sd; abstol, reltol)
    state_machine = make_state_machine(V_sd)
    callback = VectorContinuousCallback(
        sscs_condition!,
        make_affect_sscs!(state_machine),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(Plant.melibeNew, u0, SSCS_ODE_TSPAN, p)
    solve(prob, Tsit5(), callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)
    return state_machine[:scs]
end

function periodicity(xs::Vector{Int})
    if isempty(xs)
        return nothing
    end
    return findfirst(
        period -> all(xs .== xs[mod1.(1:length(xs), period)]),
        1:min(MAX_PERIODICITY_PRUNE, length(xs)),
    )
end

function compute_htop(T_scs::Vector{Int}, gamma_scs::Vector{Int})
    if length(T_scs) < 2 || isempty(gamma_scs)
        return 0.0, Int[], Int[]
    end

    T_kneading = itinerary_to_kneading_sequence(SSCS_to_itinerary(T_scs[2:end]))
    gamma_kneading = itinerary_to_kneading_sequence(SSCS_to_itinerary(gamma_scs))

    if isempty(T_kneading) || isempty(gamma_kneading)
        return 0.0, T_kneading, gamma_kneading
    end

    ℓ = maximum(vcat(gamma_kneading, T_kneading))
    K = min(length(gamma_kneading), length(T_kneading))
    if K < 2
        return 0.0, T_kneading, gamma_kneading
    end

    if K >= KNEADING_DETERMINANT_PRUNE_WINDOW
        T_tail = T_kneading[end-KNEADING_DETERMINANT_PRUNE_WINDOW+1:end]
        gamma_tail = gamma_kneading[end-KNEADING_DETERMINANT_PRUNE_WINDOW+1:end]
        if !isnothing(periodicity(T_tail)) && !isnothing(periodicity(gamma_tail))
            return 0.0, T_kneading, gamma_kneading
        end
    end

    htop = 0.0

    if ℓ == 2
        kneading_det = zeros(Integer, K)
        kneading_det[1] = 1
        sign1 = -1
        for k in 2:K
            lap1 = gamma_kneading[k - 1]
            if lap1 == 2
                kneading_det[k] = 2 * sign1
                sign1 = -sign1
            end
        end

        r = try
            smallest_root(convert(Vector{Int}, kneading_det))
        catch err
            if err isa BoundsError
                1.0
            else
                rethrow(err)
            end
        end
        htop = max(0.0, -log(r))
    elseif ℓ > 2
        kneading_matrix = zeros(Integer, 2, ℓ - 1, K)
        kneading_matrix[1, 1, 1] = 1
        kneading_matrix[2, 1, 1] = -1
        kneading_matrix[2, 2, 1] = 1

        sign1 = -1
        sign2 = 1
        for k in 2:K
            lap1 = gamma_kneading[k - 1]
            if lap1 > 1
                kneading_matrix[1, lap1 - 1, k] = 2 * sign1
                if iseven(lap1)
                    sign1 = -sign1
                end
            end

            lap2 = T_kneading[k]
            if lap2 > 1
                kneading_matrix[2, lap2 - 1, k] = 2 * sign2
                if iseven(lap2)
                    sign2 = -sign2
                end
            end
        end

        function coeff(j, k)
            if isodd(j)
                return isodd(k) ? 0 : (1 - j) ÷ 2
            end
            return isodd(k) ? (k - 1) ÷ 2 : (k - j) ÷ 2
        end

        det = Integer[]
        for j in 2:ℓ
            for k in j+1:ℓ
                factor1 = kneading_matrix[1, j - 1, :]
                factor2 = kneading_matrix[2, k - 1, :]
                factor3 = kneading_matrix[1, k - 1, :]
                factor4 = kneading_matrix[2, j - 1, :]
                result = scale(
                    coeff(j, k),
                    add(
                        multiply(factor1, factor2),
                        scale(-1, multiply(factor3, factor4)),
                    ),
                )
                det = add(det, result)
            end
        end

        det = convert(Vector{Int64}, det[1:K])
        r = try
            smallest_root(det)
        catch err
            if err isa BoundsError
                1.0
            else
                rethrow(err)
            end
        end
        htop = max(0.0, -log(r))
    end

    return htop, T_kneading, gamma_kneading
end

function run_point(delta_ca::Float64)::SweepResult
    p = build_params(DELTA_X, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)

    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    if isempty(T_scs) || abs(T_scs[1]) != 1
        error("T SSCS does not start with ±1: $(T_scs)")
    end

    gamma_scs = compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)

    lz_complexity = normalized_LZ76_complexity(gamma_scs)
    htop, T_kneading, gamma_kneading = compute_htop(T_scs, gamma_scs)

    return SweepResult(delta_ca, lz_complexity, htop, T_scs, gamma_scs, T_kneading, gamma_kneading)
end

function write_results(path::String, results::Vector{SweepResult}, failures::Vector{Pair{Float64, String}})
    open(path, "w") do io
        println(io, "delta_ca\tlz_complexity\thtop\tT_scs\tgamma_scs\tT_kneading\tgamma_kneading")
        for result in results
            @printf(
                io,
                "%.6f\t%.10f\t%.10f\t%s\t%s\t%s\t%s\n",
                result.delta_ca,
                result.lz_complexity,
                result.htop,
                join(result.T_scs, ","),
                join(result.gamma_scs, ","),
                join(result.T_kneading, ","),
                join(result.gamma_kneading, ","),
            )
        end
        if !isempty(failures)
            println(io)
            println(io, "# failures")
            for (delta_ca, message) in failures
                @printf(io, "%.6f\t%s\n", delta_ca, message)
            end
        end
    end
end

function main()
    println("Running reduced continuous critical itinerary smoke test with the exact Plant model.")
    println("Sweep size: $(length(DELTA_CAS))")

    results = SweepResult[]
    failures = Pair{Float64, String}[]

    for (idx, delta_ca) in enumerate(DELTA_CAS)
        @printf("Sweeping [%d/%d] ΔCa=%.4f\n", idx, length(DELTA_CAS), delta_ca)
        try
            result = run_point(delta_ca)
            push!(results, result)
            @printf(
                "  lz=%.6f htop=%.6f T_scs=%s gamma_scs=%s\n",
                result.lz_complexity,
                result.htop,
                string(result.T_scs),
                string(result.gamma_scs),
            )
        catch err
            message = sprint(showerror, err)
            push!(failures, delta_ca => message)
            @printf("  failed: %s\n", message)
        end
    end

    if isempty(results)
        error("No sweep points completed successfully.")
    end

    output_path = joinpath(ATTEMPT_ROOT, "smoke_results.tsv")
    write_results(output_path, results, failures)
    println("Saved smoke results to $(output_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
