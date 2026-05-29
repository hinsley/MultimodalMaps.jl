using Pkg

const ATTEMPT10_ROOT = @__DIR__
const ATTEMPT09_ROOT = normpath(joinpath(ATTEMPT10_ROOT, "..", "attempt-009"))
const REPO_ROOT_010 = normpath(joinpath(ATTEMPT10_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_010)

include(joinpath(ATTEMPT09_ROOT, "main.jl"))

using Base.Threads
using Colors
using Printf

env_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

function env_optional_float(name::String)
    return haskey(ENV, name) ? parse(Float64, ENV[name]) : nothing
end

const GRID_NX = parse(Int, get(ENV, "ATTEMPT010_NX", "4"))
const GRID_NY = parse(Int, get(ENV, "ATTEMPT010_NY", "5"))
const DELTA_X_MIN_010 = env_float("ATTEMPT010_DELTA_X_MIN", -1.5)
const DELTA_X_MAX_010 = env_float("ATTEMPT010_DELTA_X_MAX", -0.5)
const DELTA_CA_MIN_010 = env_float("ATTEMPT010_DELTA_CA_MIN", -33.0)
const DELTA_CA_MAX_010 = env_float("ATTEMPT010_DELTA_CA_MAX", -20.0)
const DELTA_X_TICK_STEP_010 = env_optional_float("ATTEMPT010_DELTA_X_TICK_STEP")
const DELTA_CA_TICK_STEP_010 = env_optional_float("ATTEMPT010_DELTA_CA_TICK_STEP")
const DELTA_XS_010 = collect(range(DELTA_X_MIN_010, DELTA_X_MAX_010, length=GRID_NX))
const DELTA_CAS_010 = collect(range(DELTA_CA_MIN_010, DELTA_CA_MAX_010, length=GRID_NY))
const EXECUTION_MODE = get(ENV, "ATTEMPT010_EXECUTION", "threaded")
const RUN_BENCHMARK = get(ENV, "ATTEMPT010_BENCHMARK", "1") != "0"
const OUTPUT_TAG = get(ENV, "ATTEMPT010_OUTPUT_TAG", "lowres")
const ATTEMPT10_MAX_SEQ_LENGTH = parse(Int, get(ENV, "ATTEMPT010_MAX_SEQ_LENGTH", string(MAX_SEQ_LENGTH)))
const T0_LOCAL_REFINE_STEPS = parse(Int, get(ENV, "ATTEMPT011_T0_LOCAL_REFINE_STEPS", "5"))
const T0_LOCAL_CA_WINDOW = env_float("ATTEMPT011_T0_CA_WINDOW", 3.0e-3)
const T0_LOCAL_CA_TOL = env_float("ATTEMPT011_T0_CA_TOL", 1.0e-5)
const SOLVER_010 = Tsit5()

struct SSCSScanResult
    delta_x::Float64
    delta_ca::Float64
    T_scs::Vector{Int}
    gamma_scs::Vector{Int}
    encoding::Union{BigInt, Nothing}
    category_id::Union{Int, Nothing}
    error_message::Union{Nothing, String}
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

function encode_sscs_pair(T_scs::Vector{Int}, gamma_scs::Vector{Int})::BigInt
    return cantor_pair(encode_sequence(T_scs), encode_sequence(gamma_scs))
end

mutable struct SSCSStateMachine010
    scs::Vector{Int}
    count::Int
    last_symbol::EventSymbol
    last2_symbol::EventSymbol
    V_sd::Float64
end

struct T0ContinuationSeed
    V::Float64
    Ca::Float64
end

function make_state_machine(V_sd::Float64)
    scs = Int[]
    sizehint!(scs, ATTEMPT10_MAX_SEQ_LENGTH)
    return SSCSStateMachine010(scs, 0, Void, Void, V_sd)
end

function make_affect_sscs!(state_machine::SSCSStateMachine010)
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

            if length(state_machine.scs) >= ATTEMPT10_MAX_SEQ_LENGTH
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

function make_ca_min_callback(x_eq_SF::Float64)
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

function initialize_T_Ca0_from_seed(
    p,
    x_eq_SF::Float64,
    gamma_sd_minus0::SVector{6, Float64},
    seed::T0ContinuationSeed,
)::Tuple{SVector{6, Float64}, Int}
    x_guess = Plant.xinf(p, seed.V) - 1.0e-4
    a = seed.Ca - T0_LOCAL_CA_WINDOW
    b = seed.Ca + T0_LOCAL_CA_WINDOW
    callback = make_ca_min_callback(x_eq_SF)

    golden_ratio = (sqrt(5) - 1) / 2
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    fc = return_voltage_at_ca_min(p, c, x_guess, callback)
    fd = return_voltage_at_ca_min(p, d, x_guess, callback)

    iterations = 0
    while iterations < T0_LOCAL_REFINE_STEPS && abs(b - a) > T0_LOCAL_CA_TOL
        if fc > fd
            b = d
            d = c
            fd = fc
            c = b - golden_ratio * (b - a)
            fc = return_voltage_at_ca_min(p, c, x_guess, callback)
        else
            a = c
            c = d
            fc = fd
            d = a + golden_ratio * (b - a)
            fd = return_voltage_at_ca_min(p, d, x_guess, callback)
        end
        iterations += 1
    end

    T_Ca0 = fc >= fd ? c : d
    u0 = EquilibriaSubset.dune(p, x_guess, T_Ca0)
    return SVector{6, Float64}(Tuple(Float64.(u0))), iterations
end

function return_voltage_at_ca_min(p, Ca0::Float64, x0::Float64, callback)::Float64
    u0 = EquilibriaSubset.dune(p, x0, Ca0)
    prob = ODEProblem(Plant.melibeNew!, u0, TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
    return Float64(sol.u[end][6])
end

return_voltage_at_ca_min(p, Ca0::Float64, x0::Float64, x_eq_SF::Float64) =
    return_voltage_at_ca_min(p, Ca0, x0, make_ca_min_callback(x_eq_SF))

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
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=MAP_RESOLUTION))
    u0s = SVector{6, Float64}[
        SVector{6, Float64}((
            Plant.xinf(p, V) - 1.0e-4,
            0.0,
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
    solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)
    return state_machine.scs
end

function run_sscs_pair(delta_x::Float64, delta_ca::Float64)::SSCSScanResult
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)

    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    if isempty(T_scs) || abs(T_scs[1]) != 1
        error("T SSCS does not start with ±1: $(T_scs)")
    end

    gamma_scs = compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)
    encoding = encode_sscs_pair(T_scs, gamma_scs)

    return SSCSScanResult(delta_x, delta_ca, T_scs, gamma_scs, encoding, nothing, nothing)
end

function assign_category_ids(results::Vector{SSCSScanResult})
    encodings = sort(unique(result.encoding for result in results if !isnothing(result.encoding)))
    category_lookup = Dict(encoding => idx for (idx, encoding) in enumerate(encodings))

    categorized = SSCSScanResult[]
    for result in results
        category_id = isnothing(result.encoding) ? nothing : category_lookup[result.encoding]
        push!(
            categorized,
            SSCSScanResult(
                result.delta_x,
                result.delta_ca,
                result.T_scs,
                result.gamma_scs,
                result.encoding,
                category_id,
                result.error_message,
            ),
        )
    end

    return categorized, encodings
end

function collect_scan_points()
    points = Tuple{Float64, Float64}[]
    for delta_ca in DELTA_CAS_010
        for delta_x in DELTA_XS_010
            push!(points, (delta_x, delta_ca))
        end
    end
    return points
end

function build_category_matrix(results::Vector{SSCSScanResult})
    matrix = fill(NaN, length(DELTA_CAS_010), length(DELTA_XS_010))

    for result in results
        i = findmin(abs.(DELTA_CAS_010 .- result.delta_ca))[2]
        j = findmin(abs.(DELTA_XS_010 .- result.delta_x))[2]
        if !isapprox(DELTA_CAS_010[i], result.delta_ca; atol=1e-5, rtol=0.0)
            error("ΔCa=$(result.delta_ca) does not align with plotting grid.")
        end
        if !isapprox(DELTA_XS_010[j], result.delta_x; atol=1e-5, rtol=0.0)
            error("Δx=$(result.delta_x) does not align with plotting grid.")
        end
        if isnothing(result.category_id)
            matrix[i, j] = NaN
        else
            matrix[i, j] = Float64(result.category_id)
        end
    end

    return matrix
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

function save_heatmap(path::String, results::Vector{SSCSScanResult}, unique_encodings::Vector{BigInt})
    matrix = build_category_matrix(results)
    fig = Figure(size=(1100, 850))
    ax = Axis(
        fig[1, 1],
        xlabel="ΔCa",
        ylabel="Δx",
    )

    if isempty(unique_encodings)
        error("No successful encodings available for plotting.")
    end

    hm = heatmap!(
        ax,
        DELTA_CAS_010,
        DELTA_XS_010,
        matrix;
        colormap=:glasbey_bw_n256,
        colorrange=(1, length(unique_encodings)),
        interpolate=false,
        nan_color=RGBf(0.15, 0.15, 0.15),
    )

    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig)
end

function run_point_safe(delta_x::Float64, delta_ca::Float64)
    try
        return run_sscs_pair(delta_x, delta_ca)
    catch err
        return SSCSScanResult(delta_x, delta_ca, Int[], Int[], nothing, nothing, sprint(showerror, err))
    end
end

function run_scan(points; parallel::Bool, verbose::Bool)
    results = Vector{SSCSScanResult}(undef, length(points))

    if parallel && nthreads() > 1
        Threads.@threads :dynamic for idx in eachindex(points)
            delta_x, delta_ca = points[idx]
            results[idx] = run_point_safe(delta_x, delta_ca)
        end
    else
        for idx in eachindex(points)
            delta_x, delta_ca = points[idx]
            results[idx] = run_point_safe(delta_x, delta_ca)
            if verbose
                if isnothing(results[idx].error_message)
                    @printf(
                        "  ok Δx=%.4f ΔCa=%.4f encoding=%s\n",
                        delta_x,
                        delta_ca,
                        string(results[idx].encoding),
                    )
                else
                    @printf("  failed Δx=%.4f ΔCa=%.4f: %s\n", delta_x, delta_ca, results[idx].error_message)
                end
            end
        end
    end

    return results
end

function benchmark_sections(delta_x::Float64, delta_ca::Float64)
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)

    GC.gc()
    gamma_sd_seconds = @elapsed saddle_data = compute_gamma_sd_minus0(p)
    init_T0_seconds = @elapsed T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    T_sscs_seconds = @elapsed compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    gamma_sscs_seconds = @elapsed compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)

    return (; gamma_sd_seconds, init_T0_seconds, T_sscs_seconds, gamma_sscs_seconds)
end

function write_benchmark(
    path::String,
    serial_seconds::Float64,
    threaded_seconds::Union{Float64, Nothing},
    point_count::Int,
    section_benchmark,
)
    total_points_target = 2000 * 2000
    open(path, "w") do io
        println(io, "attempt-010 benchmark summary")
        println(io, "threads\t$(nthreads())")
        println(io, "benchmark_points\t$(point_count)")
        println(io, @sprintf("serial_seconds\t%.6f", serial_seconds))
        println(io, @sprintf("serial_seconds_per_point\t%.6f", serial_seconds / point_count))
        println(
            io,
            @sprintf(
                "serial_estimated_minutes_for_2000x2000\t%.2f",
                (serial_seconds / point_count) * total_points_target / 60,
            ),
        )
        if !isnothing(threaded_seconds)
            println(io, @sprintf("threaded_seconds\t%.6f", threaded_seconds))
            println(io, @sprintf("threaded_seconds_per_point\t%.6f", threaded_seconds / point_count))
            println(
                io,
                @sprintf(
                    "threaded_estimated_minutes_for_2000x2000\t%.2f",
                    (threaded_seconds / point_count) * total_points_target / 60,
                ),
            )
            println(io, @sprintf("threaded_speedup\t%.4f", serial_seconds / threaded_seconds))
        end
        println(io, @sprintf("gamma_sd_seconds\t%.6f", section_benchmark.gamma_sd_seconds))
        println(io, @sprintf("init_T0_seconds\t%.6f", section_benchmark.init_T0_seconds))
        println(io, @sprintf("T_sscs_seconds\t%.6f", section_benchmark.T_sscs_seconds))
        println(io, @sprintf("gamma_sscs_seconds\t%.6f", section_benchmark.gamma_sscs_seconds))
    end
end

function benchmark_scan(points)
    run_point_safe(points[1]...)
    serial_seconds = @elapsed begin
        run_scan(points; parallel=false, verbose=false)
    end

    threaded_seconds = nothing
    if nthreads() > 1
        threaded_seconds = @elapsed begin
            run_scan(points; parallel=true, verbose=false)
        end
    end

    benchmark_idx = cld(length(points), 2)
    section_benchmark = benchmark_sections(points[benchmark_idx]...)

    return serial_seconds, threaded_seconds, length(points), section_benchmark
end

function write_results(path::String, results::Vector{SSCSScanResult})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tcategory_id\tencoding\tT_scs\tgamma_scs\tstatus")
        for result in results
            status = isnothing(result.error_message) ? "ok" : "error: " * result.error_message
            println(
                io,
                join([
                    @sprintf("%.6f", result.delta_x),
                    @sprintf("%.6f", result.delta_ca),
                    isnothing(result.category_id) ? "" : string(result.category_id),
                    isnothing(result.encoding) ? "" : string(result.encoding),
                    join(result.T_scs, ","),
                    join(result.gamma_scs, ","),
                    status,
                ], '\t'),
            )
        end
    end
end

function write_legend(path::String, results::Vector{SSCSScanResult})
    seen = Dict{Int, SSCSScanResult}()
    counts = Dict{Int, Int}()
    for result in results
        if !isnothing(result.category_id)
            seen[result.category_id] = result
            counts[result.category_id] = get(counts, result.category_id, 0) + 1
        end
    end

    open(path, "w") do io
        println(io, "category_id\tcount\tencoding\tT_scs\tgamma_scs")
        for category_id in sort(collect(keys(seen)))
            result = seen[category_id]
            println(
                io,
                join([
                    string(category_id),
                    string(counts[category_id]),
                    string(result.encoding),
                    join(result.T_scs, ","),
                    join(result.gamma_scs, ","),
                ], '\t'),
            )
        end
    end
end

function main()
    println("Running low-resolution 2D SSCS encoding scan for attempt-010.")
    println("Grid: $(length(DELTA_XS_010)) Δx points × $(length(DELTA_CAS_010)) ΔCa points")
    println("Execution mode: $(EXECUTION_MODE), Julia threads: $(nthreads())")
    println("Benchmark enabled: $(RUN_BENCHMARK), output tag: $(OUTPUT_TAG), max sequence length: $(ATTEMPT10_MAX_SEQ_LENGTH)")

    scan_points = collect_scan_points()
    serial_seconds = NaN
    threaded_seconds = nothing
    benchmark_points = 0
    section_benchmark = nothing
    if RUN_BENCHMARK
        serial_seconds, threaded_seconds, benchmark_points, section_benchmark = benchmark_scan(scan_points)
    end

    raw_results = run_scan(scan_points; parallel=(EXECUTION_MODE == "threaded"), verbose=(EXECUTION_MODE != "threaded"))

    categorized_results, unique_encodings = assign_category_ids(raw_results)
    success_count = count(result -> !isnothing(result.category_id), categorized_results)
    println("Successful points: $(success_count) / $(length(categorized_results))")
    println("Unique SSCS-pair encodings: $(length(unique_encodings))")

    results_path = joinpath(ATTEMPT10_ROOT, "$(OUTPUT_TAG)_encoding_results.tsv")
    legend_path = joinpath(ATTEMPT10_ROOT, "$(OUTPUT_TAG)_encoding_legend.tsv")
    plot_path = joinpath(ATTEMPT10_ROOT, "$(OUTPUT_TAG)_encoding_heatmap.png")
    benchmark_path = joinpath(ATTEMPT10_ROOT, "$(OUTPUT_TAG)_benchmark_summary.txt")

    write_results(results_path, categorized_results)
    write_legend(legend_path, categorized_results)
    save_heatmap(plot_path, categorized_results, unique_encodings)
    if RUN_BENCHMARK
        write_benchmark(benchmark_path, serial_seconds, threaded_seconds, benchmark_points, section_benchmark)
    end

    println("Saved pointwise results to $(results_path)")
    println("Saved encoding legend to $(legend_path)")
    println("Saved heatmap to $(plot_path)")
    if RUN_BENCHMARK
        println("Saved benchmark summary to $(benchmark_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
