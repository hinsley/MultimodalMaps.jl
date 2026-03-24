using Pkg

const ATTEMPT19_REPAIR_ROOT = @__DIR__
const REPO_ROOT_019 = normpath(joinpath(ATTEMPT19_REPAIR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_019)

ENV["ATTEMPT019_MAP_RESOLUTION"] = get(ENV, "ATTEMPT019_MAP_RESOLUTION", "40")
ENV["ATTEMPT019_OUTPUT_TAG"] = get(ENV, "ATTEMPT019_OUTPUT_TAG", "grid500_ordinal5_remap40")

include(joinpath(ATTEMPT19_REPAIR_ROOT, "contours.jl"))

using Base.Threads
using Printf

const INPUT_TAG = get(ENV, "ATTEMPT019_REPAIR_INPUT_TAG", "grid500_ordinal5")
const INPUT_SWEEP_DIR = joinpath(ATTEMPT19_ROOT, "$(INPUT_TAG)_columns")
const INPUT_RESULTS_PATH = joinpath(ATTEMPT19_ROOT, "$(INPUT_TAG)_results.tsv")
const REPAIR_LOG_LOCK = ReentrantLock()

struct RepairStats19
    repaired_points::Int
    repaired_runs::Int
    residual_errors::Int
    remap_restarts::Int
end

struct DenseMapCandidate19
    reference_u0::SVector{6, Float64}
    map_index::Int
end

input_column_path(col_idx::Int) = joinpath(INPUT_SWEEP_DIR, @sprintf("column_%04d.tsv", col_idx))
parse_bigint_or_nothing(field::AbstractString) = isempty(field) ? nothing : parse(BigInt, field)

function parse_column_result(fields::Vector{SubString{String}})
    delta_x = parse(Float64, fields[1])
    delta_ca = parse(Float64, fields[2])
    T0_V = isempty(fields[3]) ? NaN : parse(Float64, fields[3])
    T0_Ca = isempty(fields[4]) ? NaN : parse(Float64, fields[4])
    T0_method = fields[5]
    ordinal_encoding = parse_bigint_or_nothing(fields[6])
    ca_minima = parse_float_field(fields[7])
    pattern = parse_pattern_field(fields[8])
    status = fields[9]

    if status == "ok"
        return ContourScanResult19(delta_x, delta_ca, T0_V, T0_Ca, T0_method, ca_minima, pattern, ordinal_encoding, nothing)
    end

    prefix = "error: "
    error_message = startswith(status, prefix) ? status[length(prefix) + 1:end] : status
    return ContourScanResult19(delta_x, delta_ca, NaN, NaN, "", Float64[], Int[], nothing, error_message)
end

function load_column_results(path::String)
    results = ContourScanResult19[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            push!(results, parse_column_result(split(line, '\t')))
        end
    end
    return results
end

function error_runs(results::Vector{ContourScanResult19})
    runs = Tuple{Int, Int}[]
    idx = length(results)
    while idx >= 1
        if isnothing(results[idx].error_message)
            idx -= 1
            continue
        end

        run_top = idx
        while idx >= 1 && !isnothing(results[idx].error_message)
            idx -= 1
        end
        run_bottom = idx + 1
        push!(runs, (run_top, run_bottom))
    end
    return runs
end

function dense_candidate_u0s(p, x_eq_SF::Float64, gamma_sd_minus0::SVector{6, Float64})
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
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=ATTEMPT17_MAP_RESOLUTION))
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
    for u0 in u0s
        local_prob = ODEProblem(Plant.melibeNew, u0, TSPAN, p)
        local_sol = solve(local_prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
        push!(return_ca_mins, Float64(local_sol.u[end][5]))
    end

    candidates = DenseMapCandidate19[]
    for idx in 2:(length(u0s) - 1)
        center = return_ca_mins[idx]
        left = return_ca_mins[idx - 1]
        right = return_ca_mins[idx + 1]
        if (center > left && center > right) || (center < left && center < right)
            push!(candidates, DenseMapCandidate19(u0s[idx], idx))
        end
    end

    sort!(candidates; by=candidate -> candidate.map_index)
    return candidates
end

function refine_candidate_u0(p, x_eq_SF::Float64, candidate::DenseMapCandidate19)
    callback = make_ca_min_callback(x_eq_SF)
    reference_u0 = candidate.reference_u0
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

function run_contour_point_multicrit(delta_x::Float64, delta_ca::Float64)::ContourScanResult19
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)

    candidates = dense_candidate_u0s(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    isempty(candidates) && error("No dense return-map critical points found.")

    last_error = nothing
    for candidate in candidates
        try
            T0 = refine_candidate_u0(p, saddle_data.x_eq_SF, candidate)
            return finalize_contour_point(
                delta_x,
                delta_ca,
                p,
                saddle_data,
                T0,
                @sprintf("densemulti:%d", candidate.map_index),
            )
        catch err
            last_error = sprint(showerror, err)
        end
    end

    error(isnothing(last_error) ? "No valid dense-map candidate found." : last_error)
end

function run_contour_point_multicrit_safe(delta_x::Float64, delta_ca::Float64)::ContourScanResult19
    try
        return run_contour_point_multicrit(delta_x, delta_ca)
    catch err
        return ContourScanResult19(delta_x, delta_ca, NaN, NaN, "", Float64[], Int[], nothing, sprint(showerror, err))
    end
end

function repair_run!(
    repaired::Vector{ContourScanResult19},
    original::Vector{ContourScanResult19},
    run_top::Int,
    run_bottom::Int,
)
    previous_successful = run_top < length(repaired) ? repaired[run_top + 1] : nothing
    if !isnothing(previous_successful) && !isnothing(previous_successful.error_message)
        previous_successful = nothing
    end

    repaired_points = 0
    residual_errors = 0
    remap_restarts = 0
    need_dense_restart = true

    for row_idx in run_top:-1:run_bottom
        original_result = original[row_idx]
        delta_x = original_result.delta_x
        delta_ca = original_result.delta_ca

        result = nothing
        if !need_dense_restart && !isnothing(previous_successful)
            candidate_seed = make_candidate_seed(previous_successful)
            result = run_contour_point_safe(delta_x, delta_ca, candidate_seed)
            if !isnothing(result.error_message)
                need_dense_restart = true
            end
        end

        if need_dense_restart || isnothing(result)
            result = run_contour_point_multicrit_safe(delta_x, delta_ca)
            remap_restarts += 1
        end

        repaired[row_idx] = result
        repaired_points += 1

        if isnothing(result.error_message)
            previous_successful = result
            need_dense_restart = false
        else
            previous_successful = nothing
            need_dense_restart = true
            residual_errors += 1
        end
    end

    return repaired_points, residual_errors, remap_restarts
end

function repair_column(col_idx::Int)
    input_path = input_column_path(col_idx)
    output_path = column_path(col_idx)

    if !isfile(input_path)
        error("Missing input column file: $(input_path)")
    end

    original = load_column_results(input_path)
    runs = error_runs(original)
    if isempty(runs)
        cp(input_path, output_path; force=true)
        return RepairStats19(0, 0, 0, 0)
    end

    repaired = copy(original)
    repaired_points = 0
    residual_errors = 0
    remap_restarts = 0

    for (run_top, run_bottom) in runs
        points, residual, restarts = repair_run!(repaired, original, run_top, run_bottom)
        repaired_points += points
        residual_errors += residual
        remap_restarts += restarts
    end

    write_row(output_path, repaired)
    return RepairStats19(repaired_points, length(runs), residual_errors, remap_restarts)
end

function summarize_original_errors()
    if !isfile(INPUT_RESULTS_PATH)
        return 0
    end

    error_count = 0
    open(INPUT_RESULTS_PATH, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, '\t')
            if fields[end] != "ok"
                error_count += 1
            end
        end
    end
    return error_count
end

function main()
    println("Running attempt-019 segment repair.")
    println("Input tag: $(INPUT_TAG)")
    println("Output tag: $(CONTOUR_OUTPUT_TAG)")
    println("Dense remap resolution: $(ATTEMPT17_MAP_RESOLUTION)")
    println("Input columns: $(INPUT_SWEEP_DIR)")
    println("Output columns: $(SWEEP_DIR)")
    flush(stdout)

    mkpath(SWEEP_DIR)

    per_column = Vector{RepairStats19}(undef, length(DELTA_CAS_010))

    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_010)
        stats = repair_column(col_idx)
        per_column[col_idx] = stats
        if stats.repaired_points > 0
            lock(REPAIR_LOG_LOCK)
            try
                @printf(
                    "Repaired column %d/%d (ΔCa=%.6f): %d points across %d failed runs, residual errors=%d, remap restarts=%d\n",
                    col_idx,
                    length(DELTA_CAS_010),
                    DELTA_CAS_010[col_idx],
                    stats.repaired_points,
                    stats.repaired_runs,
                    stats.residual_errors,
                    stats.remap_restarts,
                )
                flush(stdout)
            finally
                unlock(REPAIR_LOG_LOCK)
            end
        end
    end

    counts, examples, error_count = collect_legend()
    lookup = write_legend(legend_path(), counts, examples)
    grid = build_category_grid(lookup)

    write_final_results(results_path(), lookup)
    save_contour_plot(plot_path(), grid)

    original_error_count = summarize_original_errors()
    total_repaired_points = sum(stats.repaired_points for stats in per_column)
    total_repaired_runs = sum(stats.repaired_runs for stats in per_column)
    total_remap_restarts = sum(stats.remap_restarts for stats in per_column)

    println("Original errored points: $(original_error_count)")
    println("Reprocessed points: $(total_repaired_points)")
    println("Failed runs repaired: $(total_repaired_runs)")
    println("Dense remap restarts used: $(total_remap_restarts)")
    println("Residual errors after repair: $(error_count)")
    println("Saved repaired results to $(results_path())")
    println("Saved repaired contour plot to $(plot_path())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
