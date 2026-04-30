using Pkg

const ATTEMPT51_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_051_CONTOUR = normpath(joinpath(ATTEMPT51_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_051_CONTOUR)

include(joinpath(ATTEMPT51_CONTOUR_ROOT, "main.jl"))

using Base.Threads
using Printf

const CONTOUR_OUTPUT_TAG = get(ENV, "ATTEMPT051_OUTPUT_TAG", "grid1000_seq12_tmax1e5_gh0p000_prefixes_remap40_newmodel")
const SWEEP_DIR = joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_columns")
const T_COLOR = RGBAf(0.86, 0.16, 0.12, 0.75)
const GAMMA_COLOR = RGBAf(0.12, 0.28, 0.88, 0.70)
const LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT051_CONTOUR_LINEWIDTH", "0.35"))
const PLOT_WIDTH = parse(Int, get(ENV, "ATTEMPT051_PLOT_WIDTH", "1600"))
const PLOT_HEIGHT = parse(Int, get(ENV, "ATTEMPT051_PLOT_HEIGHT", "1200"))
const PLOT_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT051_PLOT_PX_PER_UNIT", "2.0"))
const AXIS_LABEL_SIZE = parse(Int, get(ENV, "ATTEMPT051_AXIS_LABEL_SIZE", "34"))
const AXIS_TITLE_SIZE = parse(Int, get(ENV, "ATTEMPT051_AXIS_TITLE_SIZE", "40"))
const TICK_LABEL_SIZE = parse(Int, get(ENV, "ATTEMPT051_TICK_LABEL_SIZE", "24"))
const PLOT_ONLY_RESULTS = get(ENV, "ATTEMPT051_PLOT_ONLY_RESULTS", "")
const MAX_PREFIX_PLOT_LENGTH = min(ATTEMPT10_MAX_SEQ_LENGTH, parse(Int, get(ENV, "ATTEMPT051_MAX_PREFIX_PLOT_LENGTH", "12")))
const PREFIX_LENGTHS = collect(1:MAX_PREFIX_PLOT_LENGTH)
const LOG_LOCK = ReentrantLock()

struct ContourScanResult
    delta_x::Float64
    delta_ca::Float64
    T_scs::Vector{Int}
    gamma_scs::Vector{Int}
    T0_V::Float64
    T0_Ca::Float64
    T0_method::String
    T_encoding::Union{BigInt, Nothing}
    gamma_encoding::Union{BigInt, Nothing}
    error_message::Union{Nothing, String}
end

function finalize_contour_point(
    delta_x::Float64,
    delta_ca::Float64,
    p,
    saddle_data,
    T0::SVector{6, Float64},
    T0_method::String,
)::ContourScanResult
    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    gamma_scs = compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)

    return ContourScanResult(
        delta_x,
        delta_ca,
        T_scs,
        gamma_scs,
        Float64(T0[6]),
        Float64(T0[5]),
        T0_method,
        encode_sequence(T_scs),
        encode_sequence(gamma_scs),
        nothing,
    )
end

function run_contour_point(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::ContourScanResult
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
            method = @sprintf("continued:%d", iterations)
            return finalize_contour_point(delta_x, delta_ca, p, saddle_data, T0, method)
        catch
            # Fall through to the full initializer.
        end
    end

    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    return finalize_contour_point(delta_x, delta_ca, p, saddle_data, T0, "full")
end

function run_contour_point_safe(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::ContourScanResult
    try
        return run_contour_point(delta_x, delta_ca, candidate_seed)
    catch err
        return ContourScanResult(delta_x, delta_ca, Int[], Int[], NaN, NaN, "", nothing, nothing, sprint(showerror, err))
    end
end

make_candidate_seed(previous_successful::Union{Nothing, ContourScanResult}) =
    isnothing(previous_successful) ? nothing : T0ContinuationSeed(previous_successful.T0_V, previous_successful.T0_Ca)

column_path(col_idx::Int) = joinpath(SWEEP_DIR, @sprintf("column_%04d.tsv", col_idx))

function row_is_complete(path::String, expected_points::Int)
    if !isfile(path)
        return false
    end

    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_row(path::String, results::Vector{ContourScanResult})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tT_encoding\tgamma_encoding\tT_scs\tgamma_scs\tstatus")
        for result in results
            status = isnothing(result.error_message) ? "ok" : "error: " * result.error_message
            println(
                io,
                join([
                    @sprintf("%.6f", result.delta_x),
                    @sprintf("%.6f", result.delta_ca),
                    isfinite(result.T0_V) ? @sprintf("%.6f", result.T0_V) : "",
                    isfinite(result.T0_Ca) ? @sprintf("%.6f", result.T0_Ca) : "",
                    result.T0_method,
                    isnothing(result.T_encoding) ? "" : string(result.T_encoding),
                    isnothing(result.gamma_encoding) ? "" : string(result.gamma_encoding),
                    join(result.T_scs, ","),
                    join(result.gamma_scs, ","),
                    status,
                ], '\t'),
            )
        end
    end
end

function run_column(col_idx::Int, delta_ca::Float64, total_cols::Int, total_rows::Int)
    path = column_path(col_idx)
    if row_is_complete(path, total_rows)
        lock(LOG_LOCK)
        try
            @printf("Skipping completed column %d/%d (ΔCa=%.6f)\n", col_idx, total_cols, delta_ca)
            flush(stdout)
        finally
            unlock(LOG_LOCK)
        end
        return
    end

    started = time()
    column_results = Vector{ContourScanResult}(undef, total_rows)
    previous_successful = nothing
    for row_idx in length(DELTA_XS_010):-1:1
        delta_x = DELTA_XS_010[row_idx]
        candidate_seed = make_candidate_seed(previous_successful)
        result = run_contour_point_safe(delta_x, delta_ca, candidate_seed)
        column_results[row_idx] = result
        if isnothing(result.error_message)
            previous_successful = result
        end
    end
    write_row(path, column_results)
    ok_count = count(result -> isnothing(result.error_message), column_results)

    lock(LOG_LOCK)
    try
        @printf(
            "Saved column %d/%d (ΔCa=%.6f) with %d/%d successful points in %.2f s\n",
            col_idx,
            total_cols,
            delta_ca,
            ok_count,
            total_rows,
            time() - started,
        )
        flush(stdout)
    finally
        unlock(LOG_LOCK)
    end
end

function run_or_resume_columns()
    mkpath(SWEEP_DIR)
    total_cols = length(DELTA_CAS_010)
    total_rows = length(DELTA_XS_010)

    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_010)
        delta_ca = DELTA_CAS_010[col_idx]
        run_column(col_idx, delta_ca, total_cols, total_rows)
    end
end

function parse_sequence_field(field::AbstractString)
    return isempty(field) ? Int[] : parse.(Int, split(field, ","))
end

function encode_prefix(seq::Vector{Int}, prefix_length::Int)::BigInt
    code = BigInt(0)
    for idx in min(prefix_length, length(seq)):-1:1
        code = cantor_pair(zigzag_encode(seq[idx]), code) + 1
    end
    return code
end

prefix_sequence(seq::Vector{Int}, prefix_length::Int) = seq[1:min(prefix_length, length(seq))]

prefix_plot_path(prefix_length::Int) =
    joinpath(ATTEMPT51_ROOT, @sprintf("%s_prefix%02d_contours.png", CONTOUR_OUTPUT_TAG, prefix_length))

prefix_T_legend_path(prefix_length::Int) =
    joinpath(ATTEMPT51_ROOT, @sprintf("%s_prefix%02d_T_legend.tsv", CONTOUR_OUTPUT_TAG, prefix_length))

prefix_gamma_legend_path(prefix_length::Int) =
    joinpath(ATTEMPT51_ROOT, @sprintf("%s_prefix%02d_gamma_legend.tsv", CONTOUR_OUTPUT_TAG, prefix_length))

function scan_results_file(pass_fn, results_path::String)
    open(results_path, "r") do io
        readline(io)
        for line in eachline(io)
            pass_fn(split(line, '\t'))
        end
    end
end

function scan_column_files(pass_fn)
    for col_idx in eachindex(DELTA_CAS_010)
        path = column_path(col_idx)
        if !row_is_complete(path, length(DELTA_XS_010))
            error("Missing or incomplete column file: $(path)")
        end
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                fields = split(line, '\t')
                pass_fn(fields)
            end
        end
    end
end

function collect_legends()
    T_counts = Dict{BigInt, Int}()
    gamma_counts = Dict{BigInt, Int}()
    T_examples = Dict{BigInt, Vector{Int}}()
    gamma_examples = Dict{BigInt, Vector{Int}}()
    error_count = 0

    scan_column_files() do fields
        status = fields[10]
        if status != "ok"
            error_count += 1
            return
        end

        T_encoding = parse(BigInt, fields[6])
        gamma_encoding = parse(BigInt, fields[7])
        T_counts[T_encoding] = get(T_counts, T_encoding, 0) + 1
        gamma_counts[gamma_encoding] = get(gamma_counts, gamma_encoding, 0) + 1

        if !haskey(T_examples, T_encoding)
            T_examples[T_encoding] = parse_sequence_field(fields[8])
        end
        if !haskey(gamma_examples, gamma_encoding)
            gamma_examples[gamma_encoding] = parse_sequence_field(fields[9])
        end
    end

    return T_counts, gamma_counts, T_examples, gamma_examples, error_count
end

function nearest_index(values::Vector{Float64}, target::Float64, label::String)
    idx = findmin(abs.(values .- target))[2]
    if !isapprox(values[idx], target; atol=1e-5, rtol=0.0)
        error("$(label)=$(target) does not align with plotting grid.")
    end
    return idx
end

function build_category_grids(T_lookup::Dict{BigInt, Int}, gamma_lookup::Dict{BigInt, Int})
    T_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    gamma_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))

    scan_column_files() do fields
        delta_x = parse(Float64, fields[1])
        delta_ca = parse(Float64, fields[2])
        status = fields[10]
        x_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
        y_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")

        if filled[x_idx, y_idx]
            error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
        end
        filled[x_idx, y_idx] = true

        if status == "ok"
            T_grid[x_idx, y_idx] = T_lookup[parse(BigInt, fields[6])]
            gamma_grid[x_idx, y_idx] = gamma_lookup[parse(BigInt, fields[7])]
        end
    end

    if !all(filled)
        error("One or more contour grid entries were not filled.")
    end

    return T_grid, gamma_grid
end

function write_final_results(path::String, T_lookup::Dict{BigInt, Int}, gamma_lookup::Dict{BigInt, Int})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tT_category_id\tT_encoding\tgamma_category_id\tgamma_encoding\tT_scs\tgamma_scs\tstatus")
        scan_column_files() do fields
            status = fields[10]
            if status == "ok"
                T_encoding = parse(BigInt, fields[6])
                gamma_encoding = parse(BigInt, fields[7])
                T_category_id = T_lookup[T_encoding]
                gamma_category_id = gamma_lookup[gamma_encoding]
                println(
                    io,
                    join([
                        fields[1],
                        fields[2],
                        fields[3],
                        fields[4],
                        fields[5],
                        string(T_category_id),
                        fields[6],
                        string(gamma_category_id),
                        fields[7],
                        fields[8],
                        fields[9],
                        status,
                    ], '\t'),
                )
            else
                println(
                    io,
                    join([fields[1], fields[2], fields[3], fields[4], fields[5], "", "", "", "", fields[8], fields[9], status], '\t'),
                )
            end
        end
    end
end

function write_single_legend(path::String, counts::Dict{BigInt, Int}, examples::Dict{BigInt, Vector{Int}})
    encodings = sort(collect(keys(counts)))
    lookup = Dict(encoding => idx for (idx, encoding) in enumerate(encodings))

    open(path, "w") do io
        println(io, "category_id\tcount\tencoding\tsequence")
        for encoding in encodings
            println(
                io,
                join([
                    string(lookup[encoding]),
                    string(counts[encoding]),
                    string(encoding),
                    join(examples[encoding], ","),
                ], '\t'),
            )
        end
    end

    return lookup
end

function collect_prefix_legends(results_path::String, prefix_lengths::Vector{Int})
    T_counts = [Dict{BigInt, Int}() for _ in prefix_lengths]
    gamma_counts = [Dict{BigInt, Int}() for _ in prefix_lengths]
    T_examples = [Dict{BigInt, Vector{Int}}() for _ in prefix_lengths]
    gamma_examples = [Dict{BigInt, Vector{Int}}() for _ in prefix_lengths]
    error_count = 0

    scan_results_file(results_path) do fields
        status = fields[12]
        if status != "ok"
            error_count += 1
            return
        end

        T_scs = parse_sequence_field(fields[10])
        gamma_scs = parse_sequence_field(fields[11])

        for (idx, prefix_length) in enumerate(prefix_lengths)
            T_encoding = encode_prefix(T_scs, prefix_length)
            gamma_encoding = encode_prefix(gamma_scs, prefix_length)

            T_counts[idx][T_encoding] = get(T_counts[idx], T_encoding, 0) + 1
            gamma_counts[idx][gamma_encoding] = get(gamma_counts[idx], gamma_encoding, 0) + 1

            if !haskey(T_examples[idx], T_encoding)
                T_examples[idx][T_encoding] = prefix_sequence(T_scs, prefix_length)
            end
            if !haskey(gamma_examples[idx], gamma_encoding)
                gamma_examples[idx][gamma_encoding] = prefix_sequence(gamma_scs, prefix_length)
            end
        end
    end

    return T_counts, gamma_counts, T_examples, gamma_examples, error_count
end

function build_prefix_category_grids(
    results_path::String,
    prefix_lengths::Vector{Int},
    T_lookups::Vector{Dict{BigInt, Int}},
    gamma_lookups::Vector{Dict{BigInt, Int}},
)
    T_grids = [fill(0, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in prefix_lengths]
    gamma_grids = [fill(0, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in prefix_lengths]
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))

    scan_results_file(results_path) do fields
        delta_x = parse(Float64, fields[1])
        delta_ca = parse(Float64, fields[2])
        status = fields[12]
        x_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
        y_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")

        if filled[x_idx, y_idx]
            error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
        end
        filled[x_idx, y_idx] = true

        if status != "ok"
            return
        end

        T_scs = parse_sequence_field(fields[10])
        gamma_scs = parse_sequence_field(fields[11])
        for (idx, prefix_length) in enumerate(prefix_lengths)
            T_grids[idx][x_idx, y_idx] = T_lookups[idx][encode_prefix(T_scs, prefix_length)]
            gamma_grids[idx][x_idx, y_idx] = gamma_lookups[idx][encode_prefix(gamma_scs, prefix_length)]
        end
    end

    if !all(filled)
        error("One or more prefix contour grid entries were not filled.")
    end

    return T_grids, gamma_grids
end

function write_prefix_outputs(results_path::String, prefix_lengths::Vector{Int})
    T_counts, gamma_counts, T_examples, gamma_examples, error_count = collect_prefix_legends(results_path, prefix_lengths)
    T_lookups = Vector{Dict{BigInt, Int}}(undef, length(prefix_lengths))
    gamma_lookups = Vector{Dict{BigInt, Int}}(undef, length(prefix_lengths))

    for (idx, prefix_length) in enumerate(prefix_lengths)
        T_lookups[idx] = write_single_legend(prefix_T_legend_path(prefix_length), T_counts[idx], T_examples[idx])
        gamma_lookups[idx] = write_single_legend(prefix_gamma_legend_path(prefix_length), gamma_counts[idx], gamma_examples[idx])
    end

    T_grids, gamma_grids = build_prefix_category_grids(results_path, prefix_lengths, T_lookups, gamma_lookups)
    prefix_summaries = NamedTuple[]

    for (idx, prefix_length) in enumerate(prefix_lengths)
        plot_path = prefix_plot_path(prefix_length)
        save_contour_plot(
            plot_path,
            T_grids[idx],
            gamma_grids[idx];
            title=@sprintf("SSCS prefix %02d contour overlay", prefix_length),
        )
        push!(prefix_summaries, (
            prefix_length=prefix_length,
            T_unique=length(T_lookups[idx]),
            gamma_unique=length(gamma_lookups[idx]),
            plot_path=plot_path,
            T_legend_path=prefix_T_legend_path(prefix_length),
            gamma_legend_path=prefix_gamma_legend_path(prefix_length),
        ))
    end

    return prefix_summaries, error_count
end

function edge_point(edge::Int, x0::Float64, x1::Float64, y0::Float64, y1::Float64)
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    if edge == 0
        return x0, ym
    elseif edge == 1
        return xm, y1
    elseif edge == 2
        return x1, ym
    else
        return xm, y0
    end
end

function case_segments(mask_case::Int)
    if mask_case == 1
        return ((3, 0),)
    elseif mask_case == 2
        return ((0, 1),)
    elseif mask_case == 3
        return ((3, 1),)
    elseif mask_case == 4
        return ((1, 2),)
    elseif mask_case == 5
        return ((3, 0), (1, 2))
    elseif mask_case == 6
        return ((0, 2),)
    elseif mask_case == 12
        return ((3, 2),)
    elseif mask_case == 8
        return ((2, 3),)
    elseif mask_case == 9
        return ((0, 2),)
    elseif mask_case == 10
        return ((2, 3), (0, 1))
    elseif mask_case == 11
        return ((1, 2),)
    elseif mask_case == 12
        return ((1, 3),)
    elseif mask_case == 13
        return ((0, 1),)
    elseif mask_case == 14
        return ((3, 0),)
    else
        return ()
    end
end

function normalize_segment(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
    if x1 < x2 || (x1 == x2 && y1 <= y2)
        return (x1, y1, x2, y2)
    else
        return (x2, y2, x1, y1)
    end
end

function push_unique_segment!(segments::Vector{NTuple{4, Float64}}, segment::NTuple{4, Float64})
    for existing in segments
        if existing == segment
            return
        end
    end
    push!(segments, segment)
end

function categorical_marching_squares(grid::Matrix{Int}, x_values::Vector{Float64}, y_values::Vector{Float64})
    xs = Float32[]
    ys = Float32[]
    sizehint!(xs, 3 * (length(x_values) - 1) * (length(y_values) - 1))
    sizehint!(ys, 3 * (length(x_values) - 1) * (length(y_values) - 1))

    local_segments = NTuple{4, Float64}[]
    sizehint!(local_segments, 8)

    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
            y0 = y_values[y_idx]
            y1 = y_values[y_idx + 1]

            bottom_left = grid[x_idx, y_idx]
            bottom_right = grid[x_idx + 1, y_idx]
            top_right = grid[x_idx + 1, y_idx + 1]
            top_left = grid[x_idx, y_idx + 1]

            if bottom_left == bottom_right == top_right == top_left
                continue
            end

            empty!(local_segments)
            categories = (
                bottom_left,
                bottom_right != bottom_left ? bottom_right : 0,
                (top_right != bottom_left && top_right != bottom_right) ? top_right : 0,
                (top_left != bottom_left && top_left != bottom_right && top_left != top_right) ? top_left : 0,
            )

            for category in categories
                if category <= 0
                    continue
                end

                mask_case =
                    (bottom_left == category ? 1 : 0) +
                    (bottom_right == category ? 2 : 0) +
                    (top_right == category ? 4 : 0) +
                    (top_left == category ? 8 : 0)

                for (edge_a, edge_b) in case_segments(mask_case)
                    x_a, y_a = edge_point(edge_a, x0, x1, y0, y1)
                    x_b, y_b = edge_point(edge_b, x0, x1, y0, y1)
                    push_unique_segment!(local_segments, normalize_segment(x_a, y_a, x_b, y_b))
                end
            end

            for (x_a, y_a, x_b, y_b) in local_segments
                push!(xs, Float32(x_a), Float32(x_b), NaN32)
                push!(ys, Float32(y_a), Float32(y_b), NaN32)
            end
        end
    end

    return xs, ys
end

function save_contour_plot(path::String, T_grid::Matrix{Int}, gamma_grid::Matrix{Int}; title::String)
    fig = Figure(size=(PLOT_WIDTH, PLOT_HEIGHT))
    ax = Axis(
        fig[1, 1],
        title=title,
        xlabel="ΔCa",
        ylabel="Δx",
        titlesize=AXIS_TITLE_SIZE,
        xlabelsize=AXIS_LABEL_SIZE,
        ylabelsize=AXIS_LABEL_SIZE,
        xticklabelsize=TICK_LABEL_SIZE,
        yticklabelsize=TICK_LABEL_SIZE,
    )

    T_xs, T_ys = categorical_marching_squares(T_grid, DELTA_CAS_010, DELTA_XS_010)
    gamma_xs, gamma_ys = categorical_marching_squares(gamma_grid, DELTA_CAS_010, DELTA_XS_010)

    lines!(ax, T_xs, T_ys; color=T_COLOR, linewidth=LINEWIDTH)
    lines!(ax, gamma_xs, gamma_ys; color=GAMMA_COLOR, linewidth=LINEWIDTH)
    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT)
end

function main()
    if !isempty(PLOT_ONLY_RESULTS)
        println("Running attempt-051 prefix contour replay.")
        println("Plot-only results source: $(PLOT_ONLY_RESULTS)")
        println("Output tag: $(CONTOUR_OUTPUT_TAG), prefix lengths: $(join(PREFIX_LENGTHS, ", "))")
        flush(stdout)

        prefix_summaries, error_count = write_prefix_outputs(PLOT_ONLY_RESULTS, PREFIX_LENGTHS)
        println("Replay source error rows: $(error_count)")
        for summary in prefix_summaries
            println(
                "Saved prefix $(summary.prefix_length) plot to $(summary.plot_path) " *
                "(unique T=$(summary.T_unique), unique gamma=$(summary.gamma_unique))",
            )
        end
        return
    end

    println("Running attempt-051 contour scan.")
    println("Grid: $(length(DELTA_CAS_010)) ΔCa points × $(length(DELTA_XS_010)) Δx points")
    println(
        "Julia threads: $(nthreads()), output tag: $(CONTOUR_OUTPUT_TAG), " *
        "max sequence length: $(ATTEMPT10_MAX_SEQ_LENGTH), remap resolution: $(ATTEMPT51_MAP_RESOLUTION)",
    )
    println("Model overrides: g_h=$(ATTEMPT51_G_H), tau_y=$(ATTEMPT51_TAU_Y), SSCS tmax=$(ATTEMPT51_SSCS_TMAX)")
    println("T0 continuation refinement steps: $(T0_LOCAL_REFINE_STEPS), Ca-window: $(T0_LOCAL_CA_WINDOW)")
    println("Column checkpoint directory: $(SWEEP_DIR)")
    flush(stdout)

    run_or_resume_columns()

    T_counts, gamma_counts, T_examples, gamma_examples, error_count = collect_legends()
    T_lookup = write_single_legend(joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_T_legend.tsv"), T_counts, T_examples)
    gamma_lookup = write_single_legend(joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_gamma_legend.tsv"), gamma_counts, gamma_examples)
    T_grid, gamma_grid = build_category_grids(T_lookup, gamma_lookup)

    results_path = joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_results.tsv")
    plot_path = joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_contours.png")

    write_final_results(results_path, T_lookup, gamma_lookup)
    save_contour_plot(plot_path, T_grid, gamma_grid; title="SSCS contour overlay (12 symbols)")
    prefix_summaries, _ = write_prefix_outputs(results_path, PREFIX_LENGTHS)

    total_points = length(DELTA_CAS_010) * length(DELTA_XS_010)
    println("Successful points: $(total_points - error_count) / $(total_points)")
    println("Unique T encodings: $(length(T_lookup))")
    println("Unique gamma encodings: $(length(gamma_lookup))")
    println("Saved merged results to $(results_path)")
    println("Saved T legend to $(joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_T_legend.tsv"))")
    println("Saved gamma legend to $(joinpath(ATTEMPT51_ROOT, "$(CONTOUR_OUTPUT_TAG)_gamma_legend.tsv"))")
    println("Saved contour plot to $(plot_path)")
    for summary in prefix_summaries
        println(
            "Saved prefix $(summary.prefix_length) plot to $(summary.plot_path) " *
            "(unique T=$(summary.T_unique), unique gamma=$(summary.gamma_unique))",
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
