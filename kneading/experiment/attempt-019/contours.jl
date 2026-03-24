using Pkg

const ATTEMPT19_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_019 = normpath(joinpath(ATTEMPT19_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_019)

include(joinpath(ATTEMPT19_CONTOUR_ROOT, "main.jl"))

using Base.Threads
using Printf

const CONTOUR_OUTPUT_TAG = get(ENV, "ATTEMPT019_OUTPUT_TAG", "grid500_ordinal5")
const SWEEP_DIR = joinpath(ATTEMPT19_ROOT, "$(CONTOUR_OUTPUT_TAG)_columns")
const T_COLOR = RGBAf(0.86, 0.16, 0.12, 0.75)
const LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT019_CONTOUR_LINEWIDTH", "0.8"))
const FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT019_FIG_WIDTH", "1100"))
const FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT019_FIG_HEIGHT", "850"))
const PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT019_PX_PER_UNIT", "1.0"))
const LOG_LOCK = ReentrantLock()

struct ContourScanResult19
    delta_x::Float64
    delta_ca::Float64
    T0_V::Float64
    T0_Ca::Float64
    T0_method::String
    ca_minima::Vector{Float64}
    ordinal_pattern::Vector{Int}
    ordinal_encoding::Union{BigInt, Nothing}
    error_message::Union{Nothing, String}
end

function finalize_contour_point(
    delta_x::Float64,
    delta_ca::Float64,
    p,
    saddle_data,
    T0::SVector{6, Float64},
    T0_method::String,
)::ContourScanResult19
    ca_minima = collect_T_ca_minima(p, T0, saddle_data.x_eq_SF)
    pattern = ordinal_pattern(ca_minima)

    return ContourScanResult19(
        delta_x,
        delta_ca,
        Float64(T0[6]),
        Float64(T0[5]),
        T0_method,
        ca_minima,
        pattern,
        encode_sequence(pattern),
        nothing,
    )
end

function run_contour_point(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::ContourScanResult19
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
)::ContourScanResult19
    try
        return run_contour_point(delta_x, delta_ca, candidate_seed)
    catch err
        return ContourScanResult19(delta_x, delta_ca, NaN, NaN, "", Float64[], Int[], nothing, sprint(showerror, err))
    end
end

make_candidate_seed(previous_successful::Union{Nothing, ContourScanResult19}) =
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

function join_float_list(values::Vector{Float64})
    return join((@sprintf("%.9f", value) for value in values), ",")
end

function write_row(path::String, results::Vector{ContourScanResult19})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tordinal_encoding\tca_minima\tordinal_pattern\tstatus")
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
                    isnothing(result.ordinal_encoding) ? "" : string(result.ordinal_encoding),
                    join_float_list(result.ca_minima),
                    join(result.ordinal_pattern, ","),
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
    column_results = Vector{ContourScanResult19}(undef, total_rows)
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

parse_pattern_field(field::AbstractString) = isempty(field) ? Int[] : parse.(Int, split(field, ","))
parse_float_field(field::AbstractString) = isempty(field) ? Float64[] : parse.(Float64, split(field, ","))

function scan_column_files(pass_fn)
    for col_idx in eachindex(DELTA_CAS_010)
        path = column_path(col_idx)
        if !row_is_complete(path, length(DELTA_XS_010))
            error("Missing or incomplete column file: $(path)")
        end
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(split(line, '\t'))
            end
        end
    end
end

function collect_legend()
    counts = Dict{BigInt, Int}()
    examples = Dict{BigInt, Vector{Int}}()
    error_count = 0

    scan_column_files() do fields
        status = fields[9]
        if status != "ok"
            error_count += 1
            return
        end

        encoding = parse(BigInt, fields[6])
        counts[encoding] = get(counts, encoding, 0) + 1
        if !haskey(examples, encoding)
            examples[encoding] = parse_pattern_field(fields[8])
        end
    end

    return counts, examples, error_count
end

function nearest_index(values::Vector{Float64}, target::Float64, label::String)
    idx = findmin(abs.(values .- target))[2]
    if !isapprox(values[idx], target; atol=1e-5, rtol=0.0)
        error("$(label)=$(target) does not align with plotting grid.")
    end
    return idx
end

function build_category_grid(lookup::Dict{BigInt, Int})
    grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))

    scan_column_files() do fields
        delta_x = parse(Float64, fields[1])
        delta_ca = parse(Float64, fields[2])
        status = fields[9]
        x_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
        y_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")

        if filled[x_idx, y_idx]
            error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
        end
        filled[x_idx, y_idx] = true

        if status == "ok"
            grid[x_idx, y_idx] = lookup[parse(BigInt, fields[6])]
        end
    end

    if !all(filled)
        error("One or more contour grid entries were not filled.")
    end

    return grid
end

function results_path()
    return joinpath(ATTEMPT19_ROOT, "$(CONTOUR_OUTPUT_TAG)_results.tsv")
end

function legend_path()
    return joinpath(ATTEMPT19_ROOT, "$(CONTOUR_OUTPUT_TAG)_legend.tsv")
end

function plot_path()
    return joinpath(ATTEMPT19_ROOT, "$(CONTOUR_OUTPUT_TAG)_contours.png")
end

function write_final_results(path::String, lookup::Dict{BigInt, Int})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tordinal_category_id\tordinal_encoding\tca_minima\tordinal_pattern\tstatus")
        scan_column_files() do fields
            status = fields[9]
            if status == "ok"
                encoding = parse(BigInt, fields[6])
                println(
                    io,
                    join([
                        fields[1],
                        fields[2],
                        fields[3],
                        fields[4],
                        fields[5],
                        string(lookup[encoding]),
                        fields[6],
                        fields[7],
                        fields[8],
                        status,
                    ], '\t'),
                )
            else
                println(io, join([fields[1], fields[2], fields[3], fields[4], fields[5], "", "", fields[7], fields[8], status], '\t'))
            end
        end
    end
end

function write_legend(path::String, counts::Dict{BigInt, Int}, examples::Dict{BigInt, Vector{Int}})
    encodings = sort(collect(keys(counts)))
    lookup = Dict(encoding => idx for (idx, encoding) in enumerate(encodings))

    open(path, "w") do io
        println(io, "category_id\tcount\tencoding\tordinal_pattern")
        for encoding in encodings
            println(io, join([string(lookup[encoding]), string(counts[encoding]), string(encoding), join(examples[encoding], ",")], '\t'))
        end
    end

    return lookup
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
    elseif mask_case == 7
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

function save_contour_plot(path::String, grid::Matrix{Int})
    fig = Figure(size=(FIG_WIDTH, FIG_HEIGHT))
    ax = Axis(fig[1, 1], xlabel="ΔCa", ylabel="Δx")

    xs, ys = categorical_marching_squares(grid, DELTA_CAS_010, DELTA_XS_010)
    lines!(ax, xs, ys; color=T_COLOR, linewidth=LINEWIDTH)
    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig; px_per_unit=PX_PER_UNIT)
end

function main()
    println("Running attempt-019 ordinal-pattern contour scan.")
    println("Grid: $(length(DELTA_CAS_010)) ΔCa points × $(length(DELTA_XS_010)) Δx points")
    println(
        "Julia threads: $(nthreads()), output tag: $(CONTOUR_OUTPUT_TAG), " *
        "ordinal length: $(ATTEMPT19_ORDINAL_LENGTH), remap resolution: $(ATTEMPT17_MAP_RESOLUTION)",
    )
    println("T0 continuation refinement steps: $(T0_LOCAL_REFINE_STEPS), Ca-window: $(T0_LOCAL_CA_WINDOW)")
    println("Column checkpoint directory: $(SWEEP_DIR)")
    flush(stdout)

    run_or_resume_columns()

    counts, examples, error_count = collect_legend()
    lookup = write_legend(legend_path(), counts, examples)
    grid = build_category_grid(lookup)

    write_final_results(results_path(), lookup)
    save_contour_plot(plot_path(), grid)

    total_points = length(DELTA_CAS_010) * length(DELTA_XS_010)
    println("Successful points: $(total_points - error_count) / $(total_points)")
    println("Unique ordinal encodings: $(length(lookup))")
    println("Saved merged results to $(results_path())")
    println("Saved legend to $(legend_path())")
    println("Saved contour plot to $(plot_path())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
