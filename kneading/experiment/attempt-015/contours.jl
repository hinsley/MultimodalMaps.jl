using Pkg

const ATTEMPT15_ROOT = @__DIR__
const REPO_ROOT_015 = normpath(joinpath(ATTEMPT15_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_015)

include(joinpath(ATTEMPT15_ROOT, "main.jl"))

using Base.Threads
using Printf

const CONTOUR_OUTPUT_TAG = get(ENV, "ATTEMPT015_OUTPUT_TAG", "T_only_dca40_to_37")
const SWEEP_DIR = joinpath(ATTEMPT15_ROOT, "$(CONTOUR_OUTPUT_TAG)_columns")
const T_COLOR = RGBAf(0.86, 0.16, 0.12, 0.75)
const LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT015_CONTOUR_LINEWIDTH", "0.8"))
const DELTA_X_MIN_015 = env_float("ATTEMPT015_DELTA_X_MIN", -1.5)
const DELTA_X_MAX_015 = env_float("ATTEMPT015_DELTA_X_MAX", -0.5)
const GRID_NX_015 = parse(Int, get(ENV, "ATTEMPT015_NX", "500"))
const DELTA_CA_WINDOW_MIN_015 = env_float("ATTEMPT015_DELTA_CA_MIN", -40.0)
const DELTA_CA_WINDOW_MAX_015 = env_float("ATTEMPT015_DELTA_CA_MAX", -37.0)
const DELTA_X_TICK_STEP_015 = env_optional_float("ATTEMPT015_DELTA_X_TICK_STEP")
const DELTA_CA_TICK_STEP_015 = env_optional_float("ATTEMPT015_DELTA_CA_TICK_STEP")
const ATTEMPT11_DELTA_XS = collect(range(-1.5, -0.5, length=500))
const ATTEMPT11_DELTA_CAS = collect(range(-45.0, -20.0, length=500))
const DELTA_XS_015 = collect(range(DELTA_X_MIN_015, DELTA_X_MAX_015, length=GRID_NX_015))
const DELTA_CAS_015 = Float64[
    delta_ca for delta_ca in ATTEMPT11_DELTA_CAS
    if DELTA_CA_WINDOW_MIN_015 <= delta_ca <= DELTA_CA_WINDOW_MAX_015
]
const LOG_LOCK = ReentrantLock()

if DELTA_XS_015 != ATTEMPT11_DELTA_XS
    error("attempt-015 requires the exact 500-point attempt-011 Delta x grid.")
end
if length(DELTA_CAS_015) != 60
    error("Expected 60 Delta Ca values from the attempt-011 grid inside [-40, -37], got $(length(DELTA_CAS_015)).")
end

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

function write_column(path::String, results::Vector{TContourScanResult})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tT_encoding\tT_scs\tstatus")
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
                    join(result.T_scs, ","),
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
            @printf("Skipping completed column %d/%d (Delta Ca=%.6f)\n", col_idx, total_cols, delta_ca)
            flush(stdout)
        finally
            unlock(LOG_LOCK)
        end
        return
    end

    started = time()
    column_results = Vector{TContourScanResult}(undef, total_rows)
    previous_successful = nothing
    for row_idx in length(DELTA_XS_015):-1:1
        delta_x = DELTA_XS_015[row_idx]
        candidate_seed = make_candidate_seed(previous_successful)
        result = run_T_point_safe(delta_x, delta_ca, candidate_seed)
        column_results[row_idx] = result
        if isnothing(result.error_message)
            previous_successful = result
        end
    end
    write_column(path, column_results)
    ok_count = count(result -> isnothing(result.error_message), column_results)

    lock(LOG_LOCK)
    try
        @printf(
            "Saved column %d/%d (Delta Ca=%.6f) with %d/%d successful points in %.2f s\n",
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
    total_cols = length(DELTA_CAS_015)
    total_rows = length(DELTA_XS_015)

    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_015)
        run_column(col_idx, DELTA_CAS_015[col_idx], total_cols, total_rows)
    end
end

function parse_sequence_field(field::AbstractString)
    return isempty(field) ? Int[] : parse.(Int, split(field, ","))
end

function scan_column_files(pass_fn)
    for col_idx in eachindex(DELTA_CAS_015)
        path = column_path(col_idx)
        if !row_is_complete(path, length(DELTA_XS_015))
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
    T_counts = Dict{BigInt, Int}()
    T_examples = Dict{BigInt, Vector{Int}}()
    error_count = 0

    scan_column_files() do fields
        status = fields[8]
        if status != "ok"
            error_count += 1
            return
        end

        T_encoding = parse(BigInt, fields[6])
        T_counts[T_encoding] = get(T_counts, T_encoding, 0) + 1
        if !haskey(T_examples, T_encoding)
            T_examples[T_encoding] = parse_sequence_field(fields[7])
        end
    end

    return T_counts, T_examples, error_count
end

function nearest_index(values::Vector{Float64}, target::Float64, label::String)
    idx = findmin(abs.(values .- target))[2]
    if !isapprox(values[idx], target; atol=1e-5, rtol=0.0)
        error("$(label)=$(target) does not align with plotting grid.")
    end
    return idx
end

function build_T_grid(T_lookup::Dict{BigInt, Int})
    T_grid = fill(0, length(DELTA_CAS_015), length(DELTA_XS_015))
    filled = falses(length(DELTA_CAS_015), length(DELTA_XS_015))

    scan_column_files() do fields
        delta_x = parse(Float64, fields[1])
        delta_ca = parse(Float64, fields[2])
        status = fields[8]
        x_idx = nearest_index(DELTA_CAS_015, delta_ca, "Delta Ca")
        y_idx = nearest_index(DELTA_XS_015, delta_x, "Delta x")

        if filled[x_idx, y_idx]
            error("Duplicate grid entry for Delta x=$(delta_x), Delta Ca=$(delta_ca).")
        end
        filled[x_idx, y_idx] = true

        if status == "ok"
            T_grid[x_idx, y_idx] = T_lookup[parse(BigInt, fields[6])]
        end
    end

    if !all(filled)
        error("One or more T contour grid entries were not filled.")
    end

    return T_grid
end

function write_final_results(path::String, T_lookup::Dict{BigInt, Int})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tT_category_id\tT_encoding\tT_scs\tstatus")
        scan_column_files() do fields
            status = fields[8]
            if status == "ok"
                T_encoding = parse(BigInt, fields[6])
                T_category_id = T_lookup[T_encoding]
                println(io, join([fields[1], fields[2], fields[3], fields[4], fields[5], string(T_category_id), fields[6], fields[7], status], '\t'))
            else
                println(io, join([fields[1], fields[2], fields[3], fields[4], fields[5], "", "", fields[7], status], '\t'))
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

function save_contour_plot(path::String, T_grid::Matrix{Int})
    fig = Figure(size=(1100, 850))
    ax = Axis(fig[1, 1], xlabel="Delta Ca", ylabel="Delta x")
    T_xs, T_ys = categorical_marching_squares(T_grid, DELTA_CAS_015, DELTA_XS_015)
    lines!(ax, T_xs, T_ys; color=T_COLOR, linewidth=LINEWIDTH)
    ax.xticks = fixed_ticks(DELTA_CAS_015, "%.1f", DELTA_CA_TICK_STEP_015)
    ax.yticks = fixed_ticks(DELTA_XS_015, "%.1f", DELTA_X_TICK_STEP_015)
    save(path, fig)
end

function main()
    println("Running attempt-015 T-only contour scan.")
    println("Grid: $(length(DELTA_CAS_015)) Delta Ca points x $(length(DELTA_XS_015)) Delta x points")
    println("Julia threads: $(nthreads()), output tag: $(CONTOUR_OUTPUT_TAG), max sequence length: $(ATTEMPT10_MAX_SEQ_LENGTH)")
    println(
        @sprintf(
            "Delta Ca subset copied from attempt-011 grid: %d values from %.12f to %.12f",
            length(DELTA_CAS_015),
            first(DELTA_CAS_015),
            last(DELTA_CAS_015),
        ),
    )
    println("Column checkpoint directory: $(SWEEP_DIR)")
    flush(stdout)

    run_or_resume_columns()

    T_counts, T_examples, error_count = collect_legend()
    T_legend_path = joinpath(ATTEMPT15_ROOT, "$(CONTOUR_OUTPUT_TAG)_T_legend.tsv")
    T_lookup = write_single_legend(T_legend_path, T_counts, T_examples)
    T_grid = build_T_grid(T_lookup)

    results_path = joinpath(ATTEMPT15_ROOT, "$(CONTOUR_OUTPUT_TAG)_results.tsv")
    plot_path = joinpath(ATTEMPT15_ROOT, "$(CONTOUR_OUTPUT_TAG)_contours.png")

    write_final_results(results_path, T_lookup)
    save_contour_plot(plot_path, T_grid)

    total_points = length(DELTA_CAS_015) * length(DELTA_XS_015)
    println("Successful points: $(total_points - error_count) / $(total_points)")
    println("Unique T encodings: $(length(T_lookup))")
    println("Saved merged results to $(results_path)")
    println("Saved T legend to $(T_legend_path)")
    println("Saved contour plot to $(plot_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
