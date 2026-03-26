using Pkg

const ATTEMPT21_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_021_CONTOUR = normpath(joinpath(ATTEMPT21_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_021_CONTOUR)

include(joinpath(ATTEMPT21_CONTOUR_ROOT, "main.jl"))

using Base.Threads

const CONTOUR_OUTPUT_TAG = get(ENV, "ATTEMPT021_OUTPUT_TAG", "grid1000_branch8_shimizu_morioka")
const SWEEP_DIR = joinpath(ATTEMPT21_ROOT, "$(CONTOUR_OUTPUT_TAG)_columns")
const BLUE_COLOR = RGBAf(0.1, 0.34, 0.9, 0.9)
const RED_COLOR = RGBAf(0.88, 0.18, 0.12, 0.9)
const LOG_LOCK = ReentrantLock()
const ALPHA_INDEX_021 = Dict(@sprintf("%.6f", value) => idx for (idx, value) in enumerate(ALPHAS_021))
const LAMBDA_INDEX_021 = Dict(@sprintf("%.6f", value) => idx for (idx, value) in enumerate(LAMBDAS_021))

column_path(col_idx::Int) = joinpath(SWEEP_DIR, @sprintf("column_%04d.tsv", col_idx))
results_path() = joinpath(ATTEMPT21_ROOT, "$(CONTOUR_OUTPUT_TAG)_results.tsv")
blue_legend_path() = joinpath(ATTEMPT21_ROOT, "$(CONTOUR_OUTPUT_TAG)_blue_legend.tsv")
red_legend_path() = joinpath(ATTEMPT21_ROOT, "$(CONTOUR_OUTPUT_TAG)_red_legend.tsv")
plot_path() = joinpath(ATTEMPT21_ROOT, "$(CONTOUR_OUTPUT_TAG)_contours.png")

function row_is_complete(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_column(path::String, results::Vector{SMScanResult21})
    open(path, "w") do io
        println(io, "alpha\tlambda\tmaxima_count\tblue_mask\tred_mask\tzmax_values\treturn_times\tstatus")
        for result in results
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    string(result.maxima_count),
                    string(UInt32(result.blue_mask)),
                    string(UInt32(result.red_mask)),
                    join_float_list(result.zmax_values),
                    join_float_list(result.return_times),
                    result.status,
                ], '\t'),
            )
        end
    end
end

function parse_result(fields::Vector{SubString{String}})
    return SMScanResult21(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        UInt16(parse(UInt32, fields[4])),
        UInt16(parse(UInt32, fields[5])),
        parse_float_list(fields[6]),
        parse_float_list(fields[7]),
        String(fields[8]),
    )
end

function run_column(col_idx::Int, alpha::Float64, total_cols::Int, total_rows::Int)
    path = column_path(col_idx)
    if row_is_complete(path, total_rows)
        lock(LOG_LOCK)
        try
            @printf("Skipping completed column %d/%d (alpha=%.6f)\n", col_idx, total_cols, alpha)
            flush(stdout)
        finally
            unlock(LOG_LOCK)
        end
        return
    end

    started = time()
    column_results = Vector{SMScanResult21}(undef, total_rows)
    okish_count = 0
    for row_idx in eachindex(LAMBDAS_021)
        result = scan_orbit(alpha, LAMBDAS_021[row_idx])
        column_results[row_idx] = result
        if result.status == "ok" || result.status == "short"
            okish_count += 1
        end
    end
    write_column(path, column_results)

    lock(LOG_LOCK)
    try
        @printf(
            "Saved column %d/%d (alpha=%.6f) with %d/%d usable points in %.2f s\n",
            col_idx,
            total_cols,
            alpha,
            okish_count,
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
    total_cols = length(ALPHAS_021)
    total_rows = length(LAMBDAS_021)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_021)
        run_column(col_idx, ALPHAS_021[col_idx], total_cols, total_rows)
    end
end

function scan_column_files(pass_fn)
    for col_idx in eachindex(ALPHAS_021)
        path = column_path(col_idx)
        row_is_complete(path, length(LAMBDAS_021)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_result(split(line, '\t')))
            end
        end
    end
end

function write_legend(path::String, counts::Dict{UInt16, Int})
    open(path, "w") do io
        println(io, "category_id\tmask\tcount\tbit_indices")
        for (cat_idx, mask) in enumerate(sort!(collect(keys(counts))))
            bits = Int[]
            for bit_idx in 0:15
                if (mask >> bit_idx) & UInt16(1) == UInt16(1)
                    push!(bits, bit_idx + 1)
                end
            end
            println(io, join([string(cat_idx), string(UInt32(mask)), string(counts[mask]), join(bits, ",")], '\t'))
        end
    end

    lookup = Dict{UInt16, Int}()
    for (cat_idx, mask) in enumerate(sort!(collect(keys(counts))))
        lookup[mask] = cat_idx
    end
    return lookup
end

function build_category_grids()
    blue_counts = Dict{UInt16, Int}()
    red_counts = Dict{UInt16, Int}()

    scan_column_files() do result
        if result.status == "ok" || result.status == "short"
            blue_counts[result.blue_mask] = get(blue_counts, result.blue_mask, 0) + 1
            red_counts[result.red_mask] = get(red_counts, result.red_mask, 0) + 1
        end
    end

    blue_lookup = write_legend(blue_legend_path(), blue_counts)
    red_lookup = write_legend(red_legend_path(), red_counts)

    blue_grid = fill(0, length(ALPHAS_021), length(LAMBDAS_021))
    red_grid = fill(0, length(ALPHAS_021), length(LAMBDAS_021))
    filled = falses(length(ALPHAS_021), length(LAMBDAS_021))

    scan_column_files() do result
        alpha_idx = get(ALPHA_INDEX_021, @sprintf("%.6f", result.alpha), nothing)
        lambda_idx = get(LAMBDA_INDEX_021, @sprintf("%.6f", result.lambda), nothing)
        if isnothing(alpha_idx) || isnothing(lambda_idx)
            error("Result point does not align with plotting grid.")
        end
        filled[alpha_idx, lambda_idx] && error("Duplicate grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[alpha_idx, lambda_idx] = true
        if result.status == "ok" || result.status == "short"
            blue_grid[alpha_idx, lambda_idx] = blue_lookup[result.blue_mask]
            red_grid[alpha_idx, lambda_idx] = red_lookup[result.red_mask]
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return blue_grid, red_grid, blue_lookup, red_lookup
end

function write_final_results(path::String)
    open(path, "w") do io_out
        println(io_out, "alpha\tlambda\tmaxima_count\tblue_mask\tred_mask\tzmax_values\treturn_times\tstatus")
        for col_idx in eachindex(ALPHAS_021)
            open(column_path(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
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
        existing == segment && return
    end
    push!(segments, segment)
end

function categorical_marching_squares(grid::Matrix{Int}, x_values::Vector{Float64}, y_values::Vector{Float64})
    xs = Float32[]
    ys = Float32[]
    local_segments = NTuple{4, Float64}[]

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
                category <= 0 && continue
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

function save_contour_plot(path::String, blue_grid::Matrix{Int}, red_grid::Matrix{Int})
    fig = Figure(size=(ATTEMPT021_FIG_WIDTH, ATTEMPT021_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka right-branch scan: near-saddle minima (blue) and z-max criticality (red)",
    )
    blue_xs, blue_ys = categorical_marching_squares(blue_grid, ALPHAS_021, LAMBDAS_021)
    red_xs, red_ys = categorical_marching_squares(red_grid, ALPHAS_021, LAMBDAS_021)
    lines!(ax, blue_xs, blue_ys; color=BLUE_COLOR, linewidth=ATTEMPT021_LINEWIDTH)
    lines!(ax, red_xs, red_ys; color=RED_COLOR, linewidth=ATTEMPT021_LINEWIDTH)
    xlims!(ax, ATTEMPT021_ALPHA_MIN, ATTEMPT021_ALPHA_MAX)
    ylims!(ax, ATTEMPT021_LAMBDA_MIN, ATTEMPT021_LAMBDA_MAX)
    save(path, fig; px_per_unit=ATTEMPT021_PX_PER_UNIT)
end

function main()
    println("Running attempt-021 Shimizu-Morioka branch contour scan.")
    println("Grid: $(length(ALPHAS_021)) alpha points × $(length(LAMBDAS_021)) lambda points")
    println("Threads: $(nthreads()), max z-maxima per orbit: $(ATTEMPT021_MAX_ZMAX)")
    println("Sweep dir: $(SWEEP_DIR)")
    println("Near-saddle thresholds: speed <= $(ATTEMPT021_NEAR_SADDLE_SPEED), radius <= $(ATTEMPT021_NEAR_SADDLE_RADIUS)")
    flush(stdout)

    run_or_resume_columns()

    blue_grid, red_grid, _, _ = build_category_grids()
    write_final_results(results_path())
    save_contour_plot(plot_path(), blue_grid, red_grid)

    usable = 0
    total = length(ALPHAS_021) * length(LAMBDAS_021)
    scan_column_files() do result
        if result.status == "ok" || result.status == "short"
            usable += 1
        end
    end

    println("Usable points: $(usable) / $(total)")
    println("Saved merged results to $(results_path())")
    println("Saved blue legend to $(blue_legend_path())")
    println("Saved red legend to $(red_legend_path())")
    println("Saved contour plot to $(plot_path())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
