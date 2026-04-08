using Pkg

const ATTEMPT31_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_031_CONTOUR = normpath(joinpath(ATTEMPT31_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_031_CONTOUR)

include(joinpath(ATTEMPT31_CONTOUR_ROOT, "main.jl"))

using Base.Threads
using CairoMakie
using Colors
using Printf

const CONTOUR_OUTPUT_TAG_031 = get(
    ENV,
    "ATTEMPT031_OUTPUT_TAG",
    "grid500_branch16_xpartialabsx_plot7_shimizu_morioka_cpu",
)
const SWEEP_SOURCE_TAG_031 = get(ENV, "ATTEMPT031_SWEEP_TAG", CONTOUR_OUTPUT_TAG_031)
const SWEEP_DIR_031 = joinpath(ATTEMPT31_ROOT, "$(SWEEP_SOURCE_TAG_031)_columns")
const LOG_LOCK_031 = ReentrantLock()
const OVERLAY_ITERATE_START_031 = max(1, parse(Int, get(ENV, "ATTEMPT031_OVERLAY_ITERATE_START", "2")))
const OVERLAY_ITERATE_END_031 = min(
    ATTEMPT031_MAX_EVENT_ITERATES,
    parse(Int, get(ENV, "ATTEMPT031_OVERLAY_ITERATE_END", "7")),
)
const RUN_COLUMNS_031 = lowercase(get(ENV, "ATTEMPT031_RUN_COLUMNS", "true")) in ("1", "true", "yes")
const WRITE_MERGED_RESULTS_031 = lowercase(get(ENV, "ATTEMPT031_WRITE_MERGED_RESULTS", "true")) in ("1", "true", "yes")
const WRITE_LEGEND_031 = lowercase(get(ENV, "ATTEMPT031_WRITE_LEGEND", "true")) in ("1", "true", "yes")

column_path_031(col_idx::Int) = joinpath(SWEEP_DIR_031, @sprintf("column_%04d.tsv", col_idx))
results_path_031() = joinpath(ATTEMPT31_ROOT, "$(CONTOUR_OUTPUT_TAG_031)_results.tsv")
legend_path_031() = joinpath(ATTEMPT31_ROOT, "$(CONTOUR_OUTPUT_TAG_031)_iterate_colors.tsv")
plot_path_031() = joinpath(ATTEMPT31_ROOT, "$(CONTOUR_OUTPUT_TAG_031)_contours.png")

function row_is_complete_031(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_column_031(path::String, results::Vector{SMSensitivityResult31})
    open(path, "w") do io
        println(
            io,
            "alpha\tlambda\tabsxmax_count\tabsxmax_sensitivity_values\tabsxmax_return_times\tabsxmax_states\tstatus",
        )
        for result in results
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    string(result.absxmax_count),
                    join_float_list_031(result.absxmax_sensitivity_values),
                    join_float_list_031(result.absxmax_return_times),
                    join_state_list_031(result.absxmax_states),
                    result.status,
                ], '\t'),
            )
        end
    end
end

function parse_result_031(fields::Vector{SubString{String}})
    length(fields) == 7 || error("Expected 7 fields, got $(length(fields))")
    return SMSensitivityResult31(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        parse_float_list_031(fields[4]),
        parse_float_list_031(fields[5]),
        parse_state_list_031(fields[6]),
        String(fields[7]),
    )
end

function run_column_031(col_idx::Int, alpha::Float64, total_cols::Int, total_rows::Int)
    path = column_path_031(col_idx)
    if row_is_complete_031(path, total_rows)
        lock(LOG_LOCK_031)
        try
            @printf("Skipping completed column %d/%d (alpha=%.6f)\n", col_idx, total_cols, alpha)
            flush(stdout)
        finally
            unlock(LOG_LOCK_031)
        end
        return
    end

    started = time()
    column_results = Vector{SMSensitivityResult31}(undef, total_rows)
    usable = 0
    for row_idx in eachindex(LAMBDAS_031)
        result = scan_orbit_031(alpha, LAMBDAS_031[row_idx])
        column_results[row_idx] = result
        result_usable_031(result) && (usable += 1)
    end
    write_column_031(path, column_results)

    lock(LOG_LOCK_031)
    try
        @printf(
            "Saved column %d/%d (alpha=%.6f) with %d/%d usable points in %.2f s\n",
            col_idx,
            total_cols,
            alpha,
            usable,
            total_rows,
            time() - started,
        )
        flush(stdout)
    finally
        unlock(LOG_LOCK_031)
    end
end

function run_or_resume_columns_031()
    mkpath(SWEEP_DIR_031)
    total_cols = length(ALPHAS_031)
    total_rows = length(LAMBDAS_031)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_031)
        run_column_031(col_idx, ALPHAS_031[col_idx], total_cols, total_rows)
    end
end

function scan_column_files_031(pass_fn)
    for col_idx in eachindex(ALPHAS_031)
        path = column_path_031(col_idx)
        row_is_complete_031(path, length(LAMBDAS_031)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_result_031(split(line, '\t')))
            end
        end
    end
end

function write_final_results_031(path::String)
    open(path, "w") do io_out
        println(io_out, "alpha\tlambda\tabsxmax_count\tabsxmax_sensitivity_values\tabsxmax_return_times\tabsxmax_states\tstatus")
        for col_idx in eachindex(ALPHAS_031)
            open(column_path_031(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
end

function nearest_axis_index_031(value::Float64, axis::Vector{Float64}, axis_name::String)
    idx = searchsortedfirst(axis, value)
    candidates = Int[]
    idx <= length(axis) && push!(candidates, idx)
    idx > 1 && push!(candidates, idx - 1)
    isempty(candidates) && error("No candidate index found for $(axis_name)=$(value)")

    best_idx = candidates[1]
    best_error = abs(axis[best_idx] - value)
    for candidate in candidates[2:end]
        candidate_error = abs(axis[candidate] - value)
        if candidate_error < best_error
            best_idx = candidate
            best_error = candidate_error
        end
    end
    return best_idx
end

function build_iterate_grids_031(count_fn, values_fn)
    grids = [fill(NaN, length(LAMBDAS_031), length(ALPHAS_031)) for _ in 1:ATTEMPT031_MAX_EVENT_ITERATES]
    filled = falses(length(LAMBDAS_031), length(ALPHAS_031))

    scan_column_files_031() do result
        alpha_idx = nearest_axis_index_031(result.alpha, ALPHAS_031, "alpha")
        lambda_idx = nearest_axis_index_031(result.lambda, LAMBDAS_031, "lambda")
        filled[lambda_idx, alpha_idx] = true

        if result_usable_031(result)
            values = values_fn(result)
            for iterate in 1:min(count_fn(result), ATTEMPT031_MAX_EVENT_ITERATES, length(values))
                grids[iterate][lambda_idx, alpha_idx] = values[iterate]
            end
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return grids
end

function build_sensitivity_magnitude_grids_031(value_grids::Vector{Matrix{Float64}})
    return [abs.(grid) for grid in value_grids]
end

function edge_point_031(
    edge_id::Int,
    values::NTuple{4, Float64},
    x_tl::Float64,
    y_tl::Float64,
    x_tr::Float64,
    y_tr::Float64,
    x_br::Float64,
    y_br::Float64,
    x_bl::Float64,
    y_bl::Float64,
    level::Float64,
)
    z_tl, z_tr, z_br, z_bl = values
    if edge_id == 1
        z1, z2 = z_tl, z_tr
        x1, y1, x2, y2 = x_tl, y_tl, x_tr, y_tr
    elseif edge_id == 2
        z1, z2 = z_tr, z_br
        x1, y1, x2, y2 = x_tr, y_tr, x_br, y_br
    elseif edge_id == 3
        z1, z2 = z_br, z_bl
        x1, y1, x2, y2 = x_br, y_br, x_bl, y_bl
    else
        z1, z2 = z_bl, z_tl
        x1, y1, x2, y2 = x_bl, y_bl, x_tl, y_tl
    end

    d1 = z1 - level
    d2 = z2 - level
    if !isfinite(d1) || !isfinite(d2) || (d1 == 0.0 && d2 == 0.0)
        return nothing
    elseif d1 == 0.0
        theta = 0.0
    elseif d2 == 0.0
        theta = 1.0
    elseif signbit(d1) == signbit(d2)
        return nothing
    else
        theta = d1 / (d1 - d2)
    end

    return ((1.0 - theta) * x1 + theta * x2, (1.0 - theta) * y1 + theta * y2)
end

function append_segment_if_031!(
    segments::Vector{NTuple{4, Float64}},
    point_a::Union{Nothing, Tuple{Float64, Float64}},
    point_b::Union{Nothing, Tuple{Float64, Float64}},
)
    (isnothing(point_a) || isnothing(point_b)) && return 0
    x1, y1 = point_a
    x2, y2 = point_b
    push!(segments, (x1, y1, x2, y2))
    return 1
end

function append_march_square_zero_segments_031!(
    segments::Vector{NTuple{4, Float64}},
    values::NTuple{4, Float64},
    x_tl::Float64,
    y_tl::Float64,
    x_tr::Float64,
    y_tr::Float64,
    x_br::Float64,
    y_br::Float64,
    x_bl::Float64,
    y_bl::Float64;
    level::Float64=0.0,
)
    z_tl, z_tr, z_br, z_bl = values
    case_idx =
        (z_tl >= level ? 8 : 0) +
        (z_tr >= level ? 4 : 0) +
        (z_br >= level ? 2 : 0) +
        (z_bl >= level ? 1 : 0)

    (case_idx == 0 || case_idx == 15) && return 0

    p1 = edge_point_031(1, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p2 = edge_point_031(2, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p3 = edge_point_031(3, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p4 = edge_point_031(4, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    points = (p1, p2, p3, p4)

    added = 0
    if case_idx == 5 || case_idx == 10
        center_value = 0.25 * (z_tl + z_tr + z_br + z_bl)
        pairing =
            case_idx == 5 ?
            (center_value >= level ? ((1, 2), (3, 4)) : ((1, 4), (2, 3))) :
            (center_value >= level ? ((1, 4), (2, 3)) : ((1, 2), (3, 4)))
        for (edge_a, edge_b) in pairing
            added += append_segment_if_031!(segments, points[edge_a], points[edge_b])
        end
        return added
    end

    pairing =
        case_idx == 1 ? ((4, 3),) :
        case_idx == 2 ? ((3, 2),) :
        case_idx == 3 ? ((4, 2),) :
        case_idx == 4 ? ((1, 2),) :
        case_idx == 6 ? ((1, 3),) :
        case_idx == 7 ? ((1, 4),) :
        case_idx == 8 ? ((1, 4),) :
        case_idx == 9 ? ((1, 3),) :
        case_idx == 11 ? ((1, 2),) :
        case_idx == 12 ? ((4, 2),) :
        case_idx == 13 ? ((3, 2),) :
        case_idx == 14 ? ((4, 3),) :
        ()

    for (edge_a, edge_b) in pairing
        added += append_segment_if_031!(segments, points[edge_a], points[edge_b])
    end
    return added
end

function iterate_segments_031(value_grid::Matrix{Float64})
    segments = NTuple{4, Float64}[]
    n_lambda_cells = length(LAMBDAS_031) - 1
    n_alpha_cells = length(ALPHAS_031) - 1

    @inbounds for j in 1:n_lambda_cells
        for i in 1:n_alpha_cells
            values = (
                value_grid[j, i],
                value_grid[j, i + 1],
                value_grid[j + 1, i + 1],
                value_grid[j + 1, i],
            )
            all(isfinite, values) || continue

            append_march_square_zero_segments_031!(
                segments,
                values,
                ALPHAS_031[i],
                LAMBDAS_031[j],
                ALPHAS_031[i + 1],
                LAMBDAS_031[j],
                ALPHAS_031[i + 1],
                LAMBDAS_031[j + 1],
                ALPHAS_031[i],
                LAMBDAS_031[j + 1],
            )
        end
    end

    return segments
end

function segments_to_polyline_031(segments::Vector{NTuple{4, Float64}})
    xs = Float64[]
    ys = Float64[]
    sizehint!(xs, 3 * length(segments))
    sizehint!(ys, 3 * length(segments))

    for (x1, y1, x2, y2) in segments
        push!(xs, x1)
        push!(ys, y1)
        push!(xs, x2)
        push!(ys, y2)
        push!(xs, NaN)
        push!(ys, NaN)
    end

    return xs, ys
end

function iterate_colors_031(iterates::Vector{Int})
    palette = distinguishable_colors(
        length(iterates),
        [RGBf(1.0, 1.0, 1.0), RGBf(0.0, 0.0, 0.0)],
    )
    return Dict(iterate => palette[idx] for (idx, iterate) in enumerate(iterates))
end

function write_legend_031(path::String, color_map)
    open(path, "w") do io
        println(io, "iterate\tr\tg\tb")
        for iterate in sort(collect(keys(color_map)))
            color = color_map[iterate]
            println(
                io,
                join([
                    string(iterate),
                    @sprintf("%.6f", color.r),
                    @sprintf("%.6f", color.g),
                    @sprintf("%.6f", color.b),
                ], '\t'),
            )
        end
    end
end

function build_contour_figure_031(iterate_segments::Dict{Int, Vector{NTuple{4, Float64}}}, color_map)
    fig = Figure(size=(ATTEMPT031_FIG_WIDTH, ATTEMPT031_FIG_HEIGHT), backgroundcolor=:white)
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka forward-sensitivity contours of ∂(x²_{n+1})/∂x_n, iterates $(OVERLAY_ITERATE_START_031):$(OVERLAY_ITERATE_END_031)",
        xgridvisible=false,
        ygridvisible=false,
    )

    for iterate in sort(collect(keys(iterate_segments)))
        segments = iterate_segments[iterate]
        isempty(segments) && continue
        xs, ys = segments_to_polyline_031(segments)
        lines!(ax, xs, ys; color=color_map[iterate], linewidth=ATTEMPT031_LINEWIDTH)
    end

    xlims!(ax, ATTEMPT031_ALPHA_MIN, ATTEMPT031_ALPHA_MAX)
    ylims!(ax, ATTEMPT031_LAMBDA_MIN, ATTEMPT031_LAMBDA_MAX)
    return fig
end

function main()
    println("Running attempt-031 Shimizu-Morioka forward-sensitivity sweep and contour plot.")
    println("Grid: $(length(ALPHAS_031)) alpha points x $(length(LAMBDAS_031)) lambda points")
    println("Threads: $(nthreads()), stored |x|-max iterates: $(ATTEMPT031_MAX_EVENT_ITERATES), contour overlay: $(OVERLAY_ITERATE_START_031):$(OVERLAY_ITERATE_END_031)")
    println("Sweep dir: $(SWEEP_DIR_031)")
    flush(stdout)

    if RUN_COLUMNS_031
        run_or_resume_columns_031()
    else
        println("Reusing existing sweep columns only (ATTEMPT031_RUN_COLUMNS=false).")
    end

    sensitivity_grids = build_iterate_grids_031(result -> result.absxmax_count, result -> result.absxmax_sensitivity_values)
    _ = build_sensitivity_magnitude_grids_031(sensitivity_grids)

    if WRITE_MERGED_RESULTS_031
        write_final_results_031(results_path_031())
    end

    iterates = collect(OVERLAY_ITERATE_START_031:OVERLAY_ITERATE_END_031)
    iterate_segment_map = Dict{Int, Vector{NTuple{4, Float64}}}()
    for iterate in iterates
        segments = iterate_segments_031(sensitivity_grids[iterate])
        iterate_segment_map[iterate] = segments
        @printf("Iterate %d: emitted_segments=%d\n", iterate, length(segments))
        flush(stdout)
    end

    color_map = iterate_colors_031(iterates)
    WRITE_LEGEND_031 && write_legend_031(legend_path_031(), color_map)

    fig = build_contour_figure_031(iterate_segment_map, color_map)
    save(plot_path_031(), fig; px_per_unit=ATTEMPT031_PX_PER_UNIT)

    usable = 0
    total = length(ALPHAS_031) * length(LAMBDAS_031)
    scan_column_files_031() do result
        result_usable_031(result) && (usable += 1)
    end

    println("Usable points: $(usable) / $(total)")
    WRITE_MERGED_RESULTS_031 ? println("Saved merged results to $(results_path_031())") : println("Skipped merged TSV writes (ATTEMPT031_WRITE_MERGED_RESULTS=false)")
    WRITE_LEGEND_031 && println("Saved iterate color legend to $(legend_path_031())")
    println("Saved contour plot to $(plot_path_031())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
