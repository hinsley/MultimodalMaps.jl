using Pkg

const ATTEMPT25_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_025_CONTOUR = normpath(joinpath(ATTEMPT25_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025_CONTOUR)

include(joinpath(ATTEMPT25_CONTOUR_ROOT, "main.jl"))

using Base.Threads
using Random

const CONTOUR_OUTPUT_TAG_025 = get(
    ENV,
    "ATTEMPT025_OUTPUT_TAG",
    "grid2000_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_retired_shimizu_morioka_cpu",
)
const SWEEP_SOURCE_TAG_025 = get(ENV, "ATTEMPT025_SWEEP_TAG", CONTOUR_OUTPUT_TAG_025)
const SWEEP_DIR_025 = joinpath(ATTEMPT25_ROOT, "$(SWEEP_SOURCE_TAG_025)_columns")
const LOG_LOCK_025 = ReentrantLock()
const PLOT_ITERATES_025 = collect(1:ATTEMPT025_PLOT_ITERATE_CAP)
const OVERLAY_ITERATE_START_025 = parse(Int, get(ENV, "ATTEMPT027_OVERLAY_ITERATE_START", "2"))
const OVERLAY_ITERATE_END_025 = min(
    ATTEMPT025_PLOT_ITERATE_CAP,
    parse(Int, get(ENV, "ATTEMPT027_OVERLAY_ITERATE_END", string(ATTEMPT025_PLOT_ITERATE_CAP))),
)
const RUN_COLUMNS_025 = lowercase(get(ENV, "ATTEMPT025_RUN_COLUMNS", "true")) in ("1", "true", "yes")
const WRITE_MERGED_RESULTS_025 = lowercase(get(ENV, "ATTEMPT025_WRITE_MERGED_RESULTS", "true")) in ("1", "true", "yes")
const WRITE_ITERATE_STATS_025 = lowercase(get(ENV, "ATTEMPT025_WRITE_ITERATE_STATS", "true")) in ("1", "true", "yes")
const DRAW_INCREMENT_OVERLAY_025 = lowercase(get(ENV, "ATTEMPT025_DRAW_INCREMENT_OVERLAY", "false")) in ("1", "true", "yes")
const WRITE_INCREMENT_COUNTS_025 = lowercase(get(ENV, "ATTEMPT025_WRITE_INCREMENT_COUNTS", "false")) in ("1", "true", "yes")
const EVAL_MISSING_025 = UInt8(0)
const EVAL_CONSTANT_025 = UInt8(1)
const EVAL_MIXED_025 = UInt8(2)
const EMPTY_FLOAT4_025 = (NaN, NaN, NaN, NaN)
const EMPTY_INT4_025 = (0, 0, 0, 0)
const EMPTY_SIGN4_025 = (Int8(0), Int8(0), Int8(0), Int8(0))

column_path_025(col_idx::Int) = joinpath(SWEEP_DIR_025, @sprintf("column_%04d.tsv", col_idx))
results_path_025() = joinpath(ATTEMPT25_ROOT, "$(CONTOUR_OUTPUT_TAG_025)_results.tsv")
legend_path_025() = joinpath(ATTEMPT25_ROOT, "$(CONTOUR_OUTPUT_TAG_025)_iterate_colors.tsv")
plot_path_025() = joinpath(ATTEMPT25_ROOT, "$(CONTOUR_OUTPUT_TAG_025)_contours.png")
iterate_stats_path_025() = joinpath(ATTEMPT25_ROOT, "$(CONTOUR_OUTPUT_TAG_025)_iterate_stats.tsv")
increment_counts_path_025() = joinpath(ATTEMPT25_ROOT, "$(CONTOUR_OUTPUT_TAG_025)_increment_counts.tsv")

struct SquareEvaluation25
    status::UInt8
    current_dot::NTuple{4, Float64}
    current_time::NTuple{4, Float64}
    next_time::NTuple{4, Float64}
    sign::NTuple{4, Int8}
    effective_iterate::NTuple{4, Int}
end

mutable struct IterateStats25
    missing_data::Int
    constant_sign::Int
    incremented::Int
    contoured_squares::Int
    emitted_segments::Int
end

struct SkipState25
    tl::Matrix{UInt8}
    tr::Matrix{UInt8}
    br::Matrix{UInt8}
    bl::Matrix{UInt8}
end

function zero_iterate_stats_025()
    return IterateStats25(0, 0, 0, 0, 0)
end

function missing_evaluation_025()
    return SquareEvaluation25(EVAL_MISSING_025, EMPTY_FLOAT4_025, EMPTY_FLOAT4_025, EMPTY_FLOAT4_025, EMPTY_SIGN4_025, EMPTY_INT4_025)
end

function row_is_complete_025(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_column_025(path::String, results::Vector{SMAbsXResult25})
    open(path, "w") do io
        println(
            io,
            "alpha\tlambda\tabsxmax_count\tabsxmax_dot_values\tabsxmax_return_times\tabsxmax_states\tstatus",
        )
        for result in results
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    string(result.absxmax_count),
                    join_float_list_025(result.absxmax_dot_values),
                    join_float_list_025(result.absxmax_return_times),
                    join_state_list_025(result.absxmax_states),
                    result.status,
                ], '\t'),
            )
        end
    end
end

function parse_result_025(fields::Vector{SubString{String}})
    length(fields) == 7 || error("Expected 7 fields, got $(length(fields))")
    return SMAbsXResult25(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        parse_float_list_025(fields[4]),
        parse_float_list_025(fields[5]),
        parse_state_list_025(fields[6]),
        String(fields[7]),
    )
end

result_usable_025(result::SMAbsXResult25) = result.status == "ok" || result.status == "short"

function run_column_025(col_idx::Int, alpha::Float64, total_cols::Int, total_rows::Int)
    path = column_path_025(col_idx)
    if row_is_complete_025(path, total_rows)
        lock(LOG_LOCK_025)
        try
            @printf("Skipping completed column %d/%d (alpha=%.6f)\n", col_idx, total_cols, alpha)
            flush(stdout)
        finally
            unlock(LOG_LOCK_025)
        end
        return
    end

    started = time()
    column_results = Vector{SMAbsXResult25}(undef, total_rows)
    usable = 0
    for row_idx in eachindex(LAMBDAS_025)
        result = scan_orbit_025(alpha, LAMBDAS_025[row_idx])
        column_results[row_idx] = result
        result_usable_025(result) && (usable += 1)
    end
    write_column_025(path, column_results)

    lock(LOG_LOCK_025)
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
        unlock(LOG_LOCK_025)
    end
end

function run_or_resume_columns_025()
    mkpath(SWEEP_DIR_025)
    total_cols = length(ALPHAS_025)
    total_rows = length(LAMBDAS_025)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_025)
        run_column_025(col_idx, ALPHAS_025[col_idx], total_cols, total_rows)
    end
end

function scan_column_files_025(pass_fn)
    for col_idx in eachindex(ALPHAS_025)
        path = column_path_025(col_idx)
        row_is_complete_025(path, length(LAMBDAS_025)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_result_025(split(line, '\t')))
            end
        end
    end
end

function write_final_results_025(path::String)
    open(path, "w") do io_out
        println(io_out, "alpha\tlambda\tabsxmax_count\tabsxmax_dot_values\tabsxmax_return_times\tabsxmax_states\tstatus")
        for col_idx in eachindex(ALPHAS_025)
            open(column_path_025(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
end

function nearest_axis_index_025(value::Float64, axis::Vector{Float64}, axis_name::String)
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

    step = length(axis) > 1 ? abs(axis[2] - axis[1]) : 1.0
    best_error <= max(step * 0.51, 1e-6) || error("Result point does not align with $(axis_name) grid: $(value)")
    return best_idx
end

function build_iterate_grids_025(count_fn, values_fn)
    grids = [fill(NaN, length(LAMBDAS_025), length(ALPHAS_025)) for _ in 1:ATTEMPT025_MAX_EVENT_ITERATES]
    filled = falses(length(LAMBDAS_025), length(ALPHAS_025))

    scan_column_files_025() do result
        alpha_idx = nearest_axis_index_025(result.alpha, ALPHAS_025, "alpha")
        lambda_idx = nearest_axis_index_025(result.lambda, LAMBDAS_025, "lambda")
        filled[lambda_idx, alpha_idx] && error("Duplicate grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[lambda_idx, alpha_idx] = true

        if result_usable_025(result)
            values = values_fn(result)
            for iterate in 1:min(count_fn(result), ATTEMPT025_MAX_EVENT_ITERATES, length(values))
                grids[iterate][lambda_idx, alpha_idx] = values[iterate]
            end
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return grids
end

function cumulative_to_interval_grids_025(cumulative_grids::Vector{Matrix{Float64}})
    interval_grids = [fill(NaN, size(cumulative_grids[1])) for _ in eachindex(cumulative_grids)]
    interval_grids[1] .= cumulative_grids[1]

    for iterate in 2:length(cumulative_grids)
        current_grid = cumulative_grids[iterate]
        previous_grid = cumulative_grids[iterate - 1]
        interval_grid = interval_grids[iterate]
        @inbounds for idx in eachindex(current_grid)
            current_time = current_grid[idx]
            previous_time = previous_grid[idx]
            if isfinite(current_time) && isfinite(previous_time)
                interval_grid[idx] = current_time - previous_time
            end
        end
    end

    return interval_grids
end

function initialize_skip_state_025()
    n_lambda_cells = length(LAMBDAS_025) - 1
    n_alpha_cells = length(ALPHAS_025) - 1
    return SkipState25(
        zeros(UInt8, n_lambda_cells, n_alpha_cells),
        zeros(UInt8, n_lambda_cells, n_alpha_cells),
        zeros(UInt8, n_lambda_cells, n_alpha_cells),
        zeros(UInt8, n_lambda_cells, n_alpha_cells),
    )
end

@inline sign_class_025(value::Float64) = value > 0.0 ? Int8(1) : value < 0.0 ? Int8(-1) : Int8(0)

function evaluate_square_025(
    j::Int,
    i::Int,
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    skip_state::SkipState25,
)
    k_tl = nominal_iterate + Int(skip_state.tl[j, i])
    k_tr = nominal_iterate + Int(skip_state.tr[j, i])
    k_br = nominal_iterate + Int(skip_state.br[j, i])
    k_bl = nominal_iterate + Int(skip_state.bl[j, i])
    ks = (k_tl, k_tr, k_br, k_bl)

    any(k -> k < 1 || k + 1 > ATTEMPT025_MAX_EVENT_ITERATES, ks) && return missing_evaluation_025()

    d_tl = dot_grids[k_tl][j, i]
    d_tr = dot_grids[k_tr][j, i + 1]
    d_br = dot_grids[k_br][j + 1, i + 1]
    d_bl = dot_grids[k_bl][j + 1, i]
    all(isfinite, (d_tl, d_tr, d_br, d_bl)) || return missing_evaluation_025()

    t_tl = time_grids[k_tl][j, i]
    t_tr = time_grids[k_tr][j, i + 1]
    t_br = time_grids[k_br][j + 1, i + 1]
    t_bl = time_grids[k_bl][j + 1, i]
    all(isfinite, (t_tl, t_tr, t_br, t_bl)) || return missing_evaluation_025()

    t2_tl = time_grids[k_tl + 1][j, i]
    t2_tr = time_grids[k_tr + 1][j, i + 1]
    t2_br = time_grids[k_br + 1][j + 1, i + 1]
    t2_bl = time_grids[k_bl + 1][j + 1, i]
    all(isfinite, (t2_tl, t2_tr, t2_br, t2_bl)) || return missing_evaluation_025()

    signs = (
        sign_class_025(d_tl),
        sign_class_025(d_tr),
        sign_class_025(d_br),
        sign_class_025(d_bl),
    )
    any(==(Int8(0)), signs) && return missing_evaluation_025()

    status = all(==(signs[1]), signs) ? EVAL_CONSTANT_025 : EVAL_MIXED_025
    return SquareEvaluation25(
        status,
        (d_tl, d_tr, d_br, d_bl),
        (t_tl, t_tr, t_br, t_bl),
        (t2_tl, t2_tr, t2_br, t2_bl),
        signs,
        ks,
    )
end

function choose_representatives_025(evaluation::SquareEvaluation25)
    neg_rep = 0
    pos_rep = 0
    for idx in 1:4
        if evaluation.sign[idx] == Int8(-1) && neg_rep == 0
            neg_rep = idx
        elseif evaluation.sign[idx] == Int8(1) && pos_rep == 0
            pos_rep = idx
        end
    end
    neg_rep == 0 && error("No negative representative found for mixed-sign square.")
    pos_rep == 0 && error("No positive representative found for mixed-sign square.")

    if evaluation.current_time[neg_rep] < evaluation.current_time[pos_rep]
        return Int8(-1), neg_rep, pos_rep
    end
    return Int8(1), pos_rep, neg_rep
end

function skip_increment_decision_025(evaluation::SquareEvaluation25)
    shorter_sign, short_rep, long_rep = choose_representatives_025(evaluation)
    err_skip = abs(evaluation.current_time[short_rep] + evaluation.next_time[short_rep] - evaluation.current_time[long_rep])
    err_noskip = abs(evaluation.current_time[short_rep] - evaluation.current_time[long_rep])
    return err_skip < err_noskip, shorter_sign
end

function increment_skip_state_for_sign_025!(
    j::Int,
    i::Int,
    signs::NTuple{4, Int8},
    shorter_sign::Int8,
    skip_state::SkipState25,
)
    if signs[1] == shorter_sign
        skip_state.tl[j, i] += UInt8(1)
    end
    if signs[2] == shorter_sign
        skip_state.tr[j, i] += UInt8(1)
    end
    if signs[3] == shorter_sign
        skip_state.br[j, i] += UInt8(1)
    end
    if signs[4] == shorter_sign
        skip_state.bl[j, i] += UInt8(1)
    end
    return nothing
end

function edge_point_025(
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

function append_segment_if_025!(
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

function append_march_square_zero_segments_025!(
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

    p1 = edge_point_025(1, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p2 = edge_point_025(2, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p3 = edge_point_025(3, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p4 = edge_point_025(4, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    points = (p1, p2, p3, p4)

    added = 0
    if case_idx == 5 || case_idx == 10
        center_value = 0.25 * (z_tl + z_tr + z_br + z_bl)
        pairing =
            case_idx == 5 ?
            (center_value >= level ? ((1, 2), (3, 4)) : ((1, 4), (2, 3))) :
            (center_value >= level ? ((1, 4), (2, 3)) : ((1, 2), (3, 4)))
        for (edge_a, edge_b) in pairing
            added += append_segment_if_025!(segments, points[edge_a], points[edge_b])
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
        added += append_segment_if_025!(segments, points[edge_a], points[edge_b])
    end
    return added
end

function process_nominal_iterate_025(
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    skip_state::SkipState25,
    increment_counts::Union{Nothing, Matrix{UInt8}}=nothing,
    excluded_segments::Union{Nothing, Vector{NTuple{4, Float64}}}=nothing,
)
    segments = NTuple{4, Float64}[]
    stats = zero_iterate_stats_025()
    n_lambda_cells = length(LAMBDAS_025) - 1
    n_alpha_cells = length(ALPHAS_025) - 1

    for j in 1:n_lambda_cells
        y_tl = Float64(LAMBDAS_025[j])
        y_bl = Float64(LAMBDAS_025[j + 1])
        for i in 1:n_alpha_cells
            evaluation = evaluate_square_025(j, i, nominal_iterate, dot_grids, time_grids, skip_state)
            if evaluation.status == EVAL_MISSING_025
                stats.missing_data += 1
                continue
            elseif evaluation.status == EVAL_CONSTANT_025
                stats.constant_sign += 1
                continue
            end

            should_increment, shorter_sign = skip_increment_decision_025(evaluation)
            if should_increment
                if excluded_segments !== nothing
                    x_tl_ex = Float64(ALPHAS_025[i])
                    x_tr_ex = Float64(ALPHAS_025[i + 1])
                    append_march_square_zero_segments_025!(
                        excluded_segments,
                        evaluation.current_dot,
                        x_tl_ex,
                        y_tl,
                        x_tr_ex,
                        y_tl,
                        x_tr_ex,
                        y_bl,
                        x_tl_ex,
                        y_bl,
                    )
                end
                increment_skip_state_for_sign_025!(j, i, evaluation.sign, shorter_sign, skip_state)
                stats.incremented += 1
                increment_counts !== nothing && (increment_counts[j, i] += UInt8(1))
                evaluation = evaluate_square_025(j, i, nominal_iterate, dot_grids, time_grids, skip_state)
                if evaluation.status == EVAL_MISSING_025
                    stats.missing_data += 1
                    continue
                elseif evaluation.status == EVAL_CONSTANT_025
                    stats.constant_sign += 1
                    continue
                end
            end

            x_tl = Float64(ALPHAS_025[i])
            x_tr = Float64(ALPHAS_025[i + 1])
            added = append_march_square_zero_segments_025!(
                segments,
                evaluation.current_dot,
                x_tl,
                y_tl,
                x_tr,
                y_tl,
                x_tr,
                y_bl,
                x_tl,
                y_bl,
            )
            if added > 0
                stats.contoured_squares += 1
                stats.emitted_segments += added
            end
        end
    end

    return segments, stats
end

function segments_to_polyline_025(segments::Vector{NTuple{4, Float64}})
    xs = Float64[]
    ys = Float64[]
    sizehint!(xs, 3 * length(segments))
    sizehint!(ys, 3 * length(segments))

    for (x1, y1, x2, y2) in segments
        push!(xs, x1); push!(xs, x2); push!(xs, NaN)
        push!(ys, y1); push!(ys, y2); push!(ys, NaN)
    end

    isempty(xs) || pop!(xs)
    isempty(ys) || pop!(ys)
    return xs, ys
end

function write_iterate_stats_025(path::String, stats::Vector{IterateStats25})
    open(path, "w") do io
        println(io, "nominal_iterate\tmissing_data_squares\tconstant_sign_squares\tincremented_squares\tcontoured_squares\temitted_segments")
        for iterate in PLOT_ITERATES_025
            stat = stats[iterate]
            println(
                io,
                join([
                    string(iterate),
                    string(stat.missing_data),
                    string(stat.constant_sign),
                    string(stat.incremented),
                    string(stat.contoured_squares),
                    string(stat.emitted_segments),
                ], '\t'),
            )
        end
    end
end

function build_retired_overlay_figure_027(
    accepted_segments::Vector{NTuple{4, Float64}},
    excluded_segments::Vector{NTuple{4, Float64}},
)
    fig = Figure(size=(ATTEMPT025_FIG_WIDTH, ATTEMPT025_FIG_HEIGHT), backgroundcolor=:black)
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka |x|-max contours, nominal iterates $(OVERLAY_ITERATE_START_025):$(OVERLAY_ITERATE_END_025), retired after first skip",
        backgroundcolor=:black,
        xlabelcolor=:white,
        ylabelcolor=:white,
        titlecolor=:white,
        xgridvisible=false,
        ygridvisible=false,
        leftspinecolor=:white,
        rightspinecolor=:white,
        topspinecolor=:white,
        bottomspinecolor=:white,
        xtickcolor=:white,
        ytickcolor=:white,
        xticklabelcolor=:white,
        yticklabelcolor=:white,
    )

    if !isempty(excluded_segments)
        xs_ex, ys_ex = segments_to_polyline_025(excluded_segments)
        lines!(ax, xs_ex, ys_ex; color=RGBAf(0.92, 0.10, 0.10, 0.95), linewidth=ATTEMPT025_LINEWIDTH)
    end

    if !isempty(accepted_segments)
        xs, ys = segments_to_polyline_025(accepted_segments)
        lines!(ax, xs, ys; color=RGBAf(1.0, 1.0, 1.0, 0.98), linewidth=ATTEMPT025_LINEWIDTH)
    end

    xlims!(ax, ATTEMPT025_ALPHA_MIN, ATTEMPT025_ALPHA_MAX)
    ylims!(ax, ATTEMPT025_LAMBDA_MIN, ATTEMPT025_LAMBDA_MAX)
    return fig
end

function process_nominal_iterate_027(
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    skip_state::SkipState25,
    retired_cells::BitMatrix,
    excluded_segments::Union{Nothing, Vector{NTuple{4, Float64}}}=nothing,
)
    segments = NTuple{4, Float64}[]
    stats = zero_iterate_stats_025()
    n_lambda_cells = length(LAMBDAS_025) - 1
    n_alpha_cells = length(ALPHAS_025) - 1

    for j in 1:n_lambda_cells
        y_tl = Float64(LAMBDAS_025[j])
        y_bl = Float64(LAMBDAS_025[j + 1])
        for i in 1:n_alpha_cells
            retired_cells[j, i] && continue

            evaluation = evaluate_square_025(j, i, nominal_iterate, dot_grids, time_grids, skip_state)
            if evaluation.status == EVAL_MISSING_025
                stats.missing_data += 1
                continue
            elseif evaluation.status == EVAL_CONSTANT_025
                stats.constant_sign += 1
                continue
            end

            should_increment, shorter_sign = skip_increment_decision_025(evaluation)
            if should_increment
                if excluded_segments !== nothing
                    x_tl_ex = Float64(ALPHAS_025[i])
                    x_tr_ex = Float64(ALPHAS_025[i + 1])
                    append_march_square_zero_segments_025!(
                        excluded_segments,
                        evaluation.current_dot,
                        x_tl_ex,
                        y_tl,
                        x_tr_ex,
                        y_tl,
                        x_tr_ex,
                        y_bl,
                        x_tl_ex,
                        y_bl,
                    )
                end
                increment_skip_state_for_sign_025!(j, i, evaluation.sign, shorter_sign, skip_state)
                retired_cells[j, i] = true
                stats.incremented += 1
                continue
            end

            x_tl = Float64(ALPHAS_025[i])
            x_tr = Float64(ALPHAS_025[i + 1])
            added = append_march_square_zero_segments_025!(
                segments,
                evaluation.current_dot,
                x_tl,
                y_tl,
                x_tr,
                y_tl,
                x_tr,
                y_bl,
                x_tl,
                y_bl,
            )
            if added > 0
                stats.contoured_squares += 1
                stats.emitted_segments += added
            end
        end
    end

    return segments, stats
end

function main()
    println("Running attempt-027 Shimizu-Morioka |x|-max recompute and retired-square contour plot.")
    println("Grid: $(length(ALPHAS_025)) alpha points x $(length(LAMBDAS_025)) lambda points")
    println("Threads: $(nthreads()), stored |x|-max iterates: $(ATTEMPT025_MAX_EVENT_ITERATES), plotted nominal iterates: $(ATTEMPT025_PLOT_ITERATE_CAP)")
    println("Sweep dir: $(SWEEP_DIR_025)")
    flush(stdout)

    if RUN_COLUMNS_025
        run_or_resume_columns_025()
    else
        println("Reusing existing sweep columns only (ATTEMPT025_RUN_COLUMNS=false).")
    end

    dot_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = cumulative_to_interval_grids_025(cumulative_time_grids)

    if WRITE_MERGED_RESULTS_025
        write_final_results_025(results_path_025())
    end

    skip_state = initialize_skip_state_025()
    retired_cells = falses(length(LAMBDAS_025) - 1, length(ALPHAS_025) - 1)
    accepted_segments = NTuple{4, Float64}[]
    excluded_segments = NTuple{4, Float64}[]
    iterate_stats = [zero_iterate_stats_025() for _ in PLOT_ITERATES_025]

    for nominal_iterate in PLOT_ITERATES_025
        local_excluded = nominal_iterate >= OVERLAY_ITERATE_START_025 ? NTuple{4, Float64}[] : nothing
        segments, stats = process_nominal_iterate_027(
            nominal_iterate,
            dot_grids,
            time_grids,
            skip_state,
            retired_cells,
            local_excluded,
        )
        iterate_stats[nominal_iterate] = stats

        if nominal_iterate >= OVERLAY_ITERATE_START_025 && nominal_iterate <= OVERLAY_ITERATE_END_025
            append!(accepted_segments, segments)
            local_excluded !== nothing && append!(excluded_segments, local_excluded)
        end

        @printf(
            "Iterate %d: missing=%d constant=%d incremented=%d contoured=%d emitted_segments=%d retired=%d\n",
            nominal_iterate,
            stats.missing_data,
            stats.constant_sign,
            stats.incremented,
            stats.contoured_squares,
            stats.emitted_segments,
            count(retired_cells),
        )
        flush(stdout)
    end

    WRITE_ITERATE_STATS_025 && write_iterate_stats_025(iterate_stats_path_025(), iterate_stats)

    fig = build_retired_overlay_figure_027(accepted_segments, excluded_segments)
    save(plot_path_025(), fig; px_per_unit=ATTEMPT025_PX_PER_UNIT)

    usable = 0
    total = length(ALPHAS_025) * length(LAMBDAS_025)
    scan_column_files_025() do result
        result_usable_025(result) && (usable += 1)
    end

    println("Usable points: $(usable) / $(total)")
    WRITE_MERGED_RESULTS_025 ? println("Saved merged results to $(results_path_025())") : println("Skipped merged TSV writes (ATTEMPT025_WRITE_MERGED_RESULTS=false)")
    WRITE_ITERATE_STATS_025 && println("Saved iterate stats to $(iterate_stats_path_025())")
    println("Saved contour plot to $(plot_path_025())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
