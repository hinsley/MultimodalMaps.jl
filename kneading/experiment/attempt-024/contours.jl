using Pkg

const ATTEMPT24_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_024_CONTOUR = normpath(joinpath(ATTEMPT24_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_024_CONTOUR)

include(joinpath(ATTEMPT24_CONTOUR_ROOT, "main.jl"))

using Base.Threads
using Random

const CONTOUR_OUTPUT_TAG_024 = get(ENV, "ATTEMPT024_OUTPUT_TAG", "grid1200_branch8_floworth_events_shimizu_morioka_cpu")
const SWEEP_SOURCE_TAG_024 = get(ENV, "ATTEMPT024_SWEEP_TAG", CONTOUR_OUTPUT_TAG_024)
const SWEEP_DIR_024 = joinpath(ATTEMPT24_ROOT, "$(SWEEP_SOURCE_TAG_024)_columns")
const LOG_LOCK_024 = ReentrantLock()
const PLOT_ITERATES_024 = collect(1:ATTEMPT024_MAX_EVENT_ITERATES)
const WRITE_MERGED_RESULTS_024 = lowercase(get(ENV, "ATTEMPT024_WRITE_MERGED_RESULTS", "true")) in ("1", "true", "yes")
const GENERATE_STANDARD_PLOTS_024 = lowercase(get(ENV, "ATTEMPT024_GENERATE_STANDARD_PLOTS", "true")) in ("1", "true", "yes")
const GENERATE_MEET_PLOT_024 = lowercase(get(ENV, "ATTEMPT024_GENERATE_MEET_PLOT", "false")) in ("1", "true", "yes")
const TANGENCY_MASK_DILATION_024 = parse(Int, get(ENV, "ATTEMPT024_TANGENCY_MASK_DILATION", "0"))
const ZMAX_TANGENCY_EPS_024 = parse(Float64, get(ENV, "ATTEMPT024_ZMAX_TANGENCY_EPS", "0.02"))
const ABSXMAX_TANGENCY_EPS_024 = parse(Float64, get(ENV, "ATTEMPT024_ABSXMAX_TANGENCY_EPS", "0.02"))
const INCLUDE_FIRST_ITERATE_TANGENCY_HEATMAP_024 = lowercase(get(ENV, "ATTEMPT024_INCLUDE_FIRST_ITERATE_TANGENCY_HEATMAP", "false")) in ("1", "true", "yes")
const FIRST_ITERATE_TANGENCY_HEATMAP_ALPHA_024 = parse(Float64, get(ENV, "ATTEMPT024_FIRST_ITERATE_TANGENCY_HEATMAP_ALPHA", "0.85"))
const FIRST_ITERATE_TANGENCY_HEATMAP_LOG_FLOOR_024 = parse(Float64, get(ENV, "ATTEMPT024_FIRST_ITERATE_TANGENCY_HEATMAP_LOG_FLOOR", "1e-12"))
const TANGENCY_HEATMAP_ITERATE_024 = parse(Int, get(ENV, "ATTEMPT024_TANGENCY_HEATMAP_ITERATE", "1"))
const INVALID_STATE_024 = SVector{3, Float64}(NaN, NaN, NaN)

column_path_024(col_idx::Int) = joinpath(SWEEP_DIR_024, @sprintf("column_%04d.tsv", col_idx))
results_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_results.tsv")
zmax_results_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_zmax_results.tsv")
absxmax_results_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_absxmax_results.tsv")
legend_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_iterate_colors.tsv")
zmax_plot_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_zmax_contours.png")
absxmax_plot_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_absxmax_contours.png")
meet_plot_path_024() = joinpath(ATTEMPT24_ROOT, "$(CONTOUR_OUTPUT_TAG_024)_meet_contours.png")

function row_is_complete_024(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_column_024(path::String, results::Vector{SMDualEventResult24})
    open(path, "w") do io
        println(
            io,
            "alpha\tlambda\tzmax_count\tzmax_dot_values\tzmax_states\tabsxmax_count\tabsxmax_dot_values\tabsxmax_states\tstatus",
        )
        for result in results
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    string(result.zmax_count),
                    join_float_list(result.zmax_dot_values),
                    join_state_list(result.zmax_states),
                    string(result.absxmax_count),
                    join_float_list(result.absxmax_dot_values),
                    join_state_list(result.absxmax_states),
                    result.status,
                ], '\t'),
            )
        end
    end
end

function parse_result_024(fields::Vector{SubString{String}})
    if length(fields) == 7
        return SMDualEventResult24(
            parse(Float64, fields[1]),
            parse(Float64, fields[2]),
            parse(Int, fields[3]),
            parse_float_list(fields[4]),
            SVector{3, Float64}[],
            parse(Int, fields[5]),
            parse_float_list(fields[6]),
            SVector{3, Float64}[],
            String(fields[7]),
        )
    end

    return SMDualEventResult24(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        parse_float_list(fields[4]),
        parse_state_list(fields[5]),
        parse(Int, fields[6]),
        parse_float_list(fields[7]),
        parse_state_list(fields[8]),
        String(fields[9]),
    )
end

function result_usable_024(result::SMDualEventResult24)
    return result.status == "ok" || result.status == "short"
end

function run_column_024(col_idx::Int, alpha::Float64, total_cols::Int, total_rows::Int)
    path = column_path_024(col_idx)
    if row_is_complete_024(path, total_rows)
        lock(LOG_LOCK_024)
        try
            @printf("Skipping completed column %d/%d (alpha=%.6f)\n", col_idx, total_cols, alpha)
            flush(stdout)
        finally
            unlock(LOG_LOCK_024)
        end
        return
    end

    started = time()
    column_results = Vector{SMDualEventResult24}(undef, total_rows)
    usable = 0
    for row_idx in eachindex(LAMBDAS_024)
        result = scan_orbit(alpha, LAMBDAS_024[row_idx])
        column_results[row_idx] = result
        result_usable_024(result) && (usable += 1)
    end
    write_column_024(path, column_results)

    lock(LOG_LOCK_024)
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
        unlock(LOG_LOCK_024)
    end
end

function run_or_resume_columns_024()
    mkpath(SWEEP_DIR_024)
    total_cols = length(ALPHAS_024)
    total_rows = length(LAMBDAS_024)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_024)
        run_column_024(col_idx, ALPHAS_024[col_idx], total_cols, total_rows)
    end
end

function scan_column_files_024(pass_fn)
    for col_idx in eachindex(ALPHAS_024)
        path = column_path_024(col_idx)
        row_is_complete_024(path, length(LAMBDAS_024)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_result_024(split(line, '\t')))
            end
        end
    end
end

function write_final_results_024(path::String)
    open(path, "w") do io_out
        println(
            io_out,
            "alpha\tlambda\tzmax_count\tzmax_dot_values\tzmax_states\tabsxmax_count\tabsxmax_dot_values\tabsxmax_states\tstatus",
        )
        for col_idx in eachindex(ALPHAS_024)
            open(column_path_024(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
end

function write_event_results_024(path::String, count_fn, values_fn, states_fn)
    open(path, "w") do io
        println(io, "alpha\tlambda\tmaxima_count\tdot_values\tstates\tstatus")
        scan_column_files_024() do result
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    string(count_fn(result)),
                    join_float_list(values_fn(result)),
                    join_state_list(states_fn(result)),
                    result.status,
                ], '\t'),
            )
        end
    end
end

function nearest_axis_index_024(value::Float64, axis::Vector{Float64}, axis_name::String)
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

function build_iterate_grids_024(count_fn, values_fn)
    grids = Dict(iterate => fill(NaN, length(LAMBDAS_024), length(ALPHAS_024)) for iterate in PLOT_ITERATES_024)
    filled = falses(length(LAMBDAS_024), length(ALPHAS_024))

    scan_column_files_024() do result
        alpha_idx = nearest_axis_index_024(result.alpha, ALPHAS_024, "alpha")
        lambda_idx = nearest_axis_index_024(result.lambda, LAMBDAS_024, "lambda")
        filled[lambda_idx, alpha_idx] && error("Duplicate grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[lambda_idx, alpha_idx] = true

        if result_usable_024(result)
            values = values_fn(result)
            for iterate in 1:min(count_fn(result), ATTEMPT024_MAX_EVENT_ITERATES, length(values))
                grids[iterate][lambda_idx, alpha_idx] = values[iterate]
            end
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return grids
end

function build_iterate_state_grids_024(count_fn, states_fn)
    grids = Dict(
        iterate => fill(INVALID_STATE_024, length(LAMBDAS_024), length(ALPHAS_024))
        for iterate in PLOT_ITERATES_024
    )
    filled = falses(length(LAMBDAS_024), length(ALPHAS_024))

    scan_column_files_024() do result
        alpha_idx = nearest_axis_index_024(result.alpha, ALPHAS_024, "alpha")
        lambda_idx = nearest_axis_index_024(result.lambda, LAMBDAS_024, "lambda")
        filled[lambda_idx, alpha_idx] && error("Duplicate state-grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[lambda_idx, alpha_idx] = true

        if result_usable_024(result)
            states = states_fn(result)
            for iterate in 1:min(count_fn(result), ATTEMPT024_MAX_EVENT_ITERATES, length(states))
                grids[iterate][lambda_idx, alpha_idx] = states[iterate]
            end
        end
    end

    all(filled) || error("One or more state-grid entries were not filled.")
    return grids
end

state_isfinite_024(state::SVector{3, Float64}) = all(isfinite, state)
@inline function zmax_tangency_score_024(state::SVector{3, Float64}, alpha::Float64, lambda::Float64)
    flow = shimizu_morioka_vector(state, @SVector [alpha, lambda])
    grad_phi = @SVector [-2.0 * state[1], 0.0, 1.0]
    return abs(dot(grad_phi, flow))
end

@inline function absxmax_tangency_score_024(state::SVector{3, Float64}, alpha::Float64, lambda::Float64)
    flow = shimizu_morioka_vector(state, @SVector [alpha, lambda])
    grad_phi = @SVector [0.0, 1.0, 0.0]
    return abs(dot(grad_phi, flow))
end

@inline function tangency_score_024(
    state::SVector{3, Float64},
    alpha::Float64,
    lambda::Float64,
    event_kind::Symbol,
)
    return event_kind == :zmax ?
        zmax_tangency_score_024(state, alpha, lambda) :
        absxmax_tangency_score_024(state, alpha, lambda)
end

@inline tangency_eps_024(event_kind::Symbol) = event_kind == :zmax ? ZMAX_TANGENCY_EPS_024 : ABSXMAX_TANGENCY_EPS_024

function tangency_cell_mask_024(
    state_grids::Dict{Int, Matrix{SVector{3, Float64}}},
    iterate::Int,
    event_kind::Symbol;
    dilation::Int=TANGENCY_MASK_DILATION_024,
)
    Z = state_grids[iterate]
    mask = falses(size(Z, 1) - 1, size(Z, 2) - 1)
    eps = tangency_eps_024(event_kind)

    for j in axes(mask, 1)
        for i in axes(mask, 2)
            state_isfinite_024(Z[j, i]) &&
                state_isfinite_024(Z[j, i + 1]) &&
                state_isfinite_024(Z[j + 1, i + 1]) &&
                state_isfinite_024(Z[j + 1, i]) || continue

            corner_alphas = (ALPHAS_024[i], ALPHAS_024[i + 1], ALPHAS_024[i + 1], ALPHAS_024[i])
            corner_lambdas = (LAMBDAS_024[j], LAMBDAS_024[j], LAMBDAS_024[j + 1], LAMBDAS_024[j + 1])
            corner_states = (Z[j, i], Z[j, i + 1], Z[j + 1, i + 1], Z[j + 1, i])
            corner_scores = ntuple(
                k -> tangency_score_024(corner_states[k], corner_alphas[k], corner_lambdas[k], event_kind),
                4,
            )

            any(score -> score <= eps, corner_scores) && any(score -> score > eps, corner_scores) || continue

            j_lo = max(firstindex(mask, 1), j - dilation)
            j_hi = min(lastindex(mask, 1), j + dilation)
            i_lo = max(firstindex(mask, 2), i - dilation)
            i_hi = min(lastindex(mask, 2), i + dilation)
            mask[j_lo:j_hi, i_lo:i_hi] .= true
        end
    end

    return mask
end

function canonicalize_segment_024(segment::NTuple{4, Float64})
    x1, y1, x2, y2 = segment
    if x1 < x2 || (x1 == x2 && y1 <= y2)
        return segment
    end
    return (x2, y2, x1, y1)
end

function edge_point_024(
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

function march_squares_zero_segments_024(
    Z::AbstractMatrix{Float64},
    x_vals::AbstractVector{<:Real},
    y_vals::AbstractVector{<:Real};
    level::Float64=0.0,
    skip_mask::Union{Nothing, BitMatrix}=nothing,
)
    segments = NTuple{4, Float64}[]
    skipped_cells = 0

    for j in 1:size(Z, 1)-1
        for i in 1:size(Z, 2)-1
            if !isnothing(skip_mask) && skip_mask[j, i]
                skipped_cells += 1
                continue
            end

            x_tl = Float64(x_vals[i]);   y_tl = Float64(y_vals[j])
            x_tr = Float64(x_vals[i + 1]); y_tr = Float64(y_vals[j])
            x_br = Float64(x_vals[i + 1]); y_br = Float64(y_vals[j + 1])
            x_bl = Float64(x_vals[i]);   y_bl = Float64(y_vals[j + 1])

            z_tl = Z[j, i]
            z_tr = Z[j, i + 1]
            z_br = Z[j + 1, i + 1]
            z_bl = Z[j + 1, i]
            all(isfinite, (z_tl, z_tr, z_br, z_bl)) || continue

            case_idx =
                (z_tl >= level ? 8 : 0) +
                (z_tr >= level ? 4 : 0) +
                (z_br >= level ? 2 : 0) +
                (z_bl >= level ? 1 : 0)

            case_idx == 0 && continue
            case_idx == 15 && continue

            values = (z_tl, z_tr, z_br, z_bl)
            points = Dict{Int, Tuple{Float64, Float64}}()
            for edge_id in 1:4
                point = edge_point_024(edge_id, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
                isnothing(point) || (points[edge_id] = point)
            end

            if case_idx == 5 || case_idx == 10
                center_value = 0.25 * (z_tl + z_tr + z_br + z_bl)
                pairing =
                    case_idx == 5 ?
                    (center_value >= level ? ((1, 2), (3, 4)) : ((1, 4), (2, 3))) :
                    (center_value >= level ? ((1, 4), (2, 3)) : ((1, 2), (3, 4)))
                for (edge_a, edge_b) in pairing
                    haskey(points, edge_a) && haskey(points, edge_b) || continue
                    x1, y1 = points[edge_a]
                    x2, y2 = points[edge_b]
                    push!(segments, (x1, y1, x2, y2))
                end
                continue
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
                haskey(points, edge_a) && haskey(points, edge_b) || continue
                x1, y1 = points[edge_a]
                x2, y2 = points[edge_b]
                push!(segments, (x1, y1, x2, y2))
            end
        end
    end

    return segments, skipped_cells
end

function contour_cell_mask_024(Z::AbstractMatrix{Float64}; level::Float64=0.0)
    mask = falses(size(Z, 1) - 1, size(Z, 2) - 1)

    for j in axes(mask, 1)
        for i in axes(mask, 2)
            z_tl = Z[j, i]
            z_tr = Z[j, i + 1]
            z_br = Z[j + 1, i + 1]
            z_bl = Z[j + 1, i]
            all(isfinite, (z_tl, z_tr, z_br, z_bl)) || continue

            case_idx =
                (z_tl >= level ? 8 : 0) +
                (z_tr >= level ? 4 : 0) +
                (z_br >= level ? 2 : 0) +
                (z_bl >= level ? 1 : 0)

            mask[j, i] = case_idx != 0 && case_idx != 15
        end
    end

    return mask
end

function build_first_iterate_segments_024(
    iterate_grids::Dict{Int, Matrix{Float64}},
    state_grids::Dict{Int, Matrix{SVector{3, Float64}}},
    event_kind::Symbol,
)
    first_segments = Dict(iterate => NTuple{4, Float64}[] for iterate in PLOT_ITERATES_024)
    seen = Set{NTuple{4, Float64}}()
    cumulative_skip_mask = nothing

    for iterate in PLOT_ITERATES_024
        raw_segments, _ = march_squares_zero_segments_024(iterate_grids[iterate], ALPHAS_024, LAMBDAS_024)
        iterate_skip_mask = tangency_cell_mask_024(state_grids, iterate, event_kind)
        if isnothing(cumulative_skip_mask)
            cumulative_skip_mask = copy(iterate_skip_mask)
        else
            cumulative_skip_mask .|= iterate_skip_mask
        end
        segments, skipped_cells = march_squares_zero_segments_024(
            iterate_grids[iterate],
            ALPHAS_024,
            LAMBDAS_024;
            skip_mask=cumulative_skip_mask,
        )

        @printf(
            "Iterate %d %s-tangency masked cells %d (cumulative %d), raw segments %d, kept segments %d\n",
            iterate,
            String(event_kind),
            count(iterate_skip_mask),
            count(cumulative_skip_mask),
            length(raw_segments),
            length(segments),
        )
        flush(stdout)

        for segment in segments
            key = canonicalize_segment_024(segment)
            key in seen && continue
            push!(seen, key)
            push!(first_segments[iterate], segment)
        end
    end

    return first_segments
end

function build_meet_first_iterate_segments_024(
    zmax_iterate_grids::Dict{Int, Matrix{Float64}},
    absxmax_iterate_grids::Dict{Int, Matrix{Float64}},
)
    meet_segments = Dict(iterate => NTuple{4, Float64}[] for iterate in PLOT_ITERATES_024)
    seen_cells = falses(length(LAMBDAS_024) - 1, length(ALPHAS_024) - 1)

    for iterate in PLOT_ITERATES_024
        zmax_mask = contour_cell_mask_024(zmax_iterate_grids[iterate])
        absxmax_mask = contour_cell_mask_024(absxmax_iterate_grids[iterate])
        common_mask = zmax_mask .& absxmax_mask
        new_mask = common_mask .& .!seen_cells
        skip_mask = .!new_mask
        segments, _ = march_squares_zero_segments_024(
            zmax_iterate_grids[iterate],
            ALPHAS_024,
            LAMBDAS_024;
            skip_mask=skip_mask,
        )

        @printf(
            "Iterate %d meet cells %d (new %d), kept segments %d\n",
            iterate,
            count(common_mask),
            count(new_mask),
            length(segments),
        )
        flush(stdout)

        meet_segments[iterate] = segments
        seen_cells .|= common_mask
    end

    return meet_segments
end

function build_first_iterate_tangency_grid_024(
    state_grids::Dict{Int, Matrix{SVector{3, Float64}}},
    event_kind::Symbol,
)
    grid = fill(NaN, length(LAMBDAS_024), length(ALPHAS_024))
    haskey(state_grids, TANGENCY_HEATMAP_ITERATE_024) ||
        error("Heatmap iterate $(TANGENCY_HEATMAP_ITERATE_024) is not available in the state grids.")
    states = state_grids[TANGENCY_HEATMAP_ITERATE_024]

    for j in axes(states, 1)
        for i in axes(states, 2)
            state = states[j, i]
            state_isfinite_024(state) || continue
            grid[j, i] = tangency_score_024(state, ALPHAS_024[i], LAMBDAS_024[j], event_kind)
        end
    end

    return grid
end

function finite_extrema_024(grid::Matrix{Float64})
    values = filter(isfinite, vec(grid))
    isempty(values) && error("No finite values available for heatmap underlay.")
    return extrema(values)
end

function log_heatmap_grid_024(grid::Matrix{Float64}; floor::Float64=FIRST_ITERATE_TANGENCY_HEATMAP_LOG_FLOOR_024)
    floor > 0.0 || error("Heatmap log floor must be positive.")
    logged = similar(grid)
    for idx in eachindex(grid)
        value = grid[idx]
        logged[idx] = isfinite(value) ? log10(max(value, floor)) : NaN
    end
    return logged
end

function segments_to_polyline_024(segments::Vector{NTuple{4, Float64}})
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

function random_iterate_colors_024()
    rng = MersenneTwister(ATTEMPT024_COLOR_SEED)
    return Dict(iterate => RGBAf(rand(rng), rand(rng), rand(rng), 0.92) for iterate in PLOT_ITERATES_024)
end

function write_color_legend_024(path::String, colors::Dict{Int, RGBAf})
    open(path, "w") do io
        println(io, "iterate\tr\tg\tb\ta")
        for iterate in PLOT_ITERATES_024
            color = colors[iterate]
            println(io, join([string(iterate), string(color.r), string(color.g), string(color.b), string(color.alpha)], '\t'))
        end
    end
end

function build_contour_figure_024(
    first_iterate_segments::Dict{Int, Vector{NTuple{4, Float64}}},
    colors::Dict{Int, RGBAf},
    title::String,
    heatmap_grid::Union{Nothing, Matrix{Float64}}=nothing,
    heatmap_label::String="",
)
    fig = Figure(size=(ATTEMPT024_FIG_WIDTH, ATTEMPT024_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title=title,
    )

    if !isnothing(heatmap_grid)
        min_value, max_value = finite_extrema_024(heatmap_grid)
        if min_value == max_value
            max_value = min_value + 1.0
        end
        plot_heatmap_grid = permutedims(heatmap_grid)
        hm = heatmap!(
            ax,
            ALPHAS_024,
            LAMBDAS_024,
            plot_heatmap_grid;
            colormap=:magma,
            colorrange=(min_value, max_value),
            alpha=FIRST_ITERATE_TANGENCY_HEATMAP_ALPHA_024,
            nan_color=RGBAf(0.0, 0.0, 0.0, 0.0),
        )
        Colorbar(fig[1, 2], hm; label=heatmap_label)
    end

    for iterate in PLOT_ITERATES_024
        xs, ys = segments_to_polyline_024(first_iterate_segments[iterate])
        lines!(ax, xs, ys; color=colors[iterate], linewidth=ATTEMPT024_LINEWIDTH)
    end

    xlims!(ax, ATTEMPT024_ALPHA_MIN, ATTEMPT024_ALPHA_MAX)
    ylims!(ax, ATTEMPT024_LAMBDA_MIN, ATTEMPT024_LAMBDA_MAX)
    return fig
end

function save_contour_plot_024(
    png_path::String,
    iterate_grids::Dict{Int, Matrix{Float64}},
    state_grids::Dict{Int, Matrix{SVector{3, Float64}}},
    colors::Dict{Int, RGBAf},
    title::String,
    event_kind::Symbol,
)
    first_iterate_segments = build_first_iterate_segments_024(iterate_grids, state_grids, event_kind)
    raw_heatmap_grid = INCLUDE_FIRST_ITERATE_TANGENCY_HEATMAP_024 ?
        build_first_iterate_tangency_grid_024(state_grids, event_kind) :
        nothing
    heatmap_grid = isnothing(raw_heatmap_grid) ? nothing : log_heatmap_grid_024(raw_heatmap_grid)
    heatmap_label = event_kind == :zmax ?
        "log10 iterate $(TANGENCY_HEATMAP_ITERATE_024) z-max tangency score" :
        "log10 iterate $(TANGENCY_HEATMAP_ITERATE_024) |x|-max tangency score"
    fig = build_contour_figure_024(first_iterate_segments, colors, title, heatmap_grid, heatmap_label)
    save(png_path, fig; px_per_unit=ATTEMPT024_PX_PER_UNIT)
    return nothing
end

function main()
    println("Running attempt-024 Shimizu-Morioka flow-orthogonal dual-event contour scan.")
    println("Grid: $(length(ALPHAS_024)) alpha points × $(length(LAMBDAS_024)) lambda points")
    println("Threads: $(nthreads()), max event iterates per family: $(ATTEMPT024_MAX_EVENT_ITERATES)")
    println("Sweep dir: $(SWEEP_DIR_024)")
    flush(stdout)

    run_or_resume_columns_024()

    zmax_iterate_grids = build_iterate_grids_024(result -> result.zmax_count, result -> result.zmax_dot_values)
    zmax_state_grids = build_iterate_state_grids_024(result -> result.zmax_count, result -> result.zmax_states)
    absxmax_iterate_grids = build_iterate_grids_024(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    absxmax_state_grids = build_iterate_state_grids_024(result -> result.absxmax_count, result -> result.absxmax_states)

    if WRITE_MERGED_RESULTS_024
        write_final_results_024(results_path_024())
        write_event_results_024(
            zmax_results_path_024(),
            result -> result.zmax_count,
            result -> result.zmax_dot_values,
            result -> result.zmax_states,
        )
        write_event_results_024(
            absxmax_results_path_024(),
            result -> result.absxmax_count,
            result -> result.absxmax_dot_values,
            result -> result.absxmax_states,
        )
    end

    colors = random_iterate_colors_024()
    write_color_legend_024(legend_path_024(), colors)
    if GENERATE_STANDARD_PLOTS_024
        save_contour_plot_024(
            zmax_plot_path_024(),
            zmax_iterate_grids,
            zmax_state_grids,
            colors,
            "Shimizu-Morioka flow-orthogonal tangent z-component zero contours at z-max events",
            :zmax,
        )
        save_contour_plot_024(
            absxmax_plot_path_024(),
            absxmax_iterate_grids,
            absxmax_state_grids,
            colors,
            "Shimizu-Morioka flow-orthogonal tangent signed x-component zero contours at |x|-max events",
            :absxmax,
        )
    end
    if GENERATE_MEET_PLOT_024
        meet_segments = build_meet_first_iterate_segments_024(zmax_iterate_grids, absxmax_iterate_grids)
        meet_fig = build_contour_figure_024(
            meet_segments,
            colors,
            "Shimizu-Morioka meet of z-max and |x|-max zero contours",
        )
        save(meet_plot_path_024(), meet_fig; px_per_unit=ATTEMPT024_PX_PER_UNIT)
    end

    usable = 0
    total = length(ALPHAS_024) * length(LAMBDAS_024)
    scan_column_files_024() do result
        result_usable_024(result) && (usable += 1)
    end

    println("Usable points: $(usable) / $(total)")
    if WRITE_MERGED_RESULTS_024
        println("Saved merged results to $(results_path_024())")
        println("Saved z-max results to $(zmax_results_path_024())")
        println("Saved |x|-max results to $(absxmax_results_path_024())")
    else
        println("Skipped merged TSV writes (ATTEMPT024_WRITE_MERGED_RESULTS=false)")
    end
    println("Saved iterate-color legend to $(legend_path_024())")
    GENERATE_STANDARD_PLOTS_024 && println("Saved z-max contour plot to $(zmax_plot_path_024())")
    GENERATE_STANDARD_PLOTS_024 && println("Saved |x|-max contour plot to $(absxmax_plot_path_024())")
    GENERATE_MEET_PLOT_024 && println("Saved meet contour plot to $(meet_plot_path_024())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
