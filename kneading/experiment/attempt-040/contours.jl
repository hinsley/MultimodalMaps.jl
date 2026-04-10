using Pkg

const ATTEMPT33_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_033_CONTOUR = normpath(joinpath(ATTEMPT33_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_033_CONTOUR)

include(joinpath(ATTEMPT33_CONTOUR_ROOT, "main.jl"))

using Base.Threads
using CairoMakie
using Random

const CONTOUR_OUTPUT_TAG_033 = get(
    ENV,
    "ATTEMPT033_OUTPUT_TAG",
    "grid500_branch16_attractorseed_anglesecantdiag_floworth_absx_plot8_shimizu_morioka_cpu",
)
const SWEEP_DIR_033 = joinpath(ATTEMPT33_ROOT, "$(CONTOUR_OUTPUT_TAG_033)_columns")
const LOG_LOCK_033 = ReentrantLock()
const PLOT_ITERATE_START_033 = parse(Int, get(ENV, "ATTEMPT033_OVERLAY_ITERATE_START", "2"))
const PLOT_ITERATE_END_033 = parse(Int, get(ENV, "ATTEMPT033_OVERLAY_ITERATE_END", "8"))
const PLOT_ITERATES_033 = collect(PLOT_ITERATE_START_033:min(PLOT_ITERATE_END_033, ATTEMPT033_MAX_EVENT_ITERATES))
const ATTEMPT033_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT033_LINEWIDTH", "0.35"))
const ATTEMPT033_FIG_WIDTH = parse(Int, get(ENV, "ATTEMPT033_FIG_WIDTH", "1200"))
const ATTEMPT033_FIG_HEIGHT = parse(Int, get(ENV, "ATTEMPT033_FIG_HEIGHT", "1200"))
const ATTEMPT033_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT033_PX_PER_UNIT", "1.0"))
const ATTEMPT033_COLOR_SEED = parse(Int, get(ENV, "ATTEMPT033_COLOR_SEED", "33"))
const ATTEMPT033_DIAGNOSTIC_ALPHA = parse(Float64, get(ENV, "ATTEMPT033_DIAGNOSTIC_ALPHA", "0.32"))

column_path_033(col_idx::Int) = joinpath(SWEEP_DIR_033, @sprintf("column_%04d.tsv", col_idx))
results_path_033() = joinpath(ATTEMPT33_ROOT, "$(CONTOUR_OUTPUT_TAG_033)_results.tsv")
legend_path_033() = joinpath(ATTEMPT33_ROOT, "$(CONTOUR_OUTPUT_TAG_033)_iterate_colors.tsv")
absx_plot_path_033() = joinpath(ATTEMPT33_ROOT, "$(CONTOUR_OUTPUT_TAG_033)_contours.png")
diagnostics_path_033() = joinpath(ATTEMPT33_ROOT, "$(CONTOUR_OUTPUT_TAG_033)_diagnostics.tsv")
diagnostic_overlay_path_033() = joinpath(ATTEMPT33_ROOT, "$(CONTOUR_OUTPUT_TAG_033)_diagnostic_overlay.png")

function row_is_complete_033(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_column_033(path::String, results::Vector{SMCriticalContourResult33})
    open(path, "w") do io
        println(io, "alpha\tlambda\tcritical_state\tcritical_kind\tcritical_source\tcritical_theta\tcritical_theta_dx\tcritical_theta_dxx\tcritical_refine_mode\tcritical_scan_mode\tcritical_refine_input_x\tcritical_scan_best_x\tcritical_scan_left_x\tcritical_scan_right_x\tcritical_scan_spacing\tcritical_trace_len\tcritical_bracketed\tabsxmax_count\tabsxmax_dot_values\tstatus")
        for result in results
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    join_state_033(result.critical_state),
                    result.critical_kind,
                    result.critical_source,
                    @sprintf("%.9f", result.critical_theta),
                    @sprintf("%.9f", result.critical_theta_dx),
                    @sprintf("%.9f", result.critical_theta_dxx),
                    result.critical_refine_mode,
                    result.critical_scan_mode,
                    @sprintf("%.9f", result.critical_refine_input_x),
                    @sprintf("%.9f", result.critical_scan_best_x),
                    @sprintf("%.9f", result.critical_scan_left_x),
                    @sprintf("%.9f", result.critical_scan_right_x),
                    @sprintf("%.9f", result.critical_scan_spacing),
                    string(result.critical_trace_len),
                    string(result.critical_bracketed),
                    string(result.absxmax_count),
                    join_float_list_033(result.absxmax_dot_values),
                    result.status,
                ], '\t'),
            )
        end
    end
end

parse_bool_033(value::AbstractString) = lowercase(strip(value)) == "true"

function parse_result_033(fields::Vector{SubString{String}})
    return SMCriticalContourResult33(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse_state_033(fields[3]),
        String(fields[4]),
        String(fields[5]),
        parse(Float64, fields[6]),
        parse(Float64, fields[7]),
        parse(Float64, fields[8]),
        String(fields[9]),
        String(fields[10]),
        parse(Float64, fields[11]),
        parse(Float64, fields[12]),
        parse(Float64, fields[13]),
        parse(Float64, fields[14]),
        parse(Float64, fields[15]),
        parse(Int, fields[16]),
        parse_bool_033(fields[17]),
        parse(Int, fields[18]),
        parse_float_list_033(fields[19]),
        String(fields[20]),
    )
end

result_usable_033(result::SMCriticalContourResult33) = result.status == "ok" || result.status == "short"

function run_column_033(col_idx::Int, alpha::Float64, total_cols::Int, total_rows::Int)
    path = column_path_033(col_idx)
    if row_is_complete_033(path, total_rows)
        lock(LOG_LOCK_033)
        try
            @printf("Skipping completed column %d/%d (alpha=%.6f)\n", col_idx, total_cols, alpha)
            flush(stdout)
        finally
            unlock(LOG_LOCK_033)
        end
        return
    end

    started = time()
    previous_seed = nothing
    column_results = Vector{SMCriticalContourResult33}(undef, total_rows)
    usable = 0
    for (step_idx, row_idx) in enumerate(reverse(eachindex(LAMBDAS_033)))
        result, previous_seed = scan_parameter_point_033(alpha, LAMBDAS_033[row_idx], step_idx, previous_seed)
        column_results[row_idx] = result
        result_usable_033(result) && (usable += 1)
    end
    write_column_033(path, column_results)

    lock(LOG_LOCK_033)
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
        unlock(LOG_LOCK_033)
    end
end

function run_or_resume_columns_033()
    mkpath(SWEEP_DIR_033)
    total_cols = length(ALPHAS_033)
    total_rows = length(LAMBDAS_033)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_033)
        run_column_033(col_idx, ALPHAS_033[col_idx], total_cols, total_rows)
    end
end

function scan_column_files_033(pass_fn)
    for col_idx in eachindex(ALPHAS_033)
        path = column_path_033(col_idx)
        row_is_complete_033(path, length(LAMBDAS_033)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_result_033(split(line, '\t')))
            end
        end
    end
end

function write_final_results_033(path::String)
    open(path, "w") do io_out
        println(io_out, "alpha\tlambda\tcritical_state\tcritical_kind\tcritical_source\tcritical_theta\tcritical_theta_dx\tcritical_theta_dxx\tcritical_refine_mode\tcritical_scan_mode\tcritical_refine_input_x\tcritical_scan_best_x\tcritical_scan_left_x\tcritical_scan_right_x\tcritical_scan_spacing\tcritical_trace_len\tcritical_bracketed\tabsxmax_count\tabsxmax_dot_values\tstatus")
        for col_idx in eachindex(ALPHAS_033)
            open(column_path_033(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
end

function nearest_axis_index_033(value::Float64, axis::Vector{Float64}, axis_name::String)
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

function build_iterate_grids_033()
    grids = Dict(iterate => fill(NaN, length(LAMBDAS_033), length(ALPHAS_033)) for iterate in PLOT_ITERATES_033)
    filled = falses(length(LAMBDAS_033), length(ALPHAS_033))

    scan_column_files_033() do result
        alpha_idx = nearest_axis_index_033(result.alpha, ALPHAS_033, "alpha")
        lambda_idx = nearest_axis_index_033(result.lambda, LAMBDAS_033, "lambda")
        filled[lambda_idx, alpha_idx] && error("Duplicate grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[lambda_idx, alpha_idx] = true

        if result_usable_033(result)
            for iterate in 1:min(result.absxmax_count, length(result.absxmax_dot_values), ATTEMPT033_MAX_EVENT_ITERATES)
                iterate in PLOT_ITERATES_033 || continue
                grids[iterate][lambda_idx, alpha_idx] = result.absxmax_dot_values[iterate]
            end
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return grids
end

function canonicalize_segment_033(segment::NTuple{4, Float64})
    x1, y1, x2, y2 = segment
    if x1 < x2 || (x1 == x2 && y1 <= y2)
        return segment
    end
    return (x2, y2, x1, y1)
end

function edge_point_033(
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

function march_squares_zero_segments_033(
    Z::AbstractMatrix{Float64},
    x_vals::AbstractVector{<:Real},
    y_vals::AbstractVector{<:Real};
    level::Float64=0.0,
)
    segments = NTuple{4, Float64}[]

    for j in 1:size(Z, 1)-1
        for i in 1:size(Z, 2)-1
            x_tl = Float64(x_vals[i]); y_tl = Float64(y_vals[j])
            x_tr = Float64(x_vals[i + 1]); y_tr = Float64(y_vals[j])
            x_br = Float64(x_vals[i + 1]); y_br = Float64(y_vals[j + 1])
            x_bl = Float64(x_vals[i]); y_bl = Float64(y_vals[j + 1])

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
                point = edge_point_033(edge_id, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
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

    return segments
end

function build_segments_033(iterate_grids::Dict{Int, Matrix{Float64}})
    segments_by_iterate = Dict(iterate => NTuple{4, Float64}[] for iterate in PLOT_ITERATES_033)
    for iterate in PLOT_ITERATES_033
        seen = Set{NTuple{4, Float64}}()
        raw_segments = march_squares_zero_segments_033(iterate_grids[iterate], ALPHAS_033, LAMBDAS_033)
        for segment in raw_segments
            key = canonicalize_segment_033(segment)
            key in seen && continue
            push!(seen, key)
            push!(segments_by_iterate[iterate], segment)
        end
        @printf("Iterate %d: emitted_segments=%d\n", iterate, length(segments_by_iterate[iterate]))
        flush(stdout)
    end
    return segments_by_iterate
end

function segments_to_polyline_033(segments::Vector{NTuple{4, Float64}})
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

function iterate_colors_033()
    rng = MersenneTwister(ATTEMPT033_COLOR_SEED)
    return Dict(iterate => RGBAf(rand(rng), rand(rng), rand(rng), 0.92) for iterate in PLOT_ITERATES_033)
end

function write_color_legend_033(path::String, colors::Dict{Int, RGBAf})
    open(path, "w") do io
        println(io, "iterate\tr\tg\tb\ta")
        for iterate in PLOT_ITERATES_033
            color = colors[iterate]
            println(io, join([string(iterate), string(color.r), string(color.g), string(color.b), string(color.alpha)], '\t'))
        end
    end
end

full_reseed_flag_033(result::SMCriticalContourResult33) =
    startswith(result.critical_source, "full_attractor_map") || result.critical_source == "fallback_full_angle"

function scan_fallback_flag_033(result::SMCriticalContourResult33)
    angle_attempted = occursin("angle_", result.critical_source)
    angle_attempted || return false
    return result.critical_scan_mode == "nearest_best_neighbor" ||
           !result.critical_bracketed ||
           result.critical_refine_mode in ("scan_grad_tol", "secant_eval_fail_best", "secant_max_iters_best")
end

function build_diagnostics_033()
    results_by_alpha = [SMCriticalContourResult33[] for _ in eachindex(ALPHAS_033)]
    scan_column_files_033() do result
        alpha_idx = nearest_axis_index_033(result.alpha, ALPHAS_033, "alpha")
        push!(results_by_alpha[alpha_idx], result)
    end

    full_reseed_points = Tuple{Float64, Float64}[]
    scan_fallback_points = Tuple{Float64, Float64}[]
    jump_points = Tuple{Float64, Float64}[]

    open(diagnostics_path_033(), "w") do io
        println(
            io,
            "alpha\tlambda\tstatus\tcritical_source\tcritical_refine_mode\tcritical_scan_mode\tcritical_x\tcritical_z\tcritical_refine_input_x\tcritical_scan_best_x\tcritical_scan_left_x\tcritical_scan_right_x\tcritical_scan_spacing\tcritical_trace_len\tcritical_bracketed\tfull_reseed_flag\tscan_fallback_flag\tx_jump_from_prev_lambda\tjump_flag",
        )

        for column in results_by_alpha
            sort!(column; by=result -> result.lambda, rev=true)
            previous = nothing
            for result in column
                full_flag = full_reseed_flag_033(result)
                fallback_flag = scan_fallback_flag_033(result)
                x_jump = NaN
                jump_flag = false
                if !isnothing(previous)
                    prev = previous::SMCriticalContourResult33
                    if result_usable_033(prev) && result_usable_033(result) && isfinite(prev.critical_state[1]) && isfinite(result.critical_state[1])
                        x_jump = abs(result.critical_state[1] - prev.critical_state[1])
                        spacing = max(abs(result.critical_scan_spacing), abs(prev.critical_scan_spacing))
                        threshold = max(0.8 * spacing, 0.012)
                        jump_flag = isfinite(threshold) && x_jump >= threshold
                    end
                end

                full_flag && push!(full_reseed_points, (result.alpha, result.lambda))
                fallback_flag && push!(scan_fallback_points, (result.alpha, result.lambda))
                jump_flag && push!(jump_points, (result.alpha, result.lambda))

                println(
                    io,
                    join([
                        @sprintf("%.6f", result.alpha),
                        @sprintf("%.6f", result.lambda),
                        result.status,
                        result.critical_source,
                        result.critical_refine_mode,
                        result.critical_scan_mode,
                        @sprintf("%.9f", result.critical_state[1]),
                        @sprintf("%.9f", result.critical_state[3]),
                        @sprintf("%.9f", result.critical_refine_input_x),
                        @sprintf("%.9f", result.critical_scan_best_x),
                        @sprintf("%.9f", result.critical_scan_left_x),
                        @sprintf("%.9f", result.critical_scan_right_x),
                        @sprintf("%.9f", result.critical_scan_spacing),
                        string(result.critical_trace_len),
                        string(result.critical_bracketed),
                        string(full_flag),
                        string(fallback_flag),
                        @sprintf("%.9f", x_jump),
                        string(jump_flag),
                    ], '\t'),
                )

                previous = result
            end
        end
    end

    return (
        full_reseed_points=full_reseed_points,
        scan_fallback_points=scan_fallback_points,
        jump_points=jump_points,
    )
end

function save_contour_plot_033(
    png_path::String,
    segments_by_iterate::Dict{Int, Vector{NTuple{4, Float64}}},
    colors::Dict{Int, RGBAf},
)
    fig = Figure(size=(ATTEMPT033_FIG_WIDTH, ATTEMPT033_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Attempt-040 |x|-max contours from x-only angle-secant critical-point seeding, iterates $(first(PLOT_ITERATES_033)):$(last(PLOT_ITERATES_033))",
    )
    for iterate in PLOT_ITERATES_033
        xs, ys = segments_to_polyline_033(segments_by_iterate[iterate])
        lines!(ax, xs, ys; color=colors[iterate], linewidth=ATTEMPT033_LINEWIDTH)
    end
    xlims!(ax, ATTEMPT033_ALPHA_MIN, ATTEMPT033_ALPHA_MAX)
    ylims!(ax, ATTEMPT033_LAMBDA_MIN, ATTEMPT033_LAMBDA_MAX)
    save(png_path, fig; px_per_unit=ATTEMPT033_PX_PER_UNIT)
    return nothing
end

function save_diagnostic_overlay_plot_033(
    png_path::String,
    segments_by_iterate::Dict{Int, Vector{NTuple{4, Float64}}},
    colors::Dict{Int, RGBAf},
    diagnostics,
)
    fig = Figure(size=(ATTEMPT033_FIG_WIDTH, ATTEMPT033_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Attempt-040 seam diagnostics over angle-secant-seeded |x|-max contours",
    )

    if !isempty(diagnostics.full_reseed_points)
        xs = first.(diagnostics.full_reseed_points)
        ys = last.(diagnostics.full_reseed_points)
        scatter!(
            ax,
            xs,
            ys;
            color=RGBAf(1.0, 0.0, 0.0, ATTEMPT033_DIAGNOSTIC_ALPHA),
            marker=:xcross,
            markersize=10,
            label="full reseed/fallback full",
        )
    end
    if !isempty(diagnostics.scan_fallback_points)
        xs = first.(diagnostics.scan_fallback_points)
        ys = last.(diagnostics.scan_fallback_points)
        scatter!(
            ax,
            xs,
            ys;
            color=RGBAf(1.0, 0.5, 0.0, ATTEMPT033_DIAGNOSTIC_ALPHA),
            marker=:rect,
            markersize=7,
            label="scan fallback / unbracketed local corrector",
        )
    end
    has_legend = false
    if !isempty(diagnostics.full_reseed_points)
        has_legend = true
    end
    if !isempty(diagnostics.scan_fallback_points)
        has_legend = true
    end
    if !isempty(diagnostics.jump_points)
        xs = first.(diagnostics.jump_points)
        ys = last.(diagnostics.jump_points)
        scatter!(
            ax,
            xs,
            ys;
            color=RGBAf(0.0, 1.0, 1.0, ATTEMPT033_DIAGNOSTIC_ALPHA),
            marker=:diamond,
            markersize=8,
            label="large x jump vs previous lambda",
        )
        has_legend = true
    end

    for iterate in PLOT_ITERATES_033
        xs, ys = segments_to_polyline_033(segments_by_iterate[iterate])
        lines!(ax, xs, ys; color=colors[iterate], linewidth=ATTEMPT033_LINEWIDTH)
    end

    xlims!(ax, ATTEMPT033_ALPHA_MIN, ATTEMPT033_ALPHA_MAX)
    ylims!(ax, ATTEMPT033_LAMBDA_MIN, ATTEMPT033_LAMBDA_MAX)
    has_legend && axislegend(ax; position=:rb)
    save(png_path, fig; px_per_unit=ATTEMPT033_PX_PER_UNIT)
    return nothing
end

function main()
    println("Running attempt-040 Shimizu-Morioka angle-secant-seeded |x|-max contour scan with seam diagnostics.")
    println("Grid: $(length(ALPHAS_033)) alpha points × $(length(LAMBDAS_033)) lambda points")
    println("Threads: $(nthreads()), max event iterates: $(ATTEMPT033_MAX_EVENT_ITERATES)")
    println("Overlay iterates: $(first(PLOT_ITERATES_033)):$(last(PLOT_ITERATES_033))")
    println("Lambda traversal within each alpha column: descending")
    println("Critical reseed period within each alpha column: $(ATTEMPT033_RESEED_PERIOD)")
    println("Local corrector: x-only safeguarded secant on the eigendirection-angle derivative")
    println("Diagnostic overlay: full reseeds vs scan fallback/unbracketed points vs large x jumps")
    println("Sweep dir: $(SWEEP_DIR_033)")
    flush(stdout)

    run_or_resume_columns_033()

    iterate_grids = build_iterate_grids_033()
    segments_by_iterate = build_segments_033(iterate_grids)
    write_final_results_033(results_path_033())
    diagnostics = build_diagnostics_033()

    colors = iterate_colors_033()
    write_color_legend_033(legend_path_033(), colors)
    save_contour_plot_033(absx_plot_path_033(), segments_by_iterate, colors)
    save_diagnostic_overlay_plot_033(diagnostic_overlay_path_033(), segments_by_iterate, colors, diagnostics)

    usable = 0
    total = length(ALPHAS_033) * length(LAMBDAS_033)
    scan_column_files_033() do result
        result_usable_033(result) && (usable += 1)
    end

    println("Usable points: $(usable) / $(total)")
    println("Saved merged results to $(results_path_033())")
    println("Saved seam diagnostics to $(diagnostics_path_033())")
    println("Saved iterate-color legend to $(legend_path_033())")
    println("Saved |x|-max contour plot to $(absx_plot_path_033())")
    println("Saved seam diagnostic overlay to $(diagnostic_overlay_path_033())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
