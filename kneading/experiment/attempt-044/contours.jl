using Pkg

const ATTEMPT044_ROOT = @__DIR__
const REPO_ROOT_044 = normpath(joinpath(ATTEMPT044_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_044)

using CairoMakie
using Base.Threads
using Printf

module Attempt027
include(normpath(joinpath(@__DIR__, "..", "attempt-027", "contours.jl")))
end

const A27 = Attempt027
const OUTPUT_TAG_044 = get(
    ENV,
    "ATTEMPT044_OUTPUT_TAG",
    "grid2000_branch16_absxskip16_plot8_forcedfirstskip_black_red_shimizu_morioka_cpu",
)
const OVERLAY_ITERATE_START_044 = parse(Int, get(ENV, "ATTEMPT044_OVERLAY_ITERATE_START", "2"))
const OVERLAY_ITERATE_END_044 = min(
    A27.ATTEMPT025_PLOT_ITERATE_CAP,
    parse(Int, get(ENV, "ATTEMPT044_OVERLAY_ITERATE_END", "8")),
)

plot_path_044() = joinpath(ATTEMPT044_ROOT, "$(OUTPUT_TAG_044)_contours.png")
stats_path_044() = joinpath(ATTEMPT044_ROOT, "$(OUTPUT_TAG_044)_iterate_stats.tsv")

@inline function increment_local_skip_044(
    skip::NTuple{4, Int},
    signs::NTuple{4, Int8},
    shorter_sign::Int8,
)
    return (
        skip[1] + (signs[1] == shorter_sign ? 1 : 0),
        skip[2] + (signs[2] == shorter_sign ? 1 : 0),
        skip[3] + (signs[3] == shorter_sign ? 1 : 0),
        skip[4] + (signs[4] == shorter_sign ? 1 : 0),
    )
end

function evaluate_square_local_044(
    j::Int,
    i::Int,
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    skip::NTuple{4, Int},
)
    k_tl = nominal_iterate + skip[1]
    k_tr = nominal_iterate + skip[2]
    k_br = nominal_iterate + skip[3]
    k_bl = nominal_iterate + skip[4]
    ks = (k_tl, k_tr, k_br, k_bl)

    any(k -> k < 1 || k > length(dot_grids) || k + 1 > length(time_grids), ks) && return A27.missing_evaluation_025()

    d_tl = dot_grids[k_tl][j, i]
    d_tr = dot_grids[k_tr][j, i + 1]
    d_br = dot_grids[k_br][j + 1, i + 1]
    d_bl = dot_grids[k_bl][j + 1, i]
    all(isfinite, (d_tl, d_tr, d_br, d_bl)) || return A27.missing_evaluation_025()

    t_tl = time_grids[k_tl][j, i]
    t_tr = time_grids[k_tr][j, i + 1]
    t_br = time_grids[k_br][j + 1, i + 1]
    t_bl = time_grids[k_bl][j + 1, i]
    all(isfinite, (t_tl, t_tr, t_br, t_bl)) || return A27.missing_evaluation_025()

    t2_tl = time_grids[k_tl + 1][j, i]
    t2_tr = time_grids[k_tr + 1][j, i + 1]
    t2_br = time_grids[k_br + 1][j + 1, i + 1]
    t2_bl = time_grids[k_bl + 1][j + 1, i]
    all(isfinite, (t2_tl, t2_tr, t2_br, t2_bl)) || return A27.missing_evaluation_025()

    signs = (
        A27.sign_class_025(d_tl),
        A27.sign_class_025(d_tr),
        A27.sign_class_025(d_br),
        A27.sign_class_025(d_bl),
    )
    any(==(Int8(0)), signs) && return A27.missing_evaluation_025()

    status = all(==(signs[1]), signs) ? A27.EVAL_CONSTANT_025 : A27.EVAL_MIXED_025
    return A27.SquareEvaluation25(
        status,
        (d_tl, d_tr, d_br, d_bl),
        (t_tl, t_tr, t_br, t_bl),
        (t2_tl, t2_tr, t2_br, t2_bl),
        signs,
        ks,
    )
end

function build_forced_firstskip_segments_044(
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
)
    n_plot = A27.ATTEMPT025_PLOT_ITERATE_CAP
    n_lambda_cells = length(A27.LAMBDAS_025) - 1
    n_alpha_cells = length(A27.ALPHAS_025) - 1
    n_threads = Threads.maxthreadid()

    black_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    red_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    earliest_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    black_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    black_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    red_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    red_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]

    Threads.@threads :dynamic for j in 1:n_lambda_cells
        tid = threadid()
        black_local = black_tls[tid]
        red_local = red_tls[tid]
        earliest_local = earliest_tls[tid]
        black_cell_local = black_cell_tls[tid]
        black_segment_local = black_segment_tls[tid]
        red_cell_local = red_cell_tls[tid]
        red_segment_local = red_segment_tls[tid]

        y_tl = Float64(A27.LAMBDAS_025[j])
        y_bl = Float64(A27.LAMBDAS_025[j + 1])

        for i in 1:n_alpha_cells
            x_tl = Float64(A27.ALPHAS_025[i])
            x_tr = Float64(A27.ALPHAS_025[i + 1])
            skip = (0, 0, 0, 0)
            earliest_nominal = 0
            earliest_evaluation = A27.missing_evaluation_025()
            later_found = false

            for nominal_iterate in OVERLAY_ITERATE_START_044:OVERLAY_ITERATE_END_044
                evaluation = evaluate_square_local_044(j, i, nominal_iterate, dot_grids, time_grids, skip)
                evaluation.status == A27.EVAL_MIXED_025 || continue

                if earliest_nominal == 0
                    earliest_nominal = nominal_iterate
                    earliest_evaluation = evaluation
                    earliest_local[nominal_iterate] += 1
                    shorter_sign, _, _ = A27.choose_representatives_025(evaluation)
                    skip = increment_local_skip_044(skip, evaluation.sign, shorter_sign)
                    continue
                end

                added = A27.append_march_square_zero_segments_025!(
                    black_local[nominal_iterate],
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
                    later_found = true
                    black_cell_local[nominal_iterate] += 1
                    black_segment_local[nominal_iterate] += added
                end
            end

            if earliest_nominal != 0 && !later_found
                added = A27.append_march_square_zero_segments_025!(
                    red_local[earliest_nominal],
                    earliest_evaluation.current_dot,
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
                    red_cell_local[earliest_nominal] += 1
                    red_segment_local[earliest_nominal] += added
                end
            end
        end
    end

    black_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    red_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    earliest_source_cells = zeros(Int, n_plot)
    black_contoured_cells = zeros(Int, n_plot)
    black_segments_count = zeros(Int, n_plot)
    red_contoured_cells = zeros(Int, n_plot)
    red_segments_count = zeros(Int, n_plot)

    for tid in 1:n_threads
        for iterate in OVERLAY_ITERATE_START_044:OVERLAY_ITERATE_END_044
            append!(black_segments_by_iter[iterate], black_tls[tid][iterate])
            append!(red_segments_by_iter[iterate], red_tls[tid][iterate])
            earliest_source_cells[iterate] += earliest_tls[tid][iterate]
            black_contoured_cells[iterate] += black_cell_tls[tid][iterate]
            black_segments_count[iterate] += black_segment_tls[tid][iterate]
            red_contoured_cells[iterate] += red_cell_tls[tid][iterate]
            red_segments_count[iterate] += red_segment_tls[tid][iterate]
        end
    end

    return (
        black_segments_by_iter,
        red_segments_by_iter,
        earliest_source_cells,
        black_contoured_cells,
        black_segments_count,
        red_contoured_cells,
        red_segments_count,
    )
end

function write_iterate_stats_044(
    path::String,
    earliest_source_cells::Vector{Int},
    black_contoured_cells::Vector{Int},
    black_segments_count::Vector{Int},
    red_contoured_cells::Vector{Int},
    red_segments_count::Vector{Int},
)
    open(path, "w") do io
        println(
            io,
            "nominal_iterate\tearliest_source_cells\tblack_contoured_cells\tblack_segments\tred_contoured_cells\tred_segments",
        )
        for iterate in OVERLAY_ITERATE_START_044:OVERLAY_ITERATE_END_044
            println(
                io,
                join([
                    string(iterate),
                    string(earliest_source_cells[iterate]),
                    string(black_contoured_cells[iterate]),
                    string(black_segments_count[iterate]),
                    string(red_contoured_cells[iterate]),
                    string(red_segments_count[iterate]),
                ], '\t'),
            )
        end
    end
end

function build_figure_044(
    black_segments::Vector{NTuple{4, Float64}},
    red_segments::Vector{NTuple{4, Float64}},
)
    fig = Figure(
        size=(A27.ATTEMPT025_FIG_WIDTH, A27.ATTEMPT025_FIG_HEIGHT),
        backgroundcolor=:white,
    )
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka |x|-max contours, nominal iterates $(OVERLAY_ITERATE_START_044):$(OVERLAY_ITERATE_END_044), first contour forces one shorter-side skip",
        backgroundcolor=:white,
        xlabelcolor=:black,
        ylabelcolor=:black,
        titlecolor=:black,
        xgridvisible=false,
        ygridvisible=false,
        leftspinecolor=:black,
        rightspinecolor=:black,
        topspinecolor=:black,
        bottomspinecolor=:black,
        xtickcolor=:black,
        ytickcolor=:black,
        xticklabelcolor=:black,
        yticklabelcolor=:black,
    )

    if !isempty(black_segments)
        xs_black, ys_black = A27.segments_to_polyline_025(black_segments)
        lines!(ax, xs_black, ys_black; color=RGBAf(0.0, 0.0, 0.0, 0.98), linewidth=A27.ATTEMPT025_LINEWIDTH)
    end

    if !isempty(red_segments)
        xs_red, ys_red = A27.segments_to_polyline_025(red_segments)
        lines!(ax, xs_red, ys_red; color=RGBAf(0.92, 0.10, 0.10, 0.96), linewidth=A27.ATTEMPT025_LINEWIDTH)
    end

    xlims!(ax, A27.ATTEMPT025_ALPHA_MIN, A27.ATTEMPT025_ALPHA_MAX)
    ylims!(ax, A27.ATTEMPT025_LAMBDA_MIN, A27.ATTEMPT025_LAMBDA_MAX)
    return fig
end

function main()
    println("Running attempt-044 first-contour-forced-skip overlay from saved attempt-027 data.")
    println("Source columns: $(A27.SWEEP_DIR_025)")
    println("Grid: $(length(A27.ALPHAS_025)) x $(length(A27.LAMBDAS_025)), overlay nominal iterates $(OVERLAY_ITERATE_START_044):$(OVERLAY_ITERATE_END_044)")
    println("Threads: $(nthreads())")
    flush(stdout)

    dot_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = A27.cumulative_to_interval_grids_025(cumulative_time_grids)

    black_segments_by_iter,
    red_segments_by_iter,
    earliest_source_cells,
    black_contoured_cells,
    black_segments_count,
    red_contoured_cells,
    red_segments_count = build_forced_firstskip_segments_044(dot_grids, time_grids)

    black_segments = NTuple{4, Float64}[]
    red_segments = NTuple{4, Float64}[]
    for iterate in OVERLAY_ITERATE_START_044:OVERLAY_ITERATE_END_044
        append!(black_segments, black_segments_by_iter[iterate])
        append!(red_segments, red_segments_by_iter[iterate])
    end

    write_iterate_stats_044(
        stats_path_044(),
        earliest_source_cells,
        black_contoured_cells,
        black_segments_count,
        red_contoured_cells,
        red_segments_count,
    )

    fig = build_figure_044(black_segments, red_segments)
    save(plot_path_044(), fig; px_per_unit=A27.ATTEMPT025_PX_PER_UNIT)

    println(
        @sprintf(
            "Saved %d black segments and %d red segments to %s",
            length(black_segments),
            length(red_segments),
            plot_path_044(),
        ),
    )
    println("Saved iterate stats to $(stats_path_044())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
