using Pkg

const ATTEMPT26_ROOT = @__DIR__
const REPO_ROOT_026 = normpath(joinpath(ATTEMPT26_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_026)

include(joinpath(ATTEMPT26_ROOT, "..", "attempt-025", "contours.jl"))

const ATTEMPT026_OUTPUT_TAG = get(
    ENV,
    "ATTEMPT026_OUTPUT_TAG",
    "grid1200_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_retired_shimizu_morioka_cpu",
)
const ATTEMPT026_ITERATE_START = parse(Int, get(ENV, "ATTEMPT026_ITERATE_START", "2"))
const ATTEMPT026_ITERATE_END = min(
    ATTEMPT025_PLOT_ITERATE_CAP,
    parse(Int, get(ENV, "ATTEMPT026_ITERATE_END", string(ATTEMPT025_PLOT_ITERATE_CAP))),
)

overlay_plot_path_026() = joinpath(ATTEMPT26_ROOT, "$(ATTEMPT026_OUTPUT_TAG).png")

function build_overlay_figure_026(
    accepted_segments::Vector{NTuple{4, Float64}},
    excluded_segments::Vector{NTuple{4, Float64}},
)
    fig = Figure(size=(ATTEMPT025_FIG_WIDTH, ATTEMPT025_FIG_HEIGHT), backgroundcolor=:black)
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka |x|-max contours, nominal iterates $(ATTEMPT026_ITERATE_START):$(ATTEMPT026_ITERATE_END), retired after first skip",
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

function process_nominal_iterate_026(
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
    println("Rendering attempt-026 monochrome overlay PNG from existing attempt-025 sweep data.")
    println("Sweep dir: $(SWEEP_DIR_025)")
    println("Processed nominal iterates: 1:$(ATTEMPT025_PLOT_ITERATE_CAP)")
    println("Rendered overlay nominal iterates: $(ATTEMPT026_ITERATE_START):$(ATTEMPT026_ITERATE_END)")
    flush(stdout)

    RUN_COLUMNS_025 && error("attempt-026 overlay is plotting-only; set ATTEMPT025_RUN_COLUMNS=false")
    (1 <= ATTEMPT026_ITERATE_START <= ATTEMPT026_ITERATE_END <= ATTEMPT025_PLOT_ITERATE_CAP) || error("Invalid overlay iterate range.")

    dot_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = cumulative_to_interval_grids_025(cumulative_time_grids)

    skip_state = initialize_skip_state_025()
    retired_cells = falses(length(LAMBDAS_025) - 1, length(ALPHAS_025) - 1)
    accepted_segments = NTuple{4, Float64}[]
    excluded_segments = NTuple{4, Float64}[]

    for nominal_iterate in PLOT_ITERATES_025
        local_excluded = nominal_iterate >= ATTEMPT026_ITERATE_START ? NTuple{4, Float64}[] : nothing
        segments, stats = process_nominal_iterate_026(
            nominal_iterate,
            dot_grids,
            time_grids,
            skip_state,
            retired_cells,
            local_excluded,
        )

        if nominal_iterate >= ATTEMPT026_ITERATE_START
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

    fig = build_overlay_figure_026(accepted_segments, excluded_segments)
    save(overlay_plot_path_026(), fig; px_per_unit=ATTEMPT025_PX_PER_UNIT)
    println("Saved overlay PNG to $(overlay_plot_path_026())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
