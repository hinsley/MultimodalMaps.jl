using Pkg

const ATTEMPT25_OVERLAY_ROOT = @__DIR__
const REPO_ROOT_025_OVERLAY = normpath(joinpath(ATTEMPT25_OVERLAY_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025_OVERLAY)

include(joinpath(ATTEMPT25_OVERLAY_ROOT, "contours.jl"))

const OVERLAY_OUTPUT_TAG_025 = get(
    ENV,
    "ATTEMPT025_OVERLAY_OUTPUT_TAG",
    "grid1200_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_overlay_shimizu_morioka_cpu",
)
const OVERLAY_ITERATE_START_025 = parse(Int, get(ENV, "ATTEMPT025_OVERLAY_ITERATE_START", "2"))
const OVERLAY_ITERATE_END_025 = min(
    ATTEMPT025_PLOT_ITERATE_CAP,
    parse(Int, get(ENV, "ATTEMPT025_OVERLAY_ITERATE_END", string(ATTEMPT025_PLOT_ITERATE_CAP))),
)

overlay_plot_path_025() = joinpath(ATTEMPT25_ROOT, "$(OVERLAY_OUTPUT_TAG_025).png")

function build_overlay_figure_025(
    accepted_segments::Vector{NTuple{4, Float64}},
    excluded_segments::Vector{NTuple{4, Float64}},
)
    fig = Figure(size=(ATTEMPT025_FIG_WIDTH, ATTEMPT025_FIG_HEIGHT), backgroundcolor=:black)
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka |x|-max contours, nominal iterates $(OVERLAY_ITERATE_START_025):$(OVERLAY_ITERATE_END_025)",
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

function main()
    println("Rendering attempt-025 monochrome overlay PNG from existing sweep data.")
    println("Sweep dir: $(SWEEP_DIR_025)")
    println("Processed nominal iterates: 1:$(ATTEMPT025_PLOT_ITERATE_CAP)")
    println("Rendered overlay nominal iterates: $(OVERLAY_ITERATE_START_025):$(OVERLAY_ITERATE_END_025)")
    flush(stdout)

    RUN_COLUMNS_025 && error("overlay_excluded_png.jl is plotting-only; set ATTEMPT025_RUN_COLUMNS=false")
    (1 <= OVERLAY_ITERATE_START_025 <= OVERLAY_ITERATE_END_025 <= ATTEMPT025_PLOT_ITERATE_CAP) || error("Invalid overlay iterate range.")

    dot_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = cumulative_to_interval_grids_025(cumulative_time_grids)

    skip_state = initialize_skip_state_025()
    accepted_segments = NTuple{4, Float64}[]
    excluded_segments = NTuple{4, Float64}[]

    for nominal_iterate in PLOT_ITERATES_025
        local_excluded = nominal_iterate >= OVERLAY_ITERATE_START_025 ? NTuple{4, Float64}[] : nothing
        segments, stats = process_nominal_iterate_025(
            nominal_iterate,
            dot_grids,
            time_grids,
            skip_state,
            nothing,
            local_excluded,
        )

        if nominal_iterate >= OVERLAY_ITERATE_START_025
            append!(accepted_segments, segments)
            local_excluded !== nothing && append!(excluded_segments, local_excluded)
        end

        @printf(
            "Iterate %d: missing=%d constant=%d incremented=%d contoured=%d emitted_segments=%d\n",
            nominal_iterate,
            stats.missing_data,
            stats.constant_sign,
            stats.incremented,
            stats.contoured_squares,
            stats.emitted_segments,
        )
        flush(stdout)
    end

    fig = build_overlay_figure_025(accepted_segments, excluded_segments)
    save(overlay_plot_path_025(), fig; px_per_unit=ATTEMPT025_PX_PER_UNIT)
    println("Saved overlay PNG to $(overlay_plot_path_025())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
