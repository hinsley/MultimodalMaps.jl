using Pkg

const ATTEMPT25_GIF_ROOT = @__DIR__
const REPO_ROOT_025_GIF = normpath(joinpath(ATTEMPT25_GIF_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025_GIF)

include(joinpath(ATTEMPT25_GIF_ROOT, "contours.jl"))

const GIF_OUTPUT_TAG_025 = get(ENV, "ATTEMPT025_GIF_OUTPUT_TAG", "grid1200_branch16_absxskip16_plot8_deltatfix_nominal_iterates_shimizu_morioka_cpu")
const GIF_FRAME_START_025 = parse(Int, get(ENV, "ATTEMPT025_GIF_FRAME_START", "1"))
const GIF_FRAME_END_025 = min(
    ATTEMPT025_PLOT_ITERATE_CAP,
    parse(Int, get(ENV, "ATTEMPT025_GIF_FRAME_END", string(ATTEMPT025_PLOT_ITERATE_CAP))),
)
const GIF_BLACK_CONTOURS_025 = lowercase(get(ENV, "ATTEMPT025_GIF_BLACK_CONTOURS", "false")) in ("1", "true", "yes")
const GIF_SHOW_EXCLUDED_025 = lowercase(get(ENV, "ATTEMPT025_GIF_SHOW_EXCLUDED", "false")) in ("1", "true", "yes")
const FRAME_DIR_025 = joinpath(ATTEMPT25_ROOT, "$(GIF_OUTPUT_TAG_025)_frames")

gif_path_025() = joinpath(ATTEMPT25_ROOT, "$(GIF_OUTPUT_TAG_025).gif")

function build_single_iterate_frame_025(
    accepted_segments::Vector{NTuple{4, Float64}},
    accepted_color::RGBAf,
    nominal_iterate::Int,
    excluded_segments::Union{Nothing, Vector{NTuple{4, Float64}}}=nothing,
)
    fig = Figure(size=(ATTEMPT025_FIG_WIDTH, ATTEMPT025_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka |x|-max contours, nominal iterate $(nominal_iterate)",
    )

    if excluded_segments !== nothing && !isempty(excluded_segments)
        xs_ex, ys_ex = segments_to_polyline_025(excluded_segments)
        lines!(ax, xs_ex, ys_ex; color=RGBAf(0.90, 0.08, 0.08, 0.95), linewidth=ATTEMPT025_LINEWIDTH)
    end

    xs, ys = segments_to_polyline_025(accepted_segments)
    lines!(ax, xs, ys; color=accepted_color, linewidth=ATTEMPT025_LINEWIDTH)

    xlims!(ax, ATTEMPT025_ALPHA_MIN, ATTEMPT025_ALPHA_MAX)
    ylims!(ax, ATTEMPT025_LAMBDA_MIN, ATTEMPT025_LAMBDA_MAX)
    return fig
end

function frame_path_025(nominal_iterate::Int)
    return joinpath(FRAME_DIR_025, @sprintf("frame_%02d.png", nominal_iterate))
end

function main()
    println("Rendering attempt-025 nominal-iterate GIF from existing sweep data.")
    println("Sweep dir: $(SWEEP_DIR_025)")
    println("Processed nominal iterates: 1:$(ATTEMPT025_PLOT_ITERATE_CAP)")
    println("Rendered GIF frames: $(GIF_FRAME_START_025):$(GIF_FRAME_END_025)")
    flush(stdout)

    RUN_COLUMNS_025 && error("nominal_iterate_gif.jl is plotting-only; set ATTEMPT025_RUN_COLUMNS=false")
    (1 <= GIF_FRAME_START_025 <= GIF_FRAME_END_025 <= ATTEMPT025_PLOT_ITERATE_CAP) || error("Invalid GIF frame iterate range.")

    dot_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = cumulative_to_interval_grids_025(cumulative_time_grids)

    colors = random_iterate_colors_025()
    mkpath(FRAME_DIR_025)

    skip_state = initialize_skip_state_025()
    for nominal_iterate in PLOT_ITERATES_025
        excluded_segments = GIF_SHOW_EXCLUDED_025 ? NTuple{4, Float64}[] : nothing
        segments, stats = process_nominal_iterate_025(nominal_iterate, dot_grids, time_grids, skip_state, nothing, excluded_segments)
        if GIF_FRAME_START_025 <= nominal_iterate <= GIF_FRAME_END_025
            frame_color = GIF_BLACK_CONTOURS_025 ? RGBAf(0.0, 0.0, 0.0, 0.95) : colors[nominal_iterate]
            fig = build_single_iterate_frame_025(segments, frame_color, nominal_iterate, excluded_segments)
            save(frame_path_025(nominal_iterate), fig; px_per_unit=ATTEMPT025_PX_PER_UNIT)
        end
        @printf(
            "Frame %d: missing=%d constant=%d incremented=%d contoured=%d emitted_segments=%d\n",
            nominal_iterate,
            stats.missing_data,
            stats.constant_sign,
            stats.incremented,
            stats.contoured_squares,
            stats.emitted_segments,
        )
        flush(stdout)
    end

    println("Saved GIF frames to $(FRAME_DIR_025)")
    println("GIF target path: $(gif_path_025())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
