using Pkg

const ATTEMPT50_MATCONT_ROOT = @__DIR__
const DEFAULT_MATCONT_RESULTS = joinpath(
    ATTEMPT50_MATCONT_ROOT,
    "gcs_results",
    "grid1000_seq12_tmax1e5_prefixes_remap40_newmodel_results.tsv",
)

if !haskey(ENV, "ATTEMPT050_FILTER_RESULTS")
    ENV["ATTEMPT050_FILTER_RESULTS"] = DEFAULT_MATCONT_RESULTS
end
if !haskey(ENV, "ATTEMPT050_FILTER_OUTPUT_DIR")
    ENV["ATTEMPT050_FILTER_OUTPUT_DIR"] = joinpath(ATTEMPT50_MATCONT_ROOT, "gcs_results")
end
if !haskey(ENV, "ATTEMPT050_FILTER_OUTPUT_TAG")
    ENV["ATTEMPT050_FILTER_OUTPUT_TAG"] = "grid1000_seq12_tmax1e5_prefixcompatible_tzero2to12"
end

include(joinpath(ATTEMPT50_MATCONT_ROOT, "plot_filtered_full_contours.jl"))

const MATCONT_CURVES_PATH = get(
    ENV,
    "ATTEMPT050_MATCONT_CURVES",
    joinpath(ATTEMPT50_MATCONT_ROOT, "gcs_results", "jack_red_hil_overlap1_matcont_overlay_curves.tsv"),
)
const MATCONT_OUTPUT_TAG = get(ENV, "ATTEMPT050_MATCONT_OUTPUT_TAG", "$(FILTER_OUTPUT_TAG)_matcont_overlay")
const MATCONT_OUTPUT_PATH = get(ENV, "ATTEMPT050_MATCONT_OUTPUT", joinpath(FILTER_OUTPUT_DIR, "$(MATCONT_OUTPUT_TAG)_contours.png"))
const MATCONT_SUMMARY_PATH = get(ENV, "ATTEMPT050_MATCONT_SUMMARY", joinpath(FILTER_OUTPUT_DIR, "$(MATCONT_OUTPUT_TAG)_summary.txt"))
const MATCONT_SOURCE_FIG = get(ENV, "ATTEMPT050_MATCONT_SOURCE_FIG", "/Users/carterhinsley/Downloads/jack_red_hil_overlap1.fig")
const MATCONT_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT050_MATCONT_LINEWIDTH", "6.0"))
const MATCONT_HALO_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT050_MATCONT_HALO_LINEWIDTH", "9.5"))
const MATCONT_MARKER_SIZE = parse(Float64, get(ENV, "ATTEMPT050_MATCONT_MARKER_SIZE", "34.0"))
const MATCONT_MARKER_STROKEWIDTH = parse(Float64, get(ENV, "ATTEMPT050_MATCONT_MARKER_STROKEWIDTH", "5.0"))
const MATCONT_AXIS_LABEL_SIZE = parse(Int, get(ENV, "ATTEMPT050_MATCONT_AXIS_LABEL_SIZE", string(AXIS_LABEL_SIZE + 6)))
const MATCONT_TICK_LABEL_SIZE = parse(Int, get(ENV, "ATTEMPT050_MATCONT_TICK_LABEL_SIZE", string(TICK_LABEL_SIZE + 4)))
const MATCONT_FIGURE_PADDING = (
    parse(Float64, get(ENV, "ATTEMPT050_MATCONT_LEFT_PADDING", "20")),
    parse(Float64, get(ENV, "ATTEMPT050_MATCONT_RIGHT_PADDING", "90")),
    parse(Float64, get(ENV, "ATTEMPT050_MATCONT_BOTTOM_PADDING", "20")),
    parse(Float64, get(ENV, "ATTEMPT050_MATCONT_TOP_PADDING", "20")),
)
const MATCONT_SUPPRESS_SIGN_ONLY = lowercase(get(ENV, "ATTEMPT050_MATCONT_SUPPRESS_SIGN_ONLY", "false")) in ("1", "true", "yes")
const MATCONT_DRAW_SIGN_ONLY = lowercase(get(ENV, "ATTEMPT050_MATCONT_DRAW_SIGN_ONLY", "false")) in ("1", "true", "yes")
const MATCONT_T_SIGN_ONLY_RGB = (0.00, 0.58, 0.18)
const MATCONT_GAMMA_SIGN_ONLY_RGB = (0.95, 0.48, 0.05)

struct MatcontCurvePart
    curve::String
    part::Int
    source_handles::String
    xs::Vector{Float64}
    ys::Vector{Float64}
end

function read_matcont_curve_parts(path::String)
    isfile(path) || error("Missing MATCONT overlay curve TSV: $(path). Run extract_matcont_overlay_curves.py first.")
    parts = Dict{Tuple{String, Int}, MatcontCurvePart}()
    order = Tuple{String, Int}[]
    open(path, "r") do io
        lookup = header_lookup(readline(io))
        for name in ("curve", "part", "source_handles", "x", "y")
            haskey(lookup, name) || error("Missing required TSV column: $(name)")
        end
        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            curve = fields[lookup["curve"]]
            part = parse(Int, fields[lookup["part"]])
            source_handles = fields[lookup["source_handles"]]
            key = (curve, part)
            if !haskey(parts, key)
                parts[key] = MatcontCurvePart(curve, part, source_handles, Float64[], Float64[])
                push!(order, key)
            end
            push!(parts[key].xs, parse(Float64, fields[lookup["x"]]))
            push!(parts[key].ys, parse(Float64, fields[lookup["y"]]))
        end
    end
    return [parts[key] for key in order]
end

function matcont_color(curve::String)
    if curve == "homSF"
        return RGBAf(0.00, 0.00, 0.00, 1.0)
    elseif curve == "orange_unlabeled"
        return RGBAf(0.00, 0.55, 0.18, 1.0)
    else
        return RGBAf(0.10, 0.10, 0.10, 1.0)
    end
end

function first_mismatch_is_sign_only(corner_sequences::NTuple{4, Vector{Int}}, mismatch_idx::Int)
    values = Int[]
    for seq in corner_sequences
        if !isempty(seq)
            push!(values, seq[mismatch_idx])
        end
    end
    unique_values = unique(values)
    if length(unique_values) != 2
        return false
    end
    a, b = unique_values
    return a != 0 && b != 0 && a == -b
end

function categorical_marching_squares_prefix_compatible_matcont(
    grid::Matrix{Int},
    x_values::Vector{Float64},
    y_values::Vector{Float64},
    sequences::Matrix{Vector{Int}},
    suppress_grid::Union{Nothing, BitMatrix}=nothing,
)
    xs_by_mismatch = [Float32[] for _ in 1:ATTEMPT10_MAX_SEQ_LENGTH]
    ys_by_mismatch = [Float32[] for _ in 1:ATTEMPT10_MAX_SEQ_LENGTH]
    sign_only_xs_by_mismatch = [Float32[] for _ in 1:ATTEMPT10_MAX_SEQ_LENGTH]
    sign_only_ys_by_mismatch = [Float32[] for _ in 1:ATTEMPT10_MAX_SEQ_LENGTH]
    for idx in 1:ATTEMPT10_MAX_SEQ_LENGTH
        sizehint!(xs_by_mismatch[idx], div(3 * (length(x_values) - 1) * (length(y_values) - 1), ATTEMPT10_MAX_SEQ_LENGTH))
        sizehint!(ys_by_mismatch[idx], div(3 * (length(x_values) - 1) * (length(y_values) - 1), ATTEMPT10_MAX_SEQ_LENGTH))
        sizehint!(sign_only_xs_by_mismatch[idx], div(3 * (length(x_values) - 1) * (length(y_values) - 1), ATTEMPT10_MAX_SEQ_LENGTH))
        sizehint!(sign_only_ys_by_mismatch[idx], div(3 * (length(x_values) - 1) * (length(y_values) - 1), ATTEMPT10_MAX_SEQ_LENGTH))
    end

    local_segments = NTuple{4, Float64}[]
    sizehint!(local_segments, 8)
    sign_only_cells_suppressed = 0

    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
            if !isnothing(suppress_grid) &&
               suppress_grid[x_idx, y_idx] &&
               suppress_grid[x_idx + 1, y_idx] &&
               suppress_grid[x_idx + 1, y_idx + 1] &&
               suppress_grid[x_idx, y_idx + 1]
                continue
            end

            corner_sequences = (
                sequences[x_idx, y_idx],
                sequences[x_idx + 1, y_idx],
                sequences[x_idx + 1, y_idx + 1],
                sequences[x_idx, y_idx + 1],
            )
            mismatch_idx = first_mismatch_index(corner_sequences)
            if isnothing(mismatch_idx) || sequences_are_prefix_compatible(corner_sequences)
                continue
            end
            sign_only_mismatch = first_mismatch_is_sign_only(corner_sequences, mismatch_idx)
            mismatch_idx = min(mismatch_idx, ATTEMPT10_MAX_SEQ_LENGTH)

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

            if sign_only_mismatch
                sign_only_cells_suppressed += 1
            end
            if sign_only_mismatch && MATCONT_SUPPRESS_SIGN_ONLY
                for (x_a, y_a, x_b, y_b) in local_segments
                    push!(sign_only_xs_by_mismatch[mismatch_idx], Float32(x_a), Float32(x_b), NaN32)
                    push!(sign_only_ys_by_mismatch[mismatch_idx], Float32(y_a), Float32(y_b), NaN32)
                end
            else
                for (x_a, y_a, x_b, y_b) in local_segments
                    push!(xs_by_mismatch[mismatch_idx], Float32(x_a), Float32(x_b), NaN32)
                    push!(ys_by_mismatch[mismatch_idx], Float32(y_a), Float32(y_b), NaN32)
                end
            end
        end
    end

    return xs_by_mismatch, ys_by_mismatch, sign_only_xs_by_mismatch, sign_only_ys_by_mismatch, sign_only_cells_suppressed
end

function save_filtered_contour_plot_with_matcont_overlay(
    path::String,
    T_grid::Matrix{Int},
    gamma_grid::Matrix{Int},
    suppress_T_grid::BitMatrix,
    T_sequences::Matrix{Vector{Int}},
    gamma_sequences::Matrix{Vector{Int}},
    overlay_parts::Vector{MatcontCurvePart},
)
    background = FILTER_TRANSPARENT_BACKGROUND ? RGBAf(1, 1, 1, 0) : RGBAf(1, 1, 1, 1)
    fig = Figure(size=(PLOT_WIDTH, PLOT_HEIGHT), backgroundcolor=background, figure_padding=MATCONT_FIGURE_PADDING)
    ax = Axis(
        fig[1, 1],
        xlabel=L"\Delta \mathrm{Ca}",
        ylabel=L"\Delta x",
        backgroundcolor=background,
        xlabelsize=MATCONT_AXIS_LABEL_SIZE,
        ylabelsize=MATCONT_AXIS_LABEL_SIZE,
        xticklabelsize=MATCONT_TICK_LABEL_SIZE,
        yticklabelsize=MATCONT_TICK_LABEL_SIZE,
    )

    T_xs_by_mismatch, T_ys_by_mismatch, T_sign_only_xs_by_mismatch, T_sign_only_ys_by_mismatch, T_sign_only_cells_suppressed =
        categorical_marching_squares_prefix_compatible_matcont(T_grid, DELTA_CAS_010, DELTA_XS_010, T_sequences, suppress_T_grid)
    gamma_xs_by_mismatch, gamma_ys_by_mismatch, gamma_sign_only_xs_by_mismatch, gamma_sign_only_ys_by_mismatch, gamma_sign_only_cells_suppressed =
        categorical_marching_squares_prefix_compatible_matcont(gamma_grid, DELTA_CAS_010, DELTA_XS_010, gamma_sequences)

    for idx in 1:ATTEMPT10_MAX_SEQ_LENGTH
        if !isempty(gamma_xs_by_mismatch[idx])
            lines!(
                ax,
                gamma_xs_by_mismatch[idx],
                gamma_ys_by_mismatch[idx];
                color=rgba_from_rgb_alpha(FILTER_GAMMA_RGB, alpha_for_mismatch_index(idx)),
                linewidth=FILTER_LINEWIDTH,
            )
        end
        if !isempty(T_xs_by_mismatch[idx])
            lines!(
                ax,
                T_xs_by_mismatch[idx],
                T_ys_by_mismatch[idx];
                color=rgba_from_rgb_alpha(FILTER_T_RGB, alpha_for_mismatch_index(idx)),
                linewidth=FILTER_LINEWIDTH,
            )
        end
    end

    if MATCONT_DRAW_SIGN_ONLY
        for idx in 1:ATTEMPT10_MAX_SEQ_LENGTH
            if !isempty(gamma_sign_only_xs_by_mismatch[idx])
                lines!(
                    ax,
                    gamma_sign_only_xs_by_mismatch[idx],
                    gamma_sign_only_ys_by_mismatch[idx];
                    color=rgba_from_rgb_alpha(MATCONT_GAMMA_SIGN_ONLY_RGB, alpha_for_mismatch_index(idx)),
                    linewidth=FILTER_LINEWIDTH,
                )
            end
            if !isempty(T_sign_only_xs_by_mismatch[idx])
                lines!(
                    ax,
                    T_sign_only_xs_by_mismatch[idx],
                    T_sign_only_ys_by_mismatch[idx];
                    color=rgba_from_rgb_alpha(MATCONT_T_SIGN_ONLY_RGB, alpha_for_mismatch_index(idx)),
                    linewidth=FILTER_LINEWIDTH,
                )
            end
        end
    end

    homsf_endpoint = nothing
    for part in overlay_parts
        lines!(
            ax,
            part.xs,
            part.ys;
            color=RGBAf(1, 1, 1, 0.82),
            linewidth=MATCONT_HALO_LINEWIDTH,
        )
        lines!(
            ax,
            part.xs,
            part.ys;
            color=matcont_color(part.curve),
            linewidth=MATCONT_LINEWIDTH,
        )
        if part.curve == "homSF"
            homsf_endpoint = (last(part.xs), last(part.ys))
        end
    end

    if !isnothing(homsf_endpoint)
        scatter!(
            ax,
            [homsf_endpoint[1]],
            [homsf_endpoint[2]];
            marker=:circle,
            markersize=MATCONT_MARKER_SIZE,
            color=RGBAf(0.90, 0.00, 0.00, 1.0),
            strokecolor=RGBAf(1, 1, 1, 1),
            strokewidth=MATCONT_MARKER_STROKEWIDTH,
        )
    end

    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    xlims!(ax, first(DELTA_CAS_010), last(DELTA_CAS_010))
    ylims!(ax, first(DELTA_XS_010), last(DELTA_XS_010))
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT)
    return T_sign_only_cells_suppressed, gamma_sign_only_cells_suppressed
end

function main()
    mkpath(dirname(MATCONT_OUTPUT_PATH))
    overlay_parts = read_matcont_curve_parts(MATCONT_CURVES_PATH)
    println("Running attempt-050 filtered full-contour replay with MATCONT overlay.")
    println("Results source: $(FILTER_RESULTS_PATH)")
    println("MATCONT curves: $(MATCONT_CURVES_PATH)")
    println("Output plot: $(MATCONT_OUTPUT_PATH)")
    flush(stdout)

    T_grid, gamma_grid, suppress_T_grid, T_sequences, gamma_sequences, ok_count, error_count, suppress_points, T_tail_unique =
        build_filtered_full_grids(FILTER_RESULTS_PATH)
    T_sign_only_cells_suppressed, gamma_sign_only_cells_suppressed = save_filtered_contour_plot_with_matcont_overlay(
        MATCONT_OUTPUT_PATH,
        T_grid,
        gamma_grid,
        suppress_T_grid,
        T_sequences,
        gamma_sequences,
        overlay_parts,
    )

    open(MATCONT_SUMMARY_PATH, "w") do io
        println(io, "results_path\t$(FILTER_RESULTS_PATH)")
        println(io, "source_fig\t$(MATCONT_SOURCE_FIG)")
        println(io, "curves_path\t$(MATCONT_CURVES_PATH)")
        println(io, "plot_path\t$(MATCONT_OUTPUT_PATH)")
        println(io, "xlim\t$(first(DELTA_CAS_010))\t$(last(DELTA_CAS_010))")
        println(io, "ylim\t$(first(DELTA_XS_010))\t$(last(DELTA_XS_010))")
        println(io, "ok_count\t$(ok_count)")
        println(io, "error_count\t$(error_count)")
        println(io, "suppressed_T_points\t$(suppress_points)")
        println(io, "T_tail_unique\t$(T_tail_unique)")
        println(io, "suppress_sign_only\t$(MATCONT_SUPPRESS_SIGN_ONLY)")
        println(io, "draw_sign_only\t$(MATCONT_DRAW_SIGN_ONLY)")
        println(io, "T_sign_only_cells_suppressed\t$(T_sign_only_cells_suppressed)")
        println(io, "gamma_sign_only_cells_suppressed\t$(gamma_sign_only_cells_suppressed)")
        for part in overlay_parts
            println(
                io,
                "overlay_part\t$(part.curve)\t$(part.part)\t$(part.source_handles)\t$(length(part.xs))\t$(minimum(part.xs))\t$(maximum(part.xs))\t$(minimum(part.ys))\t$(maximum(part.ys))",
            )
        end
    end

    println("Saved plot to $(MATCONT_OUTPUT_PATH)")
    println("Saved summary to $(MATCONT_SUMMARY_PATH)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
