using Pkg

const ATTEMPT50_LZ_ROOT = @__DIR__
const DEFAULT_LZ_RESULTS = joinpath(
    ATTEMPT50_LZ_ROOT,
    "gcs_results",
    "grid1000_seq12_tmax1e5_prefixes_remap40_newmodel_results.tsv",
)

if !haskey(ENV, "ATTEMPT050_FILTER_RESULTS")
    ENV["ATTEMPT050_FILTER_RESULTS"] = DEFAULT_LZ_RESULTS
end
if !haskey(ENV, "ATTEMPT050_FILTER_OUTPUT_DIR")
    ENV["ATTEMPT050_FILTER_OUTPUT_DIR"] = joinpath(ATTEMPT50_LZ_ROOT, "gcs_results")
end

include(joinpath(ATTEMPT50_LZ_ROOT, "plot_matcont_overlay.jl"))

const GAMMA_LZ_OUTPUT_TAG = get(
    ENV,
    "ATTEMPT050_GAMMA_LZ_OUTPUT_TAG",
    "grid1000_seq12_tmax1e5_gamma_scs_lz76_matcont_overlay",
)
const GAMMA_LZ_SEQUENCE_COLUMN = get(ENV, "ATTEMPT050_LZ_SEQUENCE_COLUMN", "gamma_scs")
const GAMMA_LZ_OUTPUT_PATH = get(
    ENV,
    "ATTEMPT050_GAMMA_LZ_OUTPUT",
    joinpath(FILTER_OUTPUT_DIR, "$(GAMMA_LZ_OUTPUT_TAG).png"),
)
const GAMMA_LZ_SUMMARY_PATH = get(
    ENV,
    "ATTEMPT050_GAMMA_LZ_SUMMARY",
    joinpath(FILTER_OUTPUT_DIR, "$(GAMMA_LZ_OUTPUT_TAG)_summary.txt"),
)
const GAMMA_LZ_HEATMAP_COLORMAP = get(ENV, "ATTEMPT050_GAMMA_LZ_COLORMAP", "thermal")
const GAMMA_LZ_TRANSIENT_SYMBOLS = parse(Int, get(ENV, "ATTEMPT050_GAMMA_LZ_TRANSIENT_SYMBOLS", "0"))
const GAMMA_LZ_NORMALIZATION_LENGTH = parse(Float64, get(ENV, "ATTEMPT050_GAMMA_LZ_NORMALIZATION_LENGTH", "1.0e5"))
const GAMMA_LZ_COLORRANGE_MIN = parse(Float64, get(ENV, "ATTEMPT050_GAMMA_LZ_COLOR_MIN", "0.0"))
const GAMMA_LZ_COLORRANGE_MAX_ENV = get(ENV, "ATTEMPT050_GAMMA_LZ_COLOR_MAX", "")
const GAMMA_LZ_FIGURE_SIZE = (
    parse(Int, get(ENV, "ATTEMPT050_GAMMA_LZ_PLOT_WIDTH", string(PLOT_WIDTH + 220))),
    parse(Int, get(ENV, "ATTEMPT050_GAMMA_LZ_PLOT_HEIGHT", string(PLOT_HEIGHT))),
)
const GAMMA_LZ_FIGURE_PADDING = (
    parse(Float64, get(ENV, "ATTEMPT050_GAMMA_LZ_LEFT_PADDING", "20")),
    parse(Float64, get(ENV, "ATTEMPT050_GAMMA_LZ_RIGHT_PADDING", "35")),
    parse(Float64, get(ENV, "ATTEMPT050_GAMMA_LZ_BOTTOM_PADDING", "20")),
    parse(Float64, get(ENV, "ATTEMPT050_GAMMA_LZ_TOP_PADDING", "20")),
)

function lz76_phrase_count(sequence::Vector{Int})::Int
    if isempty(sequence)
        return 0
    end

    complexity = 0
    i = 1
    n = length(sequence)

    while i <= n
        max_match_length = 0

        for j in 1:(i - 1)
            match_length = 0
            while (i + match_length <= n) && (sequence[j + match_length] == sequence[i + match_length])
                match_length += 1
                if j + match_length > i - 1
                    break
                end
            end
            if match_length > max_match_length
                max_match_length = match_length
            end
        end

        complexity += 1
        i += max_match_length > 0 ? max_match_length + 1 : 1
    end

    return complexity
end

function normalized_lz76_complexity_fixed_length(sequence::Vector{Int}, normalization_length::Float64)::Float64
    if isempty(sequence)
        return 0.0
    end
    n = length(sequence)
    b = length(unique(sequence))
    if b == 1
        return 0.0
    end
    complexity = lz76_phrase_count(sequence)
    return complexity * log2(n) / (normalization_length * log2(b))
end

function read_lz76_matrix(results_path::String, sequence_column::String)
    matrix = fill(NaN, length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    ok_count = 0
    error_count = 0
    finite_count = 0
    full_length_count = 0

    open(results_path, "r") do io
        lookup = header_lookup(readline(io))
        required = ("delta_x", "delta_ca", sequence_column, "status")
        for name in required
            haskey(lookup, name) || error("Missing required TSV column: $(name)")
        end

        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            ca_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
            x_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")
            if filled[ca_idx, x_idx]
                error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
            end
            filled[ca_idx, x_idx] = true

            if fields[lookup["status"]] != "ok"
                error_count += 1
                continue
            end

            ok_count += 1
            sequence = parse_sequence_field(fields[lookup[sequence_column]])
            lz_sequence = length(sequence) > GAMMA_LZ_TRANSIENT_SYMBOLS ?
                sequence[(GAMMA_LZ_TRANSIENT_SYMBOLS + 1):end] :
                Int[]
            matrix[ca_idx, x_idx] = normalized_lz76_complexity_fixed_length(
                lz_sequence,
                GAMMA_LZ_NORMALIZATION_LENGTH,
            )
            finite_count += isfinite(matrix[ca_idx, x_idx]) ? 1 : 0
            full_length_count += length(sequence) >= ATTEMPT10_MAX_SEQ_LENGTH ? 1 : 0
        end
    end

    all(filled) || error("One or more gamma LZ76 grid entries were not filled.")
    return matrix, ok_count, error_count, finite_count, full_length_count
end

function draw_matcont_overlay!(ax, overlay_parts::Vector{MatcontCurvePart})
    homsf_endpoint = nothing
    for part in overlay_parts
        lines!(
            ax,
            part.xs,
            part.ys;
            color=RGBAf(1, 1, 1, 0.88),
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
end

function save_lz76_heatmap(path::String, matrix::Matrix{Float64}, overlay_parts::Vector{MatcontCurvePart})
    finite_values = matrix[isfinite.(matrix)]
    isempty(finite_values) && error("No finite gamma_scs LZ76 values found.")
    colorrange_max = isempty(GAMMA_LZ_COLORRANGE_MAX_ENV) ?
        maximum(finite_values) :
        parse(Float64, GAMMA_LZ_COLORRANGE_MAX_ENV)

    fig = Figure(
        size=GAMMA_LZ_FIGURE_SIZE,
        backgroundcolor=RGBAf(1, 1, 1, 1),
        figure_padding=GAMMA_LZ_FIGURE_PADDING,
    )
    ax = Axis(
        fig[1, 1],
        xlabel=L"\Delta \mathrm{Ca}",
        ylabel=L"\Delta x",
        backgroundcolor=RGBAf(1, 1, 1, 1),
        xlabelsize=MATCONT_AXIS_LABEL_SIZE,
        ylabelsize=MATCONT_AXIS_LABEL_SIZE,
        xticklabelsize=MATCONT_TICK_LABEL_SIZE,
        yticklabelsize=MATCONT_TICK_LABEL_SIZE,
    )

    heat = heatmap!(
        ax,
        DELTA_CAS_010,
        DELTA_XS_010,
        matrix;
        colormap=Symbol(GAMMA_LZ_HEATMAP_COLORMAP),
        colorrange=(GAMMA_LZ_COLORRANGE_MIN, colorrange_max),
        nan_color=RGBAf(0.92, 0.92, 0.92, 1.0),
    )
    draw_matcont_overlay!(ax, overlay_parts)

    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    xlims!(ax, first(DELTA_CAS_010), last(DELTA_CAS_010))
    ylims!(ax, first(DELTA_XS_010), last(DELTA_XS_010))

    Colorbar(
        fig[1, 2],
        heat;
        label="normalized LZ76",
        labelsize=MATCONT_AXIS_LABEL_SIZE - 4,
        ticklabelsize=MATCONT_TICK_LABEL_SIZE,
    )
    colgap!(fig.layout, 18)

    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT)
    return minimum(finite_values), maximum(finite_values), colorrange_max
end

function main()
    mkpath(dirname(GAMMA_LZ_OUTPUT_PATH))
    overlay_parts = read_matcont_curve_parts(MATCONT_CURVES_PATH)
    println("Running attempt-050 $(GAMMA_LZ_SEQUENCE_COLUMN) normalized LZ76 heatmap with MATCONT overlay.")
    println("Results source: $(FILTER_RESULTS_PATH)")
    println("MATCONT curves: $(MATCONT_CURVES_PATH)")
    println("Output plot: $(GAMMA_LZ_OUTPUT_PATH)")
    flush(stdout)

    matrix, ok_count, error_count, finite_count, full_length_count =
        read_lz76_matrix(FILTER_RESULTS_PATH, GAMMA_LZ_SEQUENCE_COLUMN)
    finite_min, finite_max, colorrange_max =
        save_lz76_heatmap(GAMMA_LZ_OUTPUT_PATH, matrix, overlay_parts)

    open(GAMMA_LZ_SUMMARY_PATH, "w") do io
        println(io, "results_path\t$(FILTER_RESULTS_PATH)")
        println(io, "source_fig\t$(MATCONT_SOURCE_FIG)")
        println(io, "curves_path\t$(MATCONT_CURVES_PATH)")
        println(io, "plot_path\t$(GAMMA_LZ_OUTPUT_PATH)")
        println(io, "xlim\t$(first(DELTA_CAS_010))\t$(last(DELTA_CAS_010))")
        println(io, "ylim\t$(first(DELTA_XS_010))\t$(last(DELTA_XS_010))")
        println(io, "ok_count\t$(ok_count)")
        println(io, "error_count\t$(error_count)")
        println(io, "finite_lz_count\t$(finite_count)")
        println(io, "sequence_column\t$(GAMMA_LZ_SEQUENCE_COLUMN)")
        println(io, "sequence_full_length_count\t$(full_length_count)")
        println(io, "finite_lz_min\t$(finite_min)")
        println(io, "finite_lz_max\t$(finite_max)")
        println(io, "colorrange_min\t$(GAMMA_LZ_COLORRANGE_MIN)")
        println(io, "colorrange_max\t$(colorrange_max)")
        println(io, "colormap\t$(GAMMA_LZ_HEATMAP_COLORMAP)")
        println(io, "transient_symbols_omitted\t$(GAMMA_LZ_TRANSIENT_SYMBOLS)")
        println(io, "normalization_length\t$(GAMMA_LZ_NORMALIZATION_LENGTH)")
        for part in overlay_parts
            println(
                io,
                "overlay_part\t$(part.curve)\t$(part.part)\t$(part.source_handles)\t$(length(part.xs))\t$(minimum(part.xs))\t$(maximum(part.xs))\t$(minimum(part.ys))\t$(maximum(part.ys))",
            )
        end
    end

    println("Saved plot to $(GAMMA_LZ_OUTPUT_PATH)")
    println("Saved summary to $(GAMMA_LZ_SUMMARY_PATH)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
