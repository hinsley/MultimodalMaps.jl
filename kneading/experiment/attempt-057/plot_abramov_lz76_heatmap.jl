using Pkg

const ATTEMPT57_PLOT_ROOT = @__DIR__
const REPO_ROOT_057_PLOT = normpath(joinpath(ATTEMPT57_PLOT_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_057_PLOT)

using CairoMakie
using Colors
using Printf

const DEFAULT_RESULTS_057 = joinpath(
    ATTEMPT57_PLOT_ROOT,
    "gcs_results",
    "grid500_arbitrary_ic_gh0p000_seq20_tmax3e5_results.tsv",
)
const RESULTS_PATH_057 = get(ENV, "ATTEMPT057_ABRAMOV_RESULTS", DEFAULT_RESULTS_057)
const OUTPUT_TAG_057 = get(ENV, "ATTEMPT057_ABRAMOV_OUTPUT_TAG", "grid500_arbitrary_ic_gh0p000_abramov_lz76_matcont_overlay_thermal")
const OUTPUT_DIR_057 = get(ENV, "ATTEMPT057_ABRAMOV_OUTPUT_DIR", dirname(RESULTS_PATH_057))
const OUTPUT_PATH_057 = get(ENV, "ATTEMPT057_ABRAMOV_OUTPUT", joinpath(OUTPUT_DIR_057, "$(OUTPUT_TAG_057).png"))
const SUMMARY_PATH_057 = get(ENV, "ATTEMPT057_ABRAMOV_SUMMARY", joinpath(OUTPUT_DIR_057, "$(OUTPUT_TAG_057)_summary.txt"))
const CURVES_PATH_057 = get(
    ENV,
    "ATTEMPT057_MATCONT_CURVES",
    joinpath(ATTEMPT57_PLOT_ROOT, "..", "attempt-050", "gcs_results", "jack_red_hil_overlap1_matcont_overlay_curves.tsv"),
)
const DRAW_MATCONT_OVERLAY_057 = lowercase(get(ENV, "ATTEMPT057_DRAW_MATCONT_OVERLAY", "true")) in ("1", "true", "yes")
const TRANSIENT_SYMBOLS_057 = parse(Int, get(ENV, "ATTEMPT057_ABRAMOV_TRANSIENT_SYMBOLS", "0"))
const TRANSIENT_TIME_UNITS_057 = parse(Float64, get(ENV, "ATTEMPT057_ABRAMOV_TRANSIENT_TIME_UNITS", "0.0"))
const ZERO_TERMINAL_RUN_LENGTH_057 = parse(Int, get(ENV, "ATTEMPT057_ABRAMOV_ZERO_TERMINAL_RUN_LENGTH", "4"))
const TIME_UNIT_SCALE_TO_SECONDS_057 = parse(Float64, get(ENV, "ATTEMPT057_ABRAMOV_TIME_UNIT_SCALE_TO_SECONDS", "1000.0"))
const COLORMAP_057 = Symbol(get(ENV, "ATTEMPT057_ABRAMOV_COLORMAP", "thermal"))
const LOG10_COLOR_057 = lowercase(get(ENV, "ATTEMPT057_ABRAMOV_LOG10_COLOR", "false")) in ("1", "true", "yes")
const COLOR_MIN_ENV_057 = get(ENV, "ATTEMPT057_ABRAMOV_COLOR_MIN", "")
const COLOR_MIN_057 = isempty(COLOR_MIN_ENV_057) ? 0.0 : parse(Float64, COLOR_MIN_ENV_057)
const COLOR_MAX_ENV_057 = get(ENV, "ATTEMPT057_ABRAMOV_COLOR_MAX", "")
const PLOT_WIDTH_057 = parse(Int, get(ENV, "ATTEMPT057_ABRAMOV_PLOT_WIDTH", "1820"))
const PLOT_HEIGHT_057 = parse(Int, get(ENV, "ATTEMPT057_ABRAMOV_PLOT_HEIGHT", "1200"))
const PX_PER_UNIT_057 = parse(Float64, get(ENV, "ATTEMPT057_ABRAMOV_PX_PER_UNIT", "2.0"))
const AXIS_LABEL_SIZE_057 = parse(Int, get(ENV, "ATTEMPT057_ABRAMOV_AXIS_LABEL_SIZE", "40"))
const TICK_LABEL_SIZE_057 = parse(Int, get(ENV, "ATTEMPT057_ABRAMOV_TICK_LABEL_SIZE", "28"))
const MATCONT_LINEWIDTH_057 = parse(Float64, get(ENV, "ATTEMPT057_MATCONT_LINEWIDTH", "6.0"))
const MATCONT_HALO_LINEWIDTH_057 = parse(Float64, get(ENV, "ATTEMPT057_MATCONT_HALO_LINEWIDTH", "9.5"))
const MATCONT_MARKER_SIZE_057 = parse(Float64, get(ENV, "ATTEMPT057_MATCONT_MARKER_SIZE", "34.0"))
const MATCONT_MARKER_STROKEWIDTH_057 = parse(Float64, get(ENV, "ATTEMPT057_MATCONT_MARKER_STROKEWIDTH", "5.0"))

struct MatcontCurvePart57
    curve::String
    part::Int
    source_handles::String
    xs::Vector{Float64}
    ys::Vector{Float64}
end

function header_lookup_057(header::AbstractString)
    names = split(header, '\t'; keepempty=true)
    return Dict(name => idx for (idx, name) in pairs(names))
end

parse_int_vector_057(field::AbstractString) =
    isempty(field) ? Int[] : parse.(Int, split(field, ","))

parse_float_vector_057(field::AbstractString) =
    isempty(field) ? Float64[] : parse.(Float64, split(field, ","))

function lz76_phrase_count_057(sequence::Vector{Int})::Int
    isempty(sequence) && return 0
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
            max_match_length = max(max_match_length, match_length)
        end
        complexity += 1
        i += max_match_length > 0 ? max_match_length + 1 : 1
    end
    return complexity
end

function terminal_run_length_057(sequence::Vector{Int})::Int
    isempty(sequence) && return 0
    last_symbol = last(sequence)
    run_length = 0
    for symbol in Iterators.reverse(sequence)
        symbol == last_symbol || break
        run_length += 1
    end
    return run_length
end

function abramov_lz76_057(sequence::Vector{Int}, intervals::Vector{Float64})
    length(sequence) == length(intervals) || return NaN
    if ZERO_TERMINAL_RUN_LENGTH_057 > 0 && terminal_run_length_057(sequence) >= ZERO_TERMINAL_RUN_LENGTH_057
        return 0.0
    end
    transient_index = TRANSIENT_SYMBOLS_057
    if TRANSIENT_TIME_UNITS_057 > 0
        cumulative_time = 0.0
        for (idx, interval) in pairs(intervals)
            cumulative_time += interval
            cumulative_time <= TRANSIENT_TIME_UNITS_057 || break
            transient_index = max(transient_index, idx)
        end
    end
    if length(sequence) <= transient_index
        return 0.0
    end
    used_sequence = sequence[(transient_index + 1):end]
    used_intervals = intervals[(transient_index + 1):end]
    total_time = sum(used_intervals) / TIME_UNIT_SCALE_TO_SECONDS_057
    if isempty(used_sequence) || !(isfinite(total_time)) || total_time <= 0
        return 0.0
    end
    n = length(used_sequence)
    complexity = lz76_phrase_count_057(used_sequence)
    return complexity * log2(n) / total_time
end

function read_matrix_057(path::String)
    rows = Tuple{Float64, Float64, Float64}[]
    ok_count = 0
    error_count = 0
    finite_count = 0
    full_count = 0
    min_len = typemax(Int)
    max_len = 0
    open(path, "r") do io
        lookup = header_lookup_057(readline(io))
        required = ("delta_x", "delta_ca", "arbitrary_scs", "event_intervals", "status")
        for name in required
            haskey(lookup, name) || error("Missing required column: $(name)")
        end
        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            if fields[lookup["status"]] != "ok"
                error_count += 1
                push!(rows, (delta_ca, delta_x, NaN))
                continue
            end
            ok_count += 1
            scs = parse_int_vector_057(fields[lookup["arbitrary_scs"]])
            intervals = parse_float_vector_057(fields[lookup["event_intervals"]])
            value = abramov_lz76_057(scs, intervals)
            finite_count += isfinite(value) ? 1 : 0
            full_count += length(scs) >= 20 ? 1 : 0
            min_len = min(min_len, length(scs))
            max_len = max(max_len, length(scs))
            push!(rows, (delta_ca, delta_x, value))
        end
    end

    delta_cas = sort(unique(row[1] for row in rows))
    delta_xs = sort(unique(row[2] for row in rows))
    matrix = fill(NaN, length(delta_cas), length(delta_xs))
    ca_lookup = Dict(value => idx for (idx, value) in pairs(delta_cas))
    x_lookup = Dict(value => idx for (idx, value) in pairs(delta_xs))
    for (delta_ca, delta_x, value) in rows
        matrix[ca_lookup[delta_ca], x_lookup[delta_x]] = value
    end
    return delta_cas, delta_xs, matrix, ok_count, error_count, finite_count, full_count, min_len, max_len
end

function read_matcont_curve_parts_057(path::String)
    parts = Dict{Tuple{String, Int}, MatcontCurvePart57}()
    order = Tuple{String, Int}[]
    open(path, "r") do io
        lookup = header_lookup_057(readline(io))
        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            curve = fields[lookup["curve"]]
            part = parse(Int, fields[lookup["part"]])
            key = (curve, part)
            if !haskey(parts, key)
                parts[key] = MatcontCurvePart57(curve, part, fields[lookup["source_handles"]], Float64[], Float64[])
                push!(order, key)
            end
            push!(parts[key].xs, parse(Float64, fields[lookup["x"]]))
            push!(parts[key].ys, parse(Float64, fields[lookup["y"]]))
        end
    end
    return [parts[key] for key in order]
end

matcont_color_057(curve::String) =
    curve == "homSF" ? RGBAf(0, 0, 0, 1) :
    curve == "orange_unlabeled" ? RGBAf(0.00, 0.55, 0.18, 1.0) :
    RGBAf(0.1, 0.1, 0.1, 1)

function draw_matcont_overlay_057!(ax, overlay_parts)
    homsf_endpoint = nothing
    for part in overlay_parts
        lines!(ax, part.xs, part.ys; color=RGBAf(1, 1, 1, 0.88), linewidth=MATCONT_HALO_LINEWIDTH_057)
        lines!(ax, part.xs, part.ys; color=matcont_color_057(part.curve), linewidth=MATCONT_LINEWIDTH_057)
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
            markersize=MATCONT_MARKER_SIZE_057,
            color=RGBAf(0.90, 0.00, 0.00, 1.0),
            strokecolor=RGBAf(1, 1, 1, 1),
            strokewidth=MATCONT_MARKER_STROKEWIDTH_057,
        )
    end
end

function fixed_ticks_057(values::Vector{Float64}, fmt::String, step::Float64)
    ticks = collect(first(values):step:last(values))
    if isempty(ticks) || last(ticks) < last(values) - 1.0e-9
        push!(ticks, last(values))
    end
    pfmt = Printf.Format(fmt)
    return ticks, [Printf.format(pfmt, tick) for tick in ticks]
end

function main()
    mkpath(dirname(OUTPUT_PATH_057))
    delta_cas, delta_xs, matrix, ok_count, error_count, finite_count, full_count, min_len, max_len =
        read_matrix_057(RESULTS_PATH_057)
    finite_values = matrix[isfinite.(matrix)]
    isempty(finite_values) && error("No finite Abramov-normalized LZ76 values found.")
    positive_values = finite_values[finite_values .> 0]
    matrix_for_plot = if LOG10_COLOR_057
        map(value -> isfinite(value) && value > 0 ? log10(value) : NaN, matrix)
    else
        matrix
    end
    finite_plot_values = matrix_for_plot[isfinite.(matrix_for_plot)]
    color_min = if !isempty(COLOR_MIN_ENV_057)
        COLOR_MIN_057
    elseif LOG10_COLOR_057
        isempty(finite_plot_values) ? parse(Float64, get(ENV, "ATTEMPT057_ABRAMOV_EMPTY_LOG_COLOR_MIN", "-1.0")) : minimum(finite_plot_values)
    else
        COLOR_MIN_057
    end
    color_max = if !isempty(COLOR_MAX_ENV_057)
        parse(Float64, COLOR_MAX_ENV_057)
    elseif isempty(finite_plot_values)
        parse(Float64, get(ENV, "ATTEMPT057_ABRAMOV_EMPTY_LOG_COLOR_MAX", "0.0"))
    else
        maximum(finite_plot_values)
    end
    overlay_parts = DRAW_MATCONT_OVERLAY_057 ? read_matcont_curve_parts_057(CURVES_PATH_057) : MatcontCurvePart57[]

    fig = Figure(size=(PLOT_WIDTH_057, PLOT_HEIGHT_057), figure_padding=(20, 35, 20, 20), backgroundcolor=RGBAf(1, 1, 1, 1))
    ax = Axis(
        fig[1, 1],
        xlabel=L"\Delta \mathrm{Ca}",
        ylabel=L"\Delta x",
        xlabelsize=AXIS_LABEL_SIZE_057,
        ylabelsize=AXIS_LABEL_SIZE_057,
        xticklabelsize=TICK_LABEL_SIZE_057,
        yticklabelsize=TICK_LABEL_SIZE_057,
    )
    heat = heatmap!(
        ax,
        delta_cas,
        delta_xs,
        matrix_for_plot;
        colormap=COLORMAP_057,
        colorrange=(color_min, color_max),
        nan_color=RGBAf(0.92, 0.92, 0.92, 1),
    )
    DRAW_MATCONT_OVERLAY_057 && draw_matcont_overlay_057!(ax, overlay_parts)
    ax.xticks = fixed_ticks_057(delta_cas, "%.0f", 5.0)
    ax.yticks = fixed_ticks_057(delta_xs, "%.1f", 0.1)
    xlims!(ax, first(delta_cas), last(delta_cas))
    ylims!(ax, first(delta_xs), last(delta_xs))
    colorbar_label = LOG10_COLOR_057 ? "log10(bits/s) (Abramov LZ76)" : "bits/s (Abramov LZ76)"
    Colorbar(fig[1, 2], heat; label=colorbar_label, labelsize=AXIS_LABEL_SIZE_057 - 4, ticklabelsize=TICK_LABEL_SIZE_057)
    colgap!(fig.layout, 18)
    save(OUTPUT_PATH_057, fig; px_per_unit=PX_PER_UNIT_057)

    open(SUMMARY_PATH_057, "w") do io
        println(io, "results_path\t$(RESULTS_PATH_057)")
        println(io, "plot_path\t$(OUTPUT_PATH_057)")
        println(io, "curves_path\t$(CURVES_PATH_057)")
        println(io, "draw_matcont_overlay\t$(DRAW_MATCONT_OVERLAY_057)")
        println(io, "ok_count\t$(ok_count)")
        println(io, "error_count\t$(error_count)")
        println(io, "finite_count\t$(finite_count)")
        println(io, "full_length_count\t$(full_count)")
        println(io, "transient_symbols_omitted\t$(TRANSIENT_SYMBOLS_057)")
        println(io, "transient_time_units_omitted\t$(TRANSIENT_TIME_UNITS_057)")
        println(io, "zero_terminal_run_length\t$(ZERO_TERMINAL_RUN_LENGTH_057)")
        println(io, "time_unit_scale_to_seconds\t$(TIME_UNIT_SCALE_TO_SECONDS_057)")
        println(io, "sequence_length_min\t$(min_len)")
        println(io, "sequence_length_max\t$(max_len)")
        println(io, "finite_min\t$(minimum(finite_values))")
        println(io, "finite_max\t$(maximum(finite_values))")
        println(io, "positive_finite_count\t$(length(positive_values))")
        println(io, "positive_finite_min\t$(isempty(positive_values) ? NaN : minimum(positive_values))")
        println(io, "positive_finite_max\t$(isempty(positive_values) ? NaN : maximum(positive_values))")
        println(io, "color_quantity\t$(LOG10_COLOR_057 ? "log10_abramov_normalized_lz76" : "abramov_normalized_lz76")")
        println(io, "colorrange_min\t$(color_min)")
        println(io, "colorrange_max\t$(color_max)")
        println(io, "normalization\tlz76_phrase_count * log2(n) / (sum(used_event_intervals) / time_unit_scale_to_seconds)")
    end
    println("Saved plot to $(OUTPUT_PATH_057)")
    println("Saved summary to $(SUMMARY_PATH_057)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
