using Pkg

const ATTEMPT17_REPLOT_ROOT = @__DIR__
const REPO_ROOT_017 = normpath(joinpath(ATTEMPT17_REPLOT_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_017)

ENV["ATTEMPT017_CONTOUR_LINEWIDTH"] = get(ENV, "ATTEMPT017_CONTOUR_LINEWIDTH", "0.3333335")
ENV["ATTEMPT017_FIG_WIDTH"] = get(ENV, "ATTEMPT017_FIG_WIDTH", "2200")
ENV["ATTEMPT017_FIG_HEIGHT"] = get(ENV, "ATTEMPT017_FIG_HEIGHT", "1700")
ENV["ATTEMPT017_PX_PER_UNIT"] = get(ENV, "ATTEMPT017_PX_PER_UNIT", "2.0")

include(joinpath(ATTEMPT17_REPLOT_ROOT, "contours.jl"))

const INPUT_RESULTS_PATH = joinpath(ATTEMPT17_ROOT, get(ENV, "ATTEMPT017_REPLOT_INPUT_RESULTS", "grid1200_seq10_prefixes_remap40_results.tsv"))
const OUTPUT_STEM = get(ENV, "ATTEMPT017_REDPLOT_OUTPUT_STEM", "grid1200_seq10_prefixes_remap40_savedsscs_redonly_ignore_tfirst_tail3_blackbg_thicker")
const OUTPUT_PLOT_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_contours.png")
const OUTPUT_SUMMARY_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_summary.txt")
const OUTPUT_T_LEGEND_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_T_legend.tsv")
const OPAQUE_T_COLOR = RGBAf(1.0, 0.0, 0.0, 1.0)
const T_TAIL_PREFIX_LENGTH = parse(Int, get(ENV, "ATTEMPT017_T_TAIL_PREFIX_LENGTH", "3"))
const DISPLAY_DELTA_X_MIN = parse(Float64, get(ENV, "ATTEMPT017_DISPLAY_DELTA_X_MIN", "-2.2"))
const DISPLAY_DELTA_X_MAX = parse(Float64, get(ENV, "ATTEMPT017_DISPLAY_DELTA_X_MAX", "0.2"))

const T_STATUS_PATTERN = r"^error: T SSCS does not start with ±1: \[(.*)\]$"
const T_STATUS_EMPTY_PATTERN = r"^error: T SSCS does not start with ±1: Int64\[\]$"

struct SavedTRow
    delta_x::Float64
    delta_ca::Float64
    T_scs::Union{Nothing, Vector{Int}}
    status::String
    recovered_T::Bool
end

function parse_bracketed_ints(inner::AbstractString)
    stripped = strip(inner)
    isempty(stripped) && return Int[]
    return [parse(Int, strip(token)) for token in split(stripped, ",")]
end

function parse_saved_T_from_status(status::AbstractString)
    if occursin(T_STATUS_EMPTY_PATTERN, status)
        return Int[]
    end

    match_result = match(T_STATUS_PATTERN, status)
    if isnothing(match_result)
        return nothing
    end
    return parse_bracketed_ints(match_result.captures[1])
end

function encode_T_ignoring_first(seq::Vector{Int})
    tail = length(seq) >= 2 ? seq[2:end] : Int[]
    return encode_sequence(first(tail, min(length(tail), T_TAIL_PREFIX_LENGTH)))
end

function read_saved_rows(path::String)
    rows = SavedTRow[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, '\t')
            delta_x = parse(Float64, fields[1])
            delta_ca = parse(Float64, fields[2])
            status = fields[12]

            if status == "ok"
                push!(rows, SavedTRow(
                    delta_x,
                    delta_ca,
                    parse_sequence_field(fields[10]),
                    status,
                    false,
                ))
            else
                T_scs = parse_saved_T_from_status(status)
                push!(rows, SavedTRow(
                    delta_x,
                    delta_ca,
                    T_scs,
                    status,
                    !isnothing(T_scs),
                ))
            end
        end
    end
    return rows
end

function write_T_legend(path::String, counts::Dict{BigInt, Int}, examples::Dict{BigInt, Vector{Int}})
    encodings = sort(collect(keys(counts)))
    lookup = Dict(encoding => idx for (idx, encoding) in enumerate(encodings))
    open(path, "w") do io
        println(io, "category_id\tcount\tencoding\tsequence")
        for encoding in encodings
            println(io, join([
                string(lookup[encoding]),
                string(counts[encoding]),
                string(encoding),
                join(examples[encoding], ","),
            ], '\t'))
        end
    end
    return lookup
end

function build_T_grid(rows::Vector{SavedTRow})
    T_counts = Dict{BigInt, Int}()
    T_examples = Dict{BigInt, Vector{Int}}()

    recovered_T_rows = 0
    usable_T_rows = 0
    untouched_ok_rows = 0
    unrecoverable_rows = 0

    for row in rows
        if row.status == "ok"
            untouched_ok_rows += 1
        end
        if !isnothing(row.T_scs)
            usable_T_rows += 1
            if row.recovered_T
                recovered_T_rows += 1
            end
            T_encoding = encode_T_ignoring_first(row.T_scs)
            T_counts[T_encoding] = get(T_counts, T_encoding, 0) + 1
            if !haskey(T_examples, T_encoding)
                T_examples[T_encoding] = row.T_scs
            end
        else
            unrecoverable_rows += 1
        end
    end

    T_lookup = write_T_legend(OUTPUT_T_LEGEND_PATH, T_counts, T_examples)

    T_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    gamma_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))

    for row in rows
        x_idx = nearest_index(DELTA_CAS_010, row.delta_ca, "ΔCa")
        y_idx = nearest_index(DELTA_XS_010, row.delta_x, "Δx")
        filled[x_idx, y_idx] && error("Duplicate row for Δx=$(row.delta_x), ΔCa=$(row.delta_ca)")
        filled[x_idx, y_idx] = true

        if !isnothing(row.T_scs)
            T_grid[x_idx, y_idx] = T_lookup[encode_T_ignoring_first(row.T_scs)]
        end
    end

    all(filled) || error("Not all grid entries were read from saved results.")

    return (
        T_grid=T_grid,
        gamma_grid=gamma_grid,
        recovered_T_rows=recovered_T_rows,
        usable_T_rows=usable_T_rows,
        untouched_ok_rows=untouched_ok_rows,
        unrecoverable_rows=unrecoverable_rows,
        unique_T=length(T_lookup),
    )
end

function write_summary(path::String, rows::Vector{SavedTRow}, stats)
    open(path, "w") do io
        println(io, "input_results=$(INPUT_RESULTS_PATH)")
        println(io, "output_plot=$(OUTPUT_PLOT_PATH)")
        println(io, "total_rows=$(length(rows))")
        println(io, "ok_rows_preserved=$(stats.untouched_ok_rows)")
        println(io, "usable_T_rows=$(stats.usable_T_rows)")
        println(io, "recovered_T_rows_from_error_status=$(stats.recovered_T_rows)")
        println(io, "rows_still_missing_T=$(stats.unrecoverable_rows)")
        println(io, "unique_T_categories_ignore_first=$(stats.unique_T)")
        println(io, "T_tail_prefix_length=$(T_TAIL_PREFIX_LENGTH)")
        println(io, "display_delta_x_min=$(DISPLAY_DELTA_X_MIN)")
        println(io, "display_delta_x_max=$(DISPLAY_DELTA_X_MAX)")
        println(io, "note=Red contours only. T categories ignore the first T symbol and use only the first $(T_TAIL_PREFIX_LENGTH) symbols of the remaining tail. No gamma contours are drawn.")
    end
end

function save_red_only_contour_plot(path::String, T_grid::Matrix{Int})
    fig = Figure(size=(FIG_WIDTH, FIG_HEIGHT), backgroundcolor=:black)
    ax = Axis(
        fig[1, 1],
        xlabel="ΔCa",
        ylabel="Δx",
        backgroundcolor=:black,
        xlabelcolor=:white,
        ylabelcolor=:white,
        xticklabelcolor=:white,
        yticklabelcolor=:white,
        xtickcolor=:white,
        ytickcolor=:white,
        leftspinecolor=:white,
        bottomspinecolor=:white,
        rightspinecolor=:white,
        topspinecolor=:white,
        xgridcolor=RGBAf(1.0, 1.0, 1.0, 0.12),
        ygridcolor=RGBAf(1.0, 1.0, 1.0, 0.12),
    )

    T_xs, T_ys = categorical_marching_squares(T_grid, DELTA_CAS_010, DELTA_XS_010)
    lines!(ax, T_xs, T_ys; color=OPAQUE_T_COLOR, linewidth=LINEWIDTH)
    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    ylims!(ax, DISPLAY_DELTA_X_MIN, DISPLAY_DELTA_X_MAX)
    save(path, fig; px_per_unit=PX_PER_UNIT)
end

function main()
    rows = read_saved_rows(INPUT_RESULTS_PATH)
    stats = build_T_grid(rows)
    save_red_only_contour_plot(OUTPUT_PLOT_PATH, stats.T_grid)
    write_summary(OUTPUT_SUMMARY_PATH, rows, stats)

    println("Saved red-only contour plot to $(OUTPUT_PLOT_PATH)")
    println("Saved summary to $(OUTPUT_SUMMARY_PATH)")
    println("Usable T rows: $(stats.usable_T_rows) / $(length(rows))")
    println("Recovered T rows from error status: $(stats.recovered_T_rows)")
    println("Rows still missing T: $(stats.unrecoverable_rows)")
    println("Unique T categories (ignoring first symbol): $(stats.unique_T)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
