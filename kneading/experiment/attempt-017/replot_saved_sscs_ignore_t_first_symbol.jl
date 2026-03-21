using Pkg

const ATTEMPT17_REPLOT_ROOT = @__DIR__
const REPO_ROOT_017 = normpath(joinpath(ATTEMPT17_REPLOT_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_017)

ENV["ATTEMPT017_CONTOUR_LINEWIDTH"] = get(ENV, "ATTEMPT017_CONTOUR_LINEWIDTH", "0.4")
ENV["ATTEMPT017_FIG_WIDTH"] = get(ENV, "ATTEMPT017_FIG_WIDTH", "2200")
ENV["ATTEMPT017_FIG_HEIGHT"] = get(ENV, "ATTEMPT017_FIG_HEIGHT", "1700")
ENV["ATTEMPT017_PX_PER_UNIT"] = get(ENV, "ATTEMPT017_PX_PER_UNIT", "2.0")

include(joinpath(ATTEMPT17_REPLOT_ROOT, "contours.jl"))

using Printf

const INPUT_RESULTS_PATH = joinpath(ATTEMPT17_ROOT, get(ENV, "ATTEMPT017_REPLOT_INPUT_RESULTS", "grid1200_seq10_prefixes_remap40_results.tsv"))
const OUTPUT_STEM = get(ENV, "ATTEMPT017_REPLOT_OUTPUT_STEM", "grid1200_seq10_prefixes_remap40_savedsscs_ignore_tfirst")
const OUTPUT_PLOT_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_contours.png")
const OUTPUT_SUMMARY_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_summary.txt")
const OUTPUT_T_LEGEND_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_T_legend.tsv")
const OUTPUT_GAMMA_LEGEND_PATH = joinpath(ATTEMPT17_ROOT, OUTPUT_STEM * "_gamma_legend.tsv")

const T_STATUS_PATTERN = r"^error: T SSCS does not start with ±1: \[(.*)\]$"
const T_STATUS_EMPTY_PATTERN = r"^error: T SSCS does not start with ±1: Int64\[\]$"

struct SavedSSCSRow
    delta_x::Float64
    delta_ca::Float64
    T_scs::Union{Nothing, Vector{Int}}
    gamma_scs::Union{Nothing, Vector{Int}}
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
    return encode_sequence(tail)
end

function read_saved_rows(path::String)
    rows = SavedSSCSRow[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            fields = split(line, '\t')
            delta_x = parse(Float64, fields[1])
            delta_ca = parse(Float64, fields[2])
            status = fields[12]

            if status == "ok"
                push!(rows, SavedSSCSRow(
                    delta_x,
                    delta_ca,
                    parse_sequence_field(fields[10]),
                    parse_sequence_field(fields[11]),
                    status,
                    false,
                ))
            else
                T_scs = parse_saved_T_from_status(status)
                push!(rows, SavedSSCSRow(
                    delta_x,
                    delta_ca,
                    T_scs,
                    nothing,
                    status,
                    !isnothing(T_scs),
                ))
            end
        end
    end
    return rows
end

function write_custom_legend(path::String, counts::Dict{BigInt, Int}, examples::Dict{BigInt, Vector{Int}})
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

function build_custom_grids(rows::Vector{SavedSSCSRow})
    T_counts = Dict{BigInt, Int}()
    gamma_counts = Dict{BigInt, Int}()
    T_examples = Dict{BigInt, Vector{Int}}()
    gamma_examples = Dict{BigInt, Vector{Int}}()

    recovered_T_rows = 0
    usable_T_rows = 0
    usable_gamma_rows = 0
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
        end
        if !isnothing(row.gamma_scs)
            usable_gamma_rows += 1
            gamma_encoding = encode_sequence(row.gamma_scs)
            gamma_counts[gamma_encoding] = get(gamma_counts, gamma_encoding, 0) + 1
            if !haskey(gamma_examples, gamma_encoding)
                gamma_examples[gamma_encoding] = row.gamma_scs
            end
        elseif isnothing(row.T_scs)
            unrecoverable_rows += 1
        end
    end

    T_lookup = write_custom_legend(OUTPUT_T_LEGEND_PATH, T_counts, T_examples)
    gamma_lookup = write_custom_legend(OUTPUT_GAMMA_LEGEND_PATH, gamma_counts, gamma_examples)

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
        if !isnothing(row.gamma_scs)
            gamma_grid[x_idx, y_idx] = gamma_lookup[encode_sequence(row.gamma_scs)]
        end
    end

    all(filled) || error("Not all grid entries were read from saved results.")

    return (
        T_grid=T_grid,
        gamma_grid=gamma_grid,
        recovered_T_rows=recovered_T_rows,
        usable_T_rows=usable_T_rows,
        usable_gamma_rows=usable_gamma_rows,
        untouched_ok_rows=untouched_ok_rows,
        unrecoverable_rows=unrecoverable_rows,
        unique_T=length(T_lookup),
        unique_gamma=length(gamma_lookup),
    )
end

function write_summary(path::String, rows::Vector{SavedSSCSRow}, stats)
    open(path, "w") do io
        println(io, "input_results=$(INPUT_RESULTS_PATH)")
        println(io, "output_plot=$(OUTPUT_PLOT_PATH)")
        println(io, "total_rows=$(length(rows))")
        println(io, "ok_rows_preserved=$(stats.untouched_ok_rows)")
        println(io, "usable_T_rows=$(stats.usable_T_rows)")
        println(io, "recovered_T_rows_from_error_status=$(stats.recovered_T_rows)")
        println(io, "usable_gamma_rows=$(stats.usable_gamma_rows)")
        println(io, "rows_still_missing_both=$(stats.unrecoverable_rows)")
        println(io, "unique_T_categories_ignore_first=$(stats.unique_T)")
        println(io, "unique_gamma_categories_saved_only=$(stats.unique_gamma)")
        println(io, "note=T categories ignore the first T symbol; gamma categories use only saved gamma_scs with no recomputation.")
    end
end

function main()
    rows = read_saved_rows(INPUT_RESULTS_PATH)
    stats = build_custom_grids(rows)
    save_contour_plot(OUTPUT_PLOT_PATH, stats.T_grid, stats.gamma_grid)
    write_summary(OUTPUT_SUMMARY_PATH, rows, stats)

    println("Saved contour plot to $(OUTPUT_PLOT_PATH)")
    println("Saved summary to $(OUTPUT_SUMMARY_PATH)")
    println("Usable T rows: $(stats.usable_T_rows) / $(length(rows))")
    println("Recovered T rows from error status: $(stats.recovered_T_rows)")
    println("Usable gamma rows: $(stats.usable_gamma_rows) / $(length(rows))")
    println("Rows still missing both T and gamma: $(stats.unrecoverable_rows)")
    println("Unique T categories (ignoring first symbol): $(stats.unique_T)")
    println("Unique gamma categories (saved only): $(stats.unique_gamma)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
