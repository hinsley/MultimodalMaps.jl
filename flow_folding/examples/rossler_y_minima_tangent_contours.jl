include(joinpath(@__DIR__, "rossler_common.jl"))
include(joinpath(@__DIR__, "..", "..", "scans", "contours.jl"))

using Printf

const DEFAULT_SCAN_RESULTS = joinpath(
    @__DIR__,
    "..",
    "results",
    "rossler_y_minima_tangent_scan",
    "coarse_scan.tsv",
)
const DEFAULT_CONTOUR_DIR = joinpath(
    @__DIR__,
    "..",
    "results",
    "rossler_y_minima_tangent_scan",
    "contours",
)

struct ScanRow
    a::Float64
    c::Float64
    b::Float64
    status::String
    events::Int
    word::String
    code::Int
    period::Int
    gamma::Float64
    max_time::Float64
    first_time::Float64
    last_time::Float64
    min_y::Float64
    max_y::Float64
end

_parse_float(value::AbstractString) = isempty(value) ? NaN : parse(Float64, value)
_field(parts, index, default="") = index <= length(parts) ? parts[index] : default

function read_scan_results(path::AbstractString)
    rows = ScanRow[]
    open(path, "r") do io
        header = split(readline(io), '\t')
        cols = Dict(name => idx for (idx, name) in enumerate(header))
        max_time_col = get(cols, "max_time", get(cols, "t_end", 0))
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, '\t')
            push!(
                rows,
                ScanRow(
                    parse(Float64, parts[cols["a"]]),
                    parse(Float64, parts[cols["c"]]),
                    parse(Float64, parts[cols["b"]]),
                    parts[cols["status"]],
                    parse(Int, parts[cols["events"]]),
                    parts[cols["word"]],
                    parse(Int, parts[cols["code"]]),
                    parse(Int, parts[cols["period"]]),
                    _parse_float(parts[cols["gamma"]]),
                    max_time_col == 0 ? NaN : _parse_float(_field(parts, max_time_col)),
                    _parse_float(parts[cols["first_time"]]),
                    _parse_float(parts[cols["last_time"]]),
                    _parse_float(parts[cols["min_y"]]),
                    _parse_float(parts[cols["max_y"]]),
                ),
            )
        end
    end
    return rows
end

function sorted_axes(rows)
    return sort(unique(row.c for row in rows)), sort(unique(row.a for row in rows))
end

axis_lookup(values) = Dict(value => idx for (idx, value) in enumerate(values))

function completed_word_length(rows)
    lengths = [length(row.word) for row in rows if row.status == "ok"]
    isempty(lengths) && return 0
    return maximum(lengths)
end

function symbol_grid(rows, symbol_index::Int, c_values, a_values)
    c_lookup = axis_lookup(c_values)
    a_lookup = axis_lookup(a_values)
    grid = fill(NaN, length(a_values), length(c_values))
    for row in rows
        row.status == "ok" || continue
        length(row.word) >= symbol_index || continue
        grid[a_lookup[row.a], c_lookup[row.c]] = row.word[symbol_index] == '1' ? 1.0 : -1.0
    end
    return grid
end

function prefix_code(word::AbstractString, prefix_length::Int)
    length(word) >= prefix_length || return -1
    code = 0
    for idx in 1:prefix_length
        code = 2 * code + (word[idx] == '1' ? 1 : 0)
    end
    return code
end

function prefix_grid(rows, prefix_length::Int, c_values, a_values)
    c_lookup = axis_lookup(c_values)
    a_lookup = axis_lookup(a_values)
    grid = fill(-1, length(a_values), length(c_values))
    for row in rows
        row.status == "ok" || continue
        grid[a_lookup[row.a], c_lookup[row.c]] = prefix_code(row.word, prefix_length)
    end
    return grid
end

function ok_code_grid(rows, c_values, a_values)
    c_lookup = axis_lookup(c_values)
    a_lookup = axis_lookup(a_values)
    grid = fill(-1, length(a_values), length(c_values))
    for row in rows
        row.status == "ok" || continue
        grid[a_lookup[row.a], c_lookup[row.c]] = row.code
    end
    return grid
end

function nan_polyline_segments(xs, ys)
    segments = NTuple{4, Float64}[]
    last_point = nothing
    for (x, y) in zip(xs, ys)
        if isnan(x) || isnan(y)
            last_point = nothing
            continue
        end
        if !isnothing(last_point)
            x0, y0 = last_point
            push!(segments, (x0, y0, Float64(x), Float64(y)))
        end
        last_point = (Float64(x), Float64(y))
    end
    return segments
end

function category_boundary_segments(grid, c_values, a_values)
    xs, ys = march_squares_simple(grid, c_values, a_values)
    return nan_polyline_segments(xs, ys)
end

function symbol_segments(grid, c_values, a_values)
    return march_squares_zero_segments(grid, c_values, a_values; level=0.0)
end

function contour_color(index::Int, total::Int)
    palette = [
        "#b23a2e",
        "#2f66b3",
        "#238b62",
        "#8f4cb3",
        "#c37a1f",
        "#258ea6",
        "#cc4c8a",
        "#59662b",
    ]
    return palette[((index - 1) % length(palette)) + 1]
end

svg_escape(text) = replace(replace(replace(string(text), "&" => "&amp;"), "<" => "&lt;"), ">" => "&gt;")

function svg_plot(path; title, rows, c_values, a_values, contours, symbol_index=nothing)
    width = 1100.0
    height = 760.0
    left = 78.0
    right = 236.0
    top = 62.0
    bottom = 72.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    c_min, c_max = extrema(c_values)
    a_min, a_max = extrema(a_values)

    xpix(c) = left + (Float64(c) - c_min) / (c_max - c_min) * plot_width
    ypix(a) = top + (a_max - Float64(a)) / (a_max - a_min) * plot_height

    open(path, "w") do io
        println(io, """<svg xmlns="http://www.w3.org/2000/svg" width="$(Int(width))" height="$(Int(height))" viewBox="0 0 $(Int(width)) $(Int(height))">""")
        println(io, """<rect width="100%" height="100%" fill="white"/>""")
        println(io, """<text x="$(width / 2)" y="30" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="20" font-weight="650">$(svg_escape(title))</text>""")

        for tick in range(c_min, c_max; length=6)
            x = xpix(tick)
            println(io, """<line x1="$x" y1="$top" x2="$x" y2="$(top + plot_height)" stroke="#e1e5e0" stroke-width="1"/>""")
            println(io, """<text x="$x" y="$(top + plot_height + 25)" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="12" fill="#59635d">$(round(tick; digits=2))</text>""")
        end
        for tick in range(a_min, a_max; length=6)
            y = ypix(tick)
            println(io, """<line x1="$left" y1="$y" x2="$(left + plot_width)" y2="$y" stroke="#e1e5e0" stroke-width="1"/>""")
            println(io, """<text x="$(left - 12)" y="$(y + 4)" text-anchor="end" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="12" fill="#59635d">$(round(tick; digits=3))</text>""")
        end
        println(io, """<rect x="$left" y="$top" width="$plot_width" height="$plot_height" fill="none" stroke="#17201c" stroke-width="1.3"/>""")

        if !isnothing(symbol_index)
            for row in rows
                if row.status == "ok" && length(row.word) >= symbol_index
                    fill = row.word[symbol_index] == '1' ? "#b23a2e" : "#2f66b3"
                    println(io, """<circle cx="$(xpix(row.c))" cy="$(ypix(row.a))" r="4" fill="$fill" fill-opacity="0.28" stroke="none"/>""")
                end
            end
        else
            for row in rows
                row.status == "ok" || continue
                println(io, """<circle cx="$(xpix(row.c))" cy="$(ypix(row.a))" r="3" fill="#17201c" fill-opacity="0.18" stroke="none"/>""")
            end
        end

        legend_y = top + 20
        for (idx, contour) in enumerate(contours)
            segments, color, label, stroke_width = contour
            for (x1, y1, x2, y2) in segments
                println(io, """<line x1="$(xpix(x1))" y1="$(ypix(y1))" x2="$(xpix(x2))" y2="$(ypix(y2))" stroke="$color" stroke-width="$stroke_width" stroke-linecap="round"/>""")
            end
            println(io, """<line x1="$(left + plot_width + 25)" y1="$legend_y" x2="$(left + plot_width + 55)" y2="$legend_y" stroke="$color" stroke-width="$stroke_width" stroke-linecap="round"/>""")
            println(io, """<text x="$(left + plot_width + 64)" y="$(legend_y + 4)" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="12" fill="#17201c">$(svg_escape(label))</text>""")
            legend_y += 22
        end

        println(io, """<text x="$(left + plot_width / 2)" y="$(height - 18)" text-anchor="middle" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="15">c</text>""")
        println(io, """<text x="24" y="$(top + plot_height / 2)" text-anchor="middle" transform="rotate(-90 24 $(top + plot_height / 2))" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif" font-size="15">a</text>""")
        println(io, """</svg>""")
    end
end

function write_word_legend(path, rows)
    counts = Dict{String, Int}()
    for row in rows
        row.status == "ok" || continue
        counts[row.word] = get(counts, row.word, 0) + 1
    end
    ordered = sort(collect(counts); by=x -> (-x.second, x.first))
    open(path, "w") do io
        println(io, "word\tcount\tcode\tperiod\tgamma")
        for (word, count) in ordered
            bits = [char == '1' ? 1 : 0 for char in collect(word)]
            code = foldl((acc, bit) -> 2 * acc + bit, bits; init=0)
            println(io, join((word, count, code, least_period(bits), binary_sequence_value(bits)), '\t'))
        end
    end
end

function write_summary(
    path,
    rows,
    symbol_counts,
    prefix_counts;
    results_path,
    output_dir,
    scan_seconds=NaN,
    write_tsv_seconds=NaN,
    contour_generation_seconds=NaN,
)
    ok = count(row -> row.status == "ok", rows)
    total = length(rows)
    max_times = [row.max_time for row in rows if isfinite(row.max_time)]
    max_time = isempty(max_times) ? NaN : first(max_times)
    open(path, "w") do io
        println(io, "results_path\t$(results_path)")
        println(io, "output_dir\t$(output_dir)")
        println(io, "total_points\t$(total)")
        println(io, "ok_points\t$(ok)")
        println(io, "max_time_limited_points\t$(total - ok)")
        println(io, "word_length\t$(completed_word_length(rows))")
        println(io, "max_time\t$(max_time)")
        println(io, "scan_seconds\t$(scan_seconds)")
        println(io, "write_tsv_seconds\t$(write_tsv_seconds)")
        println(io, "contour_generation_seconds\t$(contour_generation_seconds)")
        for (idx, count) in enumerate(symbol_counts)
            println(io, @sprintf("symbol%02d_segments\t%d", idx, count))
        end
        for (idx, count) in enumerate(prefix_counts)
            println(io, @sprintf("prefix%02d_boundary_segments\t%d", idx, count))
        end
    end
end

function write_all_contours(
    results_path::AbstractString=DEFAULT_SCAN_RESULTS;
    output_dir::AbstractString=DEFAULT_CONTOUR_DIR,
    stem="coarse_scan",
    scan_seconds=NaN,
    write_tsv_seconds=NaN,
)
    contour_started = time()
    rows = read_scan_results(results_path)
    isempty(rows) && error("No scan rows found in $(results_path)")
    c_values, a_values = sorted_axes(rows)
    n_symbols = completed_word_length(rows)
    n_symbols > 0 || error("No completed kneading words found in $(results_path)")
    mkpath(output_dir)

    symbol_counts = Int[]
    all_contours = []
    for symbol_index in 1:n_symbols
        grid = symbol_grid(rows, symbol_index, c_values, a_values)
        segments = symbol_segments(grid, c_values, a_values)
        color = contour_color(symbol_index, n_symbols)
        push!(symbol_counts, length(segments))
        push!(all_contours, (segments, color, @sprintf("symbol %02d", symbol_index), 2.2))
        svg_plot(
            joinpath(output_dir, @sprintf("%s_symbol%02d_contours.svg", stem, symbol_index));
            title=@sprintf("Rössler y-min tangent symbol %02d zero contour", symbol_index),
            rows=rows,
            c_values=c_values,
            a_values=a_values,
            contours=[(segments, color, @sprintf("symbol %02d", symbol_index), 2.5)],
            symbol_index=symbol_index,
        )
    end
    svg_plot(
        joinpath(output_dir, "$(stem)_all_symbol_contours.svg");
        title="Rössler y-min tangent kneading symbol contours",
        rows=rows,
        c_values=c_values,
        a_values=a_values,
        contours=all_contours,
    )

    prefix_counts = Int[]
    for prefix_length in 1:n_symbols
        grid = prefix_grid(rows, prefix_length, c_values, a_values)
        segments = category_boundary_segments(grid, c_values, a_values)
        color = contour_color(prefix_length, n_symbols)
        push!(prefix_counts, length(segments))
        svg_plot(
            joinpath(output_dir, @sprintf("%s_prefix%02d_contours.svg", stem, prefix_length));
            title=@sprintf("Rössler y-min tangent prefix %02d boundaries", prefix_length),
            rows=rows,
            c_values=c_values,
            a_values=a_values,
            contours=[(segments, color, @sprintf("prefix %02d", prefix_length), 1.8)],
        )
    end

    word_segments = category_boundary_segments(ok_code_grid(rows, c_values, a_values), c_values, a_values)
    svg_plot(
        joinpath(output_dir, "$(stem)_word_boundary_contours.svg");
        title="Rössler y-min tangent 8-symbol word boundaries",
        rows=rows,
        c_values=c_values,
        a_values=a_values,
        contours=[(word_segments, "#111111", "8-symbol word boundary", 1.6)],
    )

    write_word_legend(joinpath(output_dir, "$(stem)_word_legend.tsv"), rows)
    contour_generation_seconds = time() - contour_started
    write_summary(
        joinpath(output_dir, "$(stem)_contour_summary.tsv"),
        rows,
        symbol_counts,
        prefix_counts;
        results_path=results_path,
        output_dir=output_dir,
        scan_seconds=scan_seconds,
        write_tsv_seconds=write_tsv_seconds,
        contour_generation_seconds=contour_generation_seconds,
    )
    return output_dir
end

function main()
    results_path = get(ENV, "MM_FLOW_FOLDING_CONTOUR_INPUT", DEFAULT_SCAN_RESULTS)
    output_dir = get(ENV, "MM_FLOW_FOLDING_CONTOUR_DIR", DEFAULT_CONTOUR_DIR)
    written = write_all_contours(results_path; output_dir=output_dir)
    @printf("wrote contours in %s\n", written)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
