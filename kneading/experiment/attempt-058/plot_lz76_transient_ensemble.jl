using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7")
const EVENTS_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_events.tsv")
const CONVERGENCE_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_convergence.tsv")
const ENSEMBLE_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_lz76_transient_discard_ensemble.tsv")
const SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_lz76_transient_discard_summary.tsv")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_lz76_transient_discard_ensemble.png")
const MAX_DISCARD_SECONDS_058 = parse(Float64, get(ENV, "ATTEMPT058_LZ76_MAX_DISCARD_SECONDS", "2500.0"))
const N_DISCARDS_058 = parse(Int, get(ENV, "ATTEMPT058_LZ76_N_DISCARDS", "26"))
const MIN_SYMBOLS_058 = parse(Int, get(ENV, "ATTEMPT058_LZ76_MIN_SYMBOLS", "2"))
const PLOT_WIDTH_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_WIDTH", "1800"))
const PLOT_HEIGHT_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_HEIGHT", "1100"))

function read_tsv_058(path::String)
    lines = readlines(path)
    header = split(first(lines), '\t')
    rows = [split(line, '\t') for line in lines[2:end] if !isempty(strip(line))]
    return header, rows
end

function column_index_058(header, name::String)
    idx = findfirst(==(name), header)
    idx === nothing && error("Column $(name) not found in $(join(header, ", ")).")
    return idx
end

function read_events_058()
    header, rows = read_tsv_058(EVENTS_PATH_058)
    symbol_idx = column_index_058(header, "symbol")
    time_seconds_idx = column_index_058(header, "time_seconds")
    symbols = Int[parse(Int, row[symbol_idx]) for row in rows]
    times_seconds = Float64[parse(Float64, row[time_seconds_idx]) for row in rows]
    return symbols, times_seconds
end

function read_final_lle_058()
    header, rows = read_tsv_058(CONVERGENCE_PATH_058)
    lyap_idx = column_index_058(header, "lambda1_bits_per_second")
    return parse(Float64, rows[end][lyap_idx])
end

function lz76_phrase_count_058(xs::AbstractVector{<:Integer})::Int
    n = length(xs)
    n == 0 && return 0
    phrases = Set{Tuple{Vararg{Int}}}()
    count = 0
    i = 1
    while i <= n
        j = i
        phrase = Tuple(Int[xs[k] for k in i:j])
        while phrase in phrases && j < n
            j += 1
            phrase = Tuple(Int[xs[k] for k in i:j])
        end
        push!(phrases, phrase)
        count += 1
        i = j + 1
    end
    return count
end

function discard_values_058(final_time_seconds::Float64)
    max_discard = min(MAX_DISCARD_SECONDS_058, final_time_seconds * 0.8)
    return collect(range(0.0, max_discard; length=N_DISCARDS_058))
end

function lz76_curve_for_discard_058(
    symbols::Vector{Int},
    times_seconds::Vector{Float64},
    discard_seconds::Float64,
)
    start_idx = searchsortedfirst(times_seconds, discard_seconds)
    curve_t = Float64[]
    curve_lz = Float64[]
    curve_n = Int[]
    curve_phrases = Int[]
    for end_idx in start_idx:length(symbols)
        n_used = end_idx - start_idx + 1
        n_used < MIN_SYMBOLS_058 && continue
        elapsed_seconds = times_seconds[end_idx] - discard_seconds
        elapsed_seconds > 0 || continue
        phrase_count = lz76_phrase_count_058(@view symbols[start_idx:end_idx])
        bits_per_second = phrase_count * log2(n_used) / elapsed_seconds
        push!(curve_t, times_seconds[end_idx])
        push!(curve_lz, bits_per_second)
        push!(curve_n, n_used)
        push!(curve_phrases, phrase_count)
    end
    return (; t=curve_t, lz=curve_lz, n=curve_n, phrases=curve_phrases)
end

function build_ensemble_058(symbols::Vector{Int}, times_seconds::Vector{Float64})
    discards = discard_values_058(times_seconds[end])
    rows = NamedTuple[]
    summary = NamedTuple[]
    for discard in discards
        curve = lz76_curve_for_discard_058(symbols, times_seconds, discard)
        isempty(curve.lz) && continue
        for i in eachindex(curve.lz)
            push!(rows, (;
                discard_seconds=discard,
                time_seconds=curve.t[i],
                lz76_bits_per_second=curve.lz[i],
                n_symbols=curve.n[i],
                phrase_count=curve.phrases[i],
            ))
        end
        push!(summary, (;
            discard_seconds=discard,
            final_time_seconds=curve.t[end],
            final_lz76_bits_per_second=curve.lz[end],
            final_n_symbols=curve.n[end],
            final_phrase_count=curve.phrases[end],
        ))
    end
    return rows, summary
end

function write_ensemble_058(rows)
    open(ENSEMBLE_PATH_058, "w") do io
        println(io, "discard_seconds\ttime_seconds\tlz76_bits_per_second\tn_symbols\tphrase_count")
        for row in rows
            println(io, join((
                @sprintf("%.12g", row.discard_seconds),
                @sprintf("%.12g", row.time_seconds),
                @sprintf("%.12g", row.lz76_bits_per_second),
                row.n_symbols,
                row.phrase_count,
            ), '\t'))
        end
    end
end

function write_summary_058(summary, final_lle::Float64, mean_final_lz::Float64)
    open(SUMMARY_PATH_058, "w") do io
        println(io, "discard_seconds\tfinal_time_seconds\tfinal_lz76_bits_per_second\tfinal_n_symbols\tfinal_phrase_count\tmean_final_lz76_bits_per_second\tfinal_lle_bits_per_second")
        for row in summary
            println(io, join((
                @sprintf("%.12g", row.discard_seconds),
                @sprintf("%.12g", row.final_time_seconds),
                @sprintf("%.12g", row.final_lz76_bits_per_second),
                row.final_n_symbols,
                row.final_phrase_count,
                @sprintf("%.12g", mean_final_lz),
                @sprintf("%.12g", final_lle),
            ), '\t'))
        end
    end
end

function plot_ensemble_058(summary, final_lle::Float64, mean_final_lz::Float64, symbols, times_seconds)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="time after original transient (s)",
        ylabel="entropy rate (bits/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    )

    curve_line = nothing
    for row in summary
        curve = lz76_curve_for_discard_058(symbols, times_seconds, row.discard_seconds)
        local line = lines!(ax, curve.t, curve.lz; color=(:red, 0.18), linewidth=1.8)
        curve_line === nothing && (curve_line = line)
    end
    mean_line = hlines!(ax, [mean_final_lz]; color=:red, linewidth=3.2, linestyle=:dash)
    lle_line = hlines!(ax, [final_lle]; color=:black, linewidth=3.2, linestyle=:dash)

    xlims!(ax, 0, times_seconds[end] * 1.01)
    final_values = [row.final_lz76_bits_per_second for row in summary]
    y_values = vcat(final_values, [final_lle, mean_final_lz])
    y_min = max(0.0, minimum(y_values) - 0.08)
    y_max = maximum(y_values) + 0.14
    ylims!(ax, y_min, y_max)

    Legend(
        fig[1, 1],
        [curve_line, mean_line, lle_line],
        ["Abramov LZ76 discard ensemble", "mean final Abramov LZ76", "final LLE"],
        labelsize=21,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )

    save(PLOT_PATH_058, fig, px_per_unit=2)
end

function main()
    symbols, times_seconds = read_events_058()
    final_lle = read_final_lle_058()
    rows, summary = build_ensemble_058(symbols, times_seconds)
    final_lz_values = [row.final_lz76_bits_per_second for row in summary]
    mean_final_lz = sum(final_lz_values) / length(final_lz_values)
    write_ensemble_058(rows)
    write_summary_058(summary, final_lle, mean_final_lz)
    plot_ensemble_058(summary, final_lle, mean_final_lz, symbols, times_seconds)
    @info "Wrote LZ76 transient-discard ensemble" plot=PLOT_PATH_058 ensemble=ENSEMBLE_PATH_058 summary=SUMMARY_PATH_058 n_discards=length(summary) mean_final_lz final_lle
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
