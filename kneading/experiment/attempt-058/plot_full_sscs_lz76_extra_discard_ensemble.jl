using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const DISCARD_FRACTION_058 = parse(Float64, get(ENV, "ATTEMPT058_LZ76_EXTRA_DISCARD_FRACTION", "0.10"))
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const SUFFIX_058 = @sprintf("full_sscs_lz76_discard%02d", round(Int, 100 * DISCARD_FRACTION_058))
const SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_$(SUFFIX_058)_summary.tsv")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_$(SUFFIX_058)_convergence.png")
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
    idx === nothing && error("Column $(name) not found.")
    return idx
end

function trajectory_indices_058()
    paths = sort(filter(path -> occursin(r"traj\d+_events\.tsv$", basename(path)), readdir(TRAJECTORY_DIR_058; join=true)))
    return [parse(Int, match(r"traj(\d+)_events\.tsv$", basename(path)).captures[1]) for path in paths]
end

events_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_events.tsv", idx))
convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_convergence.tsv", idx))
discard_convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_%s.tsv", idx, SUFFIX_058))

function read_events_058(idx::Int)
    header, rows = read_tsv_058(events_path_058(idx))
    time_idx = column_index_058(header, "time")
    symbol_idx = column_index_058(header, "symbol")
    times = Float64[parse(Float64, row[time_idx]) for row in rows]
    symbols = Int[parse(Int, row[symbol_idx]) for row in rows]
    return times, symbols
end

function read_lle_convergence_058(idx::Int)
    header, rows = read_tsv_058(convergence_path_058(idx))
    time_idx = column_index_058(header, "time")
    time_seconds_idx = column_index_058(header, "time_seconds")
    lle_idx = column_index_058(header, "lambda1_nats_per_second")
    return (;
        time=Float64[parse(Float64, row[time_idx]) for row in rows],
        time_seconds=Float64[parse(Float64, row[time_seconds_idx]) for row in rows],
        lle=Float64[parse(Float64, row[lle_idx]) for row in rows],
    )
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

function lz76_nats_per_second_after_discard_058(symbols::Vector{Int}, times::Vector{Float64}, start_idx::Int, end_idx::Int)
    n = end_idx - start_idx + 1
    n <= 1 && return NaN, NaN, 0, 0, NaN
    mean_interval_seconds = ((times[end_idx] - times[start_idx]) * 1.0e-3) / (n - 1)
    mean_interval_seconds > 0 || return NaN, NaN, 0, 0, NaN
    window = @view symbols[start_idx:end_idx]
    alphabet_size = length(unique(window))
    alphabet_size > 1 || return NaN, NaN, alphabet_size, 0, mean_interval_seconds
    phrase_count = lz76_phrase_count_058(window)
    nats_per_symbol = phrase_count * log(n) / n
    bits_per_symbol = nats_per_symbol / log(2)
    return nats_per_symbol / mean_interval_seconds, bits_per_symbol / mean_interval_seconds, alphabet_size, phrase_count, mean_interval_seconds
end

function compute_discard_convergence_058(idx::Int)
    event_times, symbols = read_events_058(idx)
    lle = read_lle_convergence_058(idx)
    discard_count = floor(Int, DISCARD_FRACTION_058 * length(symbols))
    start_idx = discard_count + 1
    lz_nats = Float64[]
    lz_bits = Float64[]
    alphabet_sizes = Int[]
    phrase_counts = Int[]
    mean_intervals = Float64[]
    n_symbols_used = Int[]
    n_symbols_total = Int[]
    for t in lle.time
        end_idx = searchsortedlast(event_times, t)
        lz_nat, lz_bit, alphabet_size, phrases, mean_interval = lz76_nats_per_second_after_discard_058(symbols, event_times, start_idx, end_idx)
        push!(lz_nats, lz_nat)
        push!(lz_bits, lz_bit)
        push!(alphabet_sizes, alphabet_size)
        push!(phrase_counts, phrases)
        push!(mean_intervals, mean_interval)
        push!(n_symbols_used, max(0, end_idx - discard_count))
        push!(n_symbols_total, end_idx)
    end
    path = discard_convergence_path_058(idx)
    open(path, "w") do io
        println(io, "time\ttime_seconds\tlambda1_nats_per_second\tfull_sscs_lz76_nats_per_second\tfull_sscs_lz76_bits_per_second\tn_symbols_total\tn_symbols_discarded\tn_symbols_used\talphabet_size\tlz76_phrase_count\tmean_event_interval_seconds")
        for i in eachindex(lle.time)
            println(io, join((
                @sprintf("%.12g", lle.time[i]),
                @sprintf("%.12g", lle.time_seconds[i]),
                @sprintf("%.12g", lle.lle[i]),
                isfinite(lz_nats[i]) ? @sprintf("%.12g", lz_nats[i]) : "NaN",
                isfinite(lz_bits[i]) ? @sprintf("%.12g", lz_bits[i]) : "NaN",
                n_symbols_total[i],
                discard_count,
                n_symbols_used[i],
                alphabet_sizes[i],
                phrase_counts[i],
                isfinite(mean_intervals[i]) ? @sprintf("%.12g", mean_intervals[i]) : "NaN",
            ), '\t'))
        end
    end
    return (; idx, time_seconds=lle.time_seconds, lle=lle.lle, lz_nats, lz_bits, alphabet_sizes, phrase_counts, n_symbols_total, n_symbols_used, discard_count, path)
end

function plot_discard_ensemble_058(results)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="time (s)",
        ylabel="entropy rate (nats/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    )
    lle_handle = nothing
    lz_handle = nothing
    for result in results
        local a = lines!(ax, result.time_seconds, result.lle; color=(:black, 0.35), linewidth=2.0)
        finite_lz = isfinite.(result.lz_nats)
        local b = lines!(ax, result.time_seconds[finite_lz], result.lz_nats[finite_lz]; color=(:firebrick3, 0.45), linewidth=2.0)
        lle_handle === nothing && (lle_handle = a)
        lz_handle === nothing && (lz_handle = b)
    end
    Legend(
        fig[1, 1],
        [lle_handle, lz_handle],
        ["LLE", @sprintf("full SSCS Abramov LZ76, first %.0f%% symbols discarded", 100 * DISCARD_FRACTION_058)],
        labelsize=22,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )
    save(PLOT_PATH_058, fig, px_per_unit=2)
end

function write_summary_058(results)
    open(SUMMARY_PATH_058, "w") do io
        println(io, "trajectory\tfinal_lambda1_nats_per_second\tfinal_full_sscs_lz76_nats_per_second\tfinal_full_sscs_lz76_bits_per_second\tfinal_n_symbols_total\tfinal_n_symbols_discarded\tfinal_n_symbols_used\tfinal_alphabet_size\tfinal_lz76_phrase_count\tconvergence_path")
        for result in results
            println(io, join((
                result.idx,
                @sprintf("%.12g", result.lle[end]),
                isfinite(result.lz_nats[end]) ? @sprintf("%.12g", result.lz_nats[end]) : "NaN",
                isfinite(result.lz_bits[end]) ? @sprintf("%.12g", result.lz_bits[end]) : "NaN",
                result.n_symbols_total[end],
                result.discard_count,
                result.n_symbols_used[end],
                result.alphabet_sizes[end],
                result.phrase_counts[end],
                result.path,
            ), '\t'))
        end
    end
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory event files found in $(TRAJECTORY_DIR_058).")
    results = [compute_discard_convergence_058(idx) for idx in indices]
    write_summary_058(results)
    plot_discard_ensemble_058(results)
    mean_lle = sum(result.lle[end] for result in results) / length(results)
    mean_lz = sum(result.lz_nats[end] for result in results) / length(results)
    mean_alphabet = sum(result.alphabet_sizes[end] for result in results) / length(results)
    @info "Wrote full SSCS LZ76 extra-discard ensemble" plot=PLOT_PATH_058 summary=SUMMARY_PATH_058 n=length(results) discard_fraction=DISCARD_FRACTION_058 mean_lle mean_lz mean_alphabet
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
