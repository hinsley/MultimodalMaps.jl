using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_lz76_measurement_window_convergence_bits_per_second.png")
const SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_lz76_measurement_window_summary.tsv")
const PLOT_WIDTH_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_WIDTH", "1260"))
const PLOT_HEIGHT_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_HEIGHT", "550"))
const SECONDS_PER_TIME_UNIT_058 = 1.0e-3

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
    paths = sort(filter(path -> occursin(r"traj\d+_convergence\.tsv$", basename(path)), readdir(TRAJECTORY_DIR_058; join=true)))
    return [parse(Int, match(r"traj(\d+)_convergence\.tsv$", basename(path)).captures[1]) for path in paths]
end

convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, "traj$(lpad(string(idx), 2, '0'))_convergence.tsv")
events_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, "traj$(lpad(string(idx), 2, '0'))_events.tsv")
corrected_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, "traj$(lpad(string(idx), 2, '0'))_lz76_measurement_window.tsv")

function read_convergence_058(idx::Int)
    header, rows = read_tsv_058(convergence_path_058(idx))
    time_idx = column_index_058(header, "time")
    time_seconds_idx = column_index_058(header, "time_seconds")
    lle_bits_idx = column_index_058(header, "lambda1_bits_per_second")
    return (;
        idx,
        time=Float64[parse(Float64, row[time_idx]) for row in rows],
        time_seconds=Float64[parse(Float64, row[time_seconds_idx]) for row in rows],
        lle_bits=Float64[parse(Float64, row[lle_bits_idx]) for row in rows],
    )
end

function read_events_058(idx::Int)
    header, rows = read_tsv_058(events_path_058(idx))
    time_idx = column_index_058(header, "time")
    symbol_idx = column_index_058(header, "symbol")
    return (;
        time=Float64[parse(Float64, row[time_idx]) for row in rows],
        symbols=Int[parse(Int, row[symbol_idx]) for row in rows],
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

function lz76_measurement_window_bits_per_second_058(symbols::Vector{Int}, event_times::Vector{Float64}, n::Int)
    n <= 1 && return NaN, NaN, NaN, 0, 0
    total_elapsed_seconds = event_times[n] * SECONDS_PER_TIME_UNIT_058
    total_elapsed_seconds > 0 || return NaN, NaN, NaN, 0, 0
    mean_event_interval_seconds = total_elapsed_seconds / n
    alphabet_size = length(unique(@view symbols[1:n]))
    alphabet_size > 1 || return NaN, total_elapsed_seconds, mean_event_interval_seconds, alphabet_size, 0
    phrase_count = lz76_phrase_count_058(@view symbols[1:n])
    bits_per_symbol = phrase_count * log2(n) / n
    return bits_per_symbol / mean_event_interval_seconds, total_elapsed_seconds, mean_event_interval_seconds, alphabet_size, phrase_count
end

function compute_corrected_convergence_058(idx::Int)
    convergence = read_convergence_058(idx)
    events = read_events_058(idx)
    lz_bits = Float64[]
    elapsed_seconds = Float64[]
    mean_event_intervals = Float64[]
    alphabet_sizes = Int[]
    phrase_counts = Int[]
    n_symbols = Int[]
    for t in convergence.time
        n = searchsortedlast(events.time, t)
        lz, elapsed, mean_event_interval, alphabet_size, phrases = lz76_measurement_window_bits_per_second_058(events.symbols, events.time, n)
        push!(lz_bits, lz)
        push!(elapsed_seconds, elapsed)
        push!(mean_event_intervals, mean_event_interval)
        push!(alphabet_sizes, alphabet_size)
        push!(phrase_counts, phrases)
        push!(n_symbols, n)
    end
    path = corrected_path_058(idx)
    open(path, "w") do io
        println(io, "time\ttime_seconds\tlambda1_bits_per_second\tlz76_measurement_window_bits_per_second\tn_symbols\tlz76_phrase_count\talphabet_size\tlz76_elapsed_seconds\tlz76_mean_event_interval_seconds")
        for i in eachindex(convergence.time)
            println(io, join((
                @sprintf("%.12g", convergence.time[i]),
                @sprintf("%.12g", convergence.time_seconds[i]),
                @sprintf("%.12g", convergence.lle_bits[i]),
                isfinite(lz_bits[i]) ? @sprintf("%.12g", lz_bits[i]) : "NaN",
                n_symbols[i],
                phrase_counts[i],
                alphabet_sizes[i],
                isfinite(elapsed_seconds[i]) ? @sprintf("%.12g", elapsed_seconds[i]) : "NaN",
                isfinite(mean_event_intervals[i]) ? @sprintf("%.12g", mean_event_intervals[i]) : "NaN",
            ), '\t'))
        end
    end
    return (; idx, time_seconds=convergence.time_seconds, lle_bits=convergence.lle_bits, lz_bits, n_symbols, phrase_counts, alphabet_sizes, mean_event_intervals, elapsed_seconds, path)
end

function plot_ensemble_bits_058(results)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel="Entropy rate estimate (bits/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
        xticks=0:2500:10000,
        limits=(nothing, (0.1, 0.55)),
    )
    lle_handle = nothing
    lz_handle = nothing
    for result in results
        local a = lines!(ax, result.time_seconds, result.lle_bits; color=(:black, 0.35), linewidth=2.0)
        finite_lz = isfinite.(result.lz_bits)
        local b = lines!(ax, result.time_seconds[finite_lz], result.lz_bits[finite_lz]; color=(:firebrick3, 0.45), linewidth=2.0)
        lle_handle === nothing && (lle_handle = a)
        lz_handle === nothing && (lz_handle = b)
    end
    Legend(
        fig[1, 1],
        [lle_handle, lz_handle],
        ["Maximal Lyapunov exponent", "SSCS Abramov LZ76"],
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
        println(io, "trajectory\tfinal_lambda1_bits_per_second\tfinal_lz76_measurement_window_bits_per_second\tfinal_n_symbols\tfinal_lz76_phrase_count\tfinal_alphabet_size\tconvergence_path")
        for result in results
            println(io, join((
                result.idx,
                @sprintf("%.12g", result.lle_bits[end]),
                isfinite(result.lz_bits[end]) ? @sprintf("%.12g", result.lz_bits[end]) : "NaN",
                result.n_symbols[end],
                result.phrase_counts[end],
                result.alphabet_sizes[end],
                result.path,
            ), '\t'))
        end
    end
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory convergence files found in $(TRAJECTORY_DIR_058).")
    results = [compute_corrected_convergence_058(idx) for idx in indices]
    write_summary_058(results)
    plot_ensemble_bits_058(results)
    mean_lle_bits = sum(result.lle_bits[end] for result in results) / length(results)
    mean_lz_bits = sum(result.lz_bits[end] for result in results) / length(results)
    @info "Wrote measurement-window-normalized full SSCS LZ76 ensemble in bits/s" plot=PLOT_PATH_058 summary=SUMMARY_PATH_058 n=length(results) mean_lle_bits mean_lz_bits ratio=mean_lz_bits / mean_lle_bits
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
