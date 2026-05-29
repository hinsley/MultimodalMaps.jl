using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_binary_sign_lz76_summary.tsv")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_binary_sign_lz76_convergence.png")
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
    idx === nothing && error("Column $(name) not found in $(path).")
    return idx
end

function trajectory_indices_058()
    paths = sort(filter(path -> occursin(r"traj\d+_events\.tsv$", basename(path)), readdir(TRAJECTORY_DIR_058; join=true)))
    return [parse(Int, match(r"traj(\d+)_events\.tsv$", basename(path)).captures[1]) for path in paths]
end

events_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_events.tsv", idx))
convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_convergence.tsv", idx))
binary_convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_binary_sign_lz76.tsv", idx))

function read_events_058(idx::Int)
    header, rows = read_tsv_058(events_path_058(idx))
    time_idx = column_index_058(header, "time")
    symbol_idx = column_index_058(header, "symbol")
    times = Float64[parse(Float64, row[time_idx]) for row in rows]
    symbols = Int[parse(Int, row[symbol_idx]) for row in rows]
    binary_symbols = Int[symbol < 0 ? -1 : 1 for symbol in symbols]
    return times, binary_symbols
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

function binary_lz76_nats_per_second_058(symbols::Vector{Int}, times::Vector{Float64}, n::Int)
    n <= 1 && return NaN, 0, NaN
    mean_interval_seconds = ((times[n] - times[1]) * 1.0e-3) / (n - 1)
    mean_interval_seconds > 0 || return NaN, 0, NaN
    phrase_count = lz76_phrase_count_058(@view symbols[1:n])
    entropy_nats_per_symbol = phrase_count * log(n) / n
    return entropy_nats_per_symbol / mean_interval_seconds, phrase_count, mean_interval_seconds
end

function compute_binary_convergence_058(idx::Int)
    event_times, binary_symbols = read_events_058(idx)
    lle = read_lle_convergence_058(idx)
    binary_lz = Float64[]
    phrase_counts = Int[]
    mean_intervals = Float64[]
    n_symbols = Int[]
    for t in lle.time
        n = searchsortedlast(event_times, t)
        lz, phrases, mean_interval = binary_lz76_nats_per_second_058(binary_symbols, event_times, n)
        push!(binary_lz, lz)
        push!(phrase_counts, phrases)
        push!(mean_intervals, mean_interval)
        push!(n_symbols, n)
    end
    path = binary_convergence_path_058(idx)
    open(path, "w") do io
        println(io, "time\ttime_seconds\tlambda1_nats_per_second\tbinary_sign_lz76_nats_per_second\tn_symbols\tbinary_lz76_phrase_count\tmean_event_interval_seconds")
        for i in eachindex(lle.time)
            println(io, join((
                @sprintf("%.12g", lle.time[i]),
                @sprintf("%.12g", lle.time_seconds[i]),
                @sprintf("%.12g", lle.lle[i]),
                isfinite(binary_lz[i]) ? @sprintf("%.12g", binary_lz[i]) : "NaN",
                n_symbols[i],
                phrase_counts[i],
                isfinite(mean_intervals[i]) ? @sprintf("%.12g", mean_intervals[i]) : "NaN",
            ), '\t'))
        end
    end
    return (; idx, time_seconds=lle.time_seconds, lle=lle.lle, binary_lz, n_symbols, phrase_counts, path)
end

function plot_binary_ensemble_058(results)
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
    binary_handle = nothing
    for result in results
        local a = lines!(ax, result.time_seconds, result.lle; color=(:black, 0.35), linewidth=2.0)
        finite_lz = isfinite.(result.binary_lz)
        local b = lines!(ax, result.time_seconds[finite_lz], result.binary_lz[finite_lz]; color=(:dodgerblue4, 0.45), linewidth=2.0)
        lle_handle === nothing && (lle_handle = a)
        binary_handle === nothing && (binary_handle = b)
    end
    Legend(
        fig[1, 1],
        [lle_handle, binary_handle],
        ["LLE", "binary sign Abramov LZ76"],
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
        println(io, "trajectory\tfinal_lambda1_nats_per_second\tfinal_binary_sign_lz76_nats_per_second\tfinal_n_symbols\tfinal_binary_lz76_phrase_count\tbinary_convergence_path")
        for result in results
            println(io, join((
                result.idx,
                @sprintf("%.12g", result.lle[end]),
                isfinite(result.binary_lz[end]) ? @sprintf("%.12g", result.binary_lz[end]) : "NaN",
                result.n_symbols[end],
                result.phrase_counts[end],
                result.path,
            ), '\t'))
        end
    end
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory event files found in $(TRAJECTORY_DIR_058).")
    results = [compute_binary_convergence_058(idx) for idx in indices]
    write_summary_058(results)
    plot_binary_ensemble_058(results)
    mean_lle = sum(result.lle[end] for result in results) / length(results)
    mean_binary_lz = sum(result.binary_lz[end] for result in results) / length(results)
    @info "Wrote binary sign LZ76 ensemble" plot=PLOT_PATH_058 summary=SUMMARY_PATH_058 n=length(results) mean_lle mean_binary_lz
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
