using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const OLD_TRANSIENT_TIME_058 = parse(Float64, get(ENV, "ATTEMPT058_OLD_TRANSIENT_TIME", "1.0e4"))
const NEW_TRANSIENT_TIME_058 = parse(Float64, get(ENV, "ATTEMPT058_NEW_TRANSIENT_TIME", "1.0e6"))
const SECONDS_PER_TIME_UNIT_058 = 1.0e-3
const ANALYSIS_START_TIME_058 = NEW_TRANSIENT_TIME_058 - OLD_TRANSIENT_TIME_058
const ANALYSIS_START_SECONDS_058 = ANALYSIS_START_TIME_058 * SECONDS_PER_TIME_UNIT_058
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_lz76_ttr1e6_postprocess_convergence_bits_per_second.png")
const SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_lz76_ttr1e6_postprocess_summary.tsv")
const PLOT_WIDTH_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_WIDTH", "1260"))
const PLOT_HEIGHT_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_HEIGHT", "550"))

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
postprocess_convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, "traj$(lpad(string(idx), 2, '0'))_ttr1e6_postprocess_bits.tsv")

function read_convergence_058(idx::Int)
    header, rows = read_tsv_058(convergence_path_058(idx))
    time_idx = column_index_058(header, "time")
    time_seconds_idx = column_index_058(header, "time_seconds")
    lle_bits_idx = column_index_058(header, "lambda1_bits_per_second")
    return (;
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

function cumulative_at_058(times_seconds::Vector{Float64}, rates_bits::Vector{Float64}, t0_seconds::Float64)
    cumulative = times_seconds .* rates_bits
    exact = findfirst(t -> isapprox(t, t0_seconds; atol=1e-9, rtol=0.0), times_seconds)
    exact !== nothing && return cumulative[exact]
    hi = searchsortedfirst(times_seconds, t0_seconds)
    (hi <= 1 || hi > length(times_seconds)) && error("Cannot interpolate cumulative exponent at $(t0_seconds) seconds.")
    lo = hi - 1
    w = (t0_seconds - times_seconds[lo]) / (times_seconds[hi] - times_seconds[lo])
    return cumulative[lo] + w * (cumulative[hi] - cumulative[lo])
end

function lz76_bits_per_second_058(symbols::Vector{Int}, times::Vector{Float64}, start_idx::Int, end_idx::Int)
    n = end_idx - start_idx + 1
    n <= 1 && return NaN, 0, 0, NaN
    mean_interval_seconds = ((times[end_idx] - times[start_idx]) * SECONDS_PER_TIME_UNIT_058) / (n - 1)
    mean_interval_seconds > 0 || return NaN, 0, 0, NaN
    window = @view symbols[start_idx:end_idx]
    phrase_count = lz76_phrase_count_058(window)
    alphabet_size = length(unique(window))
    bits_per_symbol = phrase_count * log2(n) / n
    return bits_per_symbol / mean_interval_seconds, phrase_count, alphabet_size, mean_interval_seconds
end

function compute_postprocessed_058(idx::Int)
    convergence = read_convergence_058(idx)
    events = read_events_058(idx)
    start_event_idx = searchsortedfirst(events.time, ANALYSIS_START_TIME_058)
    start_cumulative_bits = cumulative_at_058(convergence.time_seconds, convergence.lle_bits, ANALYSIS_START_SECONDS_058)

    keep = findall(>(ANALYSIS_START_SECONDS_058), convergence.time_seconds)
    time_seconds = Float64[]
    lle_bits = Float64[]
    lz_bits = Float64[]
    n_symbols = Int[]
    phrase_counts = Int[]
    alphabet_sizes = Int[]
    mean_intervals = Float64[]

    for row_idx in keep
        t_seconds = convergence.time_seconds[row_idx]
        elapsed_seconds = t_seconds - ANALYSIS_START_SECONDS_058
        push!(time_seconds, elapsed_seconds)
        push!(lle_bits, ((convergence.time_seconds[row_idx] * convergence.lle_bits[row_idx]) - start_cumulative_bits) / elapsed_seconds)

        end_event_idx = searchsortedlast(events.time, convergence.time[row_idx])
        lz, phrases, alphabet_size, mean_interval = lz76_bits_per_second_058(events.symbols, events.time, start_event_idx, end_event_idx)
        push!(lz_bits, lz)
        push!(n_symbols, max(0, end_event_idx - start_event_idx + 1))
        push!(phrase_counts, phrases)
        push!(alphabet_sizes, alphabet_size)
        push!(mean_intervals, mean_interval)
    end

    path = postprocess_convergence_path_058(idx)
    open(path, "w") do io
        println(io, "time_since_ttr1e6_seconds\tlambda1_bits_per_second\tlz76_bits_per_second\tn_symbols\talphabet_size\tlz76_phrase_count\tmean_event_interval_seconds")
        for i in eachindex(time_seconds)
            println(io, join((
                @sprintf("%.12g", time_seconds[i]),
                @sprintf("%.12g", lle_bits[i]),
                isfinite(lz_bits[i]) ? @sprintf("%.12g", lz_bits[i]) : "NaN",
                n_symbols[i],
                alphabet_sizes[i],
                phrase_counts[i],
                isfinite(mean_intervals[i]) ? @sprintf("%.12g", mean_intervals[i]) : "NaN",
            ), '\t'))
        end
    end
    return (; idx, time_seconds, lle_bits, lz_bits, n_symbols, alphabet_sizes, phrase_counts, path)
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
        println(io, "trajectory\tfinal_lambda1_bits_per_second\tfinal_lz76_bits_per_second\tfinal_n_symbols\tfinal_alphabet_size\tfinal_lz76_phrase_count\tconvergence_path")
        for result in results
            println(io, join((
                result.idx,
                @sprintf("%.12g", result.lle_bits[end]),
                isfinite(result.lz_bits[end]) ? @sprintf("%.12g", result.lz_bits[end]) : "NaN",
                result.n_symbols[end],
                result.alphabet_sizes[end],
                result.phrase_counts[end],
                result.path,
            ), '\t'))
        end
    end
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory convergence files found in $(TRAJECTORY_DIR_058).")
    results = [compute_postprocessed_058(idx) for idx in indices]
    write_summary_058(results)
    plot_ensemble_bits_058(results)
    mean_lle_bits = sum(result.lle_bits[end] for result in results) / length(results)
    mean_lz_bits = sum(result.lz_bits[end] for result in results) / length(results)
    @info "Wrote full SSCS LZ76 ttr1e6 postprocessed ensemble in bits/s" plot=PLOT_PATH_058 summary=SUMMARY_PATH_058 n=length(results) analysis_start_seconds=ANALYSIS_START_SECONDS_058 mean_lle_bits mean_lz_bits
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
