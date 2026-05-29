using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7")
const EVENTS_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_events.tsv")
const OLD_CONVERGENCE_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_convergence.tsv")
const CONVERGENCE_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_convergence_interval_lz76.tsv")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_convergence_interval_lz76.png")

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

function read_events_058()
    header, rows = read_tsv_058(EVENTS_PATH_058)
    symbol_idx = column_index_058(header, "symbol")
    time_idx = column_index_058(header, "time")
    symbols = Int[parse(Int, row[symbol_idx]) for row in rows]
    times = Float64[parse(Float64, row[time_idx]) for row in rows]
    return symbols, times
end

function read_old_convergence_058()
    header, rows = read_tsv_058(OLD_CONVERGENCE_PATH_058)
    time_idx = column_index_058(header, "time")
    time_seconds_idx = column_index_058(header, "time_seconds")
    lambda_nat_idx = column_index_058(header, "lambda1_nat_per_timeunit")
    lambda_bits_idx = column_index_058(header, "lambda1_bits_per_second")
    return (;
        time=[parse(Float64, row[time_idx]) for row in rows],
        time_seconds=[parse(Float64, row[time_seconds_idx]) for row in rows],
        lambda_nat=[parse(Float64, row[lambda_nat_idx]) for row in rows],
        lambda_bits=[parse(Float64, row[lambda_bits_idx]) for row in rows],
    )
end

function interval_lz76_bits_per_second_058(symbols::Vector{Int}, event_times::Vector{Float64}, n::Int)
    n <= 1 && return NaN, 0, NaN
    interval_seconds = (event_times[n] - event_times[1]) * 1.0e-3
    mean_interval_seconds = interval_seconds / (n - 1)
    mean_interval_seconds > 0 || return NaN, 0, NaN
    phrase_count = lz76_phrase_count_058(@view symbols[1:n])
    entropy_bits_per_symbol = phrase_count * log2(n) / n
    return entropy_bits_per_symbol / mean_interval_seconds, phrase_count, mean_interval_seconds
end

function recompute_058()
    symbols, event_times = read_events_058()
    old = read_old_convergence_058()
    lz = Float64[]
    phrase_counts = Int[]
    mean_intervals = Float64[]
    n_symbols = Int[]
    last_symbol_times = Float64[]

    for t in old.time
        n = searchsortedlast(event_times, t)
        bits_per_second, phrases, mean_interval = interval_lz76_bits_per_second_058(symbols, event_times, n)
        push!(lz, bits_per_second)
        push!(phrase_counts, phrases)
        push!(mean_intervals, mean_interval)
        push!(n_symbols, n)
        push!(last_symbol_times, n > 0 ? event_times[n] : NaN)
    end

    open(CONVERGENCE_PATH_058, "w") do io
        println(io, "time\ttime_seconds\tlambda1_nat_per_timeunit\tlambda1_bits_per_second\tlz76_bits_per_second\tn_symbols\tfirst_symbol_time\tlast_symbol_time\tmean_event_interval_seconds\tlz76_phrase_count")
        for i in eachindex(old.time)
            println(io, join((
                @sprintf("%.12g", old.time[i]),
                @sprintf("%.12g", old.time_seconds[i]),
                @sprintf("%.12g", old.lambda_nat[i]),
                @sprintf("%.12g", old.lambda_bits[i]),
                isfinite(lz[i]) ? @sprintf("%.12g", lz[i]) : "NaN",
                n_symbols[i],
                n_symbols[i] > 0 ? @sprintf("%.12g", event_times[1]) : "NaN",
                isfinite(last_symbol_times[i]) ? @sprintf("%.12g", last_symbol_times[i]) : "NaN",
                isfinite(mean_intervals[i]) ? @sprintf("%.12g", mean_intervals[i]) : "NaN",
                phrase_counts[i],
            ), '\t'))
        end
    end

    fig = Figure(size=(1600, 1000), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="time (s)",
        ylabel="entropy rate (bits/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    )
    lines!(ax, old.time_seconds, old.lambda_bits; color=:black, linewidth=3.2, label="leading Lyapunov exponent")
    finite_lz = isfinite.(lz)
    lines!(ax, old.time_seconds[finite_lz], lz[finite_lz]; color=:red, linewidth=3.2, label="Abramov LZ76")
    axislegend(ax; position=:rt, labelsize=22, framevisible=false)
    save(PLOT_PATH_058, fig, px_per_unit=2)
    return old.lambda_bits[end], lz[end], phrase_counts[end], mean_intervals[end]
end

if abspath(PROGRAM_FILE) == @__FILE__
    final_lle, final_lz, phrases, mean_interval = recompute_058()
    @info "Recomputed interval-normalized Abramov LZ76" plot=PLOT_PATH_058 convergence=CONVERGENCE_PATH_058 final_lle final_lz phrases mean_interval
end
