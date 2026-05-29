using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7")
const EVENTS_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_events.tsv")
const CONVERGENCE_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_convergence.tsv")
const BLOCK_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_conditional_block_entropy.tsv")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_convergence_with_block_entropy.png")
const MAX_BLOCK_SIZE_058 = parse(Int, get(ENV, "ATTEMPT058_MAX_BLOCK_SIZE", "8"))
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

function read_events_058()
    header, rows = read_tsv_058(EVENTS_PATH_058)
    symbol_idx = column_index_058(header, "symbol")
    time_seconds_idx = column_index_058(header, "time_seconds")
    symbols = Int[parse(Int, row[symbol_idx]) for row in rows]
    times_seconds = Float64[parse(Float64, row[time_seconds_idx]) for row in rows]
    return symbols, times_seconds
end

function read_convergence_058()
    header, rows = read_tsv_058(CONVERGENCE_PATH_058)
    time_idx = column_index_058(header, "time_seconds")
    lyap_idx = column_index_058(header, "lambda1_bits_per_second")
    lz_idx = column_index_058(header, "lz76_bits_per_second")
    times = Float64[parse(Float64, row[time_idx]) for row in rows]
    lyap = Float64[parse(Float64, row[lyap_idx]) for row in rows]
    lz = Float64[tryparse(Float64, row[lz_idx]) === nothing ? NaN : parse(Float64, row[lz_idx]) for row in rows]
    return times, lyap, lz
end

function block_entropy_bits_058(symbols::Vector{Int}, block_size::Int)::Float64
    n = length(symbols)
    block_size < 1 && error("block_size must be positive.")
    n >= block_size || return NaN

    word_counts = Dict{Tuple{Vararg{Int}}, Int}()
    sample_count = n - block_size + 1

    for start in 1:sample_count
        word = Tuple(Int[symbols[i] for i in start:(start + block_size - 1)])
        word_counts[word] = get(word_counts, word, 0) + 1
    end

    entropy = 0.0
    for count in values(word_counts)
        p = count / sample_count
        entropy -= p * log2(p)
    end
    return entropy
end

function mean_event_interval_seconds_058(times_seconds::Vector{Float64})::Float64
    length(times_seconds) >= 2 || return NaN
    total = 0.0
    for i in 2:length(times_seconds)
        total += times_seconds[i] - times_seconds[i - 1]
    end
    return total / (length(times_seconds) - 1)
end

function block_entropy_rows_058(symbols::Vector{Int}, event_times_seconds::Vector{Float64})
    n = length(symbols)
    mean_interval_seconds = mean_event_interval_seconds_058(event_times_seconds)
    max_block = min(MAX_BLOCK_SIZE_058, n - 1)
    rows = NamedTuple[]
    previous_block_entropy_bits = 0.0
    for block_size in 1:max_block
        block_entropy_bits = block_entropy_bits_058(symbols, block_size)
        conditional_entropy_bits = block_entropy_bits - previous_block_entropy_bits
        block_event_interval_seconds = block_size * mean_interval_seconds
        block_entropy_rate_bits_per_second = block_entropy_bits / block_event_interval_seconds
        conditional_entropy_rate_bits_per_second = conditional_entropy_bits / mean_interval_seconds
        sample_count = n - block_size + 1
        push!(rows, (;
            block_size=block_size,
            block_entropy_bits=block_entropy_bits,
            conditional_entropy_bits=conditional_entropy_bits,
            mean_event_interval_seconds=mean_interval_seconds,
            block_event_interval_seconds=block_event_interval_seconds,
            abramov_block_entropy_bits_per_second=block_entropy_rate_bits_per_second,
            abramov_conditional_entropy_bits_per_second=conditional_entropy_rate_bits_per_second,
            sample_count=sample_count,
        ))
        previous_block_entropy_bits = block_entropy_bits
    end
    return rows
end

function write_block_rows_058(rows)
    open(BLOCK_PATH_058, "w") do io
        println(io, "block_size\tblock_entropy_bits\tconditional_entropy_bits\tmean_event_interval_seconds\tblock_event_interval_seconds\tabramov_block_entropy_bits_per_second\tabramov_conditional_entropy_bits_per_second\tsample_count")
        for row in rows
            println(io, join((
                row.block_size,
                @sprintf("%.12g", row.block_entropy_bits),
                @sprintf("%.12g", row.conditional_entropy_bits),
                @sprintf("%.12g", row.mean_event_interval_seconds),
                @sprintf("%.12g", row.block_event_interval_seconds),
                @sprintf("%.12g", row.abramov_block_entropy_bits_per_second),
                @sprintf("%.12g", row.abramov_conditional_entropy_bits_per_second),
                row.sample_count,
            ), '\t'))
        end
    end
end

function plot_with_block_entropy_058(times, lyap, lz, block_rows)
    block_sizes = [row.block_size for row in block_rows]
    block_entropy_rates = [row.abramov_block_entropy_bits_per_second for row in block_rows]
    conditional_entropy_rates = [row.abramov_conditional_entropy_bits_per_second for row in block_rows]

    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax_time = Axis(
        fig[1, 1],
        xlabel="time (s)",
        ylabel="entropy rate (bits/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    )
    ax_block = Axis(
        fig[1, 1],
        xaxisposition=:top,
        yaxisposition=:right,
        xlabel="block size b",
        xlabelsize=30,
        xticklabelsize=22,
        xticks=1:MAX_BLOCK_SIZE_058,
        backgroundcolor=:transparent,
        tellwidth=false,
        tellheight=false,
    )
    hideydecorations!(ax_block, grid=false)
    hidespines!(ax_block, :l, :r, :b)

    lyap_line = lines!(ax_time, times, lyap; color=:black, linewidth=3.2)
    finite_lz = isfinite.(lz)
    lz_line = lines!(ax_time, times[finite_lz], lz[finite_lz]; color=:red, linewidth=3.2)
    block_line = lines!(ax_block, block_sizes, block_entropy_rates; color=:dodgerblue4, linewidth=3.4)
    scatter!(ax_block, block_sizes, block_entropy_rates; color=:dodgerblue4, markersize=8)
    cbe_line = lines!(ax_block, block_sizes, conditional_entropy_rates; color=:teal, linewidth=2.6, linestyle=:dash)
    scatter!(ax_block, block_sizes, conditional_entropy_rates; color=:teal, markersize=6)

    y_values = vcat(
        lyap[isfinite.(lyap)],
        lz[isfinite.(lz)],
        block_entropy_rates[isfinite.(block_entropy_rates)],
        conditional_entropy_rates[isfinite.(conditional_entropy_rates)],
    )
    y_min, y_max = extrema(y_values)
    margin = 0.08 * max(y_max - y_min, 1.0e-6)
    ylims!(ax_time, y_min - margin, y_max + margin)
    ylims!(ax_block, y_min - margin, y_max + margin)
    xlims!(ax_time, minimum(times), maximum(times) * 1.02)
    xlims!(ax_block, minimum(block_sizes), maximum(block_sizes))

    Legend(
        fig[1, 1],
        [lyap_line, lz_line, block_line, cbe_line],
        ["leading Lyapunov exponent", "Abramov LZ76", "Block H_b/(bτ)", "CBE ΔH_b/τ"],
        labelsize=21,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )

    save(PLOT_PATH_058, fig, px_per_unit=2)
    return PLOT_PATH_058
end

function main()
    symbols, event_times_seconds = read_events_058()
    times, lyap, lz = read_convergence_058()
    block_rows = block_entropy_rows_058(symbols, event_times_seconds)
    write_block_rows_058(block_rows)
    plot_with_block_entropy_058(times, lyap, lz, block_rows)
    @info "Wrote conditional block entropy comparison" block=BLOCK_PATH_058 plot=PLOT_PATH_058 max_block_size=MAX_BLOCK_SIZE_058
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
