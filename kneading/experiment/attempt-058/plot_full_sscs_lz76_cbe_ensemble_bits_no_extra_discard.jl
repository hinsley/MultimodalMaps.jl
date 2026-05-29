using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_lz76_cbe_no_extra_discard_convergence_bits_per_second.png")
const CBE_SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_cbe_no_extra_discard_summary.tsv")
const PLOT_WIDTH_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_WIDTH", "1260"))
const PLOT_HEIGHT_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_HEIGHT", "550"))
const MAX_BLOCK_SIZE_058 = parse(Int, get(ENV, "ATTEMPT058_MAX_BLOCK_SIZE", "8"))

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

function read_convergence_058(idx::Int)
    header, rows = read_tsv_058(convergence_path_058(idx))
    time_seconds_idx = column_index_058(header, "time_seconds")
    lle_bits_idx = column_index_058(header, "lambda1_bits_per_second")
    lz_bits_idx = column_index_058(header, "lz76_bits_per_second")
    return (;
        idx,
        time_seconds=Float64[parse(Float64, row[time_seconds_idx]) for row in rows],
        lle_bits=Float64[parse(Float64, row[lle_bits_idx]) for row in rows],
        lz_bits=Float64[parse(Float64, row[lz_bits_idx]) for row in rows],
    )
end

function read_events_058(idx::Int)
    header, rows = read_tsv_058(events_path_058(idx))
    time_seconds_idx = column_index_058(header, "time_seconds")
    symbol_idx = column_index_058(header, "symbol")
    return (;
        idx,
        time_seconds=Float64[parse(Float64, row[time_seconds_idx]) for row in rows],
        symbols=Int[parse(Int, row[symbol_idx]) for row in rows],
    )
end

function block_entropy_bits_058(symbols::Vector{Int}, block_size::Int)::Float64
    n = length(symbols)
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
    return (times_seconds[end] - times_seconds[1]) / (length(times_seconds) - 1)
end

function cbe_rows_058(events)
    mean_interval_seconds = mean_event_interval_seconds_058(events.time_seconds)
    max_block = min(MAX_BLOCK_SIZE_058, length(events.symbols) - 1)
    previous_h = 0.0
    rows = NamedTuple[]
    for block_size in 1:max_block
        h = block_entropy_bits_058(events.symbols, block_size)
        cbe_bits = h - previous_h
        cbe_bits_per_second = cbe_bits / mean_interval_seconds
        push!(rows, (;
            idx=events.idx,
            block_size,
            block_entropy_bits=h,
            conditional_entropy_bits=cbe_bits,
            abramov_cbe_bits_per_second=cbe_bits_per_second,
            mean_event_interval_seconds=mean_interval_seconds,
            sample_count=length(events.symbols) - block_size + 1,
        ))
        previous_h = h
    end
    return rows
end

function write_cbe_summary_058(rows_by_trajectory)
    open(CBE_SUMMARY_PATH_058, "w") do io
        println(io, "trajectory\tblock_size\tblock_entropy_bits\tconditional_entropy_bits\tabramov_cbe_bits_per_second\tmean_event_interval_seconds\tsample_count")
        for rows in rows_by_trajectory
            for row in rows
                println(io, join((
                    row.idx,
                    row.block_size,
                    @sprintf("%.12g", row.block_entropy_bits),
                    @sprintf("%.12g", row.conditional_entropy_bits),
                    @sprintf("%.12g", row.abramov_cbe_bits_per_second),
                    @sprintf("%.12g", row.mean_event_interval_seconds),
                    row.sample_count,
                ), '\t'))
            end
        end
    end
end

function plot_ensemble_bits_058(convergence_results, cbe_rows_by_trajectory)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax_time = Axis(
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
    ax_block = Axis(
        fig[1, 1],
        xaxisposition=:top,
        yaxisposition=:right,
        xlabel="Block size",
        xlabelsize=26,
        xticklabelsize=20,
        xticks=1:MAX_BLOCK_SIZE_058,
        backgroundcolor=:transparent,
        tellwidth=false,
        tellheight=false,
    )
    hideydecorations!(ax_block, grid=false)
    hidespines!(ax_block, :l, :r, :b)
    ylims!(ax_block, 0.1, 0.55)
    xlims!(ax_block, 1, MAX_BLOCK_SIZE_058)

    lle_handle = nothing
    lz_handle = nothing
    cbe_handle = nothing
    for result in convergence_results
        local a = lines!(ax_time, result.time_seconds, result.lle_bits; color=(:black, 0.32), linewidth=2.0)
        finite_lz = isfinite.(result.lz_bits)
        local b = lines!(ax_time, result.time_seconds[finite_lz], result.lz_bits[finite_lz]; color=(:firebrick3, 0.42), linewidth=2.0)
        lle_handle === nothing && (lle_handle = a)
        lz_handle === nothing && (lz_handle = b)
    end
    for rows in cbe_rows_by_trajectory
        block_sizes = [row.block_size for row in rows]
        cbe_rates = [row.abramov_cbe_bits_per_second for row in rows]
        local c = lines!(ax_block, block_sizes, cbe_rates; color=(:dodgerblue4, 0.36), linewidth=2.0)
        scatter!(ax_block, block_sizes, cbe_rates; color=(:dodgerblue4, 0.42), markersize=7)
        cbe_handle === nothing && (cbe_handle = c)
    end

    Legend(
        fig[1, 1],
        [lle_handle, lz_handle, cbe_handle],
        ["Maximal Lyapunov exponent", "SSCS Abramov LZ76", "SSCS Abramov CBE"],
        labelsize=20,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )
    save(PLOT_PATH_058, fig, px_per_unit=2)
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory convergence files found in $(TRAJECTORY_DIR_058).")
    convergence_results = [read_convergence_058(idx) for idx in indices]
    cbe_rows_by_trajectory = [cbe_rows_058(read_events_058(idx)) for idx in indices]
    write_cbe_summary_058(cbe_rows_by_trajectory)
    plot_ensemble_bits_058(convergence_results, cbe_rows_by_trajectory)
    mean_lle_bits = sum(result.lle_bits[end] for result in convergence_results) / length(convergence_results)
    mean_lz_bits = sum(result.lz_bits[end] for result in convergence_results) / length(convergence_results)
    final_cbe_values = [last(rows).abramov_cbe_bits_per_second for rows in cbe_rows_by_trajectory]
    mean_cbe_bits = sum(final_cbe_values) / length(final_cbe_values)
    @info "Wrote full SSCS LZ76/CBE no-extra-discard ensemble in bits/s" plot=PLOT_PATH_058 cbe_summary=CBE_SUMMARY_PATH_058 n=length(indices) max_block_size=MAX_BLOCK_SIZE_058 mean_lle_bits mean_lz_bits mean_cbe_bits
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
