using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Printf
using Random

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const N_SURROGATES_058 = parse(Int, get(ENV, "ATTEMPT058_LZ76_N_SURROGATES", "200"))
const RNG_SEED_058 = parse(Int, get(ENV, "ATTEMPT058_LZ76_SURROGATE_SEED", "58076"))
const DISCARD_FRACTION_058 = parse(Float64, get(ENV, "ATTEMPT058_LZ76_DISCARD_FRACTION", "0.0"))
const DISCARD_LABEL_058 = @sprintf("discard%02d", round(Int, 100 * DISCARD_FRACTION_058))
const OUTPUT_STEM_058 = "$(BASE_TAG_058)_full_sscs_lz76_shuffled_surrogates_$(DISCARD_LABEL_058)_n$(N_SURROGATES_058)_seed$(RNG_SEED_058)"
const SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_STEM_058)_summary.tsv")
const SURROGATES_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_STEM_058)_surrogates.tsv")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_STEM_058).png")
const PLOT_WIDTH_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_WIDTH", "1700"))
const PLOT_HEIGHT_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_HEIGHT", "1000"))

function read_tsv_058(path::String)
    lines = readlines(path)
    isempty(lines) && error("Empty TSV: $(path)")
    header = split(first(lines), '\t')
    rows = [split(line, '\t') for line in lines[2:end] if !isempty(strip(line))]
    return header, rows
end

function column_index_058(header, name::String, path::String)
    idx = findfirst(==(name), header)
    idx === nothing && error("Column $(name) not found in $(path).")
    return idx
end

function trajectory_indices_058()
    paths = sort(filter(path -> occursin(r"traj\d+_events\.tsv$", basename(path)), readdir(TRAJECTORY_DIR_058; join=true)))
    return [parse(Int, match(r"traj(\d+)_events\.tsv$", basename(path)).captures[1]) for path in paths]
end

events_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, @sprintf("traj%02d_events.tsv", idx))

function read_events_058(idx::Int)
    path = events_path_058(idx)
    header, rows = read_tsv_058(path)
    time_idx = column_index_058(header, "time", path)
    symbol_idx = column_index_058(header, "symbol", path)
    times = Float64[parse(Float64, row[time_idx]) for row in rows]
    symbols = Int[parse(Int, row[symbol_idx]) for row in rows]
    return times, symbols
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

function lz76_nats_per_second_058(symbols::AbstractVector{<:Integer}, mean_interval_seconds::Float64)
    n = length(symbols)
    n <= 1 && return NaN, 0
    mean_interval_seconds > 0 || return NaN, 0
    alphabet_size = length(unique(symbols))
    alphabet_size > 1 || return NaN, 0
    phrase_count = lz76_phrase_count_058(symbols)
    nats_per_symbol = phrase_count * log(n) / n
    return nats_per_symbol / mean_interval_seconds, phrase_count
end

function quantile_sorted_058(sorted_xs::Vector{Float64}, p::Float64)
    n = length(sorted_xs)
    n == 0 && return NaN
    n == 1 && return sorted_xs[1]
    h = 1 + (n - 1) * p
    lo = floor(Int, h)
    hi = ceil(Int, h)
    lo == hi && return sorted_xs[lo]
    return sorted_xs[lo] + (h - lo) * (sorted_xs[hi] - sorted_xs[lo])
end

function summarize_surrogates_058(values::Vector{Float64}, real_value::Float64)
    finite_values = sort(filter(isfinite, values))
    n = length(finite_values)
    n == 0 && return (; mean=NaN, std=NaN, q025=NaN, q25=NaN, median=NaN, q75=NaN, q975=NaN, percentile=NaN, z=NaN, empirical_two_sided_p=NaN)
    mean_value = sum(finite_values) / n
    std_value = n > 1 ? sqrt(sum((x - mean_value)^2 for x in finite_values) / (n - 1)) : NaN
    percentile = count(<=(real_value), finite_values) / n
    abs_observed = abs(real_value - mean_value)
    empirical_two_sided_p = (count(x -> abs(x - mean_value) >= abs_observed, finite_values) + 1) / (n + 1)
    z = isfinite(std_value) && std_value > 0 ? (real_value - mean_value) / std_value : NaN
    return (;
        mean=mean_value,
        std=std_value,
        q025=quantile_sorted_058(finite_values, 0.025),
        q25=quantile_sorted_058(finite_values, 0.25),
        median=quantile_sorted_058(finite_values, 0.50),
        q75=quantile_sorted_058(finite_values, 0.75),
        q975=quantile_sorted_058(finite_values, 0.975),
        percentile,
        z,
        empirical_two_sided_p,
    )
end

function analyze_trajectory_058(idx::Int, rng::AbstractRNG)
    times, symbols_all = read_events_058(idx)
    n_total = length(symbols_all)
    discard_count = floor(Int, DISCARD_FRACTION_058 * n_total)
    start_idx = discard_count + 1
    start_idx <= n_total || error("Discard fraction removes all symbols for trajectory $(idx).")
    symbols = symbols_all[start_idx:end]
    times_window = times[start_idx:end]
    n = length(symbols)
    mean_interval_seconds = n > 1 ? ((times_window[end] - times_window[1]) * 1.0e-3) / (n - 1) : NaN
    real_lz, real_phrases = lz76_nats_per_second_058(symbols, mean_interval_seconds)
    surrogate_values = Vector{Float64}(undef, N_SURROGATES_058)
    surrogate_phrases = Vector{Int}(undef, N_SURROGATES_058)
    shuffled = collect(symbols)
    for j in 1:N_SURROGATES_058
        copyto!(shuffled, symbols)
        shuffle!(rng, shuffled)
        surrogate_values[j], surrogate_phrases[j] = lz76_nats_per_second_058(shuffled, mean_interval_seconds)
    end
    stats = summarize_surrogates_058(surrogate_values, real_lz)
    return (;
        idx,
        n_symbols_total=n_total,
        n_symbols_used=n,
        n_symbols_discarded=discard_count,
        alphabet_size=length(unique(symbols)),
        first_event_time_seconds=times_window[1] * 1.0e-3,
        last_event_time_seconds=times_window[end] * 1.0e-3,
        mean_interval_seconds,
        real_lz,
        real_phrases,
        surrogate_values,
        surrogate_phrases,
        stats,
    )
end

function write_outputs_058(results)
    open(SUMMARY_PATH_058, "w") do io
        println(io, "trajectory\tn_symbols_total\tn_symbols_discarded\tn_symbols_used\talphabet_size\tfirst_event_time_seconds\tlast_event_time_seconds\tmean_event_interval_seconds\treal_full_sscs_lz76_nats_per_second\treal_lz76_phrase_count\tsurrogate_count\tsurrogate_mean_nats_per_second\tsurrogate_std_nats_per_second\tsurrogate_q025_nats_per_second\tsurrogate_q25_nats_per_second\tsurrogate_median_nats_per_second\tsurrogate_q75_nats_per_second\tsurrogate_q975_nats_per_second\treal_minus_surrogate_mean_nats_per_second\treal_over_surrogate_mean\treal_surrogate_z\treal_surrogate_percentile\tempirical_two_sided_p")
        for result in results
            s = result.stats
            println(io, join((
                result.idx,
                result.n_symbols_total,
                result.n_symbols_discarded,
                result.n_symbols_used,
                result.alphabet_size,
                @sprintf("%.12g", result.first_event_time_seconds),
                @sprintf("%.12g", result.last_event_time_seconds),
                @sprintf("%.12g", result.mean_interval_seconds),
                @sprintf("%.12g", result.real_lz),
                result.real_phrases,
                N_SURROGATES_058,
                @sprintf("%.12g", s.mean),
                @sprintf("%.12g", s.std),
                @sprintf("%.12g", s.q025),
                @sprintf("%.12g", s.q25),
                @sprintf("%.12g", s.median),
                @sprintf("%.12g", s.q75),
                @sprintf("%.12g", s.q975),
                @sprintf("%.12g", result.real_lz - s.mean),
                @sprintf("%.12g", result.real_lz / s.mean),
                @sprintf("%.12g", s.z),
                @sprintf("%.12g", s.percentile),
                @sprintf("%.12g", s.empirical_two_sided_p),
            ), '\t'))
        end
    end
    open(SURROGATES_PATH_058, "w") do io
        println(io, "trajectory\tsurrogate_index\tfull_sscs_lz76_nats_per_second\tlz76_phrase_count")
        for result in results
            for j in eachindex(result.surrogate_values)
                println(io, join((
                    result.idx,
                    j,
                    @sprintf("%.12g", result.surrogate_values[j]),
                    result.surrogate_phrases[j],
                ), '\t'))
            end
        end
    end
end

function plot_results_058(results)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=22)
    ax = Axis(
        fig[1, 1],
        xlabel="trajectory",
        ylabel="full SSCS Abramov LZ76 (nats/s)",
        xlabelsize=28,
        ylabelsize=28,
        xticks=([result.idx for result in results], [string(result.idx) for result in results]),
        xticklabelsize=20,
        yticklabelsize=20,
    )
    surrogate_handle = nothing
    interval_handle = nothing
    real_handle = nothing
    for result in results
        x = result.idx
        s = result.stats
        local h1 = scatter!(
            ax,
            fill(x, length(result.surrogate_values)),
            result.surrogate_values;
            color=(:gray35, 0.18),
            markersize=9,
        )
        local h2 = lines!(ax, [x, x], [s.q025, s.q975]; color=(:gray15, 0.8), linewidth=5)
        lines!(ax, [x - 0.20, x + 0.20], [s.median, s.median]; color=(:gray15, 0.9), linewidth=4)
        local h3 = scatter!(ax, [x], [result.real_lz]; color=:firebrick3, markersize=20)
        surrogate_handle === nothing && (surrogate_handle = h1)
        interval_handle === nothing && (interval_handle = h2)
        real_handle === nothing && (real_handle = h3)
    end
    Legend(
        fig[1, 1],
        [surrogate_handle, interval_handle, real_handle],
        ["shuffled surrogates", "surrogate 2.5%-97.5%", "real sequence"],
        labelsize=20,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:left,
        valign=:top,
    )
    save(PLOT_PATH_058, fig, px_per_unit=2)
end

function print_key_results_058(results)
    real_values = [result.real_lz for result in results]
    surrogate_means = [result.stats.mean for result in results]
    ratios = real_values ./ surrogate_means
    z_values = [result.stats.z for result in results]
    ps = [result.stats.empirical_two_sided_p for result in results]
    mean_real = sum(real_values) / length(real_values)
    mean_surrogate = sum(surrogate_means) / length(surrogate_means)
    mean_ratio = sum(ratios) / length(ratios)
    @info "Wrote shuffled full SSCS LZ76 surrogate comparison" summary=SUMMARY_PATH_058 surrogates=SURROGATES_PATH_058 plot=PLOT_PATH_058 trajectories=length(results) n_surrogates=N_SURROGATES_058 seed=RNG_SEED_058 discard_fraction=DISCARD_FRACTION_058 mean_real mean_surrogate mean_ratio min_z=minimum(z_values) max_z=maximum(z_values) min_p=minimum(ps) max_p=maximum(ps)
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory event files found in $(TRAJECTORY_DIR_058).")
    rng = MersenneTwister(RNG_SEED_058)
    results = [analyze_trajectory_058(idx, rng) for idx in indices]
    write_outputs_058(results)
    plot_results_058(results)
    print_key_results_058(results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
