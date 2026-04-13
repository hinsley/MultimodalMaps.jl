using Pkg

const ATTEMPT23_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_022_CONTOUR = normpath(joinpath(ATTEMPT23_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_022_CONTOUR)

include(joinpath(ATTEMPT23_CONTOUR_ROOT, "main.jl"))
include(joinpath(REPO_ROOT_022_CONTOUR, "scans", "contours.jl"))

using Base.Threads
using Random

const CONTOUR_OUTPUT_TAG = get(ENV, "ATTEMPT023_OUTPUT_TAG", "grid5000_branch16_criticality_shimizu_morioka_cpu")
const SWEEP_SOURCE_TAG = get(ENV, "ATTEMPT023_SWEEP_TAG", CONTOUR_OUTPUT_TAG)
const SWEEP_DIR = joinpath(ATTEMPT23_ROOT, "$(SWEEP_SOURCE_TAG)_columns")
const LOG_LOCK = ReentrantLock()
const CRIT_ITERATES_023 = collect(2:(ATTEMPT023_MAX_ZMAX - 1))
const ATTEMPT023_PLOT_ITERATE_MIN = parse(Int, get(ENV, "ATTEMPT023_PLOT_ITERATE_MIN", string(first(CRIT_ITERATES_023))))
const ATTEMPT023_PLOT_ITERATE_MAX = parse(Int, get(ENV, "ATTEMPT023_PLOT_ITERATE_MAX", string(min(first(CRIT_ITERATES_023) + 7, last(CRIT_ITERATES_023)))))
const PLOT_ITERATES_023 = [iterate for iterate in CRIT_ITERATES_023 if ATTEMPT023_PLOT_ITERATE_MIN <= iterate <= ATTEMPT023_PLOT_ITERATE_MAX]
const ATTEMPT023_PLOT_MODE = get(ENV, "ATTEMPT023_PLOT_MODE", "overlay")

column_path(col_idx::Int) = joinpath(SWEEP_DIR, @sprintf("column_%04d.tsv", col_idx))
results_path() = joinpath(ATTEMPT23_ROOT, "$(CONTOUR_OUTPUT_TAG)_results.tsv")
legend_path() = joinpath(ATTEMPT23_ROOT, "$(CONTOUR_OUTPUT_TAG)_iterate_colors.tsv")
plot_path() = joinpath(ATTEMPT23_ROOT, "$(CONTOUR_OUTPUT_TAG)_contours.png")

function row_is_complete(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

function write_column(path::String, results::Vector{SMCriticalityResult23})
    open(path, "w") do io
        println(io, "alpha\tlambda\tmaxima_count\tred_mask\tzmax_values\treturn_times\tstatus")
        for result in results
            println(
                io,
                join([
                    @sprintf("%.6f", result.alpha),
                    @sprintf("%.6f", result.lambda),
                    string(result.maxima_count),
                    string(UInt32(result.red_mask)),
                    join_float_list(result.zmax_values),
                    join_float_list(result.return_times),
                    result.status,
                ], '\t'),
            )
        end
    end
end

function parse_result(fields::Vector{SubString{String}})
    return SMCriticalityResult23(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        UInt16(parse(UInt32, fields[4])),
        parse_float_list(fields[5]),
        parse_float_list(fields[6]),
        String(fields[7]),
    )
end

function run_column(col_idx::Int, alpha::Float64, total_cols::Int, total_rows::Int)
    path = column_path(col_idx)
    if row_is_complete(path, total_rows)
        lock(LOG_LOCK)
        try
            @printf("Skipping completed column %d/%d (alpha=%.6f)\n", col_idx, total_cols, alpha)
            flush(stdout)
        finally
            unlock(LOG_LOCK)
        end
        return
    end

    started = time()
    column_results = Vector{SMCriticalityResult23}(undef, total_rows)
    usable = 0
    for row_idx in eachindex(LAMBDAS_023)
        result = scan_orbit(alpha, LAMBDAS_023[row_idx])
        column_results[row_idx] = result
        if result.status == "ok" || result.status == "short"
            usable += 1
        end
    end
    write_column(path, column_results)

    lock(LOG_LOCK)
    try
        @printf(
            "Saved column %d/%d (alpha=%.6f) with %d/%d usable points in %.2f s\n",
            col_idx,
            total_cols,
            alpha,
            usable,
            total_rows,
            time() - started,
        )
        flush(stdout)
    finally
        unlock(LOG_LOCK)
    end
end

function run_or_resume_columns()
    mkpath(SWEEP_DIR)
    total_cols = length(ALPHAS_023)
    total_rows = length(LAMBDAS_023)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_023)
        run_column(col_idx, ALPHAS_023[col_idx], total_cols, total_rows)
    end
end

function scan_column_files(pass_fn)
    for col_idx in eachindex(ALPHAS_023)
        path = column_path(col_idx)
        row_is_complete(path, length(LAMBDAS_023)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_result(split(line, '\t')))
            end
        end
    end
end

function write_final_results(path::String)
    open(path, "w") do io_out
        println(io_out, "alpha\tlambda\tmaxima_count\tred_mask\tzmax_values\treturn_times\tstatus")
        for col_idx in eachindex(ALPHAS_023)
            open(column_path(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
end

function nearest_axis_index(value::Float64, axis::Vector{Float64}, axis_name::String)
    idx = searchsortedfirst(axis, value)
    candidates = Int[]
    idx <= length(axis) && push!(candidates, idx)
    idx > 1 && push!(candidates, idx - 1)
    isempty(candidates) && error("No candidate index found for $(axis_name)=$(value)")

    best_idx = candidates[1]
    best_error = abs(axis[best_idx] - value)
    for candidate in candidates[2:end]
        candidate_error = abs(axis[candidate] - value)
        if candidate_error < best_error
            best_idx = candidate
            best_error = candidate_error
        end
    end

    step = length(axis) > 1 ? abs(axis[2] - axis[1]) : 1.0
    best_error <= max(step * 0.51, 1e-6) || error("Result point does not align with $(axis_name) grid: $(value)")
    return best_idx
end

function local_max_mask(values::Vector{Float64})
    mask = UInt16(0)
    for idx in 2:length(values)-1
        prev_value = values[idx - 1]
        curr_value = values[idx]
        next_value = values[idx + 1]
        if curr_value > prev_value && curr_value > next_value &&
           max(curr_value - prev_value, curr_value - next_value) > ATTEMPT023_RED_EPS
            mask |= UInt16(1) << (idx - 1)
        end
    end
    return mask
end

function return_time_local_max_mask(result::SMCriticalityResult23)
    length(result.return_times) >= 4 || return UInt16(0)
    deltas = diff(result.return_times)
    return local_max_mask(deltas)
end

function build_iterate_grids()
    zmax_grids = Dict(iterate => fill(0, length(LAMBDAS_023), length(ALPHAS_023)) for iterate in CRIT_ITERATES_023)
    dt_grids = Dict(iterate => fill(0, length(LAMBDAS_023), length(ALPHAS_023)) for iterate in CRIT_ITERATES_023)
    filled = falses(length(LAMBDAS_023), length(ALPHAS_023))

    scan_column_files() do result
        alpha_idx = nearest_axis_index(result.alpha, ALPHAS_023, "alpha")
        lambda_idx = nearest_axis_index(result.lambda, LAMBDAS_023, "lambda")
        filled[lambda_idx, alpha_idx] && error("Duplicate grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[lambda_idx, alpha_idx] = true

        if result.status == "ok" || result.status == "short"
            dt_mask = return_time_local_max_mask(result)
            for iterate in CRIT_ITERATES_023
                bit_idx = iterate - 1
                if ((result.red_mask >> bit_idx) & UInt16(1)) == UInt16(1)
                    zmax_grids[iterate][lambda_idx, alpha_idx] = 1
                end
                if ((dt_mask >> bit_idx) & UInt16(1)) == UInt16(1)
                    dt_grids[iterate][lambda_idx, alpha_idx] = 1
                end
            end
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return zmax_grids, dt_grids
end

function canonicalize_segment(segment::NTuple{4, Float64})
    x1, y1, x2, y2 = segment
    if x1 < x2 || (x1 == x2 && y1 <= y2)
        return segment
    end
    return (x2, y2, x1, y1)
end

function march_squares_segments(
    Z::AbstractMatrix,
    x_vals::AbstractVector{<:Real},
    y_vals::AbstractVector{<:Real},
)::Vector{NTuple{4, Float64}}
    segments = NTuple{4, Float64}[]

    for j in 1:size(Z, 1)-1
        for i in 1:size(Z, 2)-1
            x_tl, y_tl = x_vals[i], y_vals[j]
            x_tr, y_tr = x_vals[i+1], y_vals[j]
            x_br, y_br = x_vals[i+1], y_vals[j+1]
            x_bl, y_bl = x_vals[i], y_vals[j+1]

            x_tm, y_tm = (x_tl + x_tr) / 2, (y_tl + y_tr) / 2
            x_rm, y_rm = (x_tr + x_br) / 2, (y_tr + y_br) / 2
            x_bm, y_bm = (x_br + x_bl) / 2, (y_br + y_bl) / 2
            x_lm, y_lm = (x_bl + x_tl) / 2, (y_bl + y_tl) / 2

            z_tl = Z[j, i]
            z_tr = Z[j, i+1]
            z_br = Z[j+1, i+1]
            z_bl = Z[j+1, i]

            if z_tl != z_tr && z_tr == z_br && z_br == z_bl
                push!(segments, (x_tm, y_tm, x_lm, y_lm))
            elseif z_tr != z_tl && z_tl == z_br && z_br == z_bl
                push!(segments, (x_rm, y_rm, x_tm, y_tm))
            elseif z_br != z_tr && z_tr == z_tl && z_tl == z_bl
                push!(segments, (x_bm, y_bm, x_rm, y_rm))
            elseif z_bl != z_br && z_br == z_tr && z_tr == z_tl
                push!(segments, (x_lm, y_lm, x_bm, y_bm))
            elseif z_tl == z_tr && z_bl == z_br && z_tl != z_bl
                push!(segments, (x_lm, y_lm, x_rm, y_rm))
            elseif z_tl == z_bl && z_tr == z_br && z_tl != z_tr
                push!(segments, (x_tm, y_tm, x_bm, y_bm))
            end
        end
    end

    return segments
end

function build_first_iterate_segments(iterate_grids::Dict{Int, Matrix{Int}})
    first_segments = Dict(iterate => NTuple{4, Float64}[] for iterate in PLOT_ITERATES_023)
    seen = Set{NTuple{4, Float64}}()

    for iterate in PLOT_ITERATES_023
        for segment in march_squares_segments(iterate_grids[iterate], ALPHAS_023, LAMBDAS_023)
            key = canonicalize_segment(segment)
            key in seen && continue
            push!(seen, key)
            push!(first_segments[iterate], segment)
        end
    end

    return first_segments
end

function segments_to_polyline(segments::Vector{NTuple{4, Float64}})
    xs = Float64[]
    ys = Float64[]
    sizehint!(xs, 3 * length(segments))
    sizehint!(ys, 3 * length(segments))

    for (x1, y1, x2, y2) in segments
        push!(xs, x1)
        push!(xs, x2)
        push!(xs, NaN)
        push!(ys, y1)
        push!(ys, y2)
        push!(ys, NaN)
    end

    isempty(xs) || pop!(xs)
    isempty(ys) || pop!(ys)
    return xs, ys
end

function random_iterate_colors()
    rng = MersenneTwister(ATTEMPT023_COLOR_SEED)
    return Dict(iterate => RGBAf(rand(rng), rand(rng), rand(rng), 0.92) for iterate in PLOT_ITERATES_023)
end

function dt_iterate_colors()
    palette = [
        RGBAf(0.05, 0.35, 0.85, 0.95),
        RGBAf(0.00, 0.60, 0.80, 0.95),
        RGBAf(0.00, 0.72, 0.55, 0.95),
        RGBAf(0.10, 0.45, 0.55, 0.95),
        RGBAf(0.05, 0.20, 0.55, 0.95),
        RGBAf(0.00, 0.50, 0.35, 0.95),
        RGBAf(0.00, 0.65, 0.95, 0.95),
        RGBAf(0.20, 0.40, 0.95, 0.95),
        RGBAf(0.25, 0.70, 0.95, 0.95),
        RGBAf(0.00, 0.42, 0.68, 0.95),
        RGBAf(0.12, 0.58, 0.28, 0.95),
    ]
    return Dict(iterate => palette[idx] for (idx, iterate) in enumerate(PLOT_ITERATES_023))
end

function write_color_legend(path::String, zmax_colors::Dict{Int, RGBAf}, dt_colors::Dict{Int, RGBAf})
    open(path, "w") do io
        println(io, "family\titerate\tr\tg\tb\ta")
        for iterate in PLOT_ITERATES_023
            zmax_color = zmax_colors[iterate]
            println(io, join(["zmax", string(iterate), string(zmax_color.r), string(zmax_color.g), string(zmax_color.b), string(zmax_color.alpha)], '\t'))
            if ATTEMPT023_PLOT_MODE != "zmax_only"
                dt_color = dt_colors[iterate]
                println(io, join(["dt_max", string(iterate), string(dt_color.r), string(dt_color.g), string(dt_color.b), string(dt_color.alpha)], '\t'))
            end
        end
    end
end

function build_contour_figure(
    zmax_segments::Dict{Int, Vector{NTuple{4, Float64}}},
    dt_segments::Dict{Int, Vector{NTuple{4, Float64}}},
)
    zmax_colors = random_iterate_colors()
    dt_colors = dt_iterate_colors()
    write_color_legend(legend_path(), zmax_colors, dt_colors)

    fig = Figure(size=(ATTEMPT023_FIG_WIDTH, ATTEMPT023_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title=
            ATTEMPT023_PLOT_MODE == "dt_only" ?
            "Shimizu-Morioka Δt local-maximum contours by first iterate" :
            ATTEMPT023_PLOT_MODE == "zmax_only" ?
            "Shimizu-Morioka zmax local-extremum contours by first iterate" :
            "Shimizu-Morioka contours by first iterate: zmax extrema and Δt maxima",
    )

    for iterate in PLOT_ITERATES_023
        if ATTEMPT023_PLOT_MODE != "dt_only"
            zmax_xs, zmax_ys = segments_to_polyline(zmax_segments[iterate])
            lines!(ax, zmax_xs, zmax_ys; color=zmax_colors[iterate], linewidth=ATTEMPT023_LINEWIDTH)
        end

        if ATTEMPT023_PLOT_MODE != "zmax_only"
            dt_xs, dt_ys = segments_to_polyline(dt_segments[iterate])
            lines!(ax, dt_xs, dt_ys; color=dt_colors[iterate], linewidth=ATTEMPT023_LINEWIDTH, linestyle=:dash)
        end
    end

    xlims!(ax, ATTEMPT023_ALPHA_MIN, ATTEMPT023_ALPHA_MAX)
    ylims!(ax, ATTEMPT023_LAMBDA_MIN, ATTEMPT023_LAMBDA_MAX)
    return fig
end

function save_contour_plot(
    png_path::String,
    zmax_grids::Dict{Int, Matrix{Int}},
    dt_grids::Dict{Int, Matrix{Int}},
)
    zmax_segments = build_first_iterate_segments(zmax_grids)
    dt_segments = build_first_iterate_segments(dt_grids)
    fig = build_contour_figure(zmax_segments, dt_segments)
    save(png_path, fig; px_per_unit=ATTEMPT023_PX_PER_UNIT)
    return nothing
end

function main()
    println("Running attempt-023 Shimizu-Morioka criticality-only contour scan.")
    println("Grid: $(length(ALPHAS_023)) alpha points × $(length(LAMBDAS_023)) lambda points")
    println("Threads: $(nthreads()), max z-maxima per orbit: $(ATTEMPT023_MAX_ZMAX)")
    println("Sweep dir: $(SWEEP_DIR)")
    flush(stdout)

    run_or_resume_columns()

    zmax_grids, dt_grids = build_iterate_grids()
    write_final_results(results_path())
    save_contour_plot(plot_path(), zmax_grids, dt_grids)

    usable = 0
    total = length(ALPHAS_023) * length(LAMBDAS_023)
    scan_column_files() do result
        if result.status == "ok" || result.status == "short"
            usable += 1
        end
    end

    println("Usable points: $(usable) / $(total)")
    println("Saved merged results to $(results_path())")
    println("Saved iterate-color legend to $(legend_path())")
    println("Saved contour plot to $(plot_path())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
