using Pkg

const ATTEMPT22_CONTOUR_ROOT = @__DIR__
const REPO_ROOT_022_CONTOUR = normpath(joinpath(ATTEMPT22_CONTOUR_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_022_CONTOUR)

include(joinpath(ATTEMPT22_CONTOUR_ROOT, "main.jl"))
include(joinpath(REPO_ROOT_022_CONTOUR, "scans", "contours.jl"))

using Base.Threads
using Random

const CONTOUR_OUTPUT_TAG = get(ENV, "ATTEMPT022_OUTPUT_TAG", "grid1000_branch16_criticality_shimizu_morioka")
const SWEEP_DIR = joinpath(ATTEMPT22_ROOT, "$(CONTOUR_OUTPUT_TAG)_columns")
const LOG_LOCK = ReentrantLock()
const ALPHA_INDEX_022 = Dict(@sprintf("%.6f", value) => idx for (idx, value) in enumerate(ALPHAS_022))
const LAMBDA_INDEX_022 = Dict(@sprintf("%.6f", value) => idx for (idx, value) in enumerate(LAMBDAS_022))
const CRIT_ITERATES_022 = collect(2:(ATTEMPT022_MAX_ZMAX - 1))

column_path(col_idx::Int) = joinpath(SWEEP_DIR, @sprintf("column_%04d.tsv", col_idx))
results_path() = joinpath(ATTEMPT22_ROOT, "$(CONTOUR_OUTPUT_TAG)_results.tsv")
legend_path() = joinpath(ATTEMPT22_ROOT, "$(CONTOUR_OUTPUT_TAG)_iterate_colors.tsv")
plot_path() = joinpath(ATTEMPT22_ROOT, "$(CONTOUR_OUTPUT_TAG)_contours.png")

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

function write_column(path::String, results::Vector{SMCriticalityResult22})
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
    return SMCriticalityResult22(
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
    column_results = Vector{SMCriticalityResult22}(undef, total_rows)
    usable = 0
    for row_idx in eachindex(LAMBDAS_022)
        result = scan_orbit(alpha, LAMBDAS_022[row_idx])
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
    total_cols = length(ALPHAS_022)
    total_rows = length(LAMBDAS_022)
    Threads.@threads :dynamic for col_idx in eachindex(ALPHAS_022)
        run_column(col_idx, ALPHAS_022[col_idx], total_cols, total_rows)
    end
end

function scan_column_files(pass_fn)
    for col_idx in eachindex(ALPHAS_022)
        path = column_path(col_idx)
        row_is_complete(path, length(LAMBDAS_022)) || error("Missing or incomplete column file: $(path)")
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
        for col_idx in eachindex(ALPHAS_022)
            open(column_path(col_idx), "r") do io_in
                readline(io_in)
                for line in eachline(io_in)
                    println(io_out, line)
                end
            end
        end
    end
end

function build_iterate_grids()
    grids = Dict(iterate => fill(0, length(LAMBDAS_022), length(ALPHAS_022)) for iterate in CRIT_ITERATES_022)
    filled = falses(length(LAMBDAS_022), length(ALPHAS_022))

    scan_column_files() do result
        alpha_idx = get(ALPHA_INDEX_022, @sprintf("%.6f", result.alpha), nothing)
        lambda_idx = get(LAMBDA_INDEX_022, @sprintf("%.6f", result.lambda), nothing)
        if isnothing(alpha_idx) || isnothing(lambda_idx)
            error("Result point does not align with plotting grid.")
        end
        filled[lambda_idx, alpha_idx] && error("Duplicate grid entry at alpha=$(result.alpha), lambda=$(result.lambda)")
        filled[lambda_idx, alpha_idx] = true

        if result.status == "ok" || result.status == "short"
            for iterate in CRIT_ITERATES_022
                bit_idx = iterate - 1
                if ((result.red_mask >> bit_idx) & UInt16(1)) == UInt16(1)
                    grids[iterate][lambda_idx, alpha_idx] = 1
                end
            end
        end
    end

    all(filled) || error("One or more grid entries were not filled.")
    return grids
end

function random_iterate_colors()
    rng = MersenneTwister(ATTEMPT022_COLOR_SEED)
    return Dict(iterate => RGBAf(rand(rng), rand(rng), rand(rng), 0.92) for iterate in CRIT_ITERATES_022)
end

function write_color_legend(path::String, colors::Dict{Int, RGBAf})
    open(path, "w") do io
        println(io, "iterate\tr\tg\tb\ta")
        for iterate in CRIT_ITERATES_022
            color = colors[iterate]
            println(io, join([string(iterate), string(color.r), string(color.g), string(color.b), string(color.alpha)], '\t'))
        end
    end
end

function save_contour_plot(path::String, iterate_grids::Dict{Int, Matrix{Int}})
    colors = random_iterate_colors()
    write_color_legend(legend_path(), colors)

    fig = Figure(size=(ATTEMPT022_FIG_WIDTH, ATTEMPT022_FIG_HEIGHT))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka criticality contours by iterate",
    )

    for iterate in CRIT_ITERATES_022
        xs, ys = march_squares_simple(iterate_grids[iterate], ALPHAS_022, LAMBDAS_022)
        lines!(ax, xs, ys; color=colors[iterate], linewidth=ATTEMPT022_LINEWIDTH)
    end

    xlims!(ax, ATTEMPT022_ALPHA_MIN, ATTEMPT022_ALPHA_MAX)
    ylims!(ax, ATTEMPT022_LAMBDA_MIN, ATTEMPT022_LAMBDA_MAX)
    save(path, fig; px_per_unit=ATTEMPT022_PX_PER_UNIT)
end

function main()
    println("Running attempt-022 Shimizu-Morioka criticality-only contour scan.")
    println("Grid: $(length(ALPHAS_022)) alpha points × $(length(LAMBDAS_022)) lambda points")
    println("Threads: $(nthreads()), max z-maxima per orbit: $(ATTEMPT022_MAX_ZMAX)")
    println("Sweep dir: $(SWEEP_DIR)")
    flush(stdout)

    run_or_resume_columns()

    iterate_grids = build_iterate_grids()
    write_final_results(results_path())
    save_contour_plot(plot_path(), iterate_grids)

    usable = 0
    total = length(ALPHAS_022) * length(LAMBDAS_022)
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
