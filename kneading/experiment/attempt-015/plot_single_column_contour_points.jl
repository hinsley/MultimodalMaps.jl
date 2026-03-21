using Pkg

const ATTEMPT15_ROOT = @__DIR__
const REPO_ROOT_015 = normpath(joinpath(ATTEMPT15_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_015)

include(joinpath(ATTEMPT15_ROOT, "contours.jl"))

using Printf

const COLUMN_TARGET_DELTA_CA = parse(Float64, get(ENV, "ATTEMPT015_COLUMN_TARGET_DELTA_CA", "-37.6"))
const COLUMN_OUTPUT_TAG = get(ENV, "ATTEMPT015_COLUMN_OUTPUT_TAG", "T_only_nearest_dca37p6_column")
const COLUMN_DELTA_X_TICK_STEP = env_optional_float("ATTEMPT015_COLUMN_DELTA_X_TICK_STEP")

function build_lookup(counts::Dict{BigInt, Int})
    encodings = sort(collect(keys(counts)))
    return Dict(encoding => idx for (idx, encoding) in enumerate(encodings))
end

function unique_sorted(values::Vector{Float64})
    xs = sort(values)
    ys = Float64[]
    for value in xs
        if isempty(ys) || !isapprox(value, ys[end]; atol=1e-9, rtol=0.0)
            push!(ys, value)
        end
    end
    return ys
end

nearest_column(values::Vector{Float64}, target::Float64) = findmin(abs.(values .- target))[2]

function extract_column_contour_points(grid::Matrix{Int}, x_values::Vector{Float64}, y_values::Vector{Float64}, x_target::Float64)
    xs, ys = categorical_marching_squares(grid, x_values, y_values)
    y_hits = Float64[]
    for idx in eachindex(xs)
        x = xs[idx]
        y = ys[idx]
        if !isfinite(x) || !isfinite(y)
            continue
        end
        if isapprox(Float64(x), x_target; atol=1e-6, rtol=0.0)
            push!(y_hits, Float64(y))
        end
    end
    return unique_sorted(y_hits)
end

function write_points(path::String, x_target::Float64, y_hits::Vector{Float64})
    open(path, "w") do io
        println(io, "delta_ca\tdelta_x")
        for y in y_hits
            println(io, @sprintf("%.12f\t%.12f", x_target, y))
        end
    end
end

function save_column_plot(path::String, x_target::Float64, y_hits::Vector{Float64})
    fig = Figure(size=(720, 920))
    ax = Axis(fig[1, 1], xlabel="Delta Ca", ylabel="Delta x")
    vlines!(ax, [x_target]; color=RGBAf(0.25, 0.25, 0.25, 0.8), linewidth=1.5)
    scatter!(ax, fill(x_target, length(y_hits)), y_hits; color=T_COLOR, markersize=6)

    delta = length(DELTA_CAS_015) > 1 ? DELTA_CAS_015[2] - DELTA_CAS_015[1] : 0.05
    xpad = 2.5 * delta
    xlims!(ax, x_target - xpad, x_target + xpad)
    ylims!(ax, first(DELTA_XS_015), last(DELTA_XS_015))
    ax.xticks = ([x_target], [@sprintf("%.6f", x_target)])
    ax.yticks = fixed_ticks(DELTA_XS_015, "%.2f", isnothing(COLUMN_DELTA_X_TICK_STEP) ? 0.05 : COLUMN_DELTA_X_TICK_STEP)
    save(path, fig)
end

function main()
    T_counts, _, error_count = collect_legend()
    T_lookup = build_lookup(T_counts)
    T_grid = build_T_grid(T_lookup)

    column_idx = nearest_column(DELTA_CAS_015, COLUMN_TARGET_DELTA_CA)
    x_target = DELTA_CAS_015[column_idx]
    y_hits = extract_column_contour_points(T_grid, DELTA_CAS_015, DELTA_XS_015, x_target)

    points_path = joinpath(ATTEMPT15_ROOT, "$(COLUMN_OUTPUT_TAG)_points.tsv")
    plot_path = joinpath(ATTEMPT15_ROOT, "$(COLUMN_OUTPUT_TAG).png")
    summary_path = joinpath(ATTEMPT15_ROOT, "$(COLUMN_OUTPUT_TAG)_summary.txt")

    write_points(points_path, x_target, y_hits)
    save_column_plot(plot_path, x_target, y_hits)

    open(summary_path, "w") do io
        println(io, @sprintf("target_delta_ca\t%.12f", COLUMN_TARGET_DELTA_CA))
        println(io, @sprintf("nearest_delta_ca\t%.12f", x_target))
        println(io, "nearest_column_index\t$(column_idx)")
        println(io, "contour_point_count\t$(length(y_hits))")
        println(io, "error_rows_in_source_scan\t$(error_count)")
    end

    println(@sprintf("Target Delta Ca: %.12f", COLUMN_TARGET_DELTA_CA))
    println(@sprintf("Nearest Delta Ca: %.12f", x_target))
    println("Nearest column index: $(column_idx)")
    println("Contour points on that column: $(length(y_hits))")
    println("Saved contour points to $(points_path)")
    println("Saved plot to $(plot_path)")
    println("Saved summary to $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
