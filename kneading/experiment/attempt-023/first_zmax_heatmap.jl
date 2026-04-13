using Pkg

const ATTEMPT23_HEATMAP_ROOT = @__DIR__
const REPO_ROOT_023_HEATMAP = normpath(joinpath(ATTEMPT23_HEATMAP_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_023_HEATMAP)

include(joinpath(ATTEMPT23_HEATMAP_ROOT, "main.jl"))

using CairoMakie
using Printf

const HEATMAP_SOURCE_TAG_023 = get(ENV, "ATTEMPT023_HEATMAP_SOURCE_TAG", "grid5000_branch16_criticality_shimizu_morioka_cpu")
const HEATMAP_ITERATE_023 = parse(Int, get(ENV, "ATTEMPT023_HEATMAP_ITERATE", "1"))
const HEATMAP_OUTPUT_TAG_023 = get(ENV, "ATTEMPT023_HEATMAP_OUTPUT_TAG", "$(HEATMAP_SOURCE_TAG_023)_zmax_iter$(HEATMAP_ITERATE_023)_heatmap")
const HEATMAP_SWEEP_DIR_023 = joinpath(ATTEMPT23_ROOT, "$(HEATMAP_SOURCE_TAG_023)_columns")
const HEATMAP_OUTPUT_PATH_023 = joinpath(ATTEMPT23_ROOT, "$(HEATMAP_OUTPUT_TAG_023).png")
const HEATMAP_FIG_WIDTH_023 = parse(Int, get(ENV, "ATTEMPT023_HEATMAP_FIG_WIDTH", "1200"))
const HEATMAP_FIG_HEIGHT_023 = parse(Int, get(ENV, "ATTEMPT023_HEATMAP_FIG_HEIGHT", "1000"))
const HEATMAP_PX_PER_UNIT_023 = parse(Float64, get(ENV, "ATTEMPT023_HEATMAP_PX_PER_UNIT", "2.0"))

column_path_023(col_idx::Int) = joinpath(HEATMAP_SWEEP_DIR_023, @sprintf("column_%04d.tsv", col_idx))

function row_is_complete_023(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

@inline function parse_nth_float_023(field::SubString{String}, index::Int)
    isempty(field) && return nothing
    index >= 1 || error("Heatmap iterate index must be at least 1.")

    start_idx = firstindex(field)
    current = 1
    while true
        next_comma = findnext(==(','), field, start_idx)
        token = isnothing(next_comma) ? SubString(field, start_idx, lastindex(field)) : SubString(field, start_idx, prevind(field, next_comma))
        if current == index
            isempty(token) && return nothing
            return parse(Float64, token)
        end
        isnothing(next_comma) && return nothing
        start_idx = nextind(field, next_comma)
        current += 1
    end
end

function build_zmax_iterate_grid_023(iterate_index::Int)
    grid = fill(NaN, length(LAMBDAS_023), length(ALPHAS_023))

    for col_idx in eachindex(ALPHAS_023)
        path = column_path_023(col_idx)
        row_is_complete_023(path, length(LAMBDAS_023)) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            row_idx = 0
            for line in eachline(io)
                row_idx += 1
                fields = split(line, '\t')
                maxima_count = parse(Int, fields[3])
                status = fields[7]
                if maxima_count >= iterate_index && (status == "ok" || status == "short")
                    zmax_value = parse_nth_float_023(fields[5], iterate_index)
                    if !isnothing(zmax_value)
                        grid[row_idx, col_idx] = zmax_value
                    end
                end
            end
            row_idx == length(LAMBDAS_023) || error("Unexpected row count in $(path): $(row_idx)")
        end
    end

    return grid
end

function finite_extrema_023(grid::AbstractMatrix{Float64})
    min_value = Inf
    max_value = -Inf
    found = false
    for value in grid
        if isfinite(value)
            found = true
            value < min_value && (min_value = value)
            value > max_value && (max_value = value)
        end
    end
    found || error("No finite first-zmax values were found.")
    return min_value, max_value
end

function save_zmax_heatmap_023(grid::Matrix{Float64}, iterate_index::Int)
    min_value, max_value = finite_extrema_023(grid)

    fig = Figure(size=(HEATMAP_FIG_WIDTH_023, HEATMAP_FIG_HEIGHT_023))
    ax = Axis(
        fig[1, 1];
        xlabel="alpha",
        ylabel="lambda",
        title="Shimizu-Morioka z-max iterate $(iterate_index) value heatmap",
        backgroundcolor=RGBf(0.94, 0.94, 0.94),
    )

    hm = heatmap!(
        ax,
        ALPHAS_023,
        LAMBDAS_023,
        grid;
        colormap=:viridis,
        colorrange=(min_value, max_value),
        nan_color=RGBAf(0.0, 0.0, 0.0, 0.0),
    )
    Colorbar(fig[1, 2], hm; label="z-max iterate $(iterate_index)")

    xlims!(ax, ATTEMPT023_ALPHA_MIN, ATTEMPT023_ALPHA_MAX)
    ylims!(ax, ATTEMPT023_LAMBDA_MIN, ATTEMPT023_LAMBDA_MAX)

    save(HEATMAP_OUTPUT_PATH_023, fig; px_per_unit=HEATMAP_PX_PER_UNIT_023)
    return nothing
end

function main()
    println("Rendering attempt-023 z-max iterate $(HEATMAP_ITERATE_023) heatmap from existing sweep data.")
    println("Sweep dir: $(HEATMAP_SWEEP_DIR_023)")
    println("Output path: $(HEATMAP_OUTPUT_PATH_023)")
    flush(stdout)

    grid = build_zmax_iterate_grid_023(HEATMAP_ITERATE_023)
    save_zmax_heatmap_023(grid, HEATMAP_ITERATE_023)

    println("Saved z-max iterate $(HEATMAP_ITERATE_023) heatmap to $(HEATMAP_OUTPUT_PATH_023)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
