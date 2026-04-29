using Pkg

const ATTEMPT49_FILTER_ROOT = @__DIR__
const REPO_ROOT_049_FILTER = normpath(joinpath(ATTEMPT49_FILTER_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_049_FILTER)

include(joinpath(ATTEMPT49_FILTER_ROOT, "contours.jl"))

const FILTER_RESULTS_PATH = get(
    ENV,
    "ATTEMPT049_FILTER_RESULTS",
    joinpath(ATTEMPT49_FILTER_ROOT, "gcs_results", "grid1000_seq12_prefixes_remap40_newmodel_results.tsv"),
)
const FILTER_OUTPUT_DIR = get(ENV, "ATTEMPT049_FILTER_OUTPUT_DIR", dirname(FILTER_RESULTS_PATH))
const FILTER_OUTPUT_TAG = get(ENV, "ATTEMPT049_FILTER_OUTPUT_TAG", "grid1000_seq12_filtered_tprefix100000")
const FILTER_T_PREFIX = [1, 0, 0, 0, 0, 0]

function header_lookup(header::AbstractString)
    names = split(header, '\t'; keepempty=true)
    return Dict(name => idx for (idx, name) in pairs(names))
end

function starts_with_filter_prefix(seq::Vector{Int})
    length(seq) >= length(FILTER_T_PREFIX) || return false
    return all(seq[idx] == FILTER_T_PREFIX[idx] for idx in eachindex(FILTER_T_PREFIX))
end

function build_filtered_full_grids(results_path::String)
    T_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    gamma_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    suppress_T_grid = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    ok_count = 0
    error_count = 0
    suppress_points = 0

    open(results_path, "r") do io
        lookup = header_lookup(readline(io))
        required = ("delta_x", "delta_ca", "T_category_id", "gamma_category_id", "T_scs", "status")
        for name in required
            haskey(lookup, name) || error("Missing required TSV column: $(name)")
        end

        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            ca_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
            x_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")

            if filled[ca_idx, x_idx]
                error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
            end
            filled[ca_idx, x_idx] = true

            if fields[lookup["status"]] != "ok"
                error_count += 1
                continue
            end

            ok_count += 1
            T_grid[ca_idx, x_idx] = parse(Int, fields[lookup["T_category_id"]])
            gamma_grid[ca_idx, x_idx] = parse(Int, fields[lookup["gamma_category_id"]])

            T_scs = parse_sequence_field(fields[lookup["T_scs"]])
            suppress_T_grid[ca_idx, x_idx] = starts_with_filter_prefix(T_scs)
            suppress_points += suppress_T_grid[ca_idx, x_idx] ? 1 : 0
        end
    end

    all(filled) || error("One or more contour grid entries were not filled.")
    return T_grid, gamma_grid, suppress_T_grid, ok_count, error_count, suppress_points
end

function categorical_marching_squares_filtered(
    grid::Matrix{Int},
    x_values::Vector{Float64},
    y_values::Vector{Float64},
    suppress_grid::BitMatrix,
)
    xs = Float32[]
    ys = Float32[]
    sizehint!(xs, 3 * (length(x_values) - 1) * (length(y_values) - 1))
    sizehint!(ys, 3 * (length(x_values) - 1) * (length(y_values) - 1))

    local_segments = NTuple{4, Float64}[]
    sizehint!(local_segments, 8)

    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
            if suppress_grid[x_idx, y_idx] &&
               suppress_grid[x_idx + 1, y_idx] &&
               suppress_grid[x_idx + 1, y_idx + 1] &&
               suppress_grid[x_idx, y_idx + 1]
                continue
            end

            y0 = y_values[y_idx]
            y1 = y_values[y_idx + 1]

            bottom_left = grid[x_idx, y_idx]
            bottom_right = grid[x_idx + 1, y_idx]
            top_right = grid[x_idx + 1, y_idx + 1]
            top_left = grid[x_idx, y_idx + 1]

            if bottom_left == bottom_right == top_right == top_left
                continue
            end

            empty!(local_segments)
            categories = (
                bottom_left,
                bottom_right != bottom_left ? bottom_right : 0,
                (top_right != bottom_left && top_right != bottom_right) ? top_right : 0,
                (top_left != bottom_left && top_left != bottom_right && top_left != top_right) ? top_left : 0,
            )

            for category in categories
                if category <= 0
                    continue
                end

                mask_case =
                    (bottom_left == category ? 1 : 0) +
                    (bottom_right == category ? 2 : 0) +
                    (top_right == category ? 4 : 0) +
                    (top_left == category ? 8 : 0)

                for (edge_a, edge_b) in case_segments(mask_case)
                    x_a, y_a = edge_point(edge_a, x0, x1, y0, y1)
                    x_b, y_b = edge_point(edge_b, x0, x1, y0, y1)
                    push_unique_segment!(local_segments, normalize_segment(x_a, y_a, x_b, y_b))
                end
            end

            for (x_a, y_a, x_b, y_b) in local_segments
                push!(xs, Float32(x_a), Float32(x_b), NaN32)
                push!(ys, Float32(y_a), Float32(y_b), NaN32)
            end
        end
    end

    return xs, ys
end

function save_filtered_contour_plot(path::String, T_grid::Matrix{Int}, gamma_grid::Matrix{Int}, suppress_T_grid::BitMatrix)
    fig = Figure(size=(PLOT_WIDTH, PLOT_HEIGHT))
    ax = Axis(
        fig[1, 1],
        title="SSCS changes (t ≤ 5e4)",
        xlabel="ΔCa",
        ylabel="Δx",
        titlesize=AXIS_TITLE_SIZE,
        xlabelsize=AXIS_LABEL_SIZE,
        ylabelsize=AXIS_LABEL_SIZE,
        xticklabelsize=TICK_LABEL_SIZE,
        yticklabelsize=TICK_LABEL_SIZE,
    )

    T_xs, T_ys = categorical_marching_squares_filtered(T_grid, DELTA_CAS_010, DELTA_XS_010, suppress_T_grid)
    gamma_xs, gamma_ys = categorical_marching_squares(gamma_grid, DELTA_CAS_010, DELTA_XS_010)

    lines!(ax, T_xs, T_ys; color=T_COLOR, linewidth=LINEWIDTH)
    lines!(ax, gamma_xs, gamma_ys; color=GAMMA_COLOR, linewidth=LINEWIDTH)
    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT)
end

function main()
    mkpath(FILTER_OUTPUT_DIR)
    plot_path = joinpath(FILTER_OUTPUT_DIR, "$(FILTER_OUTPUT_TAG)_contours.png")
    summary_path = joinpath(FILTER_OUTPUT_DIR, "$(FILTER_OUTPUT_TAG)_summary.txt")

    println("Running attempt-049 filtered full-contour replay.")
    println("Results source: $(FILTER_RESULTS_PATH)")
    println("Output plot: $(plot_path)")
    println("Suppressing red marched squares whose four T SSCS corners start with $(FILTER_T_PREFIX)")
    flush(stdout)

    T_grid, gamma_grid, suppress_T_grid, ok_count, error_count, suppress_points =
        build_filtered_full_grids(FILTER_RESULTS_PATH)
    save_filtered_contour_plot(plot_path, T_grid, gamma_grid, suppress_T_grid)

    open(summary_path, "w") do io
        println(io, "results_path\t$(FILTER_RESULTS_PATH)")
        println(io, "plot_path\t$(plot_path)")
        println(io, "ok_count\t$(ok_count)")
        println(io, "error_count\t$(error_count)")
        println(io, "suppressed_T_prefix\t$(join(FILTER_T_PREFIX, ","))")
        println(io, "suppressed_T_points\t$(suppress_points)")
    end

    println("Successful rows: $(ok_count)")
    println("Error rows: $(error_count)")
    println("T grid points with suppressed prefix: $(suppress_points)")
    println("Saved plot to $(plot_path)")
    println("Saved summary to $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
