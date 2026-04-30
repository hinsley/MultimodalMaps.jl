using Pkg

const ATTEMPT50_LEGACY_FILTER_ROOT = @__DIR__
const REPO_ROOT_050_LEGACY_FILTER = normpath(joinpath(ATTEMPT50_LEGACY_FILTER_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_050_LEGACY_FILTER)

include(joinpath(ATTEMPT50_LEGACY_FILTER_ROOT, "contours.jl"))

const LEGACY_FILTER_RESULTS_PATH = get(
    ENV,
    "ATTEMPT050_LEGACY_FILTER_RESULTS",
    joinpath(
        ATTEMPT50_LEGACY_FILTER_ROOT,
        "gcs_results",
        "grid1000_seq12_tmax1e5_prefixes_remap40_newmodel_results.tsv",
    ),
)
const LEGACY_FILTER_OUTPUT_DIR = get(ENV, "ATTEMPT050_LEGACY_FILTER_OUTPUT_DIR", dirname(LEGACY_FILTER_RESULTS_PATH))
const LEGACY_FILTER_PREFIX = get(ENV, "ATTEMPT050_LEGACY_FILTER_PREFIX", "grid1000_seq12_tmax1e5")

function legacy_header_lookup(header::AbstractString)
    names = split(header, '\t'; keepempty=true)
    return Dict(name => idx for (idx, name) in pairs(names))
end

function legacy_has_prefix_100000(seq::Vector{Int})
    return length(seq) >= 6 && seq[1:6] == [1, 0, 0, 0, 0, 0]
end

function legacy_has_zero_tail_2_to_6(seq::Vector{Int})
    return length(seq) >= 6 && all(seq[idx] == 0 for idx in 2:6)
end

function legacy_tail_encoding(seq::Vector{Int})
    tail = length(seq) >= 2 ? seq[2:end] : Int[]
    return encode_sequence(tail)
end

function legacy_categorical_marching_squares(
    grid::Matrix{Int},
    x_values::Vector{Float64},
    y_values::Vector{Float64},
    suppress_grid::Union{Nothing, BitMatrix}=nothing,
)
    xs = Float32[]
    ys = Float32[]
    sizehint!(xs, 4 * (length(x_values) - 1) * (length(y_values) - 1))
    sizehint!(ys, 4 * (length(x_values) - 1) * (length(y_values) - 1))
    local_segments = NTuple{4, Float64}[]
    sizehint!(local_segments, 8)

    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
            if !isnothing(suppress_grid) &&
               suppress_grid[x_idx, y_idx] &&
               suppress_grid[x_idx + 1, y_idx] &&
               suppress_grid[x_idx + 1, y_idx + 1] &&
               suppress_grid[x_idx, y_idx + 1]
                continue
            end

            bottom_left = grid[x_idx, y_idx]
            bottom_right = grid[x_idx + 1, y_idx]
            top_right = grid[x_idx + 1, y_idx + 1]
            top_left = grid[x_idx, y_idx + 1]

            if bottom_left == bottom_right == top_right == top_left
                continue
            end

            x0y = y_values[y_idx]
            x1y = y_values[y_idx + 1]
            empty!(local_segments)
            categories = (
                bottom_left,
                bottom_right != bottom_left ? bottom_right : 0,
                (top_right != bottom_left && top_right != bottom_right) ? top_right : 0,
                (top_left != bottom_left && top_left != bottom_right && top_left != top_right) ? top_left : 0,
            )

            for category in categories
                category > 0 || continue

                mask_case =
                    (bottom_left == category ? 1 : 0) +
                    (bottom_right == category ? 2 : 0) +
                    (top_right == category ? 4 : 0) +
                    (top_left == category ? 8 : 0)

                for (edge_a, edge_b) in case_segments(mask_case)
                    x_a, y_a = edge_point(edge_a, x0, x1, x0y, x1y)
                    x_b, y_b = edge_point(edge_b, x0, x1, x0y, x1y)
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

function legacy_save_plot(path::String, T_grid::Matrix{Int}, gamma_grid::Matrix{Int}, suppress_T_grid::BitMatrix; title::String)
    fig = Figure(size=(PLOT_WIDTH, PLOT_HEIGHT))
    ax = Axis(
        fig[1, 1],
        title=title,
        xlabel="ΔCa",
        ylabel="Δx",
        titlesize=AXIS_TITLE_SIZE,
        xlabelsize=AXIS_LABEL_SIZE,
        ylabelsize=AXIS_LABEL_SIZE,
        xticklabelsize=TICK_LABEL_SIZE,
        yticklabelsize=TICK_LABEL_SIZE,
    )

    T_xs, T_ys = legacy_categorical_marching_squares(T_grid, DELTA_CAS_010, DELTA_XS_010, suppress_T_grid)
    gamma_xs, gamma_ys = legacy_categorical_marching_squares(gamma_grid, DELTA_CAS_010, DELTA_XS_010)
    lines!(ax, T_xs, T_ys; color=T_COLOR, linewidth=LINEWIDTH)
    lines!(ax, gamma_xs, gamma_ys; color=GAMMA_COLOR, linewidth=LINEWIDTH)
    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT)
end

function build_tprefix_grids(results_path::String)
    T_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    gamma_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    suppress_T_grid = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    ok_count = 0
    error_count = 0
    suppress_points = 0

    open(results_path, "r") do io
        lookup = legacy_header_lookup(readline(io))
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
            filled[ca_idx, x_idx] && error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
            filled[ca_idx, x_idx] = true

            if fields[lookup["status"]] != "ok"
                error_count += 1
                continue
            end

            ok_count += 1
            T_grid[ca_idx, x_idx] = parse(Int, fields[lookup["T_category_id"]])
            gamma_grid[ca_idx, x_idx] = parse(Int, fields[lookup["gamma_category_id"]])
            T_scs = parse_sequence_field(fields[lookup["T_scs"]])
            suppress_T_grid[ca_idx, x_idx] = legacy_has_prefix_100000(T_scs)
            suppress_points += suppress_T_grid[ca_idx, x_idx] ? 1 : 0
        end
    end

    all(filled) || error("One or more tprefix grid entries were not filled.")
    return T_grid, gamma_grid, suppress_T_grid, ok_count, error_count, suppress_points
end

function build_ttail_grids(results_path::String)
    T_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    gamma_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    suppress_T_grid = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    ok_count = 0
    error_count = 0
    suppress_points = 0
    T_tail_lookup = Dict{BigInt, Int}()

    open(results_path, "r") do io
        lookup = legacy_header_lookup(readline(io))
        required = ("delta_x", "delta_ca", "gamma_category_id", "T_scs", "status")
        for name in required
            haskey(lookup, name) || error("Missing required TSV column: $(name)")
        end

        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            ca_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
            x_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")
            filled[ca_idx, x_idx] && error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
            filled[ca_idx, x_idx] = true

            if fields[lookup["status"]] != "ok"
                error_count += 1
                continue
            end

            ok_count += 1
            T_scs = parse_sequence_field(fields[lookup["T_scs"]])
            T_tail = legacy_tail_encoding(T_scs)
            T_grid[ca_idx, x_idx] = get!(T_tail_lookup, T_tail) do
                length(T_tail_lookup) + 1
            end
            gamma_grid[ca_idx, x_idx] = parse(Int, fields[lookup["gamma_category_id"]])
            suppress_T_grid[ca_idx, x_idx] = legacy_has_zero_tail_2_to_6(T_scs)
            suppress_points += suppress_T_grid[ca_idx, x_idx] ? 1 : 0
        end
    end

    all(filled) || error("One or more ttail grid entries were not filled.")
    return T_grid, gamma_grid, suppress_T_grid, ok_count, error_count, suppress_points, length(T_tail_lookup)
end

function write_summary(path::String, pairs::Pair...)
    open(path, "w") do io
        for pair in pairs
            println(io, string(pair.first), '\t', string(pair.second))
        end
    end
end

function main()
    mkpath(LEGACY_FILTER_OUTPUT_DIR)
    println("Running attempt-050 legacy filter variant replay.")
    println("Results source: $(LEGACY_FILTER_RESULTS_PATH)")
    println("Output directory: $(LEGACY_FILTER_OUTPUT_DIR)")
    flush(stdout)

    tprefix_plot = joinpath(LEGACY_FILTER_OUTPUT_DIR, "$(LEGACY_FILTER_PREFIX)_filtered_tprefix100000_contours.png")
    tprefix_summary = joinpath(LEGACY_FILTER_OUTPUT_DIR, "$(LEGACY_FILTER_PREFIX)_filtered_tprefix100000_summary.txt")
    T_grid, gamma_grid, suppress_T_grid, ok_count, error_count, suppress_points =
        build_tprefix_grids(LEGACY_FILTER_RESULTS_PATH)
    legacy_save_plot(tprefix_plot, T_grid, gamma_grid, suppress_T_grid; title="SSCS changes (t ≤ 1e5)")
    write_summary(
        tprefix_summary,
        "results_path" => LEGACY_FILTER_RESULTS_PATH,
        "plot_path" => tprefix_plot,
        "ok_count" => ok_count,
        "error_count" => error_count,
        "suppressed_T_prefix" => "1,0,0,0,0,0",
        "suppressed_T_points" => suppress_points,
    )
    println("Saved $(tprefix_plot)")
    flush(stdout)

    ttail_plot = joinpath(LEGACY_FILTER_OUTPUT_DIR, "$(LEGACY_FILTER_PREFIX)_filtered_ttail00000_contours.png")
    ttail_summary = joinpath(LEGACY_FILTER_OUTPUT_DIR, "$(LEGACY_FILTER_PREFIX)_filtered_ttail00000_summary.txt")
    T_grid, gamma_grid, suppress_T_grid, ok_count, error_count, suppress_points, T_tail_unique =
        build_ttail_grids(LEGACY_FILTER_RESULTS_PATH)
    legacy_save_plot(ttail_plot, T_grid, gamma_grid, suppress_T_grid; title="SSCS changes (t ≤ 1e5)")
    write_summary(
        ttail_summary,
        "results_path" => LEGACY_FILTER_RESULTS_PATH,
        "plot_path" => ttail_plot,
        "ok_count" => ok_count,
        "error_count" => error_count,
        "red_category_rule" => "T_scs[2:end]",
        "suppressed_T_rule" => "all T_scs[2:6] == 0",
        "suppressed_T_points" => suppress_points,
        "T_tail_unique" => T_tail_unique,
    )
    println("Saved $(ttail_plot)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
