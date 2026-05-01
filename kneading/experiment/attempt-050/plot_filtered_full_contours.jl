using Pkg

const ATTEMPT50_FILTER_ROOT = @__DIR__
const REPO_ROOT_050_FILTER = normpath(joinpath(ATTEMPT50_FILTER_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_050_FILTER)

include(joinpath(ATTEMPT50_FILTER_ROOT, "contours.jl"))

const FILTER_RESULTS_PATH = get(
    ENV,
    "ATTEMPT050_FILTER_RESULTS",
    joinpath(ATTEMPT50_FILTER_ROOT, "grid1000_seq12_tmax1e5_prefixes_remap40_newmodel_results.tsv"),
)
const FILTER_OUTPUT_DIR = get(ENV, "ATTEMPT050_FILTER_OUTPUT_DIR", dirname(FILTER_RESULTS_PATH))
const FILTER_OUTPUT_TAG = get(ENV, "ATTEMPT050_FILTER_OUTPUT_TAG", "grid1000_seq12_tmax1e5_prefixcompatible")
const FILTER_T_ZERO_TAIL_START = 2
const FILTER_T_ZERO_TAIL_STOP = 12
const FILTER_T_RGB = (0.86, 0.16, 0.12)
const FILTER_GAMMA_RGB = (0.12, 0.28, 0.88)
const FILTER_LINEWIDTH = parse(Float64, get(ENV, "ATTEMPT050_FILTER_LINEWIDTH", "0.35"))
const FILTER_ALPHA_EXPONENT = parse(Float64, get(ENV, "ATTEMPT050_FILTER_ALPHA_EXPONENT", "0.3"))
const FILTER_TRANSPARENT_BACKGROUND = lowercase(get(ENV, "ATTEMPT050_FILTER_TRANSPARENT_BACKGROUND", "false")) in ("1", "true", "yes")

function header_lookup(header::AbstractString)
    names = split(header, '\t'; keepempty=true)
    return Dict(name => idx for (idx, name) in pairs(names))
end

function T_tail_encoding(seq::Vector{Int})
    tail = length(seq) >= 2 ? seq[2:end] : Int[]
    return encode_sequence(tail)
end

function has_zero_tail_2_to_6(seq::Vector{Int})
    length(seq) >= FILTER_T_ZERO_TAIL_STOP || return false
    return all(seq[idx] == 0 for idx in FILTER_T_ZERO_TAIL_START:FILTER_T_ZERO_TAIL_STOP)
end

function sequences_are_prefix_compatible(seqs::NTuple{4, Vector{Int}})
    nonempty = [seq for seq in seqs if !isempty(seq)]
    isempty(nonempty) && return true
    min_length = minimum(length.(nonempty))
    min_length == 0 && return true

    reference = nonempty[1]
    for seq in nonempty[2:end]
        for idx in 1:min_length
            if seq[idx] != reference[idx]
                return false
            end
        end
    end
    return true
end

function first_mismatch_index(seqs::NTuple{4, Vector{Int}})
    nonempty = [seq for seq in seqs if !isempty(seq)]
    isempty(nonempty) && return nothing
    min_length = minimum(length.(nonempty))
    min_length == 0 && return nothing

    reference = nonempty[1]
    for idx in 1:min_length
        for seq in nonempty[2:end]
            if seq[idx] != reference[idx]
                return idx
            end
        end
    end
    return nothing
end

alpha_for_mismatch_index(idx::Int) = 1.0 / (idx ^ FILTER_ALPHA_EXPONENT)

rgba_from_rgb_alpha(rgb, alpha::Float64) = RGBAf(rgb[1], rgb[2], rgb[3], alpha)

function build_filtered_full_grids(results_path::String)
    T_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    gamma_grid = fill(0, length(DELTA_CAS_010), length(DELTA_XS_010))
    suppress_T_grid = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    T_sequences = [Int[] for _ in 1:length(DELTA_CAS_010), _ in 1:length(DELTA_XS_010)]
    gamma_sequences = [Int[] for _ in 1:length(DELTA_CAS_010), _ in 1:length(DELTA_XS_010)]
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    ok_count = 0
    error_count = 0
    suppress_points = 0
    T_tail_lookup = Dict{BigInt, Int}()

    open(results_path, "r") do io
        lookup = header_lookup(readline(io))
        required = ("delta_x", "delta_ca", "T_category_id", "gamma_category_id", "T_scs", "gamma_scs", "status")
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
            T_scs = parse_sequence_field(fields[lookup["T_scs"]])
            T_tail = T_tail_encoding(T_scs)
            T_grid[ca_idx, x_idx] = get!(T_tail_lookup, T_tail) do
                length(T_tail_lookup) + 1
            end
            gamma_grid[ca_idx, x_idx] = parse(Int, fields[lookup["gamma_category_id"]])

            suppress_T_grid[ca_idx, x_idx] = has_zero_tail_2_to_6(T_scs)
            suppress_points += suppress_T_grid[ca_idx, x_idx] ? 1 : 0
            T_sequences[ca_idx, x_idx] = length(T_scs) >= 2 ? T_scs[2:end] : Int[]
            gamma_sequences[ca_idx, x_idx] = parse_sequence_field(fields[lookup["gamma_scs"]])
        end
    end

    all(filled) || error("One or more contour grid entries were not filled.")
    return T_grid, gamma_grid, suppress_T_grid, T_sequences, gamma_sequences, ok_count, error_count, suppress_points, length(T_tail_lookup)
end

function categorical_marching_squares_prefix_compatible(
    grid::Matrix{Int},
    x_values::Vector{Float64},
    y_values::Vector{Float64},
    sequences::Matrix{Vector{Int}},
    suppress_grid::Union{Nothing, BitMatrix}=nothing,
)
    xs_by_mismatch = [Float32[] for _ in 1:ATTEMPT10_MAX_SEQ_LENGTH]
    ys_by_mismatch = [Float32[] for _ in 1:ATTEMPT10_MAX_SEQ_LENGTH]
    for idx in 1:ATTEMPT10_MAX_SEQ_LENGTH
        sizehint!(xs_by_mismatch[idx], div(3 * (length(x_values) - 1) * (length(y_values) - 1), ATTEMPT10_MAX_SEQ_LENGTH))
        sizehint!(ys_by_mismatch[idx], div(3 * (length(x_values) - 1) * (length(y_values) - 1), ATTEMPT10_MAX_SEQ_LENGTH))
    end

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

            corner_sequences = (
                sequences[x_idx, y_idx],
                sequences[x_idx + 1, y_idx],
                sequences[x_idx + 1, y_idx + 1],
                sequences[x_idx, y_idx + 1],
            )
            mismatch_idx = first_mismatch_index(corner_sequences)
            if isnothing(mismatch_idx) || sequences_are_prefix_compatible(corner_sequences)
                continue
            end
            mismatch_idx = min(mismatch_idx, ATTEMPT10_MAX_SEQ_LENGTH)

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
                push!(xs_by_mismatch[mismatch_idx], Float32(x_a), Float32(x_b), NaN32)
                push!(ys_by_mismatch[mismatch_idx], Float32(y_a), Float32(y_b), NaN32)
            end
        end
    end

    return xs_by_mismatch, ys_by_mismatch
end

function save_filtered_contour_plot(
    path::String,
    T_grid::Matrix{Int},
    gamma_grid::Matrix{Int},
    suppress_T_grid::BitMatrix,
    T_sequences::Matrix{Vector{Int}},
    gamma_sequences::Matrix{Vector{Int}},
)
    background = FILTER_TRANSPARENT_BACKGROUND ? RGBAf(1, 1, 1, 0) : RGBAf(1, 1, 1, 1)
    fig = Figure(size=(PLOT_WIDTH, PLOT_HEIGHT), backgroundcolor=background)
    ax = Axis(
        fig[1, 1],
        title="SSCS changes (t ≤ 1e5)",
        xlabel="ΔCa",
        ylabel="Δx",
        backgroundcolor=background,
        titlesize=AXIS_TITLE_SIZE,
        xlabelsize=AXIS_LABEL_SIZE,
        ylabelsize=AXIS_LABEL_SIZE,
        xticklabelsize=TICK_LABEL_SIZE,
        yticklabelsize=TICK_LABEL_SIZE,
    )

    T_xs_by_mismatch, T_ys_by_mismatch =
        categorical_marching_squares_prefix_compatible(T_grid, DELTA_CAS_010, DELTA_XS_010, T_sequences, suppress_T_grid)
    gamma_xs_by_mismatch, gamma_ys_by_mismatch =
        categorical_marching_squares_prefix_compatible(gamma_grid, DELTA_CAS_010, DELTA_XS_010, gamma_sequences)

    for idx in 1:ATTEMPT10_MAX_SEQ_LENGTH
        if !isempty(gamma_xs_by_mismatch[idx])
            lines!(
                ax,
                gamma_xs_by_mismatch[idx],
                gamma_ys_by_mismatch[idx];
                color=rgba_from_rgb_alpha(FILTER_GAMMA_RGB, alpha_for_mismatch_index(idx)),
                linewidth=FILTER_LINEWIDTH,
            )
        end
        if !isempty(T_xs_by_mismatch[idx])
            lines!(
                ax,
                T_xs_by_mismatch[idx],
                T_ys_by_mismatch[idx];
                color=rgba_from_rgb_alpha(FILTER_T_RGB, alpha_for_mismatch_index(idx)),
                linewidth=FILTER_LINEWIDTH,
            )
        end
    end
    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT)
end

function main()
    mkpath(FILTER_OUTPUT_DIR)
    plot_path = joinpath(FILTER_OUTPUT_DIR, "$(FILTER_OUTPUT_TAG)_contours.png")
    summary_path = joinpath(FILTER_OUTPUT_DIR, "$(FILTER_OUTPUT_TAG)_summary.txt")

    println("Running attempt-050 filtered full-contour replay.")
    println("Results source: $(FILTER_RESULTS_PATH)")
    println("Output plot: $(plot_path)")
    println("Red categories ignore T symbol 1.")
    println("Suppressing red marched squares whose four T SSCS corners have symbols 2:12 all equal to 0.")
    println("Suppressing red and blue marched squares whose corner SSCSs are prefix-compatible.")
    flush(stdout)

    T_grid, gamma_grid, suppress_T_grid, T_sequences, gamma_sequences, ok_count, error_count, suppress_points, T_tail_unique =
        build_filtered_full_grids(FILTER_RESULTS_PATH)
    save_filtered_contour_plot(plot_path, T_grid, gamma_grid, suppress_T_grid, T_sequences, gamma_sequences)

    open(summary_path, "w") do io
        println(io, "results_path\t$(FILTER_RESULTS_PATH)")
        println(io, "plot_path\t$(plot_path)")
        println(io, "ok_count\t$(ok_count)")
        println(io, "error_count\t$(error_count)")
        println(io, "red_category_rule\tT_scs[2:end]")
        println(io, "suppressed_T_rule\tall T_scs[2:12] == 0")
        println(io, "prefix_compatibility_rule\tomit contours where all nonempty corner SSCSs match after truncation to the shortest corner sequence length")
        println(io, "suppressed_T_points\t$(suppress_points)")
        println(io, "T_tail_unique\t$(T_tail_unique)")
    end

    println("Successful rows: $(ok_count)")
    println("Error rows: $(error_count)")
    println("T grid points with suppressed symbols 2:12 all zero: $(suppress_points)")
    println("Unique red categories after dropping symbol 1: $(T_tail_unique)")
    println("Saved plot to $(plot_path)")
    println("Saved summary to $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
