using Pkg

const ATTEMPT043_ROOT = @__DIR__
const REPO_ROOT_043 = normpath(joinpath(ATTEMPT043_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_043)

using Base64
using Printf

module Attempt027
include(normpath(joinpath(@__DIR__, "..", "attempt-027", "contours.jl")))
end

const A27 = Attempt027
const OUTPUT_TAG_043 = get(
    ENV,
    "ATTEMPT046_OUTPUT_TAG",
    get(
        ENV,
        "ATTEMPT043_OUTPUT_TAG",
        "grid2000_branch16_absxskip16_plot8_forcedfirstskip_sameedge_black_red_blue_explorer_shimizu_morioka_cpu",
    ),
)
const HTML_PATH_043 = joinpath(ATTEMPT043_ROOT, "$(OUTPUT_TAG_043).html")
const STATS_PATH_043 = joinpath(ATTEMPT043_ROOT, "$(OUTPUT_TAG_043)_iterate_stats.tsv")
const MISSING_TIME_WORD_043 = UInt16(0xffff)
const TABLE_ITERATE_START_046 = 2
const TABLE_ITERATE_END_046 = 16
const TABLE_ITERATE_COUNT_046 = TABLE_ITERATE_END_046 - TABLE_ITERATE_START_046 + 1

@inline sign_code_043(value::Float64) = value > 0.0 ? UInt16(0x2) : value < 0.0 ? UInt16(0x1) : UInt16(0x0)
@inline skip_bit_043(nominal_iterate::Int) = UInt8(1) << (nominal_iterate - 2)
@inline point_linear_index_043(i::Int, j::Int, n_alpha::Int) = (j - 1) * n_alpha + i

function raw_sign_word_046(result::A27.SMAbsXResult25)
    word = UInt32(0)
    max_iter = min(result.absxmax_count, TABLE_ITERATE_END_046, length(result.absxmax_dot_values))
    for nominal_iterate in TABLE_ITERATE_START_046:TABLE_ITERATE_END_046
        code = nominal_iterate <= max_iter ? UInt32(sign_code_043(result.absxmax_dot_values[nominal_iterate])) : UInt32(0)
        word |= code << (2 * (nominal_iterate - TABLE_ITERATE_START_046))
    end
    return word
end

function monotone_sign_adjusted_dots_046(raw_values::AbstractVector{Float64})
    adjusted = Vector{Float64}(undef, length(raw_values))
    isempty(raw_values) && return adjusted
    @inbounds for idx in eachindex(raw_values)
        value = raw_values[idx]
        if !isfinite(value) || value == 0.0
            adjusted[idx] = value
        else
            adjusted[idx] = abs(value)
        end
    end

    prev_sign = isfinite(raw_values[1]) ? A27.sign_class_025(raw_values[1]) : Int8(0)
    for idx in 2:length(raw_values)
        value = raw_values[idx]
        current_sign = isfinite(value) ? A27.sign_class_025(value) : Int8(0)
        if current_sign == Int8(0) || prev_sign == Int8(0)
            adjusted[idx] = value
        else
            adjusted[idx] = current_sign == prev_sign ? abs(value) : -abs(value)
        end
        current_sign != Int8(0) && (prev_sign = current_sign)
    end
    return adjusted
end

function monotone_sign_word_046(result::A27.SMAbsXResult25)
    word = UInt32(0)
    adjusted = monotone_sign_adjusted_dots_046(result.absxmax_dot_values)
    max_iter = min(result.absxmax_count, TABLE_ITERATE_END_046, length(adjusted))
    for nominal_iterate in TABLE_ITERATE_START_046:TABLE_ITERATE_END_046
        code = nominal_iterate <= max_iter ? UInt32(sign_code_043(adjusted[nominal_iterate])) : UInt32(0)
        word |= code << (2 * (nominal_iterate - TABLE_ITERATE_START_046))
    end
    return word
end

function build_sign_words_043()
    n_alpha = length(A27.ALPHAS_025)
    n_lambda = length(A27.LAMBDAS_025)
    raw_sign_words = fill(UInt32(0), n_alpha * n_lambda)
    monotone_sign_words = fill(UInt32(0), n_alpha * n_lambda)

    for col_idx in eachindex(A27.ALPHAS_025)
        path = A27.column_path_025(col_idx)
        A27.row_is_complete_025(path, n_lambda) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            row_idx = 0
            for line in eachline(io)
                row_idx += 1
                result = A27.parse_result_025(split(line, '\t'))
                linear_idx = (row_idx - 1) * n_alpha + col_idx
                raw_sign_words[linear_idx] = raw_sign_word_046(result)
                monotone_sign_words[linear_idx] = monotone_sign_word_046(result)
            end
            row_idx == n_lambda || error("Column $(col_idx) ended at $(row_idx) rows, expected $(n_lambda)")
        end
    end

    return raw_sign_words, monotone_sign_words
end

function mark_point_skip_masks_043!(
    point_skip_masks::Vector{UInt8},
    j::Int,
    i::Int,
    signs::NTuple{4, Int8},
    shorter_sign::Int8,
    nominal_iterate::Int,
    n_alpha::Int,
)
    2 <= nominal_iterate <= 8 || return nothing
    bit = skip_bit_043(nominal_iterate)
    if signs[1] == shorter_sign
        point_skip_masks[point_linear_index_043(i, j, n_alpha)] |= bit
    end
    if signs[2] == shorter_sign
        point_skip_masks[point_linear_index_043(i + 1, j, n_alpha)] |= bit
    end
    if signs[3] == shorter_sign
        point_skip_masks[point_linear_index_043(i + 1, j + 1, n_alpha)] |= bit
    end
    if signs[4] == shorter_sign
        point_skip_masks[point_linear_index_043(i, j + 1, n_alpha)] |= bit
    end
    return nothing
end

@inline function increment_local_skip_045(
    skip::NTuple{4, Int},
    signs::NTuple{4, Int8},
    shorter_sign::Int8,
)
    return (
        skip[1] + (signs[1] == shorter_sign ? 1 : 0),
        skip[2] + (signs[2] == shorter_sign ? 1 : 0),
        skip[3] + (signs[3] == shorter_sign ? 1 : 0),
        skip[4] + (signs[4] == shorter_sign ? 1 : 0),
    )
end

function evaluate_square_local_045(
    j::Int,
    i::Int,
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    skip::NTuple{4, Int},
    flip::NTuple{4, Int},
)
    k_tl = nominal_iterate + skip[1]
    k_tr = nominal_iterate + skip[2]
    k_br = nominal_iterate + skip[3]
    k_bl = nominal_iterate + skip[4]
    ks = (k_tl, k_tr, k_br, k_bl)

    any(k -> k < 1 || k > length(dot_grids) || k + 1 > length(time_grids), ks) && return A27.missing_evaluation_025()

    d_tl = isodd(flip[1]) ? -dot_grids[k_tl][j, i] : dot_grids[k_tl][j, i]
    d_tr = isodd(flip[2]) ? -dot_grids[k_tr][j, i + 1] : dot_grids[k_tr][j, i + 1]
    d_br = isodd(flip[3]) ? -dot_grids[k_br][j + 1, i + 1] : dot_grids[k_br][j + 1, i + 1]
    d_bl = isodd(flip[4]) ? -dot_grids[k_bl][j + 1, i] : dot_grids[k_bl][j + 1, i]
    all(isfinite, (d_tl, d_tr, d_br, d_bl)) || return A27.missing_evaluation_025()

    t_tl = time_grids[k_tl][j, i]
    t_tr = time_grids[k_tr][j, i + 1]
    t_br = time_grids[k_br][j + 1, i + 1]
    t_bl = time_grids[k_bl][j + 1, i]
    all(isfinite, (t_tl, t_tr, t_br, t_bl)) || return A27.missing_evaluation_025()

    t2_tl = time_grids[k_tl + 1][j, i]
    t2_tr = time_grids[k_tr + 1][j, i + 1]
    t2_br = time_grids[k_br + 1][j + 1, i + 1]
    t2_bl = time_grids[k_bl + 1][j + 1, i]
    all(isfinite, (t2_tl, t2_tr, t2_br, t2_bl)) || return A27.missing_evaluation_025()

    signs = (
        A27.sign_class_025(d_tl),
        A27.sign_class_025(d_tr),
        A27.sign_class_025(d_br),
        A27.sign_class_025(d_bl),
    )
    any(==(Int8(0)), signs) && return A27.missing_evaluation_025()

    status = all(==(signs[1]), signs) ? A27.EVAL_CONSTANT_025 : A27.EVAL_MIXED_025
    return A27.SquareEvaluation25(
        status,
        (d_tl, d_tr, d_br, d_bl),
        (t_tl, t_tr, t_br, t_bl),
        (t2_tl, t2_tr, t2_br, t2_bl),
        signs,
        ks,
    )
end

@inline function apply_scheduled_local_state_046(
    skip::NTuple{4, Int},
    flip::NTuple{4, Int},
    scheduled_skip::Matrix{UInt8},
    scheduled_flip::Matrix{UInt8},
    nominal_iterate::Int,
)
    return (
        skip[1] + Int(scheduled_skip[1, nominal_iterate]),
        skip[2] + Int(scheduled_skip[2, nominal_iterate]),
        skip[3] + Int(scheduled_skip[3, nominal_iterate]),
        skip[4] + Int(scheduled_skip[4, nominal_iterate]),
    ),
    (
        (flip[1] + Int(scheduled_flip[1, nominal_iterate])) & 0x1,
        (flip[2] + Int(scheduled_flip[2, nominal_iterate])) & 0x1,
        (flip[3] + Int(scheduled_flip[3, nominal_iterate])) & 0x1,
        (flip[4] + Int(scheduled_flip[4, nominal_iterate])) & 0x1,
    )
end

@inline function effective_sign_at_point_046(
    row_idx::Int,
    col_idx::Int,
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    skip_count::Int,
    flip_parity::Int,
)
    effective_iterate = nominal_iterate + skip_count
    effective_iterate < 1 && return Int8(0)
    effective_iterate > length(dot_grids) && return Int8(0)
    value = dot_grids[effective_iterate][row_idx, col_idx]
    isfinite(value) || return Int8(0)
    sign = A27.sign_class_025(value)
    sign == Int8(0) && return Int8(0)
    return isodd(flip_parity) ? Int8(-sign) : sign
end

function point_sign_sequence_effective_046(
    row_idx::Int,
    col_idx::Int,
    dot_grids::Vector{Matrix{Float64}};
    first_iter::Int,
    last_iter::Int,
    active_skip::Int,
    active_flip::Int,
    scheduled_skip::AbstractVector{UInt8},
    scheduled_flip::AbstractVector{UInt8},
)
    first_iter <= last_iter || return Int8[]
    seq = Int8[]
    skip_count = active_skip
    flip_parity = active_flip
    for nominal_iterate in first_iter:last_iter
        if nominal_iterate > first_iter
            skip_count += Int(scheduled_skip[nominal_iterate])
            flip_parity = (flip_parity + Int(scheduled_flip[nominal_iterate])) & 0x1
        end
        sign = effective_sign_at_point_046(row_idx, col_idx, nominal_iterate, dot_grids, skip_count, flip_parity)
        sign == Int8(0) && break
        push!(seq, sign)
    end
    return seq
end

function build_grids_043()
    dot_grids = A27.build_iterate_grids_025(
        result -> result.absxmax_count,
        result -> monotone_sign_adjusted_dots_046(result.absxmax_dot_values),
    )
    cumulative_time_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = A27.cumulative_to_interval_grids_025(cumulative_time_grids)
    return dot_grids, time_grids
end

@inline function corner_point_indices_046(j::Int, i::Int, corner::Int)
    return corner == 1 ? (j, i) :
           corner == 2 ? (j, i + 1) :
           corner == 3 ? (j + 1, i + 1) :
           (j + 1, i)
end

function point_sign_sequence_046(
    row_idx::Int,
    col_idx::Int,
    dot_grids::Vector{Matrix{Float64}};
    first_iter::Int=2,
    last_iter::Int=16,
)
    first_iter <= last_iter || return Int8[]
    last_iter <= length(dot_grids) || return nothing
    seq = Vector{Int8}(undef, last_iter - first_iter + 1)
    seq_idx = 1
    for nominal_iterate in first_iter:last_iter
        value = dot_grids[nominal_iterate][row_idx, col_idx]
        isfinite(value) || return nothing
        sign = A27.sign_class_025(value)
        sign == Int8(0) && return nothing
        seq[seq_idx] = sign
        seq_idx += 1
    end
    return seq
end

# Classification should not erase a valid mixed square merely because one corner runs out of
# later iterate data. Build the longest common suffix actually available from the contour
# iterate onward and classify from that shared suffix only.
function point_sign_sequence_available_046(
    row_idx::Int,
    col_idx::Int,
    dot_grids::Vector{Matrix{Float64}};
    first_iter::Int=2,
    last_iter::Int=16,
)
    first_iter <= last_iter || return Int8[]
    last_iter = min(last_iter, length(dot_grids))
    seq = Int8[]
    for nominal_iterate in first_iter:last_iter
        value = dot_grids[nominal_iterate][row_idx, col_idx]
        isfinite(value) || break
        sign = A27.sign_class_025(value)
        sign == Int8(0) && break
        push!(seq, sign)
    end
    return seq
end

function common_point_sign_sequences_046(
    short_row::Int,
    short_col::Int,
    long_row::Int,
    long_col::Int,
    dot_grids::Vector{Matrix{Float64}};
    first_iter::Int,
    last_iter::Int,
)
    short_seq = point_sign_sequence_available_046(short_row, short_col, dot_grids; first_iter, last_iter)
    long_seq = point_sign_sequence_available_046(long_row, long_col, dot_grids; first_iter, last_iter)
    n_common = min(length(short_seq), length(long_seq))
    n_common == length(short_seq) || resize!(short_seq, n_common)
    n_common == length(long_seq) || resize!(long_seq, n_common)
    return short_seq, long_seq
end

# A single deletion in the monotone-sign sequence leaves one unmatched terminal symbol on
# the undeleted side because attempt-027 stores iterates only through 16.
@inline function plus_delete_matches_046(candidate::Vector{Int8}, other::Vector{Int8}, delete_idx::Int)
    candidate[delete_idx] == Int8(1) || return false
    @inbounds for idx in 1:(delete_idx - 1)
        candidate[idx] == other[idx] || return false
    end
    @inbounds for idx in delete_idx:(length(candidate) - 1)
        candidate[idx + 1] == other[idx] || return false
    end
    return true
end

@inline function minus_delete_matches_046(candidate::Vector{Int8}, other::Vector{Int8}, delete_idx::Int)
    candidate[delete_idx] == Int8(-1) || return false
    @inbounds for idx in 1:(delete_idx - 1)
        candidate[idx] == other[idx] || return false
    end
    @inbounds for idx in delete_idx:(length(candidate) - 1)
        -candidate[idx + 1] == other[idx] || return false
    end
    return true
end

function grazing_match_symbolic_046(seq_a::Vector{Int8}, seq_b::Vector{Int8}, nominal_iterate::Int)
    length(seq_a) == length(seq_b) || return nothing
    max_delete_idx = min(max(0, 9 - nominal_iterate + 1), length(seq_a) - 1)
    max_delete_idx >= 1 || return nothing
    for delete_idx in 1:max_delete_idx
        if plus_delete_matches_046(seq_a, seq_b, delete_idx)
            return (kind=:blue, mutate_side=:a, delete_idx=delete_idx)
        end
        if plus_delete_matches_046(seq_b, seq_a, delete_idx)
            return (kind=:blue, mutate_side=:b, delete_idx=delete_idx)
        end
        if minus_delete_matches_046(seq_a, seq_b, delete_idx)
            return (kind=:purple, mutate_side=:a, delete_idx=delete_idx)
        end
        if minus_delete_matches_046(seq_b, seq_a, delete_idx)
            return (kind=:purple, mutate_side=:b, delete_idx=delete_idx)
        end
    end
    return nothing
end

function grazing_match_returntime_046(evaluation::A27.SquareEvaluation25)
    should_increment, _ = A27.skip_increment_decision_025(evaluation)
    return should_increment ? (kind=:blue, mutate_side=:shorter, delete_idx=1) : nothing
end

function is_coordinate_singularity_046(seq_a::Vector{Int8}, seq_b::Vector{Int8}; window_len::Int=3)
    length(seq_a) == length(seq_b) || return false
    n = min(window_len, length(seq_a), length(seq_b))
    n == window_len || return false
    diff_idx = Int[]
    @inbounds for idx in 1:n
        seq_a[idx] == seq_b[idx] || push!(diff_idx, idx)
    end
    length(diff_idx) == 2 || return false
    diff_idx[2] == diff_idx[1] + 1 || return false
    return all(seq_a[idx] == -seq_b[idx] for idx in diff_idx)
end

@inline function real_contour_difference_count_046(seq_a::Vector{Int8}, seq_b::Vector{Int8}; window_len::Int=2)
    n = min(window_len, length(seq_a), length(seq_b))
    n == window_len || return typemax(Int)
    count = 0
    @inbounds for idx in 1:n
        count += (seq_a[idx] != seq_b[idx])
    end
    return count
end

function classify_contour_046(
    seq_a::Vector{Int8},
    seq_b::Vector{Int8},
    evaluation::A27.SquareEvaluation25,
    nominal_iterate::Int;
    grazing_mode::Symbol=:symbolic,
)
    grazing_match =
        grazing_mode == :symbolic ? grazing_match_symbolic_046(seq_a, seq_b, nominal_iterate) :
        grazing_mode == :returntime ? grazing_match_returntime_046(evaluation) :
        error("Unsupported grazing mode: $(grazing_mode)")
    grazing_match !== nothing && grazing_match.kind == :blue && return (:blue, grazing_match)
    is_coordinate_singularity_046(seq_a, seq_b; window_len=3) && return (:red, nothing)
    grazing_match !== nothing && grazing_match.kind == :purple && return (:purple, grazing_match)
    real_contour_difference_count_046(seq_a, seq_b; window_len=2) == 1 && return (:black, nothing)
    return (:green, nothing)
end

@inline function edge_pair_code_046(edge_a::Int, edge_b::Int)
    lo = min(edge_a, edge_b)
    hi = max(edge_a, edge_b)
    return UInt8((lo << 4) | hi)
end

@inline function push_unique_pair_046!(pairs::Vector{UInt8}, pair::UInt8)
    @inbounds for existing in pairs
        existing == pair && return false
    end
    push!(pairs, pair)
    return true
end

@inline function has_pair_046(pairs::Vector{UInt8}, pair::UInt8)
    @inbounds for existing in pairs
        existing == pair && return true
    end
    return false
end

@inline function pair_in_specs_046(specs::Vector{Tuple{UInt8, NTuple{4, Float64}}}, pair::UInt8)
    @inbounds for spec in specs
        spec[1] == pair && return true
    end
    return false
end

@inline function append_segment_spec_046!(
    specs::Vector{Tuple{UInt8, NTuple{4, Float64}}},
    point_a::Union{Nothing, Tuple{Float64, Float64}},
    edge_a::Int,
    point_b::Union{Nothing, Tuple{Float64, Float64}},
    edge_b::Int,
)
    (isnothing(point_a) || isnothing(point_b)) && return 0
    x1, y1 = point_a
    x2, y2 = point_b
    push!(specs, (edge_pair_code_046(edge_a, edge_b), (x1, y1, x2, y2)))
    return 1
end

function segment_specs_046(
    values::NTuple{4, Float64},
    x_tl::Float64,
    y_tl::Float64,
    x_tr::Float64,
    y_tr::Float64,
    x_br::Float64,
    y_br::Float64,
    x_bl::Float64,
    y_bl::Float64;
    level::Float64=0.0,
)
    z_tl, z_tr, z_br, z_bl = values
    case_idx =
        (z_tl >= level ? 8 : 0) +
        (z_tr >= level ? 4 : 0) +
        (z_br >= level ? 2 : 0) +
        (z_bl >= level ? 1 : 0)

    specs = Tuple{UInt8, NTuple{4, Float64}}[]
    (case_idx == 0 || case_idx == 15) && return specs

    p1 = A27.edge_point_025(1, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p2 = A27.edge_point_025(2, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p3 = A27.edge_point_025(3, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    p4 = A27.edge_point_025(4, values, x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl, level)
    points = (p1, p2, p3, p4)

    if case_idx == 5 || case_idx == 10
        center_value = 0.25 * (z_tl + z_tr + z_br + z_bl)
        pairing =
            case_idx == 5 ?
            (center_value >= level ? ((1, 2), (3, 4)) : ((1, 4), (2, 3))) :
            (center_value >= level ? ((1, 4), (2, 3)) : ((1, 2), (3, 4)))
        for (edge_a, edge_b) in pairing
            append_segment_spec_046!(specs, points[edge_a], edge_a, points[edge_b], edge_b)
        end
        return specs
    end

    pairing =
        case_idx == 1 ? ((4, 3),) :
        case_idx == 2 ? ((3, 2),) :
        case_idx == 3 ? ((4, 2),) :
        case_idx == 4 ? ((1, 2),) :
        case_idx == 6 ? ((1, 3),) :
        case_idx == 7 ? ((1, 4),) :
        case_idx == 8 ? ((1, 4),) :
        case_idx == 9 ? ((1, 3),) :
        case_idx == 11 ? ((1, 2),) :
        case_idx == 12 ? ((4, 2),) :
        case_idx == 13 ? ((3, 2),) :
        case_idx == 14 ? ((4, 3),) :
        ()

    for (edge_a, edge_b) in pairing
        append_segment_spec_046!(specs, points[edge_a], edge_a, points[edge_b], edge_b)
    end
    return specs
end

function schedule_grazing_update_046!(
    scheduled_skip::Matrix{UInt8},
    scheduled_flip::Matrix{UInt8},
    evaluation::A27.SquareEvaluation25,
    short_rep::Int,
    long_rep::Int,
    grazing_match,
    nominal_iterate::Int,
    classification_iterate_end::Int,
)
    grazing_match === nothing && return nothing
    mutate_rep =
        grazing_match.mutate_side == :a ? short_rep :
        grazing_match.mutate_side == :b ? long_rep :
        grazing_match.mutate_side == :shorter ? short_rep :
        0
    mutate_rep == 0 && return nothing

    activation_nominal = max(nominal_iterate + 1, nominal_iterate + grazing_match.delete_idx - 1)
    activation_nominal <= classification_iterate_end || return nothing

    mutate_sign = evaluation.sign[mutate_rep]
    for corner in 1:4
        evaluation.sign[corner] == mutate_sign || continue
        scheduled_skip[corner, activation_nominal] += UInt8(1)
        if grazing_match.kind == :purple
            scheduled_flip[corner, activation_nominal] = xor(scheduled_flip[corner, activation_nominal], UInt8(1))
        end
    end
    return nothing
end

function collect_sequence_classified_segments_046(
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    grazing_mode::Symbol=:symbolic,
)
    n_plot = A27.ATTEMPT025_PLOT_ITERATE_CAP
    plot_iterate_end = min(8, n_plot)
    classification_iterate_end = min(16, length(dot_grids))
    n_lambda_cells = length(A27.LAMBDAS_025) - 1
    n_alpha_cells = length(A27.ALPHAS_025) - 1
    n_threads = Threads.maxthreadid()

    black_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    red_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    blue_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    purple_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    green_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    source_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    black_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    black_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    red_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    red_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    blue_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    blue_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    purple_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    purple_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    green_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    green_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    scheduled_skip_tls = [zeros(UInt8, 4, classification_iterate_end) for _ in 1:n_threads]
    scheduled_flip_tls = [zeros(UInt8, 4, classification_iterate_end) for _ in 1:n_threads]

    Threads.@threads :dynamic for j in 1:n_lambda_cells
        tid = Threads.threadid()
        black_local = black_tls[tid]
        red_local = red_tls[tid]
        blue_local = blue_tls[tid]
        purple_local = purple_tls[tid]
        green_local = green_tls[tid]
        source_local = source_tls[tid]
        black_cell_local = black_cell_tls[tid]
        black_segment_local = black_segment_tls[tid]
        red_cell_local = red_cell_tls[tid]
        red_segment_local = red_segment_tls[tid]
        blue_cell_local = blue_cell_tls[tid]
        blue_segment_local = blue_segment_tls[tid]
        purple_cell_local = purple_cell_tls[tid]
        purple_segment_local = purple_segment_tls[tid]
        green_cell_local = green_cell_tls[tid]
        green_segment_local = green_segment_tls[tid]
        scheduled_skip_local = scheduled_skip_tls[tid]
        scheduled_flip_local = scheduled_flip_tls[tid]

        y_tl = Float64(A27.LAMBDAS_025[j])
        y_bl = Float64(A27.LAMBDAS_025[j + 1])

        for i in 1:n_alpha_cells
            x_tl = Float64(A27.ALPHAS_025[i])
            x_tr = Float64(A27.ALPHAS_025[i + 1])
            fill!(scheduled_skip_local, 0x00)
            fill!(scheduled_flip_local, 0x00)
            skip = (0, 0, 0, 0)
            flip = (0, 0, 0, 0)
            pending_red_nominal = 0
            for nominal_iterate in TABLE_ITERATE_START_046:plot_iterate_end
                skip, flip = apply_scheduled_local_state_046(skip, flip, scheduled_skip_local, scheduled_flip_local, nominal_iterate)
                if pending_red_nominal != 0
                    if nominal_iterate == pending_red_nominal + 1
                        continue
                    elseif nominal_iterate > pending_red_nominal + 1
                        pending_red_nominal = 0
                    end
                end

                evaluation = evaluate_square_local_045(j, i, nominal_iterate, dot_grids, time_grids, skip, flip)
                evaluation.status == A27.EVAL_MIXED_025 || continue

                _, short_rep, long_rep = A27.choose_representatives_025(evaluation)
                short_row, short_col = corner_point_indices_046(j, i, short_rep)
                long_row, long_col = corner_point_indices_046(j, i, long_rep)
                short_seq = point_sign_sequence_effective_046(
                    short_row,
                    short_col,
                    dot_grids;
                    first_iter=nominal_iterate,
                    last_iter=classification_iterate_end,
                    active_skip=skip[short_rep],
                    active_flip=flip[short_rep],
                    scheduled_skip=@view(scheduled_skip_local[short_rep, :]),
                    scheduled_flip=@view(scheduled_flip_local[short_rep, :]),
                )
                long_seq = point_sign_sequence_effective_046(
                    long_row,
                    long_col,
                    dot_grids;
                    first_iter=nominal_iterate,
                    last_iter=classification_iterate_end,
                    active_skip=skip[long_rep],
                    active_flip=flip[long_rep],
                    scheduled_skip=@view(scheduled_skip_local[long_rep, :]),
                    scheduled_flip=@view(scheduled_flip_local[long_rep, :]),
                )
                n_common = min(length(short_seq), length(long_seq))
                n_common == 0 && continue
                n_common == length(short_seq) || resize!(short_seq, n_common)
                n_common == length(long_seq) || resize!(long_seq, n_common)
                if isempty(short_seq) || isempty(long_seq)
                    continue
                end

                specs = segment_specs_046(
                    evaluation.current_dot,
                    x_tl,
                    y_tl,
                    x_tr,
                    y_tl,
                    x_tr,
                    y_bl,
                    x_tl,
                    y_bl,
                )
                isempty(specs) && continue

                classification, grazing_match = classify_contour_046(short_seq, long_seq, evaluation, nominal_iterate; grazing_mode)

                segment_count = length(specs)
                source_local[nominal_iterate] += 1

                if classification == :black
                    @inbounds for (_, segment) in specs
                        push!(black_local[nominal_iterate], segment)
                    end
                    black_cell_local[nominal_iterate] += 1
                    black_segment_local[nominal_iterate] += segment_count
                elseif classification == :red
                    @inbounds for (_, segment) in specs
                        push!(red_local[nominal_iterate], segment)
                    end
                    red_cell_local[nominal_iterate] += 1
                    red_segment_local[nominal_iterate] += segment_count
                elseif classification == :blue
                    @inbounds for (_, segment) in specs
                        push!(blue_local[nominal_iterate], segment)
                    end
                    blue_cell_local[nominal_iterate] += 1
                    blue_segment_local[nominal_iterate] += segment_count
                elseif classification == :purple
                    @inbounds for (_, segment) in specs
                        push!(purple_local[nominal_iterate], segment)
                    end
                    purple_cell_local[nominal_iterate] += 1
                    purple_segment_local[nominal_iterate] += segment_count
                else
                    @inbounds for (_, segment) in specs
                        push!(green_local[nominal_iterate], segment)
                    end
                    green_cell_local[nominal_iterate] += 1
                    green_segment_local[nominal_iterate] += segment_count
                end
                if classification == :blue || classification == :purple
                    schedule_grazing_update_046!(
                        scheduled_skip_local,
                        scheduled_flip_local,
                        evaluation,
                        short_rep,
                        long_rep,
                        grazing_match,
                        nominal_iterate,
                        classification_iterate_end,
                    )
                end
                if classification == :red
                    pending_red_nominal = nominal_iterate
                end
            end
        end
    end

    black_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    red_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    blue_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    purple_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    green_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    source_cells = zeros(Int, n_plot)
    black_contoured_cells = zeros(Int, n_plot)
    black_segments_count = zeros(Int, n_plot)
    red_contoured_cells = zeros(Int, n_plot)
    red_segments_count = zeros(Int, n_plot)
    blue_contoured_cells = zeros(Int, n_plot)
    blue_segments_count = zeros(Int, n_plot)
    purple_contoured_cells = zeros(Int, n_plot)
    purple_segments_count = zeros(Int, n_plot)
    green_contoured_cells = zeros(Int, n_plot)
    green_segments_count = zeros(Int, n_plot)

    for tid in 1:n_threads
        for iterate in 2:plot_iterate_end
            append!(black_segments_by_iter[iterate], black_tls[tid][iterate])
            append!(red_segments_by_iter[iterate], red_tls[tid][iterate])
            append!(blue_segments_by_iter[iterate], blue_tls[tid][iterate])
            append!(purple_segments_by_iter[iterate], purple_tls[tid][iterate])
            append!(green_segments_by_iter[iterate], green_tls[tid][iterate])
            source_cells[iterate] += source_tls[tid][iterate]
            black_contoured_cells[iterate] += black_cell_tls[tid][iterate]
            black_segments_count[iterate] += black_segment_tls[tid][iterate]
            red_contoured_cells[iterate] += red_cell_tls[tid][iterate]
            red_segments_count[iterate] += red_segment_tls[tid][iterate]
            blue_contoured_cells[iterate] += blue_cell_tls[tid][iterate]
            blue_segments_count[iterate] += blue_segment_tls[tid][iterate]
            purple_contoured_cells[iterate] += purple_cell_tls[tid][iterate]
            purple_segments_count[iterate] += purple_segment_tls[tid][iterate]
            green_contoured_cells[iterate] += green_cell_tls[tid][iterate]
            green_segments_count[iterate] += green_segment_tls[tid][iterate]
        end
    end

    point_skip_masks = fill(UInt8(0), length(A27.ALPHAS_025) * length(A27.LAMBDAS_025))

    iterate_stats = (
        source_cells=source_cells,
        black_contoured_cells=black_contoured_cells,
        black_segments_count=black_segments_count,
        red_contoured_cells=red_contoured_cells,
        red_segments_count=red_segments_count,
        blue_contoured_cells=blue_contoured_cells,
        blue_segments_count=blue_segments_count,
        purple_contoured_cells=purple_contoured_cells,
        purple_segments_count=purple_segments_count,
        green_contoured_cells=green_contoured_cells,
        green_segments_count=green_segments_count,
    )

    return black_segments_by_iter, red_segments_by_iter, blue_segments_by_iter, purple_segments_by_iter, green_segments_by_iter, point_skip_masks, iterate_stats
end

function flatten_segments_043(segments::Vector{NTuple{4, Float64}})
    flat = Vector{Float32}(undef, 4 * length(segments))
    idx = 1
    @inbounds for (x1, y1, x2, y2) in segments
        flat[idx] = Float32(x1); idx += 1
        flat[idx] = Float32(y1); idx += 1
        flat[idx] = Float32(x2); idx += 1
        flat[idx] = Float32(y2); idx += 1
    end
    return flat
end

function base64_bytes_043(values::AbstractVector{T}) where {T}
    io = IOBuffer()
    write(io, reinterpret(UInt8, values))
    return base64encode(take!(io))
end

function base64_gzip_bytes_043(values::AbstractVector{T}) where {T}
    raw = reinterpret(UInt8, values)
    tmp_path = tempname()
    write(tmp_path, raw)
    compressed = try
        read(`gzip -c $tmp_path`)
    finally
        isfile(tmp_path) && rm(tmp_path; force=true)
    end
    return base64encode(compressed)
end

function choose_time_scale_043(time_grids::Vector{Matrix{Float64}})
    max_time = 0.0
    for nominal_iterate in TABLE_ITERATE_START_046:min(TABLE_ITERATE_END_046, length(time_grids))
        grid = time_grids[nominal_iterate]
        @inbounds for value in grid
            isfinite(value) || continue
            value > max_time && (max_time = value)
        end
    end

    for scale in (1000, 200, 100, 50, 10)
        max_time <= (65534 / scale) && return scale
    end
    return 1
end

function build_time_words_043(time_grids::Vector{Matrix{Float64}})
    n_alpha = length(A27.ALPHAS_025)
    n_lambda = length(A27.LAMBDAS_025)
    time_scale = choose_time_scale_043(time_grids)
    words = fill(MISSING_TIME_WORD_043, n_alpha * n_lambda * TABLE_ITERATE_COUNT_046)

    for nominal_iterate in TABLE_ITERATE_START_046:min(TABLE_ITERATE_END_046, length(time_grids))
        grid = time_grids[nominal_iterate]
        offset = nominal_iterate - TABLE_ITERATE_START_046
        for col_idx in 1:n_alpha
            for row_idx in 1:n_lambda
                value = grid[row_idx, col_idx]
                isfinite(value) || continue
                quantized = round(Int, value * time_scale)
                0 <= quantized <= 65534 || continue
                linear_idx = ((row_idx - 1) * n_alpha + (col_idx - 1)) * TABLE_ITERATE_COUNT_046 + offset + 1
                words[linear_idx] = UInt16(quantized)
            end
        end
    end

    return words, time_scale
end

function write_iterate_stats_043(path::String, stats)
    open(path, "w") do io
        println(io, "nominal_iterate\tsource_mixed_cells\tblack_contoured_cells\tblack_segments\tred_contoured_cells\tred_segments\tblue_contoured_cells\tblue_segments\tpurple_contoured_cells\tpurple_segments\tgreen_contoured_cells\tgreen_segments")
        for iterate in 2:min(8, length(stats.source_cells))
            println(
                io,
                join([
                    string(iterate),
                    string(stats.source_cells[iterate]),
                    string(stats.black_contoured_cells[iterate]),
                    string(stats.black_segments_count[iterate]),
                    string(stats.red_contoured_cells[iterate]),
                    string(stats.red_segments_count[iterate]),
                    string(stats.blue_contoured_cells[iterate]),
                    string(stats.blue_segments_count[iterate]),
                    string(stats.purple_contoured_cells[iterate]),
                    string(stats.purple_segments_count[iterate]),
                    string(stats.green_contoured_cells[iterate]),
                    string(stats.green_segments_count[iterate]),
                ], '\t'),
            )
        end
    end
end

function write_html_043(
    path::String,
    symbolic_black_segments_b64_by_iter::Vector{String},
    symbolic_red_segments_b64_by_iter::Vector{String},
    symbolic_blue_segments_b64_by_iter::Vector{String},
    symbolic_purple_segments_b64_by_iter::Vector{String},
    symbolic_green_segments_b64_by_iter::Vector{String},
    returntime_black_segments_b64_by_iter::Vector{String},
    returntime_red_segments_b64_by_iter::Vector{String},
    returntime_blue_segments_b64_by_iter::Vector{String},
    returntime_purple_segments_b64_by_iter::Vector{String},
    returntime_green_segments_b64_by_iter::Vector{String},
    raw_sign_words_b64::String,
    monotone_sign_words_b64::String,
    skip_words_b64::String,
    time_words_gz_b64::String,
    time_scale::Int,
)
    n_alpha = length(A27.ALPHAS_025)
    n_lambda = length(A27.LAMBDAS_025)
    alpha_min = A27.ATTEMPT025_ALPHA_MIN
    alpha_max = A27.ATTEMPT025_ALPHA_MAX
    lambda_min = A27.ATTEMPT025_LAMBDA_MIN
    lambda_max = A27.ATTEMPT025_LAMBDA_MAX

    open(path, "w") do io
        print(io, """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shimizu-Morioka Cumulative-Sign Explorer</title>
  <style>
    :root {
      color-scheme: light;
      --panel: #f3f4f6;
      --ink: #111111;
      --muted: #5f6368;
      --accent: #0f766e;
      --skip: #b91c1c;
      --border: #d0d7de;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: #ffffff; color: var(--ink); font-family: Menlo, Consolas, monospace; }
    #app { display: flex; height: 100%; width: 100%; }
    #viewerPane { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }
    #toolbar {
      display: flex; align-items: center; gap: 12px; padding: 10px 14px;
      border-bottom: 1px solid var(--border); background: #fafafa; color: var(--ink);
    }
    #toolbar button {
      border: 1px solid var(--border); background: white; color: var(--ink);
      padding: 6px 10px; border-radius: 6px; cursor: pointer;
    }
    #toolbar .note { color: var(--muted); font-size: 12px; }
    #viewerWrap { position: relative; flex: 1 1 auto; min-height: 0; background: white; }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    #sidebar {
      width: 344px; max-width: 38vw; border-left: 1px solid var(--border);
      background: var(--panel); padding: 8px 9px; overflow: auto;
    }
    h1, h2 { margin: 0 0 6px 0; font-size: 14px; }
    h2 { margin-top: 9px; font-size: 12px; }
    .box { border: 1px solid var(--border); background: white; border-radius: 8px; padding: 6px 8px; }
    .kv { display: grid; grid-template-columns: 78px 1fr; gap: 2px 6px; font-size: 10px; }
    .label { color: var(--muted); }
    .mono { white-space: pre-wrap; word-break: break-word; }
	    .legend-row { display: flex; align-items: center; gap: 7px; font-size: 10px; margin: 3px 0; }
	    .swatch { width: 20px; height: 3px; border-radius: 2px; }
	    .swatch.black { background: #000000; }
	    .swatch.red { background: #c00000; }
	    .swatch.blue { background: #3b82f6; }
	    .swatch.purple { background: #7c3aed; }
	    .swatch.green { background: #008000; }
	    .swatch.cyan { background: #0ea5e9; }
    .small { font-size: 10px; color: var(--muted); }
    .chip {
      border: 1px solid var(--border); border-radius: 999px; padding: 3px 8px;
      font-size: 12px; background: white;
    }
    .chip.pos { color: #111111; }
    .chip.neg { color: #111111; }
    .chip.missing { color: #6b7280; }
    .iter-controls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; margin-top: 6px; }
    .iter-controls label { display: flex; align-items: center; gap: 4px; font-size: 10px; }
    .iter-buttons { display: grid; gap: 6px; margin-bottom: 6px; }
    .iter-button-row { display: flex; flex-wrap: wrap; gap: 5px; }
    .iter-table { width: 100%; border-collapse: collapse; font-size: 10px; table-layout: fixed; }
    .iter-table th, .iter-table td { border-bottom: 1px solid var(--border); padding: 3px 4px; text-align: left; }
    .iter-table th { color: var(--muted); font-weight: 600; background: #fbfbfb; position: sticky; top: 0; }
    .iter-row.skip { color: var(--skip); font-weight: 600; }
    .iter-row.normal { color: #111111; }
    .iter-row.late { color: #a8afb8; }
    .highlight-note { margin-top: 4px; font-size: 10px; color: var(--muted); }
    .compact-meta { margin-bottom: 4px; }
  </style>
</head>
<body>
  <div id="app">
    <section id="viewerPane">
      <div id="toolbar">
        <button id="resetView">Reset View</button>
        <button id="clearSelection">Clear Selection</button>
        <span class="note">Wheel: zoom. Drag: pan. Hover snaps to the nearest sampled parameter point. Click pins a point.</span>
      </div>
      <div id="viewerWrap">
        <canvas id="baseCanvas"></canvas>
        <canvas id="overlayCanvas"></canvas>
      </div>
    </section>
    <aside id="sidebar">
	      <h1>Attempt-046 Explorer</h1>
	      <h2>Legend</h2>
	      <div class="box">
	        <div class="legend-row"><span class="swatch black"></span><span>real contour: the two monotone sign sequences differ in exactly one place</span></div>
	        <div class="legend-row"><span class="swatch red"></span><span>coordinate singularity: two consecutive monotone signs flip and the rest matches</span></div>
	        <div class="legend-row"><span class="swatch blue"></span><span>grazing (`+` deletion): deleting one `+` in the contour-relative range `k:9` reconciles the suffix, and later nominal iterates inherit that local skipped index on the affected side</span></div>
	        <div class="legend-row"><span class="swatch purple"></span><span>grazing (`-` deletion): deleting one `-` and inverting the later suffix reconciles the contour-relative suffix, and later nominal iterates inherit both the local skipped index and the persistent suffix inversion on the affected side</span></div>
	        <div class="legend-row"><span class="swatch green"></span><span>other mixed square: not black, red, blue, or purple under the above tests</span></div>
	        <div class="legend-row"><span class="swatch cyan"></span><span>selected sampled grid point</span></div>
	        <div class="legend-row"><span class="swatch" style="background:#bcbcbc;"></span><span>four marched squares around the selected point</span></div>
	      </div>
      <h2>Contours</h2>
      <div class="box">
		        <div class="iter-buttons">
		          <div class="iter-button-row">
		            <button id="showAllIterates">Show All</button>
		            <button id="hideAllIterates">Hide All</button>
		            <button id="toggleGrazingMode">Grazing: Symbolic</button>
		          </div>
		          <div class="iter-button-row">
		            <button id="toggleBlackContours">Hide Black</button>
		            <button id="toggleGreenContours">Hide Green</button>
		            <button id="toggleRedContours">Hide Red</button>
		          </div>
		          <div class="iter-button-row">
	              <button id="toggleGreyContours">Hide Blue</button>
	              <button id="togglePurpleContours">Hide Purple</button>
		          </div>
		        </div>
        <div id="iterateControls" class="iter-controls"></div>
      </div>
      <h2>Hover</h2>
      <div class="box">
        <div id="hoverInfo" class="kv compact-meta"></div>
	        <table class="iter-table">
	          <thead>
	            <tr><th>Iter</th><th>Dot</th><th>Mono</th><th>Time</th><th>Skip</th></tr>
	          </thead>
	          <tbody id="hoverTableBody"></tbody>
	        </table>
      </div>
      <h2>Selected</h2>
      <div class="box">
	        <div id="selectedInfo" class="kv compact-meta"></div>
	        <div class="highlight-note">Old-style skip compression is disabled in this version, so the skip column should remain `no`.</div>
	        <table class="iter-table">
	          <thead>
	            <tr><th>Iter</th><th>Dot</th><th>Mono</th><th>Time</th><th>Skip</th></tr>
	          </thead>
	          <tbody id="selectedTableBody"></tbody>
	        </table>
      </div>
      <h2>View</h2>
      <div class="box">
        <div id="viewInfo" class="kv"></div>
      </div>
    </aside>
  </div>
  <script>
    const CONFIG = {
      nAlpha: $(n_alpha),
      nLambda: $(n_lambda),
      alphaMin: $(alpha_min),
      alphaMax: $(alpha_max),
      lambdaMin: $(lambda_min),
      lambdaMax: $(lambda_max)
    };
	    const SYMBOLIC_BLACK_SEGMENTS_B64_BY_ITER = {
""")
        for idx in 2:length(symbolic_black_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, symbolic_black_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
	    const SYMBOLIC_RED_SEGMENTS_B64_BY_ITER = {
""")
        for idx in 2:length(symbolic_red_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, symbolic_red_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
		    const SYMBOLIC_BLUE_SEGMENTS_B64_BY_ITER = {
		""")
        for idx in 2:length(symbolic_blue_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, symbolic_blue_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
		    const SYMBOLIC_PURPLE_SEGMENTS_B64_BY_ITER = {
		""")
        for idx in 2:length(symbolic_purple_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, symbolic_purple_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
		    const SYMBOLIC_GREEN_SEGMENTS_B64_BY_ITER = {
		""")
        for idx in 2:length(symbolic_green_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, symbolic_green_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
	    const RETURNTIME_BLACK_SEGMENTS_B64_BY_ITER = {
""")
        for idx in 2:length(returntime_black_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, returntime_black_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
	    const RETURNTIME_RED_SEGMENTS_B64_BY_ITER = {
""")
        for idx in 2:length(returntime_red_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, returntime_red_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
		    const RETURNTIME_BLUE_SEGMENTS_B64_BY_ITER = {
		""")
        for idx in 2:length(returntime_blue_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, returntime_blue_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
		    const RETURNTIME_PURPLE_SEGMENTS_B64_BY_ITER = {
		""")
        for idx in 2:length(returntime_purple_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, returntime_purple_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
		    const RETURNTIME_GREEN_SEGMENTS_B64_BY_ITER = {
		""")
        for idx in 2:length(returntime_green_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, returntime_green_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
			    const RAW_SIGN_WORDS_B64 = '""")
        print(io, raw_sign_words_b64)
        print(io, """';
	    const MONOTONE_SIGN_WORDS_B64 = '""")
        print(io, monotone_sign_words_b64)
        print(io, """';
	    const SKIP_WORDS_B64 = '""")
        print(io, skip_words_b64)
        print(io, """';
    const TIME_WORDS_GZ_B64 = '""")
        print(io, time_words_gz_b64)
        print(io, """';
    const TIME_SCALE = $(time_scale);

    function decodeBase64Bytes(b64) {
      const raw = atob(b64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
      return bytes;
    }

    function decodeUint16Array(b64) {
      const bytes = decodeBase64Bytes(b64);
      return new Uint16Array(bytes.buffer);
    }

    function decodeUint32Array(b64) {
      const bytes = decodeBase64Bytes(b64);
      return new Uint32Array(bytes.buffer);
    }

    function decodeFloat32Array(b64) {
      const bytes = decodeBase64Bytes(b64);
      return new Float32Array(bytes.buffer);
    }

    async function decodeGzipUint16Array(b64) {
      if (!b64) return new Uint16Array(0);
      if (typeof DecompressionStream === 'undefined') {
        console.warn('DecompressionStream is unavailable; per-iterate times will not be shown.');
        return new Uint16Array(0);
      }
      const bytes = decodeBase64Bytes(b64);
      const stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
      const decompressed = await new Response(stream).arrayBuffer();
      return new Uint16Array(decompressed);
    }

		    const symbolicBlackSegmentsByIter = {};
		    const symbolicRedSegmentsByIter = {};
		    const symbolicBlueSegmentsByIter = {};
		    const symbolicPurpleSegmentsByIter = {};
		    const symbolicGreenSegmentsByIter = {};
		    const returntimeBlackSegmentsByIter = {};
		    const returntimeRedSegmentsByIter = {};
		    const returntimeBlueSegmentsByIter = {};
		    const returntimePurpleSegmentsByIter = {};
		    const returntimeGreenSegmentsByIter = {};
		    for (let nominal = 2; nominal <= 8; nominal += 1) {
		      symbolicBlackSegmentsByIter[nominal] = SYMBOLIC_BLACK_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(SYMBOLIC_BLACK_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      symbolicRedSegmentsByIter[nominal] = SYMBOLIC_RED_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(SYMBOLIC_RED_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      symbolicBlueSegmentsByIter[nominal] = SYMBOLIC_BLUE_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(SYMBOLIC_BLUE_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      symbolicPurpleSegmentsByIter[nominal] = SYMBOLIC_PURPLE_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(SYMBOLIC_PURPLE_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      symbolicGreenSegmentsByIter[nominal] = SYMBOLIC_GREEN_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(SYMBOLIC_GREEN_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      returntimeBlackSegmentsByIter[nominal] = RETURNTIME_BLACK_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(RETURNTIME_BLACK_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      returntimeRedSegmentsByIter[nominal] = RETURNTIME_RED_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(RETURNTIME_RED_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      returntimeBlueSegmentsByIter[nominal] = RETURNTIME_BLUE_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(RETURNTIME_BLUE_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      returntimePurpleSegmentsByIter[nominal] = RETURNTIME_PURPLE_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(RETURNTIME_PURPLE_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		      returntimeGreenSegmentsByIter[nominal] = RETURNTIME_GREEN_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(RETURNTIME_GREEN_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
		    }
	    const rawSignWords = decodeUint32Array(RAW_SIGN_WORDS_B64);
	    const monotoneSignWords = decodeUint32Array(MONOTONE_SIGN_WORDS_B64);
	    const skipWords = decodeBase64Bytes(SKIP_WORDS_B64);
	    let timeWords = null;

    const baseCanvas = document.getElementById('baseCanvas');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const baseCtx = baseCanvas.getContext('2d');
    const overlayCtx = overlayCanvas.getContext('2d');
    const viewerWrap = document.getElementById('viewerWrap');
    const hoverInfo = document.getElementById('hoverInfo');
    const hoverTableBody = document.getElementById('hoverTableBody');
    const selectedInfo = document.getElementById('selectedInfo');
	    const selectedTableBody = document.getElementById('selectedTableBody');
	    const viewInfo = document.getElementById('viewInfo');
	    const iterateControls = document.getElementById('iterateControls');
	    const resetViewButton = document.getElementById('resetView');
			    const clearSelectionButton = document.getElementById('clearSelection');
			    const showAllIteratesButton = document.getElementById('showAllIterates');
			    const hideAllIteratesButton = document.getElementById('hideAllIterates');
				    const toggleGrazingModeButton = document.getElementById('toggleGrazingMode');
				    const toggleBlackContoursButton = document.getElementById('toggleBlackContours');
				    const toggleGreenContoursButton = document.getElementById('toggleGreenContours');
				    const toggleRedContoursButton = document.getElementById('toggleRedContours');
				    const toggleBlueContoursButton = document.getElementById('toggleGreyContours');
				    const togglePurpleContoursButton = document.getElementById('togglePurpleContours');

			    const state = {
			      view: { a0: CONFIG.alphaMin, a1: CONFIG.alphaMax, l0: CONFIG.lambdaMin, l1: CONFIG.lambdaMax },
			      hover: null,
			      selected: null,
			      dragging: null,
				      grazingMode: 'symbolic',
				      visibleIterates: new Set([2, 3, 4, 5, 6, 7, 8]),
				      showBlackContours: true,
				      showGreenContours: true,
				      showRedContours: true,
				      showBlueContours: true,
				      showPurpleContours: true
				    };

	    function currentSegmentSets() {
	      return state.grazingMode === 'symbolic' ? {
	        black: symbolicBlackSegmentsByIter,
	        red: symbolicRedSegmentsByIter,
	        blue: symbolicBlueSegmentsByIter,
	        purple: symbolicPurpleSegmentsByIter,
	        green: symbolicGreenSegmentsByIter
	      } : {
	        black: returntimeBlackSegmentsByIter,
	        red: returntimeRedSegmentsByIter,
	        blue: returntimeBlueSegmentsByIter,
	        purple: returntimePurpleSegmentsByIter,
	        green: returntimeGreenSegmentsByIter
	      };
	    }

    function cssRect() {
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      return { x: 70, y: 20, w: Math.max(50, w - 90), h: Math.max(50, h - 60) };
    }

    function resizeCanvases() {
      const dpr = window.devicePixelRatio || 1;
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      for (const canvas of [baseCanvas, overlayCanvas]) {
        canvas.width = Math.max(1, Math.floor(w * dpr));
        canvas.height = Math.max(1, Math.floor(h * dpr));
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
      }
      baseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawBase();
      drawOverlay();
    }

    function alphaStep() {
      return (CONFIG.alphaMax - CONFIG.alphaMin) / (CONFIG.nAlpha - 1);
    }

    function lambdaStep() {
      return (CONFIG.lambdaMax - CONFIG.lambdaMin) / (CONFIG.nLambda - 1);
    }

    function alphaAt(i) {
      return CONFIG.alphaMin + i * alphaStep();
    }

    function lambdaAt(j) {
      return CONFIG.lambdaMin + j * lambdaStep();
    }

    function pointIndex(i, j) {
      return j * CONFIG.nAlpha + i;
    }

    function plotX(alpha) {
      const r = cssRect();
      return r.x + (alpha - state.view.a0) * r.w / (state.view.a1 - state.view.a0);
    }

    function plotY(lambda) {
      const r = cssRect();
      return r.y + r.h - (lambda - state.view.l0) * r.h / (state.view.l1 - state.view.l0);
    }

    function screenToData(clientX, clientY) {
      const bounds = overlayCanvas.getBoundingClientRect();
      const x = clientX - bounds.left;
      const y = clientY - bounds.top;
      const r = cssRect();
      if (x < r.x || x > r.x + r.w || y < r.y || y > r.y + r.h) return null;
      const alpha = state.view.a0 + (x - r.x) * (state.view.a1 - state.view.a0) / r.w;
      const lambda = state.view.l0 + (r.h - (y - r.y)) * (state.view.l1 - state.view.l0) / r.h;
      return { alpha, lambda, x, y };
    }

    function clampInt(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function sampleNearestPoint(alpha, lambda) {
      const i = clampInt(Math.round((alpha - CONFIG.alphaMin) / alphaStep()), 0, CONFIG.nAlpha - 1);
      const j = clampInt(Math.round((lambda - CONFIG.lambdaMin) / lambdaStep()), 0, CONFIG.nLambda - 1);
      const idx = pointIndex(i, j);
	      return {
	        i,
	        j,
	        idx,
	        alpha: alphaAt(i),
	        lambda: lambdaAt(j),
	        rawSignWord: rawSignWords[idx],
	        monotoneSignWord: monotoneSignWords[idx],
	        skipWord: skipWords[idx]
	      };
	    }

    function decodeSignWord(word) {
      const result = [];
      for (let nominal = $(TABLE_ITERATE_START_046); nominal <= $(TABLE_ITERATE_END_046); nominal += 1) {
        const code = (word >>> (2 * (nominal - $(TABLE_ITERATE_START_046)))) & 0x3;
        result.push({ nominal, code });
      }
      return result;
    }

    function codeText(code) {
      return code === 2 ? '+' : code === 1 ? '-' : '·';
    }

    function codeClass(code) {
      return code === 2 ? 'pos' : code === 1 ? 'neg' : 'missing';
    }

    function setInfo(target, point, emptyText) {
      if (!point) {
        target.innerHTML = '<div class="small">' + emptyText + '</div>';
        return;
      }
	      target.innerHTML = [
	        ['alpha', point.alpha.toFixed(6)],
	        ['lambda', point.lambda.toFixed(6)],
	        ['grid index', '(' + point.i + ', ' + point.j + ')'],
	        ['flat index', String(point.idx)],
		        ['raw sign word', '0x' + point.rawSignWord.toString(16).padStart(8, '0')],
		        ['monotone sign word', '0x' + point.monotoneSignWord.toString(16).padStart(8, '0')]
	      ].map(function(pair) {
	        return '<div class="label">' + pair[0] + '</div><div class="mono">' + pair[1] + '</div>';
	      }).join('');
    }

    function decodeSkipWord(word) {
      const result = [];
      for (let nominal = $(TABLE_ITERATE_START_046); nominal <= $(TABLE_ITERATE_END_046); nominal += 1) {
        const skip = nominal <= 8 ? !!(word & (1 << (nominal - 2))) : false;
        result.push({ nominal, skip });
      }
      return result;
    }

    function decodeTimeForPoint(pointIdx, nominalIterate) {
      if (!timeWords || timeWords.length === 0) return null;
      const word = timeWords[pointIdx * $(TABLE_ITERATE_COUNT_046) + (nominalIterate - $(TABLE_ITERATE_START_046))];
      if (word === undefined || word === 0xffff) return null;
      return word / TIME_SCALE;
    }

    function formatTimeValue(value) {
      if (value === null) return '·';
      if (TIME_SCALE >= 100) return value.toFixed(2);
      if (TIME_SCALE >= 10) return value.toFixed(1);
      return value.toFixed(0);
    }

    function setPointTable(target, point, emptyText) {
      if (!point) {
        target.innerHTML = '<tr><td colspan="5" class="small">' + emptyText + '</td></tr>';
        return;
      }
	      const rawSigns = decodeSignWord(point.rawSignWord);
	      const monotoneSigns = decodeSignWord(point.monotoneSignWord);
	      const skips = decodeSkipWord(point.skipWord);
	      const rows = [];
	      for (let idx = 0; idx < rawSigns.length; idx += 1) {
	        const rawEntry = rawSigns[idx];
	        const monotoneEntry = monotoneSigns[idx];
	        const skipEntry = skips[idx];
	        const timeValue = formatTimeValue(decodeTimeForPoint(point.idx, rawEntry.nominal));
	        const rowClasses = ['iter-row', skipEntry.skip ? 'skip' : 'normal'];
	        if (rawEntry.nominal >= 9) rowClasses.push('late');
	        rows.push(
	          '<tr class="' + rowClasses.join(' ') + '">' +
	            '<td>' + rawEntry.nominal + '</td>' +
	            '<td>' + codeText(rawEntry.code) + '</td>' +
	            '<td>' + codeText(monotoneEntry.code) + '</td>' +
	            '<td>' + timeValue + '</td>' +
	            '<td>' + (skipEntry.skip ? 'yes' : 'no') + '</td>' +
	          '</tr>'
        );
      }
      target.innerHTML = rows.join('');
    }

    function niceTickStep(span, targetTicks) {
      const raw = span / Math.max(1, targetTicks);
      const power = Math.pow(10, Math.floor(Math.log10(raw)));
      const scaled = raw / power;
      let nice = 1;
      if (scaled > 5) nice = 10;
      else if (scaled > 2) nice = 5;
      else if (scaled > 1) nice = 2;
      return nice * power;
    }

    function tickValues(min, max, targetTicks) {
      const step = niceTickStep(max - min, targetTicks);
      const first = Math.ceil(min / step) * step;
      const values = [];
      for (let value = first; value <= max + 0.5 * step; value += step) values.push(value);
      return values;
    }

    function drawAxes() {
      const r = cssRect();
      baseCtx.strokeStyle = '#000000';
      baseCtx.lineWidth = 1;
      baseCtx.strokeRect(r.x, r.y, r.w, r.h);
      baseCtx.fillStyle = '#111111';
      baseCtx.font = '12px Menlo, Consolas, monospace';

      const alphaTicks = tickValues(state.view.a0, state.view.a1, 6);
      for (const alpha of alphaTicks) {
        const x = plotX(alpha);
        baseCtx.beginPath();
        baseCtx.moveTo(x, r.y + r.h);
        baseCtx.lineTo(x, r.y + r.h + 5);
        baseCtx.stroke();
        baseCtx.fillText(alpha.toFixed(3), x - 14, r.y + r.h + 18);
      }

      const lambdaTicks = tickValues(state.view.l0, state.view.l1, 7);
      for (const lambda of lambdaTicks) {
        const y = plotY(lambda);
        baseCtx.beginPath();
        baseCtx.moveTo(r.x - 5, y);
        baseCtx.lineTo(r.x, y);
        baseCtx.stroke();
        baseCtx.fillText(lambda.toFixed(3), 8, y + 4);
      }

      baseCtx.fillText('alpha', r.x + r.w / 2 - 16, r.y + r.h + 38);
      baseCtx.save();
      baseCtx.translate(18, r.y + r.h / 2 + 16);
      baseCtx.rotate(-Math.PI / 2);
      baseCtx.fillText('lambda', 0, 0);
      baseCtx.restore();
    }

    function drawSegmentArray(array, color) {
      const r = cssRect();
      const a0 = state.view.a0;
      const a1 = state.view.a1;
      const l0 = state.view.l0;
      const l1 = state.view.l1;
      baseCtx.beginPath();
      for (let idx = 0; idx < array.length; idx += 4) {
        const x1 = array[idx];
        const y1 = array[idx + 1];
        const x2 = array[idx + 2];
        const y2 = array[idx + 3];
        if ((x1 < a0 && x2 < a0) || (x1 > a1 && x2 > a1) || (y1 < l0 && y2 < l0) || (y1 > l1 && y2 > l1)) continue;
        baseCtx.moveTo(plotX(x1), plotY(y1));
        baseCtx.lineTo(plotX(x2), plotY(y2));
      }
      baseCtx.strokeStyle = color;
      baseCtx.lineWidth = 1.1;
      baseCtx.stroke();
    }

    function drawSelectedNeighborCells() {
      if (!state.selected) return;
      const cellCoords = [
        [state.selected.i - 1, state.selected.j - 1],
        [state.selected.i, state.selected.j - 1],
        [state.selected.i - 1, state.selected.j],
        [state.selected.i, state.selected.j]
      ];
      overlayCtx.save();
      overlayCtx.fillStyle = 'rgba(170, 170, 170, 0.18)';
      overlayCtx.strokeStyle = 'rgba(90, 90, 90, 0.95)';
      overlayCtx.lineWidth = 1.2;
      for (const [ci, cj] of cellCoords) {
        if (ci < 0 || cj < 0 || ci >= CONFIG.nAlpha - 1 || cj >= CONFIG.nLambda - 1) continue;
        const x0 = plotX(alphaAt(ci));
        const x1 = plotX(alphaAt(ci + 1));
        const yTop = plotY(lambdaAt(cj + 1));
        const yBot = plotY(lambdaAt(cj));
        overlayCtx.fillRect(x0, yTop, x1 - x0, yBot - yTop);
        overlayCtx.strokeRect(x0, yTop, x1 - x0, yBot - yTop);
      }
      overlayCtx.restore();
    }

    function drawBase() {
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      baseCtx.clearRect(0, 0, w, h);
      baseCtx.fillStyle = '#ffffff';
      baseCtx.fillRect(0, 0, w, h);
      const r = cssRect();
      const segments = currentSegmentSets();
      baseCtx.save();
      baseCtx.beginPath();
      baseCtx.rect(r.x, r.y, r.w, r.h);
      baseCtx.clip();
				      for (let nominal = 2; nominal <= 8; nominal += 1) {
				        if (!state.visibleIterates.has(nominal)) continue;
				        if (state.showGreenContours) drawSegmentArray(segments.green[nominal], '#008000');
				        if (state.showBlackContours) drawSegmentArray(segments.black[nominal], '#000000');
				        if (state.showRedContours) drawSegmentArray(segments.red[nominal], '#c00000');
				        if (state.showBlueContours) drawSegmentArray(segments.blue[nominal], '#3b82f6');
				        if (state.showPurpleContours) drawSegmentArray(segments.purple[nominal], '#7c3aed');
				      }
      baseCtx.restore();
      drawAxes();
      updateViewInfo();
    }

    function drawPointMarker(ctx, point, color, radius, dash) {
      if (!point) return;
      const x = plotX(point.alpha);
      const y = plotY(point.lambda);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - radius - 6, y);
      ctx.lineTo(x + radius + 6, y);
      ctx.moveTo(x, y - radius - 6);
      ctx.lineTo(x, y + radius + 6);
      ctx.stroke();
      ctx.restore();
    }

    function drawOverlay() {
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      overlayCtx.clearRect(0, 0, w, h);
      drawSelectedNeighborCells();
      drawPointMarker(overlayCtx, state.hover, '#6b7280', 4, [4, 4]);
      drawPointMarker(overlayCtx, state.selected, '#0891b2', 6, null);
      setInfo(hoverInfo, state.hover, 'No point under cursor.');
      setPointTable(hoverTableBody, state.hover, 'No point under cursor.');
      setInfo(selectedInfo, state.selected, 'No point selected.');
      setPointTable(selectedTableBody, state.selected, 'No point selected.');
    }

    function updateViewInfo() {
				      const segments = currentSegmentSets();
				      let totalBlack = 0;
				      let totalRed = 0;
				      let totalBlue = 0;
				      let totalPurple = 0;
				      let totalGreen = 0;
				      for (let nominal = 2; nominal <= 8; nominal += 1) {
				        if (!state.visibleIterates.has(nominal)) continue;
				        totalBlack += segments.black[nominal].length / 4;
				        totalRed += segments.red[nominal].length / 4;
				        totalBlue += segments.blue[nominal].length / 4;
				        totalPurple += segments.purple[nominal].length / 4;
				        totalGreen += segments.green[nominal].length / 4;
				      }
			      const visibleBlack = state.showBlackContours ? totalBlack : 0;
			      const visibleGreen = state.showGreenContours ? totalGreen : 0;
			      const visibleRed = state.showRedContours ? totalRed : 0;
			      const visibleBlue = state.showBlueContours ? totalBlue : 0;
			      const visiblePurple = state.showPurpleContours ? totalPurple : 0;
				      const rows = [
				        ['alpha range', state.view.a0.toFixed(6) + ' .. ' + state.view.a1.toFixed(6)],
				        ['lambda range', state.view.l0.toFixed(6) + ' .. ' + state.view.l1.toFixed(6)],
				        ['grid', CONFIG.nAlpha + ' x ' + CONFIG.nLambda],
				        ['grazing mode', state.grazingMode === 'symbolic' ? 'symbolic deletion' : 'return-time skip'],
				        ['visible iterates', Array.from(state.visibleIterates).sort(function(a, b) { return a - b; }).join(', ') || '(none)'],
				        ['black contours', state.showBlackContours ? 'shown' : 'hidden'],
			        ['green contours', state.showGreenContours ? 'shown' : 'hidden'],
			        ['red contours', state.showRedContours ? 'shown' : 'hidden'],
			        ['blue contours', state.showBlueContours ? 'shown' : 'hidden'],
			        ['purple contours', state.showPurpleContours ? 'shown' : 'hidden'],
			        ['segments', visibleBlack.toLocaleString() + ' black, ' + visibleRed.toLocaleString() + ' red, ' + visibleBlue.toLocaleString() + ' blue, ' + visiblePurple.toLocaleString() + ' purple, ' + visibleGreen.toLocaleString() + ' green']
			      ];
      viewInfo.innerHTML = rows.map(function(pair) {
        return '<div class="label">' + pair[0] + '</div><div class="mono">' + pair[1] + '</div>';
      }).join('');
    }

    function zoomAbout(clientX, clientY, zoomFactor) {
      const data = screenToData(clientX, clientY);
      if (!data) return;
      const aSpan = state.view.a1 - state.view.a0;
      const lSpan = state.view.l1 - state.view.l0;
      const newASpan = Math.max(alphaStep() * 8, Math.min(CONFIG.alphaMax - CONFIG.alphaMin, aSpan * zoomFactor));
      const newLSpan = Math.max(lambdaStep() * 8, Math.min(CONFIG.lambdaMax - CONFIG.lambdaMin, lSpan * zoomFactor));
      const aRatio = (data.alpha - state.view.a0) / aSpan;
      const lRatio = (data.lambda - state.view.l0) / lSpan;
      state.view.a0 = data.alpha - aRatio * newASpan;
      state.view.a1 = state.view.a0 + newASpan;
      state.view.l0 = data.lambda - lRatio * newLSpan;
      state.view.l1 = state.view.l0 + newLSpan;
      clampView();
      drawBase();
      drawOverlay();
    }

    function clampView() {
      const fullA = CONFIG.alphaMax - CONFIG.alphaMin;
      const fullL = CONFIG.lambdaMax - CONFIG.lambdaMin;
      const spanA = state.view.a1 - state.view.a0;
      const spanL = state.view.l1 - state.view.l0;
      if (state.view.a0 < CONFIG.alphaMin) {
        state.view.a0 = CONFIG.alphaMin;
        state.view.a1 = state.view.a0 + spanA;
      }
      if (state.view.a1 > CONFIG.alphaMax) {
        state.view.a1 = CONFIG.alphaMax;
        state.view.a0 = state.view.a1 - spanA;
      }
      if (state.view.l0 < CONFIG.lambdaMin) {
        state.view.l0 = CONFIG.lambdaMin;
        state.view.l1 = state.view.l0 + spanL;
      }
      if (state.view.l1 > CONFIG.lambdaMax) {
        state.view.l1 = CONFIG.lambdaMax;
        state.view.l0 = state.view.l1 - spanL;
      }
      if (spanA >= fullA) { state.view.a0 = CONFIG.alphaMin; state.view.a1 = CONFIG.alphaMax; }
      if (spanL >= fullL) { state.view.l0 = CONFIG.lambdaMin; state.view.l1 = CONFIG.lambdaMax; }
    }

    overlayCanvas.addEventListener('mousemove', function(event) {
      if (state.dragging) {
        const r = cssRect();
        const dx = event.clientX - state.dragging.clientX;
        const dy = event.clientY - state.dragging.clientY;
        const aShift = -dx * (state.view.a1 - state.view.a0) / r.w;
        const lShift = dy * (state.view.l1 - state.view.l0) / r.h;
        state.view.a0 = state.dragging.view.a0 + aShift;
        state.view.a1 = state.dragging.view.a1 + aShift;
        state.view.l0 = state.dragging.view.l0 + lShift;
        state.view.l1 = state.dragging.view.l1 + lShift;
        clampView();
        drawBase();
        drawOverlay();
        return;
      }
      const data = screenToData(event.clientX, event.clientY);
      state.hover = data ? sampleNearestPoint(data.alpha, data.lambda) : null;
      drawOverlay();
    });

    overlayCanvas.addEventListener('mouseleave', function() {
      if (!state.dragging) {
        state.hover = null;
        drawOverlay();
      }
    });

    overlayCanvas.addEventListener('mousedown', function(event) {
      if (event.button !== 0) return;
      state.dragging = {
        clientX: event.clientX,
        clientY: event.clientY,
        view: { a0: state.view.a0, a1: state.view.a1, l0: state.view.l0, l1: state.view.l1 }
      };
    });

    window.addEventListener('mouseup', function(event) {
      if (!state.dragging) return;
      const moved = Math.hypot(event.clientX - state.dragging.clientX, event.clientY - state.dragging.clientY);
      const dragState = state.dragging;
      state.dragging = null;
      if (moved < 4) {
        const data = screenToData(event.clientX, event.clientY);
        state.selected = data ? sampleNearestPoint(data.alpha, data.lambda) : null;
        drawOverlay();
      }
    });

    overlayCanvas.addEventListener('wheel', function(event) {
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 0.85 : 1.18;
      zoomAbout(event.clientX, event.clientY, zoomFactor);
    }, { passive: false });

    resetViewButton.addEventListener('click', function() {
      state.view = { a0: CONFIG.alphaMin, a1: CONFIG.alphaMax, l0: CONFIG.lambdaMin, l1: CONFIG.lambdaMax };
      drawBase();
      drawOverlay();
    });

    clearSelectionButton.addEventListener('click', function() {
      state.selected = null;
      drawOverlay();
    });

    function renderIterateControls() {
      const html = [];
      for (let nominal = 2; nominal <= 8; nominal += 1) {
        const checked = state.visibleIterates.has(nominal) ? 'checked' : '';
        html.push(
          '<label><input type="checkbox" class="iterateToggle" data-iterate="' + nominal + '" ' + checked + '>k=' + nominal + '</label>'
        );
      }
      iterateControls.innerHTML = html.join('');
      for (const input of iterateControls.querySelectorAll('.iterateToggle')) {
        input.addEventListener('change', function(event) {
          const nominal = Number(event.target.getAttribute('data-iterate'));
          if (event.target.checked) state.visibleIterates.add(nominal);
          else state.visibleIterates.delete(nominal);
          drawBase();
          drawOverlay();
        });
      }
    }

    showAllIteratesButton.addEventListener('click', function() {
      state.visibleIterates = new Set([2, 3, 4, 5, 6, 7, 8]);
      renderIterateControls();
      drawBase();
      drawOverlay();
    });

    hideAllIteratesButton.addEventListener('click', function() {
      state.visibleIterates = new Set();
      renderIterateControls();
      drawBase();
      drawOverlay();
    });

			    function updateBlackToggleButton() {
			      toggleBlackContoursButton.textContent = state.showBlackContours ? 'Hide Black' : 'Show Black';
			    }

			    function updateGrazingModeButton() {
			      toggleGrazingModeButton.textContent =
			        state.grazingMode === 'symbolic' ? 'Grazing: Symbolic' : 'Grazing: Return-Time';
			    }

		    function updateGreenToggleButton() {
		      toggleGreenContoursButton.textContent = state.showGreenContours ? 'Hide Green' : 'Show Green';
		    }

		    function updateRedToggleButton() {
		      toggleRedContoursButton.textContent = state.showRedContours ? 'Hide Red' : 'Show Red';
		    }

		    function updateBlueToggleButton() {
		      toggleBlueContoursButton.textContent = state.showBlueContours ? 'Hide Blue' : 'Show Blue';
		    }

		    function updatePurpleToggleButton() {
		      togglePurpleContoursButton.textContent = state.showPurpleContours ? 'Hide Purple' : 'Show Purple';
		    }

			    toggleBlackContoursButton.addEventListener('click', function() {
			      state.showBlackContours = !state.showBlackContours;
			      updateBlackToggleButton();
			      drawBase();
			      drawOverlay();
			    });

			    toggleGrazingModeButton.addEventListener('click', function() {
			      state.grazingMode = state.grazingMode === 'symbolic' ? 'returntime' : 'symbolic';
			      updateGrazingModeButton();
			      drawBase();
			      drawOverlay();
			    });

		    toggleGreenContoursButton.addEventListener('click', function() {
		      state.showGreenContours = !state.showGreenContours;
		      updateGreenToggleButton();
		      drawBase();
		      drawOverlay();
		    });

		    toggleRedContoursButton.addEventListener('click', function() {
		      state.showRedContours = !state.showRedContours;
		      updateRedToggleButton();
		      drawBase();
		      drawOverlay();
		    });

		    toggleBlueContoursButton.addEventListener('click', function() {
		      state.showBlueContours = !state.showBlueContours;
		      updateBlueToggleButton();
		      drawBase();
		      drawOverlay();
		    });

		    togglePurpleContoursButton.addEventListener('click', function() {
		      state.showPurpleContours = !state.showPurpleContours;
		      updatePurpleToggleButton();
		      drawBase();
		      drawOverlay();
		    });

			    window.addEventListener('resize', resizeCanvases);
			    renderIterateControls();
			    updateGrazingModeButton();
			    updateBlackToggleButton();
		    updateGreenToggleButton();
		    updateRedToggleButton();
		    updateBlueToggleButton();
		    updatePurpleToggleButton();
		    decodeGzipUint16Array(TIME_WORDS_GZ_B64)
      .then(function(words) {
        timeWords = words;
        drawOverlay();
      })
      .catch(function(error) {
        console.error('Failed to decode time payload:', error);
        timeWords = new Uint16Array(0);
        drawOverlay();
      });
    resizeCanvases();
  </script>
</body>
</html>
""")
    end
end

function main()
    println("Building attempt-046 interactive explorer from saved attempt-027 sweep.")
    println("Source columns: $(A27.SWEEP_DIR_025)")
    flush(stdout)

    raw_sign_words, monotone_sign_words = build_sign_words_043()
    println("Packed raw and monotone sign words for $(length(raw_sign_words)) sampled grid points.")
    flush(stdout)

    dot_grids, time_grids = build_grids_043()
    time_words, time_scale = build_time_words_043(time_grids)
    println("Packed $(length(time_words)) quantized return-time words at scale $(time_scale).")
    flush(stdout)

    symbolic_black_segments_by_iter,
    symbolic_red_segments_by_iter,
    symbolic_blue_segments_by_iter,
    symbolic_purple_segments_by_iter,
    symbolic_green_segments_by_iter,
    point_skip_masks,
    iterate_stats =
        collect_sequence_classified_segments_046(dot_grids, time_grids, :symbolic)
    symbolic_total_black = sum(length, symbolic_black_segments_by_iter)
    symbolic_total_red = sum(length, symbolic_red_segments_by_iter)
    symbolic_total_blue = sum(length, symbolic_blue_segments_by_iter)
    symbolic_total_purple = sum(length, symbolic_purple_segments_by_iter)
    symbolic_total_green = sum(length, symbolic_green_segments_by_iter)
    println("Collected symbolic-mode segments: $(symbolic_total_black) black, $(symbolic_total_red) red, $(symbolic_total_blue) blue, $(symbolic_total_purple) purple, $(symbolic_total_green) green.")
    flush(stdout)

    returntime_black_segments_by_iter,
    returntime_red_segments_by_iter,
    returntime_blue_segments_by_iter,
    returntime_purple_segments_by_iter,
    returntime_green_segments_by_iter,
    _,
    _ =
        collect_sequence_classified_segments_046(dot_grids, time_grids, :returntime)
    returntime_total_black = sum(length, returntime_black_segments_by_iter)
    returntime_total_red = sum(length, returntime_red_segments_by_iter)
    returntime_total_blue = sum(length, returntime_blue_segments_by_iter)
    returntime_total_purple = sum(length, returntime_purple_segments_by_iter)
    returntime_total_green = sum(length, returntime_green_segments_by_iter)
    println("Collected return-time mode segments: $(returntime_total_black) black, $(returntime_total_red) red, $(returntime_total_blue) blue, $(returntime_total_purple) purple, $(returntime_total_green) green.")
    flush(stdout)

    symbolic_black_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    symbolic_red_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    symbolic_blue_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    symbolic_purple_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    symbolic_green_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    returntime_black_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    returntime_red_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    returntime_blue_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    returntime_purple_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    returntime_green_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    for nominal_iterate in 2:min(8, A27.ATTEMPT025_PLOT_ITERATE_CAP)
        symbolic_black_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(symbolic_black_segments_by_iter[nominal_iterate]))
        symbolic_red_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(symbolic_red_segments_by_iter[nominal_iterate]))
        symbolic_blue_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(symbolic_blue_segments_by_iter[nominal_iterate]))
        symbolic_purple_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(symbolic_purple_segments_by_iter[nominal_iterate]))
        symbolic_green_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(symbolic_green_segments_by_iter[nominal_iterate]))
        returntime_black_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(returntime_black_segments_by_iter[nominal_iterate]))
        returntime_red_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(returntime_red_segments_by_iter[nominal_iterate]))
        returntime_blue_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(returntime_blue_segments_by_iter[nominal_iterate]))
        returntime_purple_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(returntime_purple_segments_by_iter[nominal_iterate]))
        returntime_green_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(returntime_green_segments_by_iter[nominal_iterate]))
    end
    raw_sign_blob = base64_bytes_043(raw_sign_words)
    monotone_sign_blob = base64_bytes_043(monotone_sign_words)
    skip_blob = base64_bytes_043(point_skip_masks)
    time_blob = base64_gzip_bytes_043(time_words)

    write_iterate_stats_043(STATS_PATH_043, iterate_stats)
    write_html_043(
        HTML_PATH_043,
        symbolic_black_blobs,
        symbolic_red_blobs,
        symbolic_blue_blobs,
        symbolic_purple_blobs,
        symbolic_green_blobs,
        returntime_black_blobs,
        returntime_red_blobs,
        returntime_blue_blobs,
        returntime_purple_blobs,
        returntime_green_blobs,
        raw_sign_blob,
        monotone_sign_blob,
        skip_blob,
        time_blob,
        time_scale,
    )

    println("Saved iterate stats to $(STATS_PATH_043)")
    println("Saved explorer HTML to $(HTML_PATH_043)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
