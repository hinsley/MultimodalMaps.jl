using Pkg

const ATTEMPT12_ROOT = @__DIR__
const ATTEMPT11_ROOT = normpath(joinpath(ATTEMPT12_ROOT, "..", "attempt-011"))
const REPO_ROOT_012 = normpath(joinpath(ATTEMPT12_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_012)

include(joinpath(ATTEMPT11_ROOT, "inspect_return_section.jl"))

using Base.Threads
using CairoMakie
using Colors
using Printf
using Statistics

const COLUMN_DELTA_CA = parse(Float64, get(ENV, "ATTEMPT012_COLUMN_DELTA_CA", "-38.386774"))
const COLUMN_POINT_COUNT = parse(Int, get(ENV, "ATTEMPT012_COLUMN_POINTS", "80"))
const COLUMN_DELTA_X_MAX = parse(Float64, get(ENV, "ATTEMPT012_COLUMN_DELTA_X_MAX", "-0.5"))
const COLUMN_DELTA_X_MIN = parse(Float64, get(ENV, "ATTEMPT012_COLUMN_DELTA_X_MIN", "-1.5"))

const PATCH_N_CA = parse(Int, get(ENV, "ATTEMPT012_PATCH_N_CA", "10"))
const PATCH_N_X = parse(Int, get(ENV, "ATTEMPT012_PATCH_N_X", "10"))
const PATCH_DELTA_CA_MIN = parse(Float64, get(ENV, "ATTEMPT012_PATCH_DELTA_CA_MIN", "-43.95"))
const PATCH_DELTA_CA_MAX = parse(Float64, get(ENV, "ATTEMPT012_PATCH_DELTA_CA_MAX", "-36.43"))
const PATCH_DELTA_X_MIN = parse(Float64, get(ENV, "ATTEMPT012_PATCH_DELTA_X_MIN", "-1.50"))
const PATCH_DELTA_X_MAX = parse(Float64, get(ENV, "ATTEMPT012_PATCH_DELTA_X_MAX", "-0.96"))

const MINIMAL_SAVEAT = parse(Float64, get(ENV, "ATTEMPT012_MINIMAL_SAVEAT", string(SECTION_TSPAN[2])))
const FAST_ABSTOL = parse(Float64, get(ENV, "ATTEMPT012_FAST_ABSTOL", "1e-8"))
const FAST_RELTOL = parse(Float64, get(ENV, "ATTEMPT012_FAST_RELTOL", "1e-8"))
const GOLD_ABSTOL = parse(Float64, get(ENV, "ATTEMPT012_GOLD_ABSTOL", "1e-8"))
const GOLD_RELTOL = parse(Float64, get(ENV, "ATTEMPT012_GOLD_RELTOL", "1e-8"))
const FAST_H_MIN = parse(Float64, get(ENV, "ATTEMPT012_H_MIN", "2e-4"))
const FAST_H_MAX = parse(Float64, get(ENV, "ATTEMPT012_H_MAX", "2e-2"))
const FAST_IMAGE_ABS_TOL = parse(Float64, get(ENV, "ATTEMPT012_IMAGE_ABS_TOL", "5e-4"))
const FAST_IMAGE_REL_FACTOR = parse(Float64, get(ENV, "ATTEMPT012_IMAGE_REL_FACTOR", "0.25"))
const LOCAL_SCAN_POINTS_1 = parse(Int, get(ENV, "ATTEMPT012_LOCAL_SCAN_POINTS_1", "9"))
const LOCAL_SCAN_POINTS_2 = parse(Int, get(ENV, "ATTEMPT012_LOCAL_SCAN_POINTS_2", "17"))
const GLOBAL_SCAN_POINTS = parse(Int, get(ENV, "ATTEMPT012_GLOBAL_SCAN_POINTS", "121"))
const GLOBAL_REFINE_ITERS = parse(Int, get(ENV, "ATTEMPT012_GLOBAL_REFINE_ITERS", "3"))
const LOCAL_REFINE_ITERS = parse(Int, get(ENV, "ATTEMPT012_LOCAL_REFINE_ITERS", "1"))
const PROOF_PX_PER_UNIT = parse(Float64, get(ENV, "ATTEMPT012_PX_PER_UNIT", "2.0"))

const FAIL_MASK_COLOR = RGBAf(0.22, 0.22, 0.22, 0.25)
const T_FAST_COLOR = RGBAf(0.86, 0.16, 0.12, 0.85)
const T_GOLD_COLOR = RGBAf(0.55, 0.0, 0.0, 0.95)
const MODE_COLORS = Dict(
    "predict" => RGBAf(0.10, 0.55, 0.18, 0.95),
    "fallback_local" => RGBAf(0.92, 0.56, 0.05, 0.95),
    "fallback_global" => RGBAf(0.12, 0.32, 0.82, 0.95),
    "failed" => RGBAf(0.55, 0.0, 0.0, 0.95),
)

const COLUMN_DELTA_XS = collect(range(COLUMN_DELTA_X_MAX, COLUMN_DELTA_X_MIN, length=COLUMN_POINT_COUNT))
const PATCH_DELTA_CAS = collect(range(PATCH_DELTA_CA_MIN, PATCH_DELTA_CA_MAX, length=PATCH_N_CA))
const PATCH_DELTA_XS = collect(range(PATCH_DELTA_X_MAX, PATCH_DELTA_X_MIN, length=PATCH_N_X))

Base.@kwdef struct SectionReturnEval
    s::Float64
    ok::Bool = false
    spike_count::Int = 0
    return_s::Float64 = NaN
    end_u::Union{Nothing, SVector{6, Float64}} = nothing
end

Base.@kwdef struct T0SolveResult
    success::Bool = false
    s_pre::Float64 = NaN
    s_img::Float64 = NaN
    T0::Union{Nothing, SVector{6, Float64}} = nothing
    mode::String = "failed"
    return_solves::Int = 0
    elapsed_seconds::Float64 = 0.0
    message::String = ""
end

Base.@kwdef struct ProofPointResult
    delta_x::Float64
    delta_ca::Float64
    fast::T0SolveResult
    gold::T0SolveResult
    gamma_scs::Vector{Int}
    fast_T_scs::Vector{Int}
    gold_T_scs::Vector{Int}
end

function format_sequence(seq::Vector{Int})
    isempty(seq) && return ""
    return join(seq, ",")
end

function mode_color(mode::String)
    return get(MODE_COLORS, mode, RGBAf(0.3, 0.3, 0.3, 1.0))
end

function fixed_delta_xs()
    return COLUMN_DELTA_XS
end

function section_return_eval(
    p,
    section,
    s::Float64;
    abstol::Float64,
    reltol::Float64,
)::SectionReturnEval
    sol, target_phase, spike_count = solve_section_from_s(
        p,
        section,
        s;
        saveat=MINIMAL_SAVEAT,
        abstol=abstol,
        reltol=reltol,
    )
    if !returned_to_section(sol, target_phase)
        return SectionReturnEval(s=s)
    end
    return SectionReturnEval(
        s=s,
        ok=true,
        spike_count=spike_count,
        return_s=ray_coordinate(sol.u[end], section),
        end_u=state6(sol.u[end]),
    )
end

function local_minimum_indices(evals::Vector{SectionReturnEval}; spike_count::Int=1)
    idxs = Int[]
    for idx in 2:(length(evals) - 1)
        left = evals[idx - 1]
        mid = evals[idx]
        right = evals[idx + 1]
        if left.ok &&
           mid.ok &&
           right.ok &&
           left.spike_count == spike_count &&
           mid.spike_count == spike_count &&
           right.spike_count == spike_count &&
           left.return_s > mid.return_s < right.return_s
            push!(idxs, idx)
        end
    end
    return idxs
end

function choose_candidate_index(
    evals::Vector{SectionReturnEval},
    candidates::Vector{Int};
    hint_s::Union{Nothing, Float64}=nothing,
)
    isempty(candidates) && return nothing
    if isnothing(hint_s)
        return candidates[1]
    end
    _, best_pos = findmin(abs.(getfield.(evals[candidates], :s) .- hint_s))
    return candidates[best_pos]
end

function secant_predict(x0::Float64, y0::Float64, x1::Float64, y1::Float64, x::Float64)
    if !isfinite(y0) || !isfinite(y1) || x1 == x0
        return y1
    end
    return y1 + (x - x1) * (y1 - y0) / (x1 - x0)
end

function refinement_candidate(
    left::SectionReturnEval,
    mid::SectionReturnEval,
    right::SectionReturnEval,
)
    x1, f1 = left.s, left.return_s
    x2, f2 = mid.s, mid.return_s
    x3, f3 = right.s, right.return_s
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1.0e-14
        return 0.5 * (left.s + right.s)
    end
    a = (x3 * (f2 - f1) + x2 * (f1 - f3) + x1 * (f3 - f2)) / denom
    b = (x3^2 * (f1 - f2) + x2^2 * (f3 - f1) + x1^2 * (f2 - f3)) / denom
    if !isfinite(a) || abs(a) < 1.0e-14
        return 0.5 * (left.s + right.s)
    end
    s_new = -b / (2a)
    if !isfinite(s_new) || !(left.s < s_new < right.s)
        return 0.5 * (left.s + right.s)
    end
    return s_new
end

function refine_local_minimum(
    p,
    section,
    left::SectionReturnEval,
    mid::SectionReturnEval,
    right::SectionReturnEval;
    max_iters::Int,
    abstol::Float64,
    reltol::Float64,
)
    solves = 0
    current_left = left
    current_mid = mid
    current_right = right
    for _ in 1:max_iters
        s_new = clamp(refinement_candidate(current_left, current_mid, current_right), current_left.s, current_right.s)
        if !(current_left.s < s_new < current_right.s)
            break
        end
        trial = section_return_eval(p, section, s_new; abstol=abstol, reltol=reltol)
        solves += 1
        if !trial.ok || trial.spike_count != 1
            break
        end
        candidates = sort([current_left, current_mid, current_right, trial]; by=eval -> eval.s)
        minima = local_minimum_indices(candidates)
        if isempty(minima)
            break
        end
        best_idx = minima[1]
        current_left = candidates[best_idx - 1]
        current_mid = candidates[best_idx]
        current_right = candidates[best_idx + 1]
    end
    return current_mid, solves
end

function global_t0_solve_impl(
    p,
    section;
    hint_s::Union{Nothing, Float64}=nothing,
)::T0SolveResult
    solve_count = 0
    ss = collect(range(SECTION_S_MIN, SECTION_S_MAX, length=GLOBAL_SCAN_POINTS))
    evals = SectionReturnEval[]
    sizehint!(evals, length(ss))
    for s in ss
        push!(evals, section_return_eval(p, section, s; abstol=GOLD_ABSTOL, reltol=GOLD_RELTOL))
    end
    solve_count += length(ss)

    candidates = local_minimum_indices(evals)
    candidate_idx = choose_candidate_index(evals, candidates; hint_s=hint_s)
    if isnothing(candidate_idx)
        return T0SolveResult(
            success=false,
            mode="failed",
            return_solves=solve_count,
            message="No 1-spike local minimum found on the global section scan.",
        )
    end

    refined, extra_solves = refine_local_minimum(
        p,
        section,
        evals[candidate_idx - 1],
        evals[candidate_idx],
        evals[candidate_idx + 1];
        max_iters=GLOBAL_REFINE_ITERS,
        abstol=GOLD_ABSTOL,
        reltol=GOLD_RELTOL,
    )
    solve_count += extra_solves
    return T0SolveResult(
        success=true,
        s_pre=refined.s,
        s_img=refined.return_s,
        T0=lift_section_point(p, section, refined.return_s),
        mode="fallback_global",
        return_solves=solve_count,
    )
end

function global_t0_solve(
    p,
    section;
    hint_s::Union{Nothing, Float64}=nothing,
)::T0SolveResult
    result = Ref{T0SolveResult}()
    elapsed = @elapsed begin
        result[] = global_t0_solve_impl(p, section; hint_s=hint_s)
    end
    base = result[]
    return T0SolveResult(;
        success=base.success,
        s_pre=base.s_pre,
        s_img=base.s_img,
        T0=base.T0,
        mode=base.mode,
        return_solves=base.return_solves,
        elapsed_seconds=elapsed,
        message=base.message,
    )
end

function local_window_scan(
    p,
    section,
    s_pred::Float64,
    h::Float64;
    hint_s::Float64,
)::Tuple{Union{Nothing, T0SolveResult}, Int}
    windows = (
        (points=LOCAL_SCAN_POINTS_1, scale=3.0),
        (points=LOCAL_SCAN_POINTS_2, scale=6.0),
    )
    total_solves = 0
    for (window_idx, window) in enumerate(windows)
        left = max(SECTION_S_MIN, s_pred - window.scale * h)
        right = min(SECTION_S_MAX, s_pred + window.scale * h)
        if !(left < right)
            continue
        end
        ss = collect(range(left, right, length=window.points))
        evals = SectionReturnEval[]
        sizehint!(evals, length(ss))
        for s in ss
            push!(evals, section_return_eval(p, section, s; abstol=FAST_ABSTOL, reltol=FAST_RELTOL))
        end
        total_solves += length(ss)
        candidates = local_minimum_indices(evals)
        candidate_idx = choose_candidate_index(evals, candidates; hint_s=hint_s)
        if isnothing(candidate_idx)
            continue
        end
        refine_iters = window_idx == 1 ? LOCAL_REFINE_ITERS : 0
        refined, extra_solves = refine_local_minimum(
            p,
            section,
            evals[candidate_idx - 1],
            evals[candidate_idx],
            evals[candidate_idx + 1];
            max_iters=refine_iters,
            abstol=FAST_ABSTOL,
            reltol=FAST_RELTOL,
        )
        total_solves += extra_solves
        if refined.ok && refined.spike_count == 1
            return (
                T0SolveResult(
                    success=true,
                    s_pre=refined.s,
                    s_img=refined.return_s,
                    T0=lift_section_point(p, section, refined.return_s),
                    mode="fallback_local",
                    return_solves=total_solves,
                ),
                total_solves,
            )
        end
    end
    return nothing, total_solves
end

function fast_t0_solve_impl(
    p,
    section,
    delta_x::Float64,
    prev2_delta_x::Float64,
    prev2::T0SolveResult,
    prev1_delta_x::Float64,
    prev1::T0SolveResult,
)::T0SolveResult
    solve_count = 0
    s_pred = clamp(secant_predict(prev2_delta_x, prev2.s_pre, prev1_delta_x, prev1.s_pre, delta_x), SECTION_S_MIN, SECTION_S_MAX)
    R_pred = secant_predict(prev2_delta_x, prev2.s_img, prev1_delta_x, prev1.s_img, delta_x)
    h = clamp(0.5 * abs(prev1.s_pre - prev2.s_pre), FAST_H_MIN, FAST_H_MAX)

    s_triplet = sort(unique([
        clamp(s_pred - h, SECTION_S_MIN, SECTION_S_MAX),
        clamp(s_pred, SECTION_S_MIN, SECTION_S_MAX),
        clamp(s_pred + h, SECTION_S_MIN, SECTION_S_MAX),
    ]))

    if length(s_triplet) == 3
        evals = [section_return_eval(p, section, s; abstol=FAST_ABSTOL, reltol=FAST_RELTOL) for s in s_triplet]
        solve_count += 3
        left, mid, right = evals
        continuity_tol = max(FAST_IMAGE_ABS_TOL, FAST_IMAGE_REL_FACTOR * abs(prev1.s_img - prev2.s_img))
        if left.ok &&
           mid.ok &&
           right.ok &&
           left.spike_count == 1 &&
           mid.spike_count == 1 &&
           right.spike_count == 1 &&
           left.return_s > mid.return_s < right.return_s &&
           abs(mid.return_s - R_pred) <= continuity_tol
            return T0SolveResult(
                success=true,
                s_pre=mid.s,
                s_img=mid.return_s,
                T0=lift_section_point(p, section, mid.return_s),
                mode="predict",
                return_solves=solve_count,
            )
        end
    end

    local_result, extra_solves = local_window_scan(p, section, s_pred, h; hint_s=s_pred)
    solve_count += extra_solves
    if !isnothing(local_result)
        return T0SolveResult(
            success=true,
            s_pre=local_result.s_pre,
            s_img=local_result.s_img,
            T0=local_result.T0,
            mode="fallback_local",
            return_solves=solve_count,
        )
    end

    gold_result = global_t0_solve(p, section; hint_s=s_pred)
    return T0SolveResult(
        success=gold_result.success,
        s_pre=gold_result.s_pre,
        s_img=gold_result.s_img,
        T0=gold_result.T0,
        mode=gold_result.success ? "fallback_global" : "failed",
        return_solves=solve_count + gold_result.return_solves,
        elapsed_seconds=gold_result.elapsed_seconds,
        message=gold_result.message,
    )
end

function fast_t0_solve(
    p,
    section,
    delta_x::Float64,
    prev2_delta_x::Float64,
    prev2::T0SolveResult,
    prev1_delta_x::Float64,
    prev1::T0SolveResult,
)::T0SolveResult
    result = Ref{T0SolveResult}()
    elapsed = @elapsed begin
        result[] = fast_t0_solve_impl(p, section, delta_x, prev2_delta_x, prev2, prev1_delta_x, prev1)
    end
    base = result[]
    total_elapsed = elapsed + base.elapsed_seconds
    return T0SolveResult(;
        success=base.success,
        s_pre=base.s_pre,
        s_img=base.s_img,
        T0=base.T0,
        mode=base.mode,
        return_solves=base.return_solves,
        elapsed_seconds=total_elapsed,
        message=base.message,
    )
end

function compute_T_scs_for_result(p, saddle_data, result::T0SolveResult)
    if !result.success || isnothing(result.T0)
        return Int[]
    end
    return compute_sscs(p, result.T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
end

function point_context(delta_x::Float64, delta_ca::Float64)
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    section = equilibrium_section_data(p)
    gamma_scs = compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)
    return p, saddle_data, section, gamma_scs
end

function solve_column(delta_ca::Float64, delta_xs::Vector{Float64})
    results = ProofPointResult[]
    sizehint!(results, length(delta_xs))
    prev_fast = T0SolveResult[]
    prev_fast_xs = Float64[]
    prev_gold_hint = nothing

    for (idx, delta_x) in enumerate(delta_xs)
        p, saddle_data, section, gamma_scs = point_context(delta_x, delta_ca)
        gold = global_t0_solve(p, section; hint_s=prev_gold_hint)
        if gold.success
            prev_gold_hint = gold.s_pre
        end

        fast =
            if length(prev_fast) < 2
                T0SolveResult(
                    success=gold.success,
                    s_pre=gold.s_pre,
                    s_img=gold.s_img,
                    T0=gold.T0,
                    mode=gold.success ? "fallback_global" : "failed",
                    return_solves=gold.return_solves,
                    elapsed_seconds=gold.elapsed_seconds,
                    message=gold.message,
                )
            else
                fast_t0_solve(p, section, delta_x, prev_fast_xs[end - 1], prev_fast[end - 1], prev_fast_xs[end], prev_fast[end])
            end

        fast_T_scs = compute_T_scs_for_result(p, saddle_data, fast)
        gold_T_scs = compute_T_scs_for_result(p, saddle_data, gold)
        point = ProofPointResult(
            delta_x=delta_x,
            delta_ca=delta_ca,
            fast=fast,
            gold=gold,
            gamma_scs=gamma_scs,
            fast_T_scs=fast_T_scs,
            gold_T_scs=gold_T_scs,
        )
        push!(results, point)

        if fast.success
            push!(prev_fast, fast)
            push!(prev_fast_xs, delta_x)
            if length(prev_fast) > 2
                popfirst!(prev_fast)
                popfirst!(prev_fast_xs)
            end
        end
    end
    return results
end

function run_patch(delta_cas::Vector{Float64}, delta_xs::Vector{Float64})
    column_results = Vector{Vector{ProofPointResult}}(undef, length(delta_cas))
    @threads for idx in eachindex(delta_cas)
        column_results[idx] = solve_column(delta_cas[idx], delta_xs)
    end
    return reduce(vcat, column_results)
end

function cell_edges(values::Vector{Float64})
    if length(values) == 1
        δ = 0.5
        return [values[1] - δ, values[1] + δ]
    end
    edges = Vector{Float64}(undef, length(values) + 1)
    edges[1] = values[1] - 0.5 * (values[2] - values[1])
    for idx in 2:length(values)
        edges[idx] = 0.5 * (values[idx - 1] + values[idx])
    end
    edges[end] = values[end] + 0.5 * (values[end] - values[end - 1])
    return edges
end

function case_segments(mask_case::Int)
    if mask_case == 1
        return ((3, 0),)
    elseif mask_case == 2
        return ((0, 1),)
    elseif mask_case == 3
        return ((3, 1),)
    elseif mask_case == 4
        return ((1, 2),)
    elseif mask_case == 5
        return ((3, 0), (1, 2))
    elseif mask_case == 6
        return ((0, 2),)
    elseif mask_case == 7
        return ((3, 2),)
    elseif mask_case == 8
        return ((2, 3),)
    elseif mask_case == 9
        return ((0, 2),)
    elseif mask_case == 10
        return ((2, 3), (0, 1))
    elseif mask_case == 11
        return ((1, 2),)
    elseif mask_case == 12
        return ((1, 3),)
    elseif mask_case == 13
        return ((0, 1),)
    elseif mask_case == 14
        return ((3, 0),)
    else
        return ()
    end
end

function edge_point(edge::Int, x0::Float64, x1::Float64, y0::Float64, y1::Float64)
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    if edge == 0
        return x0, ym
    elseif edge == 1
        return xm, y1
    elseif edge == 2
        return x1, ym
    else
        return xm, y0
    end
end

function normalize_segment(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
    if x1 < x2 || (x1 == x2 && y1 <= y2)
        return (x1, y1, x2, y2)
    else
        return (x2, y2, x1, y1)
    end
end

function push_unique_segment!(segments::Vector{NTuple{4, Float64}}, segment::NTuple{4, Float64})
    for existing in segments
        existing == segment && return
    end
    push!(segments, segment)
end

function categorical_marching_squares(grid::Matrix{Int}, x_values::Vector{Float64}, y_values::Vector{Float64})
    xs = Float32[]
    ys = Float32[]
    local_segments = NTuple{4, Float64}[]
    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
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

function assign_T_categories(results::Vector{ProofPointResult}, which::Symbol)
    sequences = Dict{String, Int}()
    next_id = 1
    grid = fill(0, PATCH_N_CA, PATCH_N_X)
    fail_mask = falses(PATCH_N_CA, PATCH_N_X)
    for result in results
        ca_idx = findmin(abs.(PATCH_DELTA_CAS .- result.delta_ca))[2]
        x_idx = findmin(abs.(PATCH_DELTA_XS .- result.delta_x))[2]
        seq = which == :fast ? result.fast_T_scs : result.gold_T_scs
        success = which == :fast ? result.fast.success : result.gold.success
        if !success || isempty(seq)
            fail_mask[ca_idx, x_idx] = true
            continue
        end
        key = format_sequence(seq)
        if !haskey(sequences, key)
            sequences[key] = next_id
            next_id += 1
        end
        grid[ca_idx, x_idx] = sequences[key]
    end
    return grid, fail_mask
end

function draw_failure_mask!(ax, xs::Vector{Float64}, ys::Vector{Float64}, fail_mask::BitMatrix)
    x_edges = cell_edges(xs)
    y_edges = cell_edges(ys)
    for x_idx in eachindex(xs), y_idx in eachindex(ys)
        if !fail_mask[x_idx, y_idx]
            continue
        end
        rect = Rect(
            Point2f(x_edges[x_idx], y_edges[y_idx]),
            Vec2f(x_edges[x_idx + 1] - x_edges[x_idx], y_edges[y_idx + 1] - y_edges[y_idx]),
        )
        poly!(ax, rect; color=FAIL_MASK_COLOR, strokewidth=0)
    end
end

function fixed_ticks(values::Vector{Float64}, fmt::String)
    ticks = unique(round.(values; digits=2))
    formatter = Printf.Format(fmt)
    labels = [Printf.format(formatter, tick) for tick in ticks]
    return (ticks, labels)
end

function save_patch_contour_comparison(results::Vector{ProofPointResult}, path::String)
    fast_grid, fast_fail = assign_T_categories(results, :fast)
    gold_grid, gold_fail = assign_T_categories(results, :gold)
    fast_xs, fast_ys = categorical_marching_squares(fast_grid, PATCH_DELTA_CAS, PATCH_DELTA_XS)
    gold_xs, gold_ys = categorical_marching_squares(gold_grid, PATCH_DELTA_CAS, PATCH_DELTA_XS)

    fig = Figure(size=(1800, 820))
    for (panel_idx, xs, ys, fail_mask, title) in (
        (1, fast_xs, fast_ys, fast_fail, "Fast Red Contours"),
        (2, gold_xs, gold_ys, gold_fail, "Gold Red Contours"),
    )
        ax = Axis(fig[1, panel_idx], xlabel="ΔCa", ylabel="Δx", title=title)
        draw_failure_mask!(ax, PATCH_DELTA_CAS, PATCH_DELTA_XS, fail_mask)
        lines!(ax, xs, ys; color=T_GOLD_COLOR, linewidth=2.5)
        ax.xticks = fixed_ticks(PATCH_DELTA_CAS, "%.2f")
        ax.yticks = fixed_ticks(PATCH_DELTA_XS, "%.2f")
    end
    save(path, fig; px_per_unit=PROOF_PX_PER_UNIT)
end

function save_column_continuation_plot(results::Vector{ProofPointResult}, path::String)
    xs = [result.delta_x for result in results]
    fast_s_pre = [result.fast.s_pre for result in results]
    gold_s_pre = [result.gold.s_pre for result in results]
    fast_s_img = [result.fast.s_img for result in results]
    gold_s_img = [result.gold.s_img for result in results]

    fig = Figure(size=(1600, 1000))
    ax1 = Axis(fig[1, 1], xlabel="Δx", ylabel="s*", title="Critical Preimage Coordinate")
    ax2 = Axis(fig[2, 1], xlabel="Δx", ylabel="R(s*)", title="Iterate-1 Sweep T0 Section Coordinate")

    lines!(ax1, xs, gold_s_pre; color=:black, linewidth=2.5, label="gold")
    lines!(ax1, xs, fast_s_pre; color=T_FAST_COLOR, linewidth=2.0, linestyle=:dash, label="fast")
    lines!(ax2, xs, gold_s_img; color=:black, linewidth=2.5, label="gold")
    lines!(ax2, xs, fast_s_img; color=T_FAST_COLOR, linewidth=2.0, linestyle=:dash, label="fast")

    for result in results
        scatter!(ax1, [result.delta_x], [result.fast.s_pre]; color=mode_color(result.fast.mode), markersize=12)
        scatter!(ax2, [result.delta_x], [result.fast.s_img]; color=mode_color(result.fast.mode), markersize=12)
    end

    Legend(
        fig[1, 2],
        [
            MarkerElement(color=mode_color("predict"), marker=:circle, markersize=14),
            MarkerElement(color=mode_color("fallback_local"), marker=:circle, markersize=14),
            MarkerElement(color=mode_color("fallback_global"), marker=:circle, markersize=14),
        ],
        ["predict", "fallback_local", "fallback_global"],
    )
    save(path, fig; px_per_unit=PROOF_PX_PER_UNIT)
end

function p95(values::Vector{Float64})
    isempty(values) && return NaN
    sorted = sort(values)
    idx = clamp(ceil(Int, 0.95 * length(sorted)), 1, length(sorted))
    return sorted[idx]
end

function summarize_results(all_results::Vector{ProofPointResult}, total_seconds::Float64)
    fast_times = [result.fast.elapsed_seconds for result in all_results if result.fast.success]
    gold_times = [result.gold.elapsed_seconds for result in all_results if result.gold.success]
    fast_predict = count(result -> result.fast.mode == "predict", all_results)
    fast_local = count(result -> result.fast.mode == "fallback_local", all_results)
    fast_global = count(result -> result.fast.mode == "fallback_global", all_results)
    pair_mismatch = count(
        result -> (result.fast.success != result.gold.success) ||
                  (result.fast.success && result.gold.success && result.fast_T_scs != result.gold_T_scs),
        all_results,
    )
    shared_failures = count(result -> !result.fast.success && !result.gold.success, all_results)
    s_diffs = [
        abs(result.fast.s_pre - result.gold.s_pre) for result in all_results
        if result.fast.success && result.gold.success
    ]
    R_diffs = [
        abs(result.fast.s_img - result.gold.s_img) for result in all_results
        if result.fast.success && result.gold.success
    ]
    return (
        total_points=length(all_results),
        total_seconds=total_seconds,
        fast_median=isempty(fast_times) ? NaN : median(fast_times),
        fast_p95=p95(fast_times),
        gold_median=isempty(gold_times) ? NaN : median(gold_times),
        gold_p95=p95(gold_times),
        predict_count=fast_predict,
        fallback_local_count=fast_local,
        fallback_global_count=fast_global,
        pair_mismatch=pair_mismatch,
        shared_failures=shared_failures,
        s_median=isempty(s_diffs) ? NaN : median(s_diffs),
        s_max=isempty(s_diffs) ? NaN : maximum(s_diffs),
        R_median=isempty(R_diffs) ? NaN : median(R_diffs),
        R_max=isempty(R_diffs) ? NaN : maximum(R_diffs),
    )
end

function write_summary(path::String, label::String, summary)
    open(path, "w") do io
        println(io, "label\t$(label)")
        println(io, "total_points\t$(summary.total_points)")
        println(io, "total_seconds\t$(summary.total_seconds)")
        println(io, "fast_median_seconds\t$(summary.fast_median)")
        println(io, "fast_p95_seconds\t$(summary.fast_p95)")
        println(io, "gold_median_seconds\t$(summary.gold_median)")
        println(io, "gold_p95_seconds\t$(summary.gold_p95)")
        println(io, "predict_count\t$(summary.predict_count)")
        println(io, "fallback_local_count\t$(summary.fallback_local_count)")
        println(io, "fallback_global_count\t$(summary.fallback_global_count)")
        println(io, "pair_mismatch_count\t$(summary.pair_mismatch)")
        println(io, "shared_failure_count\t$(summary.shared_failures)")
        println(io, "s_diff_median\t$(summary.s_median)")
        println(io, "s_diff_max\t$(summary.s_max)")
        println(io, "R_diff_median\t$(summary.R_median)")
        println(io, "R_diff_max\t$(summary.R_max)")
    end
end

function write_results_tsv(path::String, results::Vector{ProofPointResult})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tfast_success\tgold_success\tfast_mode\tgold_mode\tfast_s_pre\tfast_s_img\tfast_return_solves\tfast_elapsed\tgold_s_pre\tgold_s_img\tgold_return_solves\tgold_elapsed\tgamma_scs\tfast_T_scs\tgold_T_scs\tmatch")
        for result in results
            match = (result.fast.success == result.gold.success) &&
                    (!result.fast.success || result.fast_T_scs == result.gold_T_scs)
            println(
                io,
                join([
                    string(result.delta_x),
                    string(result.delta_ca),
                    string(result.fast.success),
                    string(result.gold.success),
                    result.fast.mode,
                    result.gold.mode,
                    string(result.fast.s_pre),
                    string(result.fast.s_img),
                    string(result.fast.return_solves),
                    string(result.fast.elapsed_seconds),
                    string(result.gold.s_pre),
                    string(result.gold.s_img),
                    string(result.gold.return_solves),
                    string(result.gold.elapsed_seconds),
                    format_sequence(result.gamma_scs),
                    format_sequence(result.fast_T_scs),
                    format_sequence(result.gold_T_scs),
                    string(match),
                ], '\t'),
            )
        end
    end
end

function main()
    column_results = ProofPointResult[]
    patch_results = ProofPointResult[]
    total_elapsed = @elapsed begin
        column_results = solve_column(COLUMN_DELTA_CA, COLUMN_DELTA_XS)
        patch_results = run_patch(PATCH_DELTA_CAS, PATCH_DELTA_XS)
    end

    all_results = vcat(column_results, patch_results)
    column_summary = summarize_results(column_results, sum(result.fast.elapsed_seconds + result.gold.elapsed_seconds for result in column_results))
    patch_summary = summarize_results(patch_results, sum(result.fast.elapsed_seconds + result.gold.elapsed_seconds for result in patch_results))
    total_summary = summarize_results(all_results, total_elapsed)

    write_results_tsv(joinpath(ATTEMPT12_ROOT, "column_proof_results.tsv"), column_results)
    write_results_tsv(joinpath(ATTEMPT12_ROOT, "hook_patch_results.tsv"), patch_results)
    write_summary(joinpath(ATTEMPT12_ROOT, "column_timing_summary.txt"), "column", column_summary)
    write_summary(joinpath(ATTEMPT12_ROOT, "patch_timing_summary.txt"), "patch", patch_summary)
    write_summary(joinpath(ATTEMPT12_ROOT, "proof_timing_summary.txt"), "total", total_summary)

    save_column_continuation_plot(column_results, joinpath(ATTEMPT12_ROOT, "column_continuation_plot.png"))
    save_patch_contour_comparison(patch_results, joinpath(ATTEMPT12_ROOT, "hook_patch_red_contour_comparison.png"))

    open(joinpath(ATTEMPT12_ROOT, "mismatch_summary.txt"), "w") do io
        for (label, results) in (("column", column_results), ("patch", patch_results))
            mismatches = [
                result for result in results
                if (result.fast.success != result.gold.success) ||
                   (result.fast.success && result.gold.success && result.fast_T_scs != result.gold_T_scs)
            ]
            println(io, "[$(label)] mismatches=$(length(mismatches))")
            for result in mismatches
                println(
                    io,
                    join([
                        string(result.delta_x),
                        string(result.delta_ca),
                        result.fast.mode,
                        format_sequence(result.fast_T_scs),
                        format_sequence(result.gold_T_scs),
                    ], '\t'),
                )
            end
            println(io)
        end
    end

    println("proof_total_seconds=$(total_elapsed)")
    println("column_results_path=$(joinpath(ATTEMPT12_ROOT, "column_proof_results.tsv"))")
    println("patch_results_path=$(joinpath(ATTEMPT12_ROOT, "hook_patch_results.tsv"))")
    println("column_plot_path=$(joinpath(ATTEMPT12_ROOT, "column_continuation_plot.png"))")
    println("patch_plot_path=$(joinpath(ATTEMPT12_ROOT, "hook_patch_red_contour_comparison.png"))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
