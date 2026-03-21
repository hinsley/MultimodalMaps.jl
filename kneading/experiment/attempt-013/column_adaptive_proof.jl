using Pkg

const ATTEMPT13_ROOT = @__DIR__
const ATTEMPT11_ROOT = normpath(joinpath(ATTEMPT13_ROOT, "..", "attempt-011"))
const REPO_ROOT_013 = normpath(joinpath(ATTEMPT13_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_013)

include(joinpath(ATTEMPT11_ROOT, "inspect_return_section.jl"))

using CairoMakie
using Printf
using Statistics

env_float_013(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

const PROOF_DELTA_CA = env_float_013("ATTEMPT013_DELTA_CA", -38.386774)
const PROOF_X_MIN = env_float_013("ATTEMPT013_X_MIN", -1.5)
const PROOF_X_MAX = env_float_013("ATTEMPT013_X_MAX", -0.5)
const PROOF_POINT_COUNT = parse(Int, get(ENV, "ATTEMPT013_POINT_COUNT", "80"))
const PROOF_SEED_X = env_float_013("ATTEMPT013_SEED_X", -1.0)

const PROOF_XS = collect(range(PROOF_X_MIN, PROOF_X_MAX, length=PROOF_POINT_COUNT))
const SEED_INDEX = findmin(abs.(PROOF_XS .- PROOF_SEED_X))[2]

const SEED_SCAN_POINTS = parse(Int, get(ENV, "ATTEMPT013_SEED_SCAN_POINTS", "181"))
const LOCAL_FAST_POINTS = parse(Int, get(ENV, "ATTEMPT013_LOCAL_FAST_POINTS", "5"))
const LOCAL_FALLBACK_POINTS = parse(Int, get(ENV, "ATTEMPT013_LOCAL_FALLBACK_POINTS", "11"))
const LOCAL_ANCHOR_POINTS = parse(Int, get(ENV, "ATTEMPT013_LOCAL_ANCHOR_POINTS", "61"))
const ROOT_REFINE_ITERS = parse(Int, get(ENV, "ATTEMPT013_ROOT_REFINE_ITERS", "5"))
const ANCHOR_EVERY = parse(Int, get(ENV, "ATTEMPT013_ANCHOR_EVERY", "10"))

const PRE_H_MIN = env_float_013("ATTEMPT013_PRE_H_MIN", 2.0e-4)
const PRE_H_MAX = env_float_013("ATTEMPT013_PRE_H_MAX", 3.0e-2)
const IMAGE_CONTINUITY_ABS = env_float_013("ATTEMPT013_IMAGE_CONTINUITY_ABS", 4.0e-3)
const IMAGE_CONTINUITY_REL = env_float_013("ATTEMPT013_IMAGE_CONTINUITY_REL", 0.35)
const ROOT_RESIDUAL_TOL = env_float_013("ATTEMPT013_ROOT_RESIDUAL_TOL", 1.5e-3)

const FAST_ABSTOL_013 = env_float_013("ATTEMPT013_FAST_ABSTOL", 1.0e-8)
const FAST_RELTOL_013 = env_float_013("ATTEMPT013_FAST_RELTOL", 1.0e-8)
const ANCHOR_ABSTOL_013 = env_float_013("ATTEMPT013_ANCHOR_ABSTOL", 1.0e-8)
const ANCHOR_RELTOL_013 = env_float_013("ATTEMPT013_ANCHOR_RELTOL", 1.0e-8)
const T0_MAPRES_013 = parse(Int, get(ENV, "ATTEMPT013_T0_MAPRES", "400"))

const MINIMAL_SAVEAT_013 = parse(Float64, get(ENV, "ATTEMPT013_MINIMAL_SAVEAT", string(SECTION_TSPAN[2])))
const PLOT_PX_PER_UNIT_013 = env_float_013("ATTEMPT013_PX_PER_UNIT", 2.0)

Base.@kwdef struct SectionReturnEval013
    s::Float64
    ok::Bool = false
    spike_count::Int = 0
    return_s::Float64 = NaN
end

Base.@kwdef struct PreimageSolve013
    success::Bool = false
    s_pre::Float64 = NaN
    s_img::Float64 = NaN
    residual::Float64 = Inf
    return_solves::Int = 0
    message::String = ""
end

Base.@kwdef struct ColumnPoint013
    delta_x::Float64
    success::Bool = false
    mode::String = "failed"
    s_pre::Float64 = NaN
    s_img::Float64 = NaN
    residual::Float64 = Inf
    return_solves::Int = 0
    elapsed_seconds::Float64 = 0.0
    message::String = ""
    anchor_checked::Bool = false
    anchor_match::Bool = false
    anchor_s_pre::Float64 = NaN
    anchor_s_img::Float64 = NaN
    anchor_residual::Float64 = NaN
    s_diff::Float64 = NaN
    img_diff::Float64 = NaN
    fast_T_scs::Vector{Int} = Int[]
    anchor_T_scs::Vector{Int} = Int[]
end

function format_sequence_013(seq::Vector{Int})
    isempty(seq) && return ""
    return join(seq, ",")
end

function secant_predict_013(x0::Float64, y0::Float64, x1::Float64, y1::Float64, x::Float64)
    if !isfinite(y0) || !isfinite(y1) || x1 == x0
        return y1
    end
    return y1 + (x - x1) * (y1 - y0) / (x1 - x0)
end

function section_return_eval_013(
    p,
    section,
    s::Float64;
    abstol::Float64,
    reltol::Float64,
)::SectionReturnEval013
    sol, target_phase, spike_count = solve_section_from_s(
        p,
        section,
        s;
        saveat=MINIMAL_SAVEAT_013,
        abstol=abstol,
        reltol=reltol,
    )
    if !returned_to_section(sol, target_phase)
        return SectionReturnEval013(s=s)
    end
    return SectionReturnEval013(
        s=s,
        ok=true,
        spike_count=spike_count,
        return_s=ray_coordinate(sol.u[end], section),
    )
end

function init_T0_highres_013(p, x_eq_SF, gamma_sd_minus0; mapres::Int)
    callback = make_ca_min_callback(x_eq_SF)

    prob = ODEProblem(Plant.melibeNew, gamma_sd_minus0, TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
    gamma_sd_minus_endpoint = sol.u[end]
    gamma_sd_minus_ca_min = Float64(gamma_sd_minus_endpoint[5])
    gamma_sd_minus_ca_min_V = Float64(find_zero(
        V -> EquilibriaSubset.Ca_null_Ca(p, V) - gamma_sd_minus_ca_min,
        Float64(gamma_sd_minus_endpoint[6]),
    ))

    V_eq_SF = find_equilibria(p)[2]
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=mapres))
    u0s = SVector{6, Float64}[
        SVector{6, Float64}((
            Plant.xinf(p, V) - 1.0e-4,
            0.0,
            Plant.ninf(V),
            Plant.hinf(V),
            EquilibriaSubset.Ca_null_Ca(p, V),
            V,
        )) for V in Vs
    ]

    return_ca_mins = Float64[]
    sizehint!(return_ca_mins, length(u0s))
    for u0 in u0s
        local_prob = ODEProblem(Plant.melibeNew, u0, TSPAN, p)
        local_sol = solve(local_prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
        push!(return_ca_mins, Float64(local_sol.u[end][5]))
    end

    first_max_index = nothing
    for idx in 2:(length(return_ca_mins) - 1)
        if return_ca_mins[idx] > return_ca_mins[idx - 1] && return_ca_mins[idx] > return_ca_mins[idx + 1]
            first_max_index = idx
            break
        end
    end
    isnothing(first_max_index) && error("High-resolution T0 solve could not locate a first local maximum.")

    reference_u0 = u0s[first_max_index]
    a = reference_u0[5] - 3.0e-3
    b = reference_u0[5] + 3.0e-3
    golden_ratio = (sqrt(5) - 1) / 2
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    fc = return_voltage_at_ca_min(p, c, reference_u0[1], callback)
    fd = return_voltage_at_ca_min(p, d, reference_u0[1], callback)
    while abs(b - a) > 1.0e-10
        if fc > fd
            b = d
            d = c
            fd = fc
            c = b - golden_ratio * (b - a)
            fc = return_voltage_at_ca_min(p, c, reference_u0[1], callback)
        else
            a = c
            c = d
            fc = fd
            d = a + golden_ratio * (b - a)
            fd = return_voltage_at_ca_min(p, d, reference_u0[1], callback)
        end
    end

    T_Ca0 = (a + b) / 2
    return SVector{6, Float64}(Tuple(Float64.(EquilibriaSubset.dune(p, reference_u0[1], T_Ca0))))
end

function choose_root_pair_013(
    valid::Vector{SectionReturnEval013},
    target_s::Float64;
    hint_s::Union{Nothing, Float64}=nothing,
)
    isempty(valid) && return nothing

    residuals = [eval.return_s - target_s for eval in valid]
    sign_change_pairs = Tuple{Int, Int}[]
    for idx in 1:(length(valid) - 1)
        r1 = residuals[idx]
        r2 = residuals[idx + 1]
        if !isfinite(r1) || !isfinite(r2)
            continue
        end
        if r1 == 0.0 || r2 == 0.0 || signbit(r1) != signbit(r2)
            push!(sign_change_pairs, (idx, idx + 1))
        end
    end

    if !isempty(sign_change_pairs)
        if isnothing(hint_s)
            return valid[sign_change_pairs[1][1]], valid[sign_change_pairs[1][2]]
        end
        _, best_pos = findmin([
            abs(0.5 * (valid[i].s + valid[j].s) - hint_s) for (i, j) in sign_change_pairs
        ])
        i, j = sign_change_pairs[best_pos]
        return valid[i], valid[j]
    end

    order = sortperm(abs.(residuals))
    if length(order) == 1
        return valid[order[1]], valid[order[1]]
    end
    i = order[1]
    neighbor_candidates = [idx for idx in order[2:end] if abs(valid[idx].s - valid[i].s) > 0]
    isempty(neighbor_candidates) && return valid[i], valid[i]
    if isnothing(hint_s)
        j = neighbor_candidates[1]
    else
        _, pos = findmin(abs.([0.5 * (valid[i].s + valid[idx].s) - hint_s for idx in neighbor_candidates]))
        j = neighbor_candidates[pos]
    end
    return valid[min(i, j)], valid[max(i, j)]
end

function refine_target_root_013(
    p,
    section,
    target_s::Float64,
    left::SectionReturnEval013,
    right::SectionReturnEval013;
    max_iters::Int,
    abstol::Float64,
    reltol::Float64,
)
    solves = 0
    best = abs(left.return_s - target_s) <= abs(right.return_s - target_s) ? left : right
    current_left = left
    current_right = right
    for _ in 1:max_iters
        f_left = current_left.return_s - target_s
        f_right = current_right.return_s - target_s
        s_new =
            if current_left.s == current_right.s
                current_left.s
            elseif isfinite(f_left) && isfinite(f_right) && abs(f_right - f_left) > 1.0e-12
                clamp(current_right.s - f_right * (current_right.s - current_left.s) / (f_right - f_left), min(current_left.s, current_right.s), max(current_left.s, current_right.s))
            else
                0.5 * (current_left.s + current_right.s)
            end
        if !isfinite(s_new)
            break
        end
        trial = section_return_eval_013(p, section, s_new; abstol=abstol, reltol=reltol)
        solves += 1
        if !trial.ok || trial.spike_count != 1
            break
        end

        if abs(trial.return_s - target_s) < abs(best.return_s - target_s)
            best = trial
        end

        f_trial = trial.return_s - target_s
        if abs(f_trial) <= ROOT_RESIDUAL_TOL
            best = trial
            break
        end

        if current_left.s == current_right.s
            break
        elseif signbit(f_left) != signbit(f_trial)
            current_right = trial
        elseif signbit(f_right) != signbit(f_trial)
            current_left = trial
        else
            if abs(current_left.s - trial.s) < abs(current_right.s - trial.s)
                current_left = trial
            else
                current_right = trial
            end
        end
    end
    return best, solves
end

function one_spike_preimage_for_target_013(
    p,
    section,
    target_s::Float64,
    s_values::Vector{Float64};
    hint_s::Union{Nothing, Float64}=nothing,
    abstol::Float64,
    reltol::Float64,
    max_refine::Int,
)::PreimageSolve013
    evals = SectionReturnEval013[]
    sizehint!(evals, length(s_values))
    for s in s_values
        push!(evals, section_return_eval_013(p, section, s; abstol=abstol, reltol=reltol))
    end
    solve_count = length(s_values)

    valid = sort(
        [eval for eval in evals if eval.ok && eval.spike_count == 1];
        by=eval -> eval.s,
    )
    isempty(valid) && return PreimageSolve013(
        success=false,
        return_solves=solve_count,
        message="No 1-spike section returns in the requested window.",
    )

    pair = choose_root_pair_013(valid, target_s; hint_s=hint_s)
    isnothing(pair) && return PreimageSolve013(
        success=false,
        return_solves=solve_count,
        message="Could not choose a one-spike root candidate.",
    )
    left, right = pair
    best, extra_solves = refine_target_root_013(
        p,
        section,
        target_s,
        left,
        right;
        max_iters=max_refine,
        abstol=abstol,
        reltol=reltol,
    )
    solve_count += extra_solves
    return PreimageSolve013(
        success=true,
        s_pre=best.s,
        s_img=best.return_s,
        residual=abs(best.return_s - target_s),
        return_solves=solve_count,
    )
end

function track_point_013(
    delta_x::Float64,
    delta_ca::Float64,
    prev2::ColumnPoint013,
    prev1::ColumnPoint013,
)::ColumnPoint013
    result = Ref{ColumnPoint013}()
    elapsed = @elapsed begin
        p = build_params(delta_x, delta_ca)
        section = equilibrium_section_data(p)

        s_pre_pred =
            if prev2.success
                clamp(secant_predict_013(prev2.delta_x, prev2.s_pre, prev1.delta_x, prev1.s_pre, delta_x), SECTION_S_MIN, SECTION_S_MAX)
            else
                prev1.s_pre
            end
        s_img_pred =
            if prev2.success
                secant_predict_013(prev2.delta_x, prev2.s_img, prev1.delta_x, prev1.s_img, delta_x)
            else
                prev1.s_img
            end
        h = clamp(0.5 * abs(prev1.s_pre - (prev2.success ? prev2.s_pre : prev1.s_pre)), PRE_H_MIN, PRE_H_MAX)

        continuity_tol = max(IMAGE_CONTINUITY_ABS, IMAGE_CONTINUITY_REL * abs(prev1.s_img - (prev2.success ? prev2.s_img : prev1.s_img)))

        fast_left = max(SECTION_S_MIN, s_pre_pred - h)
        fast_right = min(SECTION_S_MAX, s_pre_pred + h)
        fast_values = fast_left == fast_right ? [fast_left] : collect(range(fast_left, fast_right, length=LOCAL_FAST_POINTS))
        fast = one_spike_preimage_for_target_013(
            p,
            section,
            s_img_pred,
            fast_values;
            hint_s=s_pre_pred,
            abstol=FAST_ABSTOL_013,
            reltol=FAST_RELTOL_013,
            max_refine=ROOT_REFINE_ITERS,
        )
        if fast.success && fast.residual <= continuity_tol
            result[] = ColumnPoint013(
                delta_x=delta_x,
                success=true,
                mode="predict",
                s_pre=fast.s_pre,
                s_img=fast.s_img,
                residual=fast.residual,
                return_solves=fast.return_solves,
            )
        else
            fallback_left = max(SECTION_S_MIN, s_pre_pred - 3h)
            fallback_right = min(SECTION_S_MAX, s_pre_pred + 3h)
            fallback_values = fallback_left == fallback_right ? [fallback_left] : collect(range(fallback_left, fallback_right, length=LOCAL_FALLBACK_POINTS))
            fallback = one_spike_preimage_for_target_013(
                p,
                section,
                s_img_pred,
                fallback_values;
                hint_s=s_pre_pred,
                abstol=FAST_ABSTOL_013,
                reltol=FAST_RELTOL_013,
                max_refine=ROOT_REFINE_ITERS,
            )
            if fallback.success && fallback.residual <= max(3 * continuity_tol, continuity_tol)
                result[] = ColumnPoint013(
                    delta_x=delta_x,
                    success=true,
                    mode="fallback_local",
                    s_pre=fallback.s_pre,
                    s_img=fallback.s_img,
                    residual=fallback.residual,
                    return_solves=fast.return_solves + fallback.return_solves,
                )
            else
                result[] = ColumnPoint013(
                    delta_x=delta_x,
                    success=false,
                    mode="failed",
                    residual=min(fast.residual, fallback.residual),
                    return_solves=fast.return_solves + fallback.return_solves,
                    message=fast.success || fallback.success ? "Residual too large for continuation acceptance." : "No valid one-spike preimage found in fast or fallback windows.",
                )
            end
        end
    end
    base = result[]
    return ColumnPoint013(
        delta_x=base.delta_x,
        success=base.success,
        mode=base.mode,
        s_pre=base.s_pre,
        s_img=base.s_img,
        residual=base.residual,
        return_solves=base.return_solves,
        elapsed_seconds=elapsed,
        message=base.message,
        anchor_checked=base.anchor_checked,
        anchor_match=base.anchor_match,
        anchor_s_pre=base.anchor_s_pre,
        anchor_s_img=base.anchor_s_img,
        anchor_residual=base.anchor_residual,
        s_diff=base.s_diff,
        img_diff=base.img_diff,
        fast_T_scs=base.fast_T_scs,
        anchor_T_scs=base.anchor_T_scs,
    )
end

function bootstrap_point_013(
    delta_x::Float64,
    delta_ca::Float64,
    prev1::ColumnPoint013,
)::ColumnPoint013
    result = Ref{ColumnPoint013}()
    elapsed = @elapsed begin
        p = build_params(delta_x, delta_ca)
        section = equilibrium_section_data(p)
        half_width = max(0.03, 6 * PRE_H_MAX)
        left = max(SECTION_S_MIN, prev1.s_pre - half_width)
        right = min(SECTION_S_MAX, prev1.s_pre + half_width)
        values = collect(range(left, right, length=LOCAL_ANCHOR_POINTS))
        boot = one_spike_preimage_for_target_013(
            p,
            section,
            prev1.s_img,
            values;
            hint_s=prev1.s_pre,
            abstol=FAST_ABSTOL_013,
            reltol=FAST_RELTOL_013,
            max_refine=ROOT_REFINE_ITERS + 1,
        )
        continuity_tol = max(6 * IMAGE_CONTINUITY_ABS, 0.02)
        if boot.success && boot.residual <= continuity_tol
            result[] = ColumnPoint013(
                delta_x=delta_x,
                success=true,
                mode="bootstrap_local",
                s_pre=boot.s_pre,
                s_img=boot.s_img,
                residual=boot.residual,
                return_solves=boot.return_solves,
            )
        else
            result[] = ColumnPoint013(
                delta_x=delta_x,
                success=false,
                mode="failed",
                residual=boot.residual,
                return_solves=boot.return_solves,
                message=boot.success ? "Bootstrap residual too large." : boot.message,
            )
        end
    end
    base = result[]
    return ColumnPoint013(
        delta_x=base.delta_x,
        success=base.success,
        mode=base.mode,
        s_pre=base.s_pre,
        s_img=base.s_img,
        residual=base.residual,
        return_solves=base.return_solves,
        elapsed_seconds=elapsed,
        message=base.message,
    )
end

function continuity_tol_from(point::ColumnPoint013, previous::ColumnPoint013)
    return max(IMAGE_CONTINUITY_ABS, IMAGE_CONTINUITY_REL * abs(point.s_img - previous.s_img))
end

function seed_point_013(delta_x::Float64, delta_ca::Float64)::ColumnPoint013
    result = Ref{ColumnPoint013}()
    elapsed = @elapsed begin
        p = build_params(delta_x, delta_ca)
        saddle_data = compute_gamma_sd_minus0(p)
        section = equilibrium_section_data(p)
        T0 = init_T0_highres_013(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0; mapres=T0_MAPRES_013)
        first_return_sol, first_target_phase, _ = solve_section_return(
            p,
            section,
            T0;
            target_returns=1,
            saveat=MINIMAL_SAVEAT_013,
            abstol=3e-6,
            reltol=3e-6,
        )
        returned_to_section(first_return_sol, first_target_phase) || error("Seed T0 did not return to the section.")
        target_s = ray_coordinate(first_return_sol.u[end], section)
        seed_values = collect(range(SECTION_S_MIN, SECTION_S_MAX, length=SEED_SCAN_POINTS))
        preimage = one_spike_preimage_for_target_013(
            p,
            section,
            target_s,
            seed_values;
            hint_s=nothing,
            abstol=ANCHOR_ABSTOL_013,
            reltol=ANCHOR_RELTOL_013,
            max_refine=ROOT_REFINE_ITERS + 2,
        )
        if !preimage.success
            result[] = ColumnPoint013(
                delta_x=delta_x,
                success=false,
                mode="seed_failed",
                return_solves=preimage.return_solves,
                residual=preimage.residual,
                message=preimage.message,
            )
        else
            result[] = ColumnPoint013(
                delta_x=delta_x,
                success=true,
                mode="seed",
                s_pre=preimage.s_pre,
                s_img=target_s,
                residual=preimage.residual,
                return_solves=preimage.return_solves,
            )
        end
    end
    base = result[]
    return ColumnPoint013(
        delta_x=base.delta_x,
        success=base.success,
        mode=base.mode,
        s_pre=base.s_pre,
        s_img=base.s_img,
        residual=base.residual,
        return_solves=base.return_solves,
        elapsed_seconds=elapsed,
        message=base.message,
        anchor_checked=base.anchor_checked,
        anchor_match=base.anchor_match,
        anchor_s_pre=base.anchor_s_pre,
        anchor_s_img=base.anchor_s_img,
        anchor_residual=base.anchor_residual,
        s_diff=base.s_diff,
        img_diff=base.img_diff,
        fast_T_scs=base.fast_T_scs,
        anchor_T_scs=base.anchor_T_scs,
    )
end

function anchor_validate_013(point::ColumnPoint013)::ColumnPoint013
    !point.success && return point

    result = Ref{ColumnPoint013}()
    elapsed = @elapsed begin
        p = build_params(point.delta_x, PROOF_DELTA_CA)
        saddle_data = compute_gamma_sd_minus0(p)
        section = equilibrium_section_data(p)
        h = clamp(4 * PRE_H_MIN, PRE_H_MIN, 0.05)
        anchor_left = max(SECTION_S_MIN, point.s_pre - max(4 * h, 0.012))
        anchor_right = min(SECTION_S_MAX, point.s_pre + max(4 * h, 0.012))
        anchor_values = collect(range(anchor_left, anchor_right, length=LOCAL_ANCHOR_POINTS))
        anchor = one_spike_preimage_for_target_013(
            p,
            section,
            point.s_img,
            anchor_values;
            hint_s=point.s_pre,
            abstol=ANCHOR_ABSTOL_013,
            reltol=ANCHOR_RELTOL_013,
            max_refine=ROOT_REFINE_ITERS + 2,
        )
        fast_T = compute_sscs(p, lift_section_point(p, section, point.s_img), saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
        anchor_T = anchor.success ? compute_sscs(p, lift_section_point(p, section, anchor.s_img), saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6) : Int[]
        result[] = ColumnPoint013(
            delta_x=point.delta_x,
            success=point.success,
            mode=point.mode,
            s_pre=point.s_pre,
            s_img=point.s_img,
            residual=point.residual,
            return_solves=point.return_solves + anchor.return_solves,
            message=point.message,
            anchor_checked=true,
            anchor_match=anchor.success && fast_T == anchor_T,
            anchor_s_pre=anchor.s_pre,
            anchor_s_img=anchor.s_img,
            anchor_residual=anchor.residual,
            s_diff=anchor.success ? abs(point.s_pre - anchor.s_pre) : NaN,
            img_diff=anchor.success ? abs(point.s_img - anchor.s_img) : NaN,
            fast_T_scs=fast_T,
            anchor_T_scs=anchor_T,
        )
    end
    base = result[]
    return ColumnPoint013(
        delta_x=base.delta_x,
        success=base.success,
        mode=base.mode,
        s_pre=base.s_pre,
        s_img=base.s_img,
        residual=base.residual,
        return_solves=base.return_solves,
        elapsed_seconds=point.elapsed_seconds + elapsed,
        message=base.message,
        anchor_checked=base.anchor_checked,
        anchor_match=base.anchor_match,
        anchor_s_pre=base.anchor_s_pre,
        anchor_s_img=base.anchor_s_img,
        anchor_residual=base.anchor_residual,
        s_diff=base.s_diff,
        img_diff=base.img_diff,
        fast_T_scs=base.fast_T_scs,
        anchor_T_scs=base.anchor_T_scs,
    )
end

function point_needs_anchor(idx::Int, point::ColumnPoint013)
    return idx == SEED_INDEX || mod(idx - 1, ANCHOR_EVERY) == 0 || point.mode != "predict" || point.residual > 0.5 * IMAGE_CONTINUITY_ABS
end

function fill_direction!(
    results::Vector{ColumnPoint013},
    indices::Vector{Int},
)
    accepted = [results[SEED_INDEX]]
    if isempty(indices)
        return
    end
    for idx in indices
        prev1 = accepted[end]
        prev2 = length(accepted) >= 2 ? accepted[end - 1] : ColumnPoint013(delta_x=prev1.delta_x, success=false)
        point =
            if prev2.success
                track_point_013(PROOF_XS[idx], PROOF_DELTA_CA, prev2, prev1)
            else
                bootstrap_point_013(PROOF_XS[idx], PROOF_DELTA_CA, prev1)
            end

        if point.success
            push!(accepted, point)
            results[idx] = point
        else
            results[idx] = point
        end
    end
end

function save_column_plot_013(results::Vector{ColumnPoint013}, path::String)
    xs = PROOF_XS
    fig = Figure(size=(1800, 1350))
    ax1 = Axis(fig[1, 1], xlabel="Δx", ylabel="s_pre", title="One-Spike Preimage Coordinate")
    ax2 = Axis(fig[2, 1], xlabel="Δx", ylabel="s_img", title="Iterate-1 T0 Section Start")
    ax3 = Axis(fig[3, 1], xlabel="Δx", ylabel="Residual / Solves", title="Continuation Residual and Return Solves")

    good = [point.success for point in results]
    x_good = [results[idx].delta_x for idx in eachindex(results) if good[idx]]
    s_pre_good = [results[idx].s_pre for idx in eachindex(results) if good[idx]]
    s_img_good = [results[idx].s_img for idx in eachindex(results) if good[idx]]
    lines!(ax1, x_good, s_pre_good; color=:black, linewidth=2.0)
    lines!(ax2, x_good, s_img_good; color=:black, linewidth=2.0)
    lines!(ax3, x_good, [results[idx].residual for idx in eachindex(results) if good[idx]]; color=:firebrick, linewidth=2.0, label="residual")
    lines!(ax3, x_good, [0.001 * results[idx].return_solves for idx in eachindex(results) if good[idx]]; color=:royalblue, linewidth=2.0, linestyle=:dash, label="0.001 × solves")

    for point in results
        if !point.success
            scatter!(ax1, [point.delta_x], [SECTION_S_MIN]; color=:red, marker=:xcross, markersize=14)
            scatter!(ax2, [point.delta_x], [SECTION_S_MIN]; color=:red, marker=:xcross, markersize=14)
            continue
        end
        color =
            point.mode == "seed" ? :purple :
            point.mode == "predict" ? :green :
            point.mode == "bootstrap_local" ? :orange :
            point.mode == "fallback_local" ? :goldenrod :
            :royalblue
        scatter!(ax1, [point.delta_x], [point.s_pre]; color=color, markersize=12)
        scatter!(ax2, [point.delta_x], [point.s_img]; color=color, markersize=12)
        if point.anchor_checked
            marker_color = point.anchor_match ? :black : :red
            scatter!(ax1, [point.delta_x], [point.s_pre]; color=marker_color, marker=:rect, markersize=8)
            scatter!(ax2, [point.delta_x], [point.s_img]; color=marker_color, marker=:rect, markersize=8)
        end
    end
    axislegend(ax3, position=:rt)
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT_013)
end

function write_results_013(path::String, results::Vector{ColumnPoint013})
    open(path, "w") do io
        println(io, "delta_x\tsuccess\tmode\ts_pre\ts_img\tresidual\treturn_solves\telapsed_seconds\tanchor_checked\tanchor_match\tanchor_s_pre\tanchor_s_img\tanchor_residual\ts_diff\timg_diff\tfast_T_scs\tanchor_T_scs\tmessage")
        for point in results
            println(
                io,
                join([
                    string(point.delta_x),
                    string(point.success),
                    point.mode,
                    string(point.s_pre),
                    string(point.s_img),
                    string(point.residual),
                    string(point.return_solves),
                    string(point.elapsed_seconds),
                    string(point.anchor_checked),
                    string(point.anchor_match),
                    string(point.anchor_s_pre),
                    string(point.anchor_s_img),
                    string(point.anchor_residual),
                    string(point.s_diff),
                    string(point.img_diff),
                    format_sequence_013(point.fast_T_scs),
                    format_sequence_013(point.anchor_T_scs),
                    replace(point.message, '\t' => ' '),
                ], '\t'),
            )
        end
    end
end

function write_summary_013(path::String, results::Vector{ColumnPoint013}, total_seconds::Float64)
    successes = [point for point in results if point.success]
    anchor_points = [point for point in results if point.anchor_checked]
    open(path, "w") do io
        println(io, "delta_ca\t$(PROOF_DELTA_CA)")
        println(io, "point_count\t$(length(results))")
        println(io, "seed_index\t$(SEED_INDEX)")
        println(io, "seed_delta_x\t$(PROOF_XS[SEED_INDEX])")
        println(io, "total_seconds\t$(total_seconds)")
        println(io, "success_count\t$(length(successes))")
        println(io, "predict_count\t$(count(point -> point.mode == "predict", results))")
        println(io, "bootstrap_local_count\t$(count(point -> point.mode == "bootstrap_local", results))")
        println(io, "fallback_local_count\t$(count(point -> point.mode == "fallback_local", results))")
        println(io, "failed_count\t$(count(point -> !point.success, results))")
        println(io, "median_point_seconds\t$(isempty(successes) ? NaN : median([point.elapsed_seconds for point in successes]))")
        println(io, "p95_point_seconds\t$(isempty(successes) ? NaN : sort([point.elapsed_seconds for point in successes])[clamp(ceil(Int, 0.95 * length(successes)), 1, length(successes))])")
        println(io, "median_return_solves\t$(isempty(successes) ? NaN : median([point.return_solves for point in successes]))")
        println(io, "anchor_count\t$(length(anchor_points))")
        println(io, "anchor_match_count\t$(count(point -> point.anchor_match, anchor_points))")
        println(io, "anchor_mismatch_count\t$(count(point -> point.anchor_checked && !point.anchor_match, anchor_points))")
        println(io, "anchor_median_s_diff\t$(isempty(anchor_points) ? NaN : median([point.s_diff for point in anchor_points if isfinite(point.s_diff)]))")
        println(io, "anchor_max_s_diff\t$(isempty(anchor_points) ? NaN : maximum([point.s_diff for point in anchor_points if isfinite(point.s_diff)]))")
        println(io, "anchor_median_img_diff\t$(isempty(anchor_points) ? NaN : median([point.img_diff for point in anchor_points if isfinite(point.img_diff)]))")
        println(io, "anchor_max_img_diff\t$(isempty(anchor_points) ? NaN : maximum([point.img_diff for point in anchor_points if isfinite(point.img_diff)]))")
    end
end

function main()
    results = [ColumnPoint013(delta_x=x) for x in PROOF_XS]

    total_seconds = @elapsed begin
        seed = seed_point_013(PROOF_XS[SEED_INDEX], PROOF_DELTA_CA)
        results[SEED_INDEX] = seed
        seed.success || error("Seed construction failed: $(seed.message)")

        fill_direction!(results, collect((SEED_INDEX - 1):-1:1))
        fill_direction!(results, collect((SEED_INDEX + 1):length(PROOF_XS)))

        for idx in eachindex(results)
            if point_needs_anchor(idx, results[idx])
                results[idx] = anchor_validate_013(results[idx])
            end
        end
    end

    results_path = joinpath(ATTEMPT13_ROOT, "column_adaptive_results.tsv")
    summary_path = joinpath(ATTEMPT13_ROOT, "column_adaptive_summary.txt")
    plot_path = joinpath(ATTEMPT13_ROOT, "column_adaptive_plot.png")

    write_results_013(results_path, results)
    write_summary_013(summary_path, results, total_seconds)
    save_column_plot_013(results, plot_path)

    println("proof_total_seconds=$(total_seconds)")
    println("results_path=$(results_path)")
    println("summary_path=$(summary_path)")
    println("plot_path=$(plot_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
