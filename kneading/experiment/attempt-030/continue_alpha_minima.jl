using Pkg

const ATTEMPT028_CONT_ROOT = @__DIR__
include(joinpath(ATTEMPT028_CONT_ROOT, "main.jl"))

using Base.Threads
using Dates
using Printf

const ATTEMPT028_CONT_N_ALPHA = parse(Int, get(ENV, "ATTEMPT028_CONT_N_ALPHA", "2000"))
const ATTEMPT028_CONT_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT028_CONT_ALPHA_MIN", "0.0"))
const ATTEMPT028_CONT_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT028_CONT_ALPHA_MAX", "0.7"))
const ATTEMPT028_CONT_LAMBDA = parse(Float64, get(ENV, "ATTEMPT028_CONT_LAMBDA", string(ATTEMPT028_LAMBDA)))
const ATTEMPT028_CONT_SEED_ALPHA = parse(Float64, get(ENV, "ATTEMPT028_CONT_SEED_ALPHA", string(ATTEMPT028_ALPHA)))
const ATTEMPT028_CONT_SEED_TARGET_S = parse(Float64, get(ENV, "ATTEMPT028_CONT_SEED_TARGET_S", "4.4944"))
const ATTEMPT028_CONT_BREAK_S_JUMP = parse(Float64, get(ENV, "ATTEMPT028_CONT_BREAK_S_JUMP", "0.05"))
const ATTEMPT028_CONT_BREAK_STATE_JUMP = parse(Float64, get(ENV, "ATTEMPT028_CONT_BREAK_STATE_JUMP", "0.05"))
const ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION = parse(Int, get(ENV, "ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION", "0"))
const ATTEMPT028_CONT_CORRECT_EVERY = parse(Int, get(ENV, "ATTEMPT028_CONT_CORRECT_EVERY", "5"))

const ATTEMPT028_CONT_ALPHAS = collect(range(
    ATTEMPT028_CONT_ALPHA_MIN,
    ATTEMPT028_CONT_ALPHA_MAX,
    length=ATTEMPT028_CONT_N_ALPHA,
))

const ATTEMPT028_CONT_TAG = @sprintf(
    "alpha_continuation_lambda%.4f_seedalpha%.4f_target%.4f",
    ATTEMPT028_CONT_LAMBDA,
    ATTEMPT028_CONT_SEED_ALPHA,
    ATTEMPT028_CONT_SEED_TARGET_S,
)
const ATTEMPT028_CONT_RESULTS_PATH = joinpath(ATTEMPT028_CONT_ROOT, "$(ATTEMPT028_CONT_TAG)_results.tsv")
const ATTEMPT028_CONT_TRACE_PATH = joinpath(ATTEMPT028_CONT_ROOT, "$(ATTEMPT028_CONT_TAG)_trace.tsv")
const ATTEMPT028_CONT_SUMMARY_PATH = joinpath(ATTEMPT028_CONT_ROOT, "$(ATTEMPT028_CONT_TAG)_summary.md")

struct ContinuationRow028
    direction::String
    step_index::Int
    alpha::Float64
    predicted_s::Float64
    branch_points::Int
    branch_s_min::Float64
    branch_s_max::Float64
    used_s0::Float64
    converged::Bool
    predictor_only::Bool
    broke::Bool
    status::String
    accepted_steps::Int
    total_trace_rows::Int
    final_s::Float64
    map_value::Float64
    first_derivative::Float64
    second_derivative::Float64
    correction_abs::Float64
    state_jump::Float64
    event_time::Float64
    current_x::Float64
    current_y::Float64
    current_z::Float64
    event_x::Float64
    event_y::Float64
    event_z::Float64
    event_time_d1::Float64
    event_time_d2::Float64
    sciml_first_derivative::Float64
    first_derivative_mismatch::Float64
end

struct TraceRow028
    direction::String
    alpha::Float64
    step_index::Int
    iter::Int
    s::Float64
    map_value::Float64
    first_derivative::Float64
    second_derivative::Float64
    raw_step::Float64
    damping::Float64
    accepted::Bool
    sciml_first_derivative::Float64
    first_derivative_mismatch::Float64
end

struct PendingStep030
    step_index::Int
    alpha::Float64
end

function failure_row_028(direction::String, step_index::Int, alpha::Float64, predicted_s::Float64, status::String)
    return ContinuationRow028(
        direction,
        step_index,
        alpha,
        predicted_s,
        0,
        NaN,
        NaN,
        NaN,
        false,
        false,
        true,
        status,
        0,
        0,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
    )
end

function predictor_row_028(
    direction::String,
    step_index::Int,
    alpha::Float64,
    predicted_s::Float64,
    used_s0::Float64,
    evaluation::ReturnMapEval028,
    prev_state::Union{Nothing, SVector{3, Float64}},
    status::String,
    broke::Bool,
)
    state_jump = isnothing(prev_state) ? 0.0 : norm(evaluation.current_state - prev_state)
    return ContinuationRow028(
        direction,
        step_index,
        alpha,
        predicted_s,
        0,
        NaN,
        NaN,
        used_s0,
        false,
        true,
        broke,
        status,
        0,
        0,
        evaluation.s,
        evaluation.value,
        evaluation.first_derivative,
        evaluation.second_derivative,
        0.0,
        state_jump,
        evaluation.event_time,
        evaluation.current_state[1],
        evaluation.current_state[2],
        evaluation.current_state[3],
        evaluation.event_state[1],
        evaluation.event_state[2],
        evaluation.event_state[3],
        evaluation.event_time_d1,
        evaluation.event_time_d2,
        evaluation.sciml_first_derivative,
        evaluation.first_derivative_mismatch,
    )
end

function carried_predictor_row_028(
    direction::String,
    step_index::Int,
    alpha::Float64,
    predicted_s::Float64,
    current_state::SVector{3, Float64},
    prev_state::Union{Nothing, SVector{3, Float64}},
    status::String,
    broke::Bool,
)
    state_jump = isnothing(prev_state) ? 0.0 : norm(current_state - prev_state)
    return ContinuationRow028(
        direction,
        step_index,
        alpha,
        predicted_s,
        0,
        NaN,
        NaN,
        predicted_s,
        false,
        true,
        broke,
        status,
        0,
        0,
        predicted_s,
        NaN,
        NaN,
        NaN,
        0.0,
        state_jump,
        NaN,
        current_state[1],
        current_state[2],
        current_state[3],
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN,
    )
end

function discrete_local_extrema_indices_028(values::Vector{Float64}; minimum::Bool)
    idxs = Int[]
    for i in 2:(length(values) - 1)
        if minimum
            values[i] <= values[i - 1] && values[i] <= values[i + 1] && push!(idxs, i)
        else
            values[i] >= values[i - 1] && values[i] >= values[i + 1] && push!(idxs, i)
        end
    end
    return idxs
end

function branch_for_alpha_028(alpha::Float64, lambda::Float64)
    times, states = collect_absxmax_events_028(alpha, lambda)
    length(times) >= ATTEMPT028_TRANSIENT_EVENTS + 50 || error("Not enough sampled |x|-maxima at alpha=$(alpha)")
    branch = build_positive_branch_028(times, states)
    branch = filter_branch_by_next_sign_028(branch, ATTEMPT028_TARGET_NEXT_SIGN)
    length(branch) >= ATTEMPT028_FIT_WINDOW + 4 || error("Not enough filtered branch points at alpha=$(alpha)")
    spline = build_natural_spline_028([point.s for point in branch], [point.z for point in branch])
    return branch, spline
end

function seed_from_target_minimum_028(branch::Vector{BranchPoint028}, target_s::Float64)
    branch_r = [point.next_r for point in branch]
    mins = discrete_local_extrema_indices_028(branch_r; minimum=true)
    isempty(mins) && error("No discrete local minima found on filtered branch")
    chosen = mins[argmin(abs.([branch[i].s - target_s for i in mins]))]
    return chosen, branch[chosen].s
end

function build_row_028(
    direction::String,
    step_index::Int,
    alpha::Float64,
    predicted_s::Float64,
    used_s0::Float64,
    trace::Vector{NewtonTrace028},
    evaluation::ReturnMapEval028,
    prev_state::Union{Nothing, SVector{3, Float64}},
    status::String,
    broke::Bool,
)
    accepted_steps = count(row -> row.accepted, trace)
    correction_abs = abs(evaluation.s - predicted_s)
    state_jump = isnothing(prev_state) ? 0.0 : norm(evaluation.current_state - prev_state)
    return ContinuationRow028(
        direction,
        step_index,
        alpha,
        predicted_s,
        0,
        NaN,
        NaN,
        used_s0,
        true,
        false,
        broke,
        status,
        accepted_steps,
        length(trace),
        evaluation.s,
        evaluation.value,
        evaluation.first_derivative,
        evaluation.second_derivative,
        correction_abs,
        state_jump,
        evaluation.event_time,
        evaluation.current_state[1],
        evaluation.current_state[2],
        evaluation.current_state[3],
        evaluation.event_state[1],
        evaluation.event_state[2],
        evaluation.event_state[3],
        evaluation.event_time_d1,
        evaluation.event_time_d2,
        evaluation.sciml_first_derivative,
        evaluation.first_derivative_mismatch,
    )
end

function process_pending_steps_030(
    direction::String,
    pending::Vector{PendingStep030},
    lambda::Float64,
    current_state::SVector{3, Float64},
    prev_corrected_state::SVector{3, Float64},
    forced_jump_used::Bool,
)
    rows = ContinuationRow028[]
    traces = TraceRow028[]
    working_state = current_state
    working_prev_corrected = prev_corrected_state
    working_forced_jump_used = forced_jump_used

    for pending_step in pending
        step_index = pending_step.step_index
        alpha = pending_step.alpha
        predicted_x = working_state[1]
        predicted_z = working_state[3]
        predicted_s = predicted_x^2
        predictor_x_str = @sprintf("%.12f", predicted_x)

        try
            trace, eval = run_damped_newton_xfixed_028(alpha, lambda, predicted_x, predicted_z, ATTEMPT028_TARGET_NEXT_SIGN)

            for row in trace
                push!(traces, TraceRow028(
                    direction,
                    alpha,
                    step_index,
                    row.iter,
                    row.s,
                    row.value,
                    row.first_derivative,
                    row.second_derivative,
                    row.step,
                    row.damping,
                    row.accepted,
                    row.sciml_first_derivative,
                    row.first_derivative_mismatch,
                ))
            end

            correction_abs = abs(eval.s - predicted_s)
            state_jump = norm(eval.current_state - working_prev_corrected)
            broke = correction_abs > ATTEMPT028_CONT_BREAK_S_JUMP || state_jump > ATTEMPT028_CONT_BREAK_STATE_JUMP
            x_str = @sprintf("%.12f", eval.current_state[1])
            s_str = @sprintf("%.12f", eval.s)
            fp_str = @sprintf("%.3e", eval.first_derivative)
            jump_s_str = @sprintf("%.3e", correction_abs)
            jump_state_str = @sprintf("%.3e", state_jump)

            if broke
                if direction == "decreasing" && !working_forced_jump_used
                    alpha_str = @sprintf("%.12f", alpha)
                    row = build_row_028(
                        direction,
                        step_index,
                        alpha,
                        predicted_s,
                        predicted_s,
                        trace,
                        eval,
                        working_prev_corrected,
                        "forced_accept_after_first_would_stop_jump",
                        false,
                    )
                    push!(rows, row)
                    println("[$direction]   forced-accepted jump at alpha=$alpha_str from predictor x=$predictor_x_str toward corrected x=$x_str s=$s_str F'=$fp_str jump_s=$jump_s_str jump_state=$jump_state_str")
                    working_state = eval.current_state
                    working_prev_corrected = eval.current_state
                    working_forced_jump_used = true
                    continue
                end
                row = predictor_row_028(
                    direction,
                    step_index,
                    alpha,
                    predicted_s,
                    predicted_s,
                    eval,
                    working_prev_corrected,
                    "would_stop_jump_break_then_predictor_only",
                    true,
                )
                push!(rows, row)
                println("[$direction]   would-stop jump at alpha=$(@sprintf("%.12f", alpha)) from predictor x=$predictor_x_str toward corrected x=$x_str s=$s_str F'=$fp_str jump_s=$jump_s_str jump_state=$jump_state_str")
                continue
            end

            row = build_row_028(direction, step_index, alpha, predicted_s, predicted_s, trace, eval, working_prev_corrected, "ok", false)
            push!(rows, row)
            println("[$direction]   corrected alpha=$(@sprintf("%.12f", alpha)) x=$x_str s=$s_str F'=$fp_str jump_s=$jump_s_str jump_state=$jump_state_str status=ok")
            working_state = eval.current_state
            working_prev_corrected = eval.current_state
        catch err
            corrector_status = sprint(showerror, err)
            row = carried_predictor_row_028(
                direction,
                step_index,
                alpha,
                predicted_s,
                working_state,
                working_prev_corrected,
                "predictor_only_after_corrector_failure: $corrector_status",
                false,
            )
            push!(rows, row)
            println("[$direction]   predictor-only after corrector failure alpha=$(@sprintf("%.12f", alpha)) x=$predictor_x_str (initial condition held fixed)")
        end
    end

    return rows, traces, working_state, working_prev_corrected, working_forced_jump_used
end

function continuation_direction_028(
    direction::String,
    alphas::Vector{Float64},
    lambda::Float64,
    seed_eval::ReturnMapEval028,
)
    rows = ContinuationRow028[]
    traces = TraceRow028[]
    current_state = seed_eval.current_state
    prev_corrected_state = seed_eval.current_state
    pending = PendingStep030[]
    forced_jump_used = false

    if ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION > 0
        alphas = alphas[1:min(end, ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION)]
    end

    for (step_index, alpha) in enumerate(alphas)
        alpha_str = @sprintf("%.12f", alpha)
        predictor_x_str = @sprintf("%.12f", current_state[1])
        println("[$direction] alpha=$alpha_str predictor_x=$predictor_x_str")
        push!(pending, PendingStep030(step_index, alpha))

        do_flush = ATTEMPT028_CONT_CORRECT_EVERY <= 1 || (step_index % ATTEMPT028_CONT_CORRECT_EVERY == 0)
        if do_flush
            println("[$direction]   backfilling $(length(pending)) pending alpha values since the last corrected point")
            flush_rows, flush_traces, current_state, prev_corrected_state, forced_jump_used = process_pending_steps_030(
                direction,
                pending,
                lambda,
                current_state,
                prev_corrected_state,
                forced_jump_used,
            )
            append!(rows, flush_rows)
            append!(traces, flush_traces)
            empty!(pending)
        end
    end

    if !isempty(pending)
        println("[$direction] final backfill of $(length(pending)) pending alpha values")
        flush_rows, flush_traces, current_state, prev_corrected_state, forced_jump_used = process_pending_steps_030(
            direction,
            pending,
            lambda,
            current_state,
            prev_corrected_state,
            forced_jump_used,
        )
        append!(rows, flush_rows)
        append!(traces, flush_traces)
    end

    return rows, traces
end

function write_results_tsv_028(path::String, seed_row::ContinuationRow028, rows::Vector{ContinuationRow028})
    ordered = [seed_row; sort(rows, by=row -> (row.alpha, row.direction == "seed" ? 0 : (row.direction == "decreasing" ? 1 : 2)))]
    open(path, "w") do io
        println(io, "direction\tstep_index\talpha\tpredicted_s\tbranch_points\tbranch_s_min\tbranch_s_max\tused_s0\tconverged\tpredictor_only\tbroke\tstatus\taccepted_steps\ttotal_trace_rows\tfinal_s\tmap_value\tfirst_derivative\tsecond_derivative\tcorrection_abs\tstate_jump\tevent_time\tcurrent_x\tcurrent_y\tcurrent_z\tevent_x\tevent_y\tevent_z\tevent_time_d1\tevent_time_d2\tsciml_first_derivative\tfirst_derivative_mismatch")
        for row in ordered
            @printf(
                io,
                "%s\t%d\t%.12f\t%.12f\t%d\t%.12f\t%.12f\t%.12f\t%s\t%s\t%s\t%s\t%d\t%d\t%.12f\t%.12f\t%.12e\t%.12e\t%.12e\t%.12e\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12f\t%.12e\t%.12e\t%.12e\t%.12e\n",
                row.direction,
                row.step_index,
                row.alpha,
                row.predicted_s,
                row.branch_points,
                row.branch_s_min,
                row.branch_s_max,
                row.used_s0,
                row.converged ? "true" : "false",
                row.predictor_only ? "true" : "false",
                row.broke ? "true" : "false",
                replace(row.status, '\t' => ' '),
                row.accepted_steps,
                row.total_trace_rows,
                row.final_s,
                row.map_value,
                row.first_derivative,
                row.second_derivative,
                row.correction_abs,
                row.state_jump,
                row.event_time,
                row.current_x,
                row.current_y,
                row.current_z,
                row.event_x,
                row.event_y,
                row.event_z,
                row.event_time_d1,
                row.event_time_d2,
                row.sciml_first_derivative,
                row.first_derivative_mismatch,
            )
        end
    end
end

function write_trace_tsv_028(path::String, seed_trace::Vector{NewtonTrace028}, dec_traces::Vector{TraceRow028}, inc_traces::Vector{TraceRow028}, seed_alpha::Float64)
    open(path, "w") do io
        println(io, "direction\talpha\tstep_index\titer\ts\tmap_value\tfirst_derivative\tsecond_derivative\traw_step\tdamping\taccepted\tsciml_first_derivative\tfirst_derivative_mismatch")
        for row in seed_trace
            @printf(
                io,
                "seed\t%.12f\t0\t%d\t%.12f\t%.12f\t%.12e\t%.12e\t%.12e\t%.6f\t%s\t%.12e\t%.12e\n",
                seed_alpha,
                row.iter,
                row.s,
                row.value,
                row.first_derivative,
                row.second_derivative,
                row.step,
                row.damping,
                row.accepted ? "true" : "false",
                row.sciml_first_derivative,
                row.first_derivative_mismatch,
            )
        end
        for row in vcat(dec_traces, inc_traces)
            @printf(
                io,
                "%s\t%.12f\t%d\t%d\t%.12f\t%.12f\t%.12e\t%.12e\t%.12e\t%.6f\t%s\t%.12e\t%.12e\n",
                row.direction,
                row.alpha,
                row.step_index,
                row.iter,
                row.s,
                row.map_value,
                row.first_derivative,
                row.second_derivative,
                row.raw_step,
                row.damping,
                row.accepted ? "true" : "false",
                row.sciml_first_derivative,
                row.first_derivative_mismatch,
            )
        end
    end
end

function write_summary_md_028(path::String, seed_row::ContinuationRow028, dec_rows::Vector{ContinuationRow028}, inc_rows::Vector{ContinuationRow028}, seed_index::Int)
    all_rows = [seed_row; dec_rows; inc_rows]
    ok_rows = [row for row in all_rows if row.converged]
    predictor_only_rows = [row for row in all_rows if row.predictor_only]
    break_rows = [row for row in all_rows if row.broke && row.direction != "seed"]
    max_corr = isempty(ok_rows) ? nothing : ok_rows[argmax(row.correction_abs for row in ok_rows)]
    max_jump = isempty(ok_rows) ? nothing : ok_rows[argmax(row.state_jump for row in ok_rows)]

    open(path, "w") do io
        println(io, "# Attempt 030 Alpha Continuation of `|x|`-Map Local Minimum")
        println(io)
        @printf(io, "- Generated: `%s`\n", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
        @printf(io, "- Fixed `lambda = %.6f`\n", ATTEMPT028_CONT_LAMBDA)
        @printf(io, "- Fixed `B = %.1f`\n", ATTEMPT028_B)
        @printf(io, "- Alpha grid matches attempt-027: `range(%.1f, %.1f, length=%d)`\n", ATTEMPT028_CONT_ALPHA_MIN, ATTEMPT028_CONT_ALPHA_MAX, ATTEMPT028_CONT_N_ALPHA)
        @printf(io, "- Exact seed alpha: `%.6f`\n", ATTEMPT028_CONT_SEED_ALPHA)
        @printf(io, "- Seed target minimum guess near `s = %.4f`\n", ATTEMPT028_CONT_SEED_TARGET_S)
        println(io, "- Predictor between correction batches is exact carry-forward of the last corrected initial condition")
        println(io, "- The corrector uses `x` as the Newton variable and keeps `z` fixed during each Newton solve")
        @printf(io, "- Corrector cadence: every `%d` alpha values\n", ATTEMPT028_CONT_CORRECT_EVERY)
        println(io, "- When a scheduled correction point is reached, the code backfills the entire pending alpha block since the last flush")
        println(io, "- Failed or would-stop correction points are left predictor-only, and later pending points are still attempted")
        @printf(io, "- Would-stop thresholds: `|Δs| > %.6f` or `||Δstate|| > %.6f`\n", ATTEMPT028_CONT_BREAK_S_JUMP, ATTEMPT028_CONT_BREAK_STATE_JUMP)
        println(io)
        println(io, "## Seed")
        println(io)
        @printf(io, "- Seed discrete-minimum branch index: `%d`\n", seed_index)
        @printf(io, "- Seed refined `s* = %.12f`\n", seed_row.final_s)
        @printf(io, "- Seed initial condition `(x,y,z) = (%.12f, %.12f, %.12f)`\n", seed_row.current_x, seed_row.current_y, seed_row.current_z)
        println(io)
        println(io, "## Continuation Outcome")
        println(io)
        @printf(io, "- Decreasing-alpha converged points: `%d`\n", count(row -> row.converged, dec_rows))
        @printf(io, "- Increasing-alpha converged points: `%d`\n", count(row -> row.converged, inc_rows))
        @printf(io, "- Predictor-only rows: `%d`\n", length(predictor_only_rows))
        @printf(io, "- Would-stop rows detected: `%d`\n", length(break_rows))
        if !isempty(break_rows)
            first_break = break_rows[1]
            @printf(io, "- First would-stop row: direction `%s`, alpha `%.12f`, status `%s`\n", first_break.direction, first_break.alpha, first_break.status)
        end
        if !isnothing(max_corr)
            row = max_corr
            @printf(io, "- Largest accepted corrector update: `%.6e` at alpha `%.12f` (%s)\n", row.correction_abs, row.alpha, row.direction)
        end
        if !isnothing(max_jump)
            row = max_jump
            @printf(io, "- Largest adjacent state jump: `%.6e` at alpha `%.12f` (%s)\n", row.state_jump, row.alpha, row.direction)
        end
        println(io)
        println(io, "Outputs:")
        println(io)
        println(io, "- Results TSV: `$(basename(ATTEMPT028_CONT_RESULTS_PATH))`")
        println(io, "- Trace TSV: `$(basename(ATTEMPT028_CONT_TRACE_PATH))`")
        println(io, "- Summary: `$(basename(ATTEMPT028_CONT_SUMMARY_PATH))`")
    end
end

function main()
    max_steps_label = ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION == 0 ? "all" : string(ATTEMPT028_CONT_MAX_STEPS_PER_DIRECTION)
    println("Attempt 030 alpha continuation")
    println("  lambda = $(ATTEMPT028_CONT_LAMBDA)")
    println("  seed alpha = $(ATTEMPT028_CONT_SEED_ALPHA)")
    println("  seed target s = $(ATTEMPT028_CONT_SEED_TARGET_S)")
    println("  break thresholds: Δs=$(ATTEMPT028_CONT_BREAK_S_JUMP), state=$(ATTEMPT028_CONT_BREAK_STATE_JUMP)")
    println("  max steps per direction = $max_steps_label")
    println()

    seed_branch, _ = branch_for_alpha_028(ATTEMPT028_CONT_SEED_ALPHA, ATTEMPT028_CONT_LAMBDA)
    seed_index, seed_s0 = seed_from_target_minimum_028(seed_branch, ATTEMPT028_CONT_SEED_TARGET_S)
    seed_x0 = seed_branch[seed_index].x
    seed_z0 = seed_branch[seed_index].z
    println("Seed discrete minimum index = $seed_index")
    @printf("Seed discrete s0 = %.12f\n", seed_s0)
    seed_trace, seed_eval = run_damped_newton_xfixed_028(
        ATTEMPT028_CONT_SEED_ALPHA,
        ATTEMPT028_CONT_LAMBDA,
        seed_x0,
        seed_z0,
        ATTEMPT028_TARGET_NEXT_SIGN,
    )
    @printf("Seed refined s* = %.12f\n", seed_eval.s)
    @printf("Seed current state = (%.12f, %.12f, %.12f)\n", seed_eval.current_state[1], seed_eval.current_state[2], seed_eval.current_state[3])
    println()

    seed_row = build_row_028(
        "seed",
        0,
        ATTEMPT028_CONT_SEED_ALPHA,
        seed_s0,
        seed_s0,
        seed_trace,
        seed_eval,
        nothing,
        "ok",
        false,
    )

    dec_alphas = reverse([alpha for alpha in ATTEMPT028_CONT_ALPHAS if alpha < ATTEMPT028_CONT_SEED_ALPHA])
    inc_alphas = [alpha for alpha in ATTEMPT028_CONT_ALPHAS if alpha > ATTEMPT028_CONT_SEED_ALPHA]

    dec_task = Threads.@spawn continuation_direction_028("decreasing", dec_alphas, ATTEMPT028_CONT_LAMBDA, seed_eval)
    inc_task = Threads.@spawn continuation_direction_028("increasing", inc_alphas, ATTEMPT028_CONT_LAMBDA, seed_eval)
    dec_rows, dec_traces = fetch(dec_task)
    inc_rows, inc_traces = fetch(inc_task)

    write_results_tsv_028(ATTEMPT028_CONT_RESULTS_PATH, seed_row, vcat(dec_rows, inc_rows))
    write_trace_tsv_028(ATTEMPT028_CONT_TRACE_PATH, seed_trace, dec_traces, inc_traces, ATTEMPT028_CONT_SEED_ALPHA)
    write_summary_md_028(ATTEMPT028_CONT_SUMMARY_PATH, seed_row, dec_rows, inc_rows, seed_index)

    println()
    println("Wrote:")
    println("  $(ATTEMPT028_CONT_RESULTS_PATH)")
    println("  $(ATTEMPT028_CONT_TRACE_PATH)")
    println("  $(ATTEMPT028_CONT_SUMMARY_PATH)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
