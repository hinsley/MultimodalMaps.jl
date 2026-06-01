include(joinpath(@__DIR__, "rossler_common.jl"))

using Printf
using LinearAlgebra: norm

function env_float(name, default)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_int(name, default)
    return parse(Int, get(ENV, name, string(default)))
end

function env_bool(name, default)
    value = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return value in ("1", "true", "yes", "on")
end

const CRITICAL_ORBIT_HEADER = [
    "a",
    "c",
    "b",
    "status",
    "events",
    "word",
    "code",
    "period",
    "gamma",
    "max_time",
    "first_time",
    "last_time",
    "min_y",
    "max_y",
    "critical_status",
    "critical_event_index",
    "critical_rho",
    "critical_residual",
    "critical_iterations",
    "critical_message",
    "critical_time",
    "critical_x",
    "critical_y",
    "critical_z",
    "critical_next_y",
    "critical_dx_current",
    "critical_dx_next",
    "critical_event_denominator",
    "critical_event_value",
    "critical_second_derivative",
    "critical_tangent_x",
    "critical_tangent_y",
    "critical_tangent_z",
    "orbit_transient_events",
    "first_event_is_return",
    "initial_event_included",
]

function write_tsv_header(io)
    println(io, join(CRITICAL_ORBIT_HEADER, '\t'))
end

function tsv_value(value)
    if value isa AbstractString
        return value
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return isfinite(value) ? @sprintf("%.12g", value) : "NaN"
    else
        return string(value)
    end
end

function write_tsv_row(io, row)
    println(io, join((tsv_value(getproperty(row, Symbol(name))) for name in CRITICAL_ORBIT_HEADER), '\t'))
end

function tsv_row_string(row)
    io = IOBuffer()
    write_tsv_row(io, row)
    return String(take!(io))
end

function finite_values(values)
    out = Float64[]
    for value in values
        if !isnothing(value) && isfinite(Float64(value))
            push!(out, Float64(value))
        end
    end
    return unique(out)
end

function critical_residual_at(problem, ray, rho; event_index, denominator_atol, critical_tspan, event_denom_atol, max_state, abstol, reltol)
    return criticality_residual(
        problem,
        ray,
        rho;
        event_index=event_index,
        denominator_atol=denominator_atol,
        tspan=(0.0, critical_tspan),
        denom_atol=event_denom_atol,
        max_state=max_state,
        abstol=abstol,
        reltol=reltol,
    )
end

function newton_correct_rho(
    problem,
    ray,
    initial_rho;
    event_index,
    tolerance,
    max_iterations,
    finite_difference_step,
    max_newton_step,
    denominator_atol,
    critical_tspan,
    event_denom_atol,
    max_state,
    abstol,
    reltol,
)
    rho = Float64(initial_rho)
    result = critical_residual_at(
        problem,
        ray,
        rho;
        event_index=event_index,
        denominator_atol=denominator_atol,
        critical_tspan=critical_tspan,
        event_denom_atol=event_denom_atol,
        max_state=max_state,
        abstol=abstol,
        reltol=reltol,
    )
    iterations = 0
    message = result.message
    while result.valid && abs(result.residual) > tolerance && iterations < max_iterations
        iterations += 1
        rp = critical_residual_at(
            problem,
            ray,
            rho + finite_difference_step;
            event_index=event_index,
            denominator_atol=denominator_atol,
            critical_tspan=critical_tspan,
            event_denom_atol=event_denom_atol,
            max_state=max_state,
            abstol=abstol,
            reltol=reltol,
        )
        rm = critical_residual_at(
            problem,
            ray,
            rho - finite_difference_step;
            event_index=event_index,
            denominator_atol=denominator_atol,
            critical_tspan=critical_tspan,
            event_denom_atol=event_denom_atol,
            max_state=max_state,
            abstol=abstol,
            reltol=reltol,
        )
        if !(rp.valid && rm.valid)
            message = "finite-difference residual failed"
            break
        end
        slope = (rp.residual - rm.residual) / (2 * finite_difference_step)
        if !isfinite(slope) || abs(slope) <= eps(Float64)
            message = "Newton slope vanished"
            break
        end
        step = clamp(result.residual / slope, -max_newton_step, max_newton_step)
        rho -= step
        result = critical_residual_at(
            problem,
            ray,
            rho;
            event_index=event_index,
            denominator_atol=denominator_atol,
            critical_tspan=critical_tspan,
            event_denom_atol=event_denom_atol,
            max_state=max_state,
            abstol=abstol,
            reltol=reltol,
        )
        message = result.message
    end
    return (rho=rho, result=result, iterations=iterations, message=message)
end

function choose_root(roots, rho_guesses, target_y)
    isempty(roots) && return nothing
    guesses = finite_values(rho_guesses)
    has_target = !isnothing(target_y) && isfinite(Float64(target_y))
    scores = map(roots) do root
        rho_score = isempty(guesses) ? 0.0 : minimum(abs(root.rho - guess) for guess in guesses)
        y_score = has_target ? 10.0 * abs(root.x_current - Float64(target_y)) : 0.0
        residual_score = 10.0 * abs(root.residual)
        rho_weight = has_target ? 0.01 : 1.0
        rho_weight * rho_score + y_score + residual_score
    end
    return roots[argmin(scores)]
end

function choose_corrected(candidates, rho_guesses, target_y; tolerance)
    isempty(candidates) && return nothing
    roots = map(candidates) do candidate
        (rho=candidate.rho, x_current=candidate.result.x_current, residual=candidate.result.residual)
    end
    selected = choose_root(roots, rho_guesses, target_y)
    isnothing(selected) && return nothing
    for candidate in candidates
        if candidate.rho == selected.rho
            if candidate.result.valid && abs(candidate.result.residual) <= tolerance
                return candidate
            end
            return nothing
        end
    end
    return nothing
end

function locate_critical_point(
    problem,
    ray;
    rho_guesses=Float64[],
    target_y=nothing,
    event_index=4,
    rho_range=(-24.0, -2.0),
    rho_samples=45,
    tolerance=1e-6,
    max_iterations=8,
    finite_difference_step=1e-4,
    max_newton_step=0.5,
    denominator_atol=1e-9,
    critical_tspan=650.0,
    event_denom_atol=1e-10,
    max_state=1e6,
    abstol=1e-9,
    reltol=1e-9,
)
    best = nothing
    candidates = []
    for guess in finite_values(rho_guesses)
        corrected = newton_correct_rho(
            problem,
            ray,
            guess;
            event_index=event_index,
            tolerance=tolerance,
            max_iterations=max_iterations,
            finite_difference_step=finite_difference_step,
            max_newton_step=max_newton_step,
            denominator_atol=denominator_atol,
            critical_tspan=critical_tspan,
            event_denom_atol=event_denom_atol,
            max_state=max_state,
            abstol=abstol,
            reltol=reltol,
        )
        if corrected.result.valid
            if abs(corrected.result.residual) <= tolerance
                push!(candidates, corrected)
            end
            if isnothing(best) || abs(corrected.result.residual) < abs(best.result.residual)
                best = corrected
            end
        end
    end
    selected = choose_corrected(candidates, rho_guesses, target_y; tolerance=tolerance)
    if !isnothing(selected)
        return selected
    end

    roots, _ = find_seeded_critical_points(
        problem,
        ray;
        event_index=event_index,
        rho_range=rho_range,
        samples=rho_samples,
        root_tol=tolerance,
        max_bisection_iterations=40,
        denominator_atol=denominator_atol,
        tspan=(0.0, critical_tspan),
        denom_atol=event_denom_atol,
        max_state=max_state,
        abstol=abstol,
        reltol=reltol,
    )
    root = choose_root(roots, rho_guesses, target_y)
    if !isnothing(root)
        corrected = newton_correct_rho(
            problem,
            ray,
            root.rho;
            event_index=event_index,
            tolerance=tolerance,
            max_iterations=max_iterations,
            finite_difference_step=finite_difference_step,
            max_newton_step=max_newton_step,
            denominator_atol=denominator_atol,
            critical_tspan=critical_tspan,
            event_denom_atol=event_denom_atol,
            max_state=max_state,
            abstol=abstol,
            reltol=reltol,
        )
        return corrected
    end

    if !isnothing(best)
        return best
    end

    fallback = critical_residual_at(
        problem,
        ray,
        first(rho_range);
        event_index=event_index,
        denominator_atol=denominator_atol,
        critical_tspan=critical_tspan,
        event_denom_atol=event_denom_atol,
        max_state=max_state,
        abstol=abstol,
        reltol=reltol,
    )
    return (rho=Float64(first(rho_range)), result=fallback, iterations=0, message="critical root scan failed")
end

function failed_row(a, c, b, status, critical_status, event_index, message; max_time)
    nan = NaN
    return (
        a=Float64(a), c=Float64(c), b=Float64(b), status=status,
        events=0, word="", code=-1, period=0, gamma=nan, max_time=Float64(max_time),
        first_time=nan, last_time=nan, min_y=nan, max_y=nan,
        critical_status=critical_status, critical_event_index=Int(event_index),
        critical_rho=nan, critical_residual=nan, critical_iterations=0,
        critical_message=message, critical_time=nan, critical_x=nan, critical_y=nan, critical_z=nan,
        critical_next_y=nan, critical_dx_current=nan, critical_dx_next=nan,
        critical_event_denominator=nan, critical_event_value=nan, critical_second_derivative=nan,
        critical_tangent_x=nan, critical_tangent_y=nan, critical_tangent_z=nan,
        orbit_transient_events=0, first_event_is_return=false, initial_event_included=false,
    )
end

function tangent_event_at_initial(problem, state, tangent, second_derivative)
    component = Float64(tangent[problem.variable_index])
    return TangentEvent(
        0.0,
        collect(Float64, state),
        Float64(state[problem.variable_index]),
        Float64(second_derivative),
        component,
        component > 0 ? 1 : (component < 0 ? -1 : 0),
    )
end

function scan_point(
    a,
    c;
    b=ROSSLER_MALYKH_B,
    event_index=4,
    rho_guesses=Float64[],
    target_y=nothing,
    word_length=8,
    dt=0.05,
    max_time=2000.0,
    critical_tspan=650.0,
    max_state=1e6,
    rho_range=(-24.0, -1.0),
    rho_samples=45,
    tolerance=1e-6,
    max_iterations=8,
    finite_difference_step=1e-4,
    max_newton_step=0.5,
    denominator_atol=1e-9,
    event_denom_atol=1e-10,
    abstol=1e-9,
    reltol=1e-9,
)
    problem = rossler_y_minima_problem(a, c; b=b)
    ray = saddle_focus_seed_ray(problem, rossler_origin_equilibrium())
    corrected = locate_critical_point(
        problem,
        ray;
        rho_guesses=rho_guesses,
        target_y=target_y,
        event_index=event_index,
        rho_range=rho_range,
        rho_samples=rho_samples,
        tolerance=tolerance,
        max_iterations=max_iterations,
        finite_difference_step=finite_difference_step,
        max_newton_step=max_newton_step,
        denominator_atol=denominator_atol,
        critical_tspan=critical_tspan,
        event_denom_atol=event_denom_atol,
        max_state=max_state,
        abstol=abstol,
        reltol=reltol,
    )
    result = corrected.result
    if !(result.valid && abs(result.residual) <= tolerance)
        return failed_row(a, c, b, "critical_fail", "critical_fail", event_index, corrected.message; max_time=max_time)
    end

    seeded_events = collect_seeded_tangent_extrema(
        problem,
        ray,
        corrected.rho;
        max_events=event_index + 1,
        tspan=(0.0, critical_tspan),
        denom_atol=event_denom_atol,
        max_state=max_state,
        abstol=abstol,
        reltol=reltol,
    )
    if length(seeded_events) < event_index + 1
        return failed_row(a, c, b, "critical_fail", "critical_event_short", event_index, "not enough seeded tangent extrema"; max_time=max_time)
    end

    critical = seeded_events[event_index]
    next_critical = seeded_events[event_index + 1]
    critical_event_value = extremum_event_value(problem, critical.u, critical.t)
    critical_second_derivative = extremum_second_derivative(problem, critical.u, critical.t)
    tangent0 = critical.tangent
    if any(!isfinite, tangent0) || norm(tangent0) <= 0
        return failed_row(a, c, b, "critical_fail", "critical_tangent_fail", event_index, "invalid critical tangent"; max_time=max_time)
    end

    initial_event = tangent_event_at_initial(problem, critical.u, tangent0, critical_second_derivative)
    return_events = word_length > 1 ? collect_tangent_extrema_rk4(
        problem,
        critical.u,
        tangent0;
        observable_index=2,
        dt=dt,
        t_end=max_time,
        transient_events=0,
        max_events=word_length - 1,
        max_state=max_state,
        min_event_time=max(1e-8, 0.5 * dt),
    ) : TangentEvent[]
    orbit_events = TangentEvent[initial_event]
    append!(orbit_events, return_events)
    bits = tangent_bits(orbit_events)
    full_word = !isnothing(bits) && length(bits) == word_length
    period = full_word ? least_period(bits) : 0
    gamma = full_word ? binary_sequence_value(bits) : NaN
    status = length(orbit_events) == word_length ? "ok" : "orbit_max_time"

    return (
        a=Float64(a),
        c=Float64(c),
        b=Float64(b),
        status=status,
        events=length(orbit_events),
        word=tangent_word(orbit_events),
        code=full_word ? tangent_word_code(orbit_events) : -1,
        period=period,
        gamma=gamma,
        max_time=Float64(max_time),
        first_time=isempty(orbit_events) ? NaN : first(orbit_events).t,
        last_time=isempty(orbit_events) ? NaN : last(orbit_events).t,
        min_y=isempty(orbit_events) ? NaN : minimum(event.value for event in orbit_events),
        max_y=isempty(orbit_events) ? NaN : maximum(event.value for event in orbit_events),
        critical_status="ok",
        critical_event_index=Int(event_index),
        critical_rho=Float64(corrected.rho),
        critical_residual=Float64(result.residual),
        critical_iterations=Int(corrected.iterations),
        critical_message="ok",
        critical_time=Float64(critical.t),
        critical_x=Float64(critical.u[1]),
        critical_y=Float64(critical.u[2]),
        critical_z=Float64(critical.u[3]),
        critical_next_y=Float64(next_critical.value),
        critical_dx_current=Float64(critical.derivative),
        critical_dx_next=Float64(next_critical.derivative),
        critical_event_denominator=Float64(critical.event_denominator),
        critical_event_value=Float64(critical_event_value),
        critical_second_derivative=Float64(critical_second_derivative),
        critical_tangent_x=Float64(tangent0[1]),
        critical_tangent_y=Float64(tangent0[2]),
        critical_tangent_z=Float64(tangent0[3]),
        orbit_transient_events=0,
        first_event_is_return=false,
        initial_event_included=true,
    )
end

function compute_column_anchors(
    c_values,
    a_start;
    b,
    event_index,
    word_length,
    dt,
    max_time,
    critical_tspan,
    max_state,
    rho_range,
    rho_samples,
    tolerance,
)
    anchors = Vector{Any}(undef, length(c_values))
    fill!(anchors, nothing)
    last_rho = nothing
    last_y = nothing
    started = time()

    @printf(
        "critical_orbit column anchors started columns=%d a_start=%.6f elapsed=%.1fs\n",
        length(c_values),
        a_start,
        0.0,
    )
    flush(stdout)

    for c_idx in eachindex(c_values)
        c = c_values[c_idx]
        row = scan_point(
            a_start,
            c;
            b=b,
            event_index=event_index,
            rho_guesses=finite_values(Union{Nothing,Float64}[last_rho]),
            target_y=last_y,
            word_length=word_length,
            dt=dt,
            max_time=max_time,
            critical_tspan=critical_tspan,
            max_state=max_state,
            rho_range=rho_range,
            rho_samples=rho_samples,
            tolerance=tolerance,
        )
        if row.critical_status == "ok"
            anchors[c_idx] = (rho=row.critical_rho, y=row.critical_y)
            last_rho = row.critical_rho
            last_y = row.critical_y
        end
        if c_idx == 1 || c_idx == length(c_values) || c_idx % 64 == 0
            elapsed = time() - started
            @printf(
                "critical_orbit column anchors %d/%d c_idx=%d c=%.6f status=%s critical=%s elapsed=%.1fs\n",
                c_idx,
                length(c_values),
                c_idx,
                c,
                row.status,
                row.critical_status,
                elapsed,
            )
            flush(stdout)
        end
    end

    return anchors
end

function run_scan_parallel_columns(;
    c_min=2.0,
    c_max=7.0,
    a_min=0.30,
    a_max=0.55,
    n_c=256,
    n_a=256,
    a_start_index=1,
    a_end_index=n_a,
    b=ROSSLER_MALYKH_B,
    event_index=4,
    word_length=8,
    dt=0.05,
    max_time=2000.0,
    critical_tspan=650.0,
    max_state=1e6,
    rho_range=(-24.0, -1.0),
    rho_samples=45,
    tolerance=1e-6,
    progress_seconds=30.0,
    output=joinpath(@__DIR__, "..", "results", "rossler_y_minima_critical_orbit_scan_256", "coarse_scan.tsv"),
)
    event_index > 0 || throw(ArgumentError("event_index must be positive"))
    word_length > 0 || throw(ArgumentError("word_length must be positive"))
    a_start_index = max(1, Int(a_start_index))
    a_end_index = min(Int(n_a), Int(a_end_index))
    a_start_index <= a_end_index || throw(ArgumentError("empty a-index range"))

    c_values = collect(range(c_min, c_max; length=n_c))
    a_values = collect(range(a_min, a_max; length=n_a))
    a_indices = collect(a_start_index:a_end_index)
    mkpath(dirname(output))

    total = length(a_indices) * n_c
    started = time()
    last_report = started - progress_seconds
    scanned = 0
    ok = 0
    progress_lock = ReentrantLock()
    columns = Vector{Vector{String}}(undef, n_c)

    anchors = compute_column_anchors(
        c_values,
        a_values[a_start_index];
        b=b,
        event_index=event_index,
        word_length=word_length,
        dt=dt,
        max_time=max_time,
        critical_tspan=critical_tspan,
        max_state=max_state,
        rho_range=rho_range,
        rho_samples=rho_samples,
        tolerance=tolerance,
    )

    @printf(
        "critical_orbit parallel columns started columns=%d a_points=%d threads=%d output=%s\n",
        length(c_values),
        length(a_indices),
        Threads.nthreads(),
        output,
    )
    flush(stdout)

    function report_progress(row, a_idx, c_idx, a, c)
        lock(progress_lock)
        try
            scanned += 1
            ok += row.status == "ok" ? 1 : 0
            now = time()
            if scanned == 1 || scanned == total || now - last_report >= progress_seconds
                elapsed = now - started
                rate = elapsed > 0 ? scanned / elapsed : NaN
                eta = isfinite(rate) && rate > 0 ? (total - scanned) / rate : NaN
                @printf(
                    "critical_orbit threaded scanned %d/%d a_idx=%d c_idx=%d a=%.6f c=%.6f status=%s critical=%s word=%s ok=%d elapsed=%.1fs rate=%.2f/s eta=%.1fs threads=%d axis=columns\n",
                    scanned,
                    total,
                    a_idx,
                    c_idx,
                    a,
                    c,
                    row.status,
                    row.critical_status,
                    row.word,
                    ok,
                    elapsed,
                    rate,
                    eta,
                    Threads.nthreads(),
                )
                flush(stdout)
                last_report = now
            end
        finally
            unlock(progress_lock)
        end
    end

    Threads.@threads for c_idx in eachindex(c_values)
        c = c_values[c_idx]
        lines = Vector{String}(undef, length(a_indices))
        last_rho = isnothing(anchors[c_idx]) ? nothing : anchors[c_idx].rho
        last_y = isnothing(anchors[c_idx]) ? nothing : anchors[c_idx].y
        for (a_pos, a_idx) in enumerate(a_indices)
            a = a_values[a_idx]
            row = scan_point(
                a,
                c;
                b=b,
                event_index=event_index,
                rho_guesses=finite_values(Union{Nothing,Float64}[last_rho]),
                target_y=last_y,
                word_length=word_length,
                dt=dt,
                max_time=max_time,
                critical_tspan=critical_tspan,
                max_state=max_state,
                rho_range=rho_range,
                rho_samples=rho_samples,
                tolerance=tolerance,
            )
            lines[a_pos] = tsv_row_string(row)
            if row.critical_status == "ok"
                last_rho = row.critical_rho
                last_y = row.critical_y
            end
            report_progress(row, a_idx, c_idx, a, c)
        end
        columns[c_idx] = lines
    end

    open(output, "w") do io
        write_tsv_header(io)
        for a_pos in eachindex(a_indices)
            for c_idx in eachindex(c_values)
                print(io, columns[c_idx][a_pos])
            end
        end
    end

    return output
end

function run_scan(;
    c_min=2.0,
    c_max=7.0,
    a_min=0.30,
    a_max=0.55,
    n_c=256,
    n_a=256,
    a_start_index=1,
    a_end_index=n_a,
    b=ROSSLER_MALYKH_B,
    event_index=4,
    word_length=8,
    dt=0.05,
    max_time=2000.0,
    critical_tspan=650.0,
    max_state=1e6,
    rho_range=(-24.0, -1.0),
    rho_samples=45,
    tolerance=1e-6,
    progress_seconds=30.0,
    serpentine=true,
    output=joinpath(@__DIR__, "..", "results", "rossler_y_minima_critical_orbit_scan_256", "coarse_scan.tsv"),
)
    event_index > 0 || throw(ArgumentError("event_index must be positive"))
    word_length > 0 || throw(ArgumentError("word_length must be positive"))
    a_start_index = max(1, Int(a_start_index))
    a_end_index = min(Int(n_a), Int(a_end_index))
    a_start_index <= a_end_index || throw(ArgumentError("empty a-index range"))

    c_values = collect(range(c_min, c_max; length=n_c))
    a_values = collect(range(a_min, a_max; length=n_a))
    mkpath(dirname(output))
    total = (a_end_index - a_start_index + 1) * n_c
    started = time()
    last_report = started - progress_seconds
    scanned = 0
    ok = 0
    previous_row_rhos = Vector{Union{Nothing,Float64}}(nothing, n_c)
    previous_row_y = Vector{Union{Nothing,Float64}}(nothing, n_c)

    open(output, "w") do io
        write_tsv_header(io)
        for a_idx in a_start_index:a_end_index
            a = a_values[a_idx]
            current_row_rhos = Vector{Union{Nothing,Float64}}(nothing, n_c)
            current_row_y = Vector{Union{Nothing,Float64}}(nothing, n_c)
            row_results = Vector{Any}(undef, n_c)
            c_order = serpentine && iseven(a_idx - a_start_index + 1) ? reverse(eachindex(c_values)) : eachindex(c_values)
            last_rho = nothing
            last_y = nothing
            for c_idx in c_order
                c = c_values[c_idx]
                rho_guesses = Union{Nothing,Float64}[
                    last_rho,
                    previous_row_rhos[c_idx],
                    c_idx > 1 ? previous_row_rhos[c_idx - 1] : nothing,
                    c_idx < n_c ? previous_row_rhos[c_idx + 1] : nothing,
                ]
                target_y = !isnothing(last_y) ? last_y : previous_row_y[c_idx]
                row = scan_point(
                    a,
                    c;
                    b=b,
                    event_index=event_index,
                    rho_guesses=finite_values(rho_guesses),
                    target_y=target_y,
                    word_length=word_length,
                    dt=dt,
                    max_time=max_time,
                    critical_tspan=critical_tspan,
                    max_state=max_state,
                    rho_range=rho_range,
                    rho_samples=rho_samples,
                    tolerance=tolerance,
                )
                row_results[c_idx] = row
                if row.critical_status == "ok"
                    current_row_rhos[c_idx] = row.critical_rho
                    current_row_y[c_idx] = row.critical_y
                    last_rho = row.critical_rho
                    last_y = row.critical_y
                end
                scanned += 1
                ok += row.status == "ok" ? 1 : 0
                now = time()
                if scanned == 1 || scanned == total || now - last_report >= progress_seconds
                    elapsed = now - started
                    rate = elapsed > 0 ? scanned / elapsed : NaN
                    eta = isfinite(rate) && rate > 0 ? (total - scanned) / rate : NaN
                    @printf(
                        "critical_orbit scanned %d/%d a_idx=%d c_idx=%d a=%.6f c=%.6f status=%s critical=%s word=%s ok=%d elapsed=%.1fs rate=%.2f/s eta=%.1fs\n",
                        scanned,
                        total,
                        a_idx,
                        c_idx,
                        a,
                        c,
                        row.status,
                        row.critical_status,
                        row.word,
                        ok,
                        elapsed,
                        rate,
                        eta,
                    )
                    flush(stdout)
                    last_report = now
                end
            end
            for c_idx in eachindex(c_values)
                write_tsv_row(io, row_results[c_idx])
            end
            flush(io)
            previous_row_rhos = current_row_rhos
            previous_row_y = current_row_y
        end
    end
    return output
end

function main()
    results_name = get(ENV, "MM_FLOW_FOLDING_RESULTS_NAME", "rossler_y_minima_critical_orbit_scan_256")
    result_dir = get(ENV, "MM_FLOW_FOLDING_RESULT_DIR", joinpath(@__DIR__, "..", "results", results_name))
    output = get(ENV, "MM_FLOW_FOLDING_OUTPUT", joinpath(result_dir, "coarse_scan.tsv"))
    parallel_axis = lowercase(strip(get(ENV, "MM_FLOW_FOLDING_PARALLEL_AXIS", Threads.nthreads() > 1 ? "columns" : "sequential")))
    common_kwargs = (
        c_min=env_float("MM_FLOW_FOLDING_C_MIN", 2.0),
        c_max=env_float("MM_FLOW_FOLDING_C_MAX", 7.0),
        a_min=env_float("MM_FLOW_FOLDING_A_MIN", 0.30),
        a_max=env_float("MM_FLOW_FOLDING_A_MAX", 0.55),
        n_c=env_int("MM_FLOW_FOLDING_NC", 256),
        n_a=env_int("MM_FLOW_FOLDING_NA", 256),
        a_start_index=env_int("MM_FLOW_FOLDING_A_START_INDEX", 1),
        a_end_index=env_int("MM_FLOW_FOLDING_A_END_INDEX", env_int("MM_FLOW_FOLDING_NA", 256)),
        b=env_float("MM_FLOW_FOLDING_B", ROSSLER_MALYKH_B),
        event_index=env_int("MM_FLOW_FOLDING_CRITICAL_EVENT_INDEX", 4),
        word_length=env_int("MM_FLOW_FOLDING_WORD_LENGTH", 8),
        dt=env_float("MM_FLOW_FOLDING_DT", 0.05),
        max_time=env_float("MM_FLOW_FOLDING_MAX_TIME", 2000.0),
        critical_tspan=env_float("MM_FLOW_FOLDING_CRITICAL_TSPAN", 650.0),
        max_state=env_float("MM_FLOW_FOLDING_MAX_STATE", 1e6),
        rho_range=(env_float("MM_FLOW_FOLDING_RHO_MIN", -24.0), env_float("MM_FLOW_FOLDING_RHO_MAX", -1.0)),
        rho_samples=env_int("MM_FLOW_FOLDING_RHO_SAMPLES", 45),
        tolerance=env_float("MM_FLOW_FOLDING_CRITICAL_TOL", 1e-6),
        progress_seconds=env_float("MM_FLOW_FOLDING_PROGRESS_SECONDS", 30.0),
        output=output,
    )
    path = if parallel_axis in ("columns", "column", "c")
        run_scan_parallel_columns(; common_kwargs...)
    elseif parallel_axis in ("sequential", "serial", "none", "false", "0")
        run_scan(;
            common_kwargs...,
            serpentine=env_bool("MM_FLOW_FOLDING_SERPENTINE", true),
        )
    else
        throw(ArgumentError("unsupported MM_FLOW_FOLDING_PARALLEL_AXIS=$(parallel_axis); use columns or sequential"))
    end
    @printf("wrote %s\n", path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
