include(joinpath(@__DIR__, "rossler_common.jl"))

using Printf

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

function scan_point(
    a,
    c;
    b=ROSSLER_MALYKH_B,
    word_length=8,
    transient_events=20,
    dt=0.05,
    max_time=450.0,
    max_state=1e6,
    u0=[1.0, 0.0, 0.0],
    tangent0=[1.0, 0.0, 0.0],
)
    problem = rossler_y_minima_problem(a, c; b=b)
    events = collect_tangent_extrema_rk4(
        problem,
        u0,
        tangent0;
        observable_index=2,
        dt=dt,
        t_end=max_time,
        transient_events=transient_events,
        max_events=word_length,
        max_state=max_state,
    )

    bits = tangent_bits(events)
    period = isnothing(bits) || length(bits) != word_length ? 0 : least_period(bits)
    gamma = isnothing(bits) || length(bits) != word_length ? NaN : binary_sequence_value(bits)
    status = length(events) == word_length ? "ok" : "max_time"
    return (
        a=Float64(a),
        c=Float64(c),
        b=Float64(b),
        status=status,
        events=length(events),
        word=tangent_word(events),
        code=tangent_word_code(events),
        period=period,
        gamma=gamma,
        max_time=Float64(max_time),
        first_time=isempty(events) ? NaN : first(events).t,
        last_time=isempty(events) ? NaN : last(events).t,
        min_y=isempty(events) ? NaN : minimum(event.value for event in events),
        max_y=isempty(events) ? NaN : maximum(event.value for event in events),
    )
end

function run_scan(;
    c_min=2.0,
    c_max=7.0,
    a_min=0.30,
    a_max=0.55,
    n_c=1024,
    n_a=1024,
    b=ROSSLER_MALYKH_B,
    word_length=8,
    transient_events=20,
    dt=0.05,
    max_time=450.0,
    max_state=1e6,
    progress_seconds=30.0,
)
    c_values = collect(range(c_min, c_max; length=n_c))
    a_values = collect(range(a_min, a_max; length=n_a))
    rows = Vector{NamedTuple}(undef, length(c_values) * length(a_values))
    k = 0
    total = length(rows)
    scan_started = time()
    last_report = scan_started - progress_seconds
    for a in a_values
        for c in c_values
            k += 1
            rows[k] = scan_point(
                a,
                c;
                b=b,
                word_length=word_length,
                transient_events=transient_events,
                dt=dt,
                max_time=max_time,
                max_state=max_state,
            )
            now = time()
            if k == 1 || k == total || now - last_report >= progress_seconds
                elapsed = now - scan_started
                rate = elapsed > 0.0 ? k / elapsed : NaN
                eta = isfinite(rate) && rate > 0.0 ? (total - k) / rate : NaN
                @printf(
                    "scanned %d/%d a=%.6f c=%.6f status=%s word=%s elapsed=%.1fs rate=%.2f/s eta=%.1fs\n",
                    k,
                    total,
                    a,
                    c,
                    rows[k].status,
                    rows[k].word,
                    elapsed,
                    rate,
                    eta,
                )
                flush(stdout)
                last_report = now
            end
        end
    end
    return rows
end

function write_tsv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "a\tc\tb\tstatus\tevents\tword\tcode\tperiod\tgamma\tmax_time\tfirst_time\tlast_time\tmin_y\tmax_y")
        for row in rows
            @printf(
                io,
                "%.12g\t%.12g\t%.12g\t%s\t%d\t%s\t%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
                row.a,
                row.c,
                row.b,
                row.status,
                row.events,
                row.word,
                row.code,
                row.period,
                row.gamma,
                row.max_time,
                row.first_time,
                row.last_time,
                row.min_y,
                row.max_y,
            )
        end
    end
    return path
end

function js_string(value)
    escaped = replace(string(value), "\\" => "\\\\", "\"" => "\\\"")
    return "\"" * escaped * "\""
end

function js_value(value)
    return value isa AbstractString ? js_string(value) : string(value)
end

function write_js_object(io, name, data; trailing_comma=true)
    println(io, "  $(name): {")
    fields = collect(pairs(data))
    for (i, pair) in enumerate(fields)
        comma = i == length(fields) ? "" : ","
        println(io, "    $(pair.first): $(js_value(pair.second))$(comma)")
    end
    println(io, "  }$(trailing_comma ? "," : "")")
end

function write_docs_data(path, rows; config, runtime=nothing)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "window.FLOW_FOLDING_ROSSLER_SCAN = {")
        write_js_object(io, "config", config)
        if !isnothing(runtime)
            write_js_object(io, "runtime", runtime)
        end
        println(io, "  rows: [")
        for (i, row) in enumerate(rows)
            comma = i == length(rows) ? "" : ","
            print(io, "    {")
            print(io, "a:$(row.a),c:$(row.c),b:$(row.b),")
            print(io, "status:$(js_string(row.status)),events:$(row.events),")
            print(io, "word:$(js_string(row.word)),code:$(row.code),period:$(row.period),")
            print(io, "gamma:$(row.gamma),max_time:$(row.max_time),first_time:$(row.first_time),last_time:$(row.last_time),")
            print(io, "min_y:$(row.min_y),max_y:$(row.max_y)")
            println(io, "}$(comma)")
        end
        println(io, "  ]")
        println(io, "};")
    end
    return path
end

function write_runtime_log(path, rows; config, runtime, output, docs_data, docs_data_written, contour_dir, contour_format)
    mkpath(dirname(path))
    ok = count(row -> row.status == "ok", rows)
    open(path, "w") do io
        println(io, "metric\tvalue")
        println(io, "output\t$(output)")
        println(io, "docs_data\t$(docs_data)")
        println(io, "docs_data_written\t$(docs_data_written)")
        println(io, "contour_dir\t$(contour_dir)")
        println(io, "contour_format\t$(contour_format)")
        println(io, "total_points\t$(length(rows))")
        println(io, "ok_points\t$(ok)")
        println(io, "max_time_limited_points\t$(length(rows) - ok)")
        for pair in pairs(config)
            println(io, "$(pair.first)\t$(pair.second)")
        end
        for pair in pairs(runtime)
            println(io, "$(pair.first)\t$(pair.second)")
        end
    end
    return path
end

function main()
    started = time()
    default_max_time = env_float("MM_FLOW_FOLDING_MAX_TIME", env_float("MM_FLOW_FOLDING_T_END", 450.0))
    config = (
        c_min=env_float("MM_FLOW_FOLDING_C_MIN", 2.0),
        c_max=env_float("MM_FLOW_FOLDING_C_MAX", 7.0),
        a_min=env_float("MM_FLOW_FOLDING_A_MIN", 0.30),
        a_max=env_float("MM_FLOW_FOLDING_A_MAX", 0.55),
        n_c=env_int("MM_FLOW_FOLDING_NC", 1024),
        n_a=env_int("MM_FLOW_FOLDING_NA", 1024),
        b=env_float("MM_FLOW_FOLDING_B", ROSSLER_MALYKH_B),
        word_length=env_int("MM_FLOW_FOLDING_WORD_LENGTH", 8),
        transient_events=env_int("MM_FLOW_FOLDING_TRANSIENT_EVENTS", 20),
        dt=env_float("MM_FLOW_FOLDING_DT", 0.05),
        max_time=default_max_time,
        max_state=env_float("MM_FLOW_FOLDING_MAX_STATE", 1e6),
        progress_seconds=env_float("MM_FLOW_FOLDING_PROGRESS_SECONDS", 30.0),
    )
    default_results_name = get(
        ENV,
        "MM_FLOW_FOLDING_RESULTS_NAME",
        config.n_c == 1024 && config.n_a == 1024 ? "rossler_y_minima_tangent_scan_1024" : "rossler_y_minima_tangent_scan",
    )
    default_results_dir = joinpath(@__DIR__, "..", "results", default_results_name)
    output = get(
        ENV,
        "MM_FLOW_FOLDING_OUTPUT",
        joinpath(default_results_dir, "coarse_scan.tsv"),
    )
    docs_data = get(
        ENV,
        "MM_FLOW_FOLDING_DOCS_DATA",
        joinpath(@__DIR__, "..", "docs", "rossler_y_minima_tangent_scan_data.js"),
    )
    contour_dir = get(
        ENV,
        "MM_FLOW_FOLDING_CONTOUR_DIR",
        joinpath(default_results_dir, "contours"),
    )
    runtime_log = get(
        ENV,
        "MM_FLOW_FOLDING_RUNTIME_LOG",
        joinpath(default_results_dir, "coarse_scan_runtime.tsv"),
    )
    generate_contours = lowercase(get(ENV, "MM_FLOW_FOLDING_GENERATE_CONTOURS", "true")) in ("1", "true", "yes")
    contour_format = lowercase(get(ENV, "MM_FLOW_FOLDING_CONTOUR_FORMAT", "png"))
    docs_data_max_points = env_int("MM_FLOW_FOLDING_DOCS_DATA_MAX_POINTS", 200000)
    write_docs_data_enabled = env_bool("MM_FLOW_FOLDING_WRITE_DOCS_DATA", config.n_c * config.n_a <= docs_data_max_points)

    scan_seconds = @elapsed rows = run_scan(; config...)
    write_tsv_seconds = @elapsed write_tsv(output, rows)
    contour_step_seconds = 0.0
    if generate_contours
        if contour_format == "svg"
            include(joinpath(@__DIR__, "rossler_y_minima_tangent_contours.jl"))
            contour_step_seconds = @elapsed Base.invokelatest(
                write_all_contours,
                output;
                output_dir=contour_dir,
                scan_seconds=scan_seconds,
                write_tsv_seconds=write_tsv_seconds,
            )
        elseif contour_format == "png"
            renderer = joinpath(@__DIR__, "rossler_y_minima_tangent_pngs.py")
            python = get(ENV, "MM_FLOW_FOLDING_PYTHON", "python3")
            png_width = env_int("MM_FLOW_FOLDING_PNG_WIDTH", 1600)
            png_height = env_int("MM_FLOW_FOLDING_PNG_HEIGHT", 1100)
            line_width_scale = env_float("MM_FLOW_FOLDING_CONTOUR_LINE_WIDTH_SCALE", 0.25)
            contour_alpha = env_float("MM_FLOW_FOLDING_CONTOUR_ALPHA", 0.10)
            contour_step_seconds = @elapsed run(
                `$(python) $(renderer) $(output) --output-dir $(contour_dir) --stem coarse_scan --scan-seconds $(scan_seconds) --write-tsv-seconds $(write_tsv_seconds) --width $(png_width) --height $(png_height) --line-width-scale $(line_width_scale) --alpha $(contour_alpha) --clean`,
            )
        else
            error("Unsupported MM_FLOW_FOLDING_CONTOUR_FORMAT=$(contour_format); expected png or svg")
        end
    end
    runtime_for_docs = (
        scan_seconds=scan_seconds,
        write_tsv_seconds=write_tsv_seconds,
        contour_step_seconds=contour_step_seconds,
        total_seconds=time() - started,
    )
    write_docs_data_seconds = 0.0
    if write_docs_data_enabled
        write_docs_data_seconds = @elapsed write_docs_data(docs_data, rows; config=config, runtime=runtime_for_docs)
    end
    runtime = (
        scan_seconds=scan_seconds,
        write_tsv_seconds=write_tsv_seconds,
        contour_step_seconds=contour_step_seconds,
        write_docs_data_seconds=write_docs_data_seconds,
        total_seconds=time() - started,
    )
    write_runtime_log(
        runtime_log,
        rows;
        config=config,
        runtime=runtime,
        output=output,
        docs_data=docs_data,
        docs_data_written=write_docs_data_enabled,
        contour_dir=contour_dir,
        contour_format=contour_format,
    )
    ok = count(row -> row.status == "ok", rows)
    @printf("wrote %s\n", output)
    write_docs_data_enabled ? @printf("wrote %s\n", docs_data) : @printf("skipped docs data %s\n", docs_data)
    @printf("wrote %s\n", runtime_log)
    generate_contours && @printf("wrote contours in %s\n", contour_dir)
    @printf("ok points: %d/%d\n", ok, length(rows))
    @printf("scan runtime: %.3f s\n", scan_seconds)
    @printf("contour step runtime: %.3f s\n", contour_step_seconds)
    @printf("total runtime: %.3f s\n", runtime.total_seconds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
