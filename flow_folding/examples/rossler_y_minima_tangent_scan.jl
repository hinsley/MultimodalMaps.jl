include(joinpath(@__DIR__, "rossler_common.jl"))

using Printf

function env_float(name, default)
    return parse(Float64, get(ENV, name, string(default)))
end

function env_int(name, default)
    return parse(Int, get(ENV, name, string(default)))
end

function scan_point(
    a,
    c;
    b=ROSSLER_MALYKH_B,
    word_length=8,
    transient_events=20,
    dt=0.05,
    t_end=450.0,
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
        t_end=t_end,
        transient_events=transient_events,
        max_events=word_length,
        max_state=max_state,
    )

    bits = tangent_bits(events)
    period = isnothing(bits) || length(bits) != word_length ? 0 : least_period(bits)
    gamma = isnothing(bits) || length(bits) != word_length ? NaN : binary_sequence_value(bits)
    status = length(events) == word_length ? "ok" : "short"
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
    n_c=31,
    n_a=21,
    b=ROSSLER_MALYKH_B,
    word_length=8,
    transient_events=20,
    dt=0.05,
    t_end=450.0,
    max_state=1e6,
)
    c_values = collect(range(c_min, c_max; length=n_c))
    a_values = collect(range(a_min, a_max; length=n_a))
    rows = Vector{NamedTuple}(undef, length(c_values) * length(a_values))
    k = 0
    total = length(rows)
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
                t_end=t_end,
                max_state=max_state,
            )
            if k == 1 || k == total || k % max(1, total ÷ 20) == 0
                @printf("scanned %d/%d a=%.6f c=%.6f status=%s word=%s\n", k, total, a, c, rows[k].status, rows[k].word)
                flush(stdout)
            end
        end
    end
    return rows
end

function write_tsv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "a\tc\tb\tstatus\tevents\tword\tcode\tperiod\tgamma\tfirst_time\tlast_time\tmin_y\tmax_y")
        for row in rows
            @printf(
                io,
                "%.12g\t%.12g\t%.12g\t%s\t%d\t%s\t%d\t%d\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
                row.a,
                row.c,
                row.b,
                row.status,
                row.events,
                row.word,
                row.code,
                row.period,
                row.gamma,
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

function write_docs_data(path, rows; config)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "window.FLOW_FOLDING_ROSSLER_SCAN = {")
        println(io, "  config: {")
        fields = collect(pairs(config))
        for (i, pair) in enumerate(fields)
            comma = i == length(fields) ? "" : ","
            value = pair.second isa AbstractString ? js_string(pair.second) : string(pair.second)
            println(io, "    $(pair.first): $(value)$(comma)")
        end
        println(io, "  },")
        println(io, "  rows: [")
        for (i, row) in enumerate(rows)
            comma = i == length(rows) ? "" : ","
            print(io, "    {")
            print(io, "a:$(row.a),c:$(row.c),b:$(row.b),")
            print(io, "status:$(js_string(row.status)),events:$(row.events),")
            print(io, "word:$(js_string(row.word)),code:$(row.code),period:$(row.period),")
            print(io, "gamma:$(row.gamma),first_time:$(row.first_time),last_time:$(row.last_time),")
            print(io, "min_y:$(row.min_y),max_y:$(row.max_y)")
            println(io, "}$(comma)")
        end
        println(io, "  ]")
        println(io, "};")
    end
    return path
end

function main()
    config = (
        c_min=env_float("MM_FLOW_FOLDING_C_MIN", 2.0),
        c_max=env_float("MM_FLOW_FOLDING_C_MAX", 7.0),
        a_min=env_float("MM_FLOW_FOLDING_A_MIN", 0.30),
        a_max=env_float("MM_FLOW_FOLDING_A_MAX", 0.55),
        n_c=env_int("MM_FLOW_FOLDING_NC", 31),
        n_a=env_int("MM_FLOW_FOLDING_NA", 21),
        b=env_float("MM_FLOW_FOLDING_B", ROSSLER_MALYKH_B),
        word_length=env_int("MM_FLOW_FOLDING_WORD_LENGTH", 8),
        transient_events=env_int("MM_FLOW_FOLDING_TRANSIENT_EVENTS", 20),
        dt=env_float("MM_FLOW_FOLDING_DT", 0.05),
        t_end=env_float("MM_FLOW_FOLDING_T_END", 450.0),
        max_state=env_float("MM_FLOW_FOLDING_MAX_STATE", 1e6),
    )
    output = get(
        ENV,
        "MM_FLOW_FOLDING_OUTPUT",
        joinpath(@__DIR__, "..", "results", "rossler_y_minima_tangent_scan", "coarse_scan.tsv"),
    )
    docs_data = get(
        ENV,
        "MM_FLOW_FOLDING_DOCS_DATA",
        joinpath(@__DIR__, "..", "docs", "rossler_y_minima_tangent_scan_data.js"),
    )

    rows = run_scan(; config...)
    write_tsv(output, rows)
    write_docs_data(docs_data, rows; config=config)
    ok = count(row -> row.status == "ok", rows)
    @printf("wrote %s\n", output)
    @printf("wrote %s\n", docs_data)
    @printf("ok points: %d/%d\n", ok, length(rows))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
