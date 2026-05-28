include(joinpath(@__DIR__, "rossler_common.jl"))

using Printf

function build_problem_ray(parameter; a=0.35, b=ROSSLER_MALYKH_B)
    c = Float64(parameter)
    problem = rossler_y_minima_problem(a, c; b=b)
    ray = saddle_focus_seed_ray(problem, rossler_origin_equilibrium())
    return problem, ray
end

function locate_initial_seed(;
    a=0.35,
    c=4.6,
    b=ROSSLER_MALYKH_B,
    event_index=1,
    rho_range=(-14.0, -3.0),
    samples=31,
    tspan=(0.0, 600.0),
)
    problem, ray = build_problem_ray(c; a=a, b=b)
    roots, residual_samples = find_seeded_critical_points(
        problem,
        ray;
        event_index=event_index,
        rho_range=rho_range,
        samples=samples,
        tspan=tspan,
    )
    return roots, residual_samples
end

function continue_along_c(;
    a=0.35,
    c_start=4.6,
    c_stop=5.2,
    steps=13,
    initial_rho=-8.0,
    b=ROSSLER_MALYKH_B,
    event_index=1,
    tspan=(0.0, 600.0),
)
    c_values = collect(range(c_start, c_stop; length=steps))
    builder = c -> build_problem_ray(c; a=a, b=b)
    return continue_seeded_critical_point(
        builder,
        c_values;
        initial_rho=initial_rho,
        event_index=event_index,
        tspan=tspan,
    )
end

function write_continuation_tsv(path, points)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "c\trho\tresidual\tx_current\tx_next\tvalid\titerations\tmessage")
        for point in points
            @printf(
                io,
                "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%s\t%d\t%s\n",
                point.parameter,
                point.rho,
                point.residual,
                point.x_current,
                point.x_next,
                point.valid,
                point.iterations,
                point.message,
            )
        end
    end
    return path
end

function main()
    a = parse(Float64, get(ENV, "MM_FLOW_FOLDING_CONT_A", "0.35"))
    c = parse(Float64, get(ENV, "MM_FLOW_FOLDING_CONT_C0", "4.6"))
    event_index = parse(Int, get(ENV, "MM_FLOW_FOLDING_CONT_EVENT_INDEX", "1"))
    roots, _ = locate_initial_seed(; a=a, c=c, event_index=event_index)
    if isempty(roots)
        @printf("no seed root found near a=%.6f c=%.6f event_index=%d\n", a, c, event_index)
        @printf("try widening MM_FLOW_FOLDING_CONT_RHO_MIN/MAX or changing the event index\n")
        return
    end

    seed = first(roots)
    @printf("seed rho=%.12g residual=%.12g x=(%.12g -> %.12g)\n", seed.rho, seed.residual, seed.x_current, seed.x_next)
    points = continue_along_c(; a=a, initial_rho=seed.rho, event_index=event_index)
    output = get(
        ENV,
        "MM_FLOW_FOLDING_CONT_OUTPUT",
        joinpath(@__DIR__, "..", "results", "rossler_y_minima_tangent_scan", "seeded_continuation.tsv"),
    )
    write_continuation_tsv(output, points)
    @printf("wrote %s\n", output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

