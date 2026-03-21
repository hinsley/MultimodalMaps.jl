using Pkg

const ATTEMPT15_ROOT = @__DIR__
const REPO_ROOT_015 = normpath(joinpath(ATTEMPT15_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_015)

include(joinpath(ATTEMPT15_ROOT, "contours.jl"))

using Printf

const TARGET_DELTA_CA = parse(Float64, get(ENV, "ATTEMPT015_PROBE_DELTA_CA", "-37.6"))
const MAP_RES_LIST = parse.(Int, split(get(ENV, "ATTEMPT015_PROBE_MAP_RES_LIST", "20,30,40,60,80,100,120,150,200,300,400"), ","))

struct ProbePointData
    delta_x::Float64
    p::SVector{17, Float64}
    saddle_data
end

function nearest_column(values::Vector{Float64}, target::Float64)
    return findmin(abs.(values .- target))[2]
end

function initialize_T_Ca0_with_mapres(
    p,
    x_eq_SF::Float64,
    gamma_sd_minus0::SVector{6, Float64},
    map_resolution::Int,
)::SVector{6, Float64}
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
    Vs = collect(range(V_eq_SF, gamma_sd_minus_ca_min_V, length=map_resolution))
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
    first_max_index = nothing
    for (idx, u0) in enumerate(u0s)
        local_prob = ODEProblem(Plant.melibeNew, u0, TSPAN, p)
        local_sol = solve(local_prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
        push!(return_ca_mins, Float64(local_sol.u[end][5]))

        if idx >= 3
            mid = idx - 1
            if return_ca_mins[mid] > return_ca_mins[mid - 1] && return_ca_mins[mid] > return_ca_mins[mid + 1]
                first_max_index = mid
                break
            end
        end
    end

    if isnothing(first_max_index)
        error("Could not locate the first local maximum of the reduced return map.")
    end

    reference_u0 = u0s[first_max_index]
    T_Ca0_guess = reference_u0[5]

    a = T_Ca0_guess - 3.0e-3
    b = T_Ca0_guess + 3.0e-3
    golden_ratio = (sqrt(5) - 1) / 2
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    fc = return_voltage_at_ca_min(p, c, reference_u0[1], callback)
    fd = return_voltage_at_ca_min(p, d, reference_u0[1], callback)
    while abs(b - a) > 1.0e-8
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
    return SVector{6, Float64}((
        reference_u0[1],
        reference_u0[2],
        reference_u0[3],
        reference_u0[4],
        T_Ca0,
        reference_u0[6],
    ))
end

function run_T_point_with_dense_remap(
    point::ProbePointData,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
    map_resolution::Int,
)::TContourScanResult
    if !isnothing(candidate_seed)
        try
            T0, iterations = initialize_T_Ca0_from_seed(
                point.p,
                point.saddle_data.x_eq_SF,
                point.saddle_data.gamma_sd_minus0,
                candidate_seed,
            )
            method = @sprintf("continued:%d", iterations)
            return finalize_T_point(point.delta_x, delta_ca, point.p, point.saddle_data, T0, method)
        catch
            # Fall through to denser remap.
        end
    end

    T0 = initialize_T_Ca0_with_mapres(point.p, point.saddle_data.x_eq_SF, point.saddle_data.gamma_sd_minus0, map_resolution)
    return finalize_T_point(point.delta_x, delta_ca, point.p, point.saddle_data, T0, @sprintf("full:%d", map_resolution))
end

function run_T_point_with_dense_remap_safe(
    point::ProbePointData,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
    map_resolution::Int,
)::TContourScanResult
    try
        return run_T_point_with_dense_remap(point, delta_ca, candidate_seed, map_resolution)
    catch err
        return TContourScanResult(point.delta_x, delta_ca, Int[], NaN, NaN, "", nothing, sprint(showerror, err))
    end
end

function contiguous_error_runs(results::Vector{TContourScanResult})
    runs = Tuple{Float64, Float64, Int}[]
    in_run = false
    run_start = 0.0
    run_count = 0
    previous_x = 0.0

    for result in sort(results; by = r -> r.delta_x)
        is_error = !isnothing(result.error_message)
        if is_error && !in_run
            in_run = true
            run_start = result.delta_x
            run_count = 1
        elseif is_error && in_run
            run_count += 1
        elseif !is_error && in_run
            push!(runs, (run_start, previous_x, run_count))
            in_run = false
            run_count = 0
        end
        previous_x = result.delta_x
    end

    if in_run
        push!(runs, (run_start, previous_x, run_count))
    end

    return runs
end

function run_column_probe(map_resolution::Int, points::Vector{ProbePointData}, delta_ca::Float64)
    results = Vector{TContourScanResult}(undef, length(points))
    previous_successful = nothing
    started = time()

    for idx in reverse(eachindex(points))
        point = points[idx]
        candidate_seed = make_candidate_seed(previous_successful)
        result = run_T_point_with_dense_remap_safe(point, delta_ca, candidate_seed, map_resolution)
        results[idx] = result
        if isnothing(result.error_message)
            previous_successful = result
        end
    end

    ok_count = count(r -> isnothing(r.error_message), results)
    runs = contiguous_error_runs(results)
    return (; map_resolution, results, ok_count, elapsed=time() - started, error_runs=runs)
end

function build_probe_points(delta_ca::Float64)
    points = ProbePointData[]
    for delta_x in DELTA_XS_015
        p = build_params(delta_x, delta_ca)
        saddle_data = compute_gamma_sd_minus0(p)
        push!(points, ProbePointData(delta_x, p, saddle_data))
    end
    return points
end

function main()
    column_idx = nearest_column(DELTA_CAS_015, TARGET_DELTA_CA)
    delta_ca = DELTA_CAS_015[column_idx]
    points = build_probe_points(delta_ca)

    summary_path = joinpath(ATTEMPT15_ROOT, "dense_remap_probe_summary.txt")
    open(summary_path, "w") do io
        println(io, @sprintf("target_delta_ca\t%.12f", TARGET_DELTA_CA))
        println(io, @sprintf("nearest_delta_ca\t%.12f", delta_ca))
        println(io, "nearest_column_index\t$(column_idx)")
        println(io, "map_res\tok_count\telapsed_seconds\terror_runs")

        println(@sprintf("Target Delta Ca: %.12f", TARGET_DELTA_CA))
        println(@sprintf("Nearest Delta Ca: %.12f", delta_ca))
        println("Testing map resolutions: $(join(MAP_RES_LIST, ", "))")

        first_full_success = nothing
        for map_resolution in MAP_RES_LIST
            probe = run_column_probe(map_resolution, points, delta_ca)
            run_desc = isempty(probe.error_runs) ? "none" : join([
                @sprintf("[%.6f, %.6f] x%d", start_x, end_x, count)
                for (start_x, end_x, count) in probe.error_runs
            ], "; ")

            println(
                @sprintf(
                    "mapres=%d ok=%d/500 elapsed=%.2fs error_runs=%s",
                    map_resolution,
                    probe.ok_count,
                    probe.elapsed,
                    run_desc,
                ),
            )
            println(io, join([string(map_resolution), string(probe.ok_count), @sprintf("%.6f", probe.elapsed), run_desc], '\t'))

            if probe.ok_count == length(points) && isnothing(first_full_success)
                first_full_success = map_resolution
            end
        end

        println(io, "first_full_success\t$(something(first_full_success, "none"))")
        println("First full-success map resolution: $(something(first_full_success, "none"))")
    end

    println("Saved summary to $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
