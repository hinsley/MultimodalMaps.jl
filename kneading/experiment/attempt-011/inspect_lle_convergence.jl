using Pkg

const ATTEMPT11_LLE_ROOT = @__DIR__
const REPO_ROOT_011_LLE = normpath(joinpath(ATTEMPT11_LLE_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_011_LLE)

include(joinpath(ATTEMPT11_LLE_ROOT, "inspect_return_section.jl"))

using CairoMakie
using DynamicalSystems
using Random
using Printf

const LLE_DELTA_X = env_float_local("ATTEMPT011_LLE_DELTA_X", -1.0)
const LLE_DELTA_CA = env_float_local("ATTEMPT011_LLE_DELTA_CA", -38.386774)
const LLE_RESULTS_PATH = get(
    ENV,
    "ATTEMPT011_LLE_RESULTS",
    joinpath(ATTEMPT11_ROOT, "grid500_seq7_prefixes_results.tsv"),
)
const LLE_OUTPUT_TAG = get(ENV, "ATTEMPT011_LLE_TAG", "llediag_dxm1_dca38p39")
const LLE_RANDOM_SEED = parse(Int, get(ENV, "ATTEMPT011_LLE_RANDOM_SEED", "1107"))
const LLE_TTR = env_float_local("ATTEMPT011_LLE_TTR", 2000.0)
const LLE_TOTAL_TIME = env_float_local("ATTEMPT011_LLE_TOTAL_TIME", 10000.0)
const LLE_DELTA_T = env_float_local("ATTEMPT011_LLE_DELTA_T", 1.0)
const LLE_D0 = env_float_local("ATTEMPT011_LLE_D0", 1.0e-9)
const LLE_D0_LOWER = env_float_local("ATTEMPT011_LLE_D0_LOWER", 1.0e-12)
const LLE_D0_UPPER = env_float_local("ATTEMPT011_LLE_D0_UPPER", 1.0e-6)
const LLE_ABSTOL = env_float_local("ATTEMPT011_LLE_ABSTOL", 1.0e-9)
const LLE_RELTOL = env_float_local("ATTEMPT011_LLE_RELTOL", 1.0e-9)
const LLE_FIGURE_SIZE = (3900, 1100)
const LLE_PX_PER_UNIT = env_float_local("ATTEMPT011_LLE_PX_PER_UNIT", 2.5)

function λdist_local(pds::ParallelDynamicalSystem)
    u1 = current_state(pds, 1)
    u2 = current_state(pds, 2)
    d = zero(eltype(u1))
    @inbounds for i in eachindex(u1)
        d += (u1[i] - u2[i])^2
    end
    return sqrt(d)
end

function λrescale_local!(pds::ParallelDynamicalSystem, a::Real)
    u1 = current_state(pds, 1)
    u2 = current_state(pds, 2)
    if ismutable(u2)
        @. u2 = u1 + (u2 - u1) / a
    else
        u2 = @. u1 + (u2 - u1) / a
    end
    set_state!(pds, u2, 2)
end

inittest_local(u, d0) = u .+ d0 / sqrt(length(u))

function lyapunov_convergence_trace(ds::DynamicalSystem, u0::AbstractVector;
    Ttr::Float64,
    T::Float64,
    Δt::Float64,
    d0::Float64,
    d0_lower::Float64,
    d0_upper::Float64,
)
    states = [copy(u0), inittest_local(u0, d0)]
    pds = ParallelDynamicalSystem(ds, states)

    while current_time(pds) - initial_time(pds) < Ttr
        step!(pds, Δt)
        successful_step(pds) || error("Lyapunov transient integration failed.")
        d = λdist_local(pds)
        d0_lower <= d <= d0_upper || λrescale_local!(pds, d / d0)
    end

    t0 = current_time(pds)
    d = λdist_local(pds)
    d == 0 && error("Initial distance between Lyapunov trajectories is zero.")
    d != d0 && λrescale_local!(pds, d / d0)

    λsum = 0.0
    times = Float64[]
    λhats = Float64[]
    xvals = Float64[]
    cavals = Float64[]
    vvals = Float64[]

    while current_time(pds) < t0 + T
        d = λdist_local(pds)
        if !(d0_lower <= d <= d0_upper)
            error("Lyapunov rescaling window violated before step.")
        end

        while d0_lower <= d <= d0_upper
            successful_step(pds) || error("Lyapunov main integration failed.")
            step!(pds, Δt, true)
            d = λdist_local(pds)
            elapsed = current_time(pds) - t0
            elapsed <= 0 && continue

            push!(times, elapsed)
            push!(λhats, (λsum + log(d / d0)) / elapsed)
            u = current_state(pds, 1)
            push!(xvals, u[1])
            push!(cavals, u[5])
            push!(vvals, u[6])

            current_time(pds) >= t0 + T && break
        end

        current_time(pds) >= t0 + T && break

        a = d / d0
        λsum += log(a)
        λrescale_local!(pds, a)
    end

    final_estimate = isempty(λhats) ? NaN : λhats[end]
    return (; times, λhats, xvals, cavals, vvals, final_estimate)
end

function main()
    rows = parse_results(LLE_RESULTS_PATH)
    saved_row = nearest_saved_row(rows, LLE_DELTA_X, LLE_DELTA_CA)
    if saved_row["status"] != "ok"
        error("Nearest saved point is not successful: $(saved_row["status"])")
    end

    delta_x = parse(Float64, saved_row["delta_x"])
    delta_ca = parse(Float64, saved_row["delta_ca"])
    p = build_params(delta_x, delta_ca)
    section = equilibrium_section_data(p)

    rng = MersenneTwister(LLE_RANDOM_SEED)
    s0 = rand(rng) * (SECTION_S_MAX - SECTION_S_MIN) + SECTION_S_MIN
    u0 = Vector{Float64}(lift_section_point(p, section, s0))

    ds = CoupledODEs(
        Plant.melibeNew!,
        copy(u0),
        p;
        diffeq=(alg=Tsit5(), abstol=LLE_ABSTOL, reltol=LLE_RELTOL),
    )

    trace_runtime = @elapsed trace = lyapunov_convergence_trace(
        ds,
        u0;
        Ttr=LLE_TTR,
        T=LLE_TOTAL_TIME,
        Δt=LLE_DELTA_T,
        d0=LLE_D0,
        d0_lower=LLE_D0_LOWER,
        d0_upper=LLE_D0_UPPER,
    )

    direct_runtime = @elapsed λ_direct = lyapunov(
        ds,
        LLE_TOTAL_TIME;
        u0=copy(u0),
        Ttr=LLE_TTR,
        Δt=LLE_DELTA_T,
        d0=LLE_D0,
        d0_lower=LLE_D0_LOWER,
        d0_upper=LLE_D0_UPPER,
    )

    tail_start = max(1, floor(Int, 0.8 * length(trace.times)))
    tail_times = trace.times[tail_start:end]
    tail_λ = trace.λhats[tail_start:end]

    plot_path = joinpath(ATTEMPT11_LLE_ROOT, "$(LLE_OUTPUT_TAG)_convergence.png")
    data_path = joinpath(ATTEMPT11_LLE_ROOT, "$(LLE_OUTPUT_TAG)_convergence.tsv")
    summary_path = joinpath(ATTEMPT11_LLE_ROOT, "$(LLE_OUTPUT_TAG)_summary.txt")

    fig = Figure(size=LLE_FIGURE_SIZE)

    ax_full = Axis(
        fig[1, 1],
        xlabel="measurement time",
        ylabel="running LLE estimate",
        title=@sprintf("LLE Convergence, Δx = %.6f, ΔCa = %.6f", delta_x, delta_ca),
    )
    lines!(ax_full, trace.times, trace.λhats; color=:black, linewidth=2.1)
    hlines!(ax_full, [trace.final_estimate]; color=:firebrick, linewidth=2.2, linestyle=:dash)

    ax_tail = Axis(
        fig[1, 2],
        xlabel="measurement time",
        ylabel="running LLE estimate",
        title="Tail Zoom (last 20%)",
    )
    lines!(ax_tail, tail_times, tail_λ; color=:black, linewidth=2.1)
    hlines!(ax_tail, [trace.final_estimate]; color=:firebrick, linewidth=2.2, linestyle=:dash)

    ax_phase = Axis(
        fig[1, 3],
        xlabel="Ca",
        ylabel="x",
        title="Measured Reference Trajectory In (Ca, x)",
    )
    lines!(ax_phase, trace.cavals, trace.xvals; color=:black, linewidth=1.5)
    scatter!(ax_phase, [trace.cavals[1]], [trace.xvals[1]]; color=:firebrick, markersize=14, marker=:diamond)
    scatter!(ax_phase, [trace.cavals[end]], [trace.xvals[end]]; color=:royalblue, markersize=16, marker=:star5)

    save_runtime = @elapsed save(plot_path, fig; px_per_unit=LLE_PX_PER_UNIT)

    open(data_path, "w") do io
        println(io, "time\tlambda_hat\tx\tCa\tV")
        for idx in eachindex(trace.times)
            println(io, "$(trace.times[idx])\t$(trace.λhats[idx])\t$(trace.xvals[idx])\t$(trace.cavals[idx])\t$(trace.vvals[idx])")
        end
    end

    open(summary_path, "w") do io
        println(io, "requested_delta_x\t$(LLE_DELTA_X)")
        println(io, "requested_delta_ca\t$(LLE_DELTA_CA)")
        println(io, "grid_delta_x\t$(delta_x)")
        println(io, "grid_delta_ca\t$(delta_ca)")
        println(io, "random_seed\t$(LLE_RANDOM_SEED)")
        println(io, "section_s0\t$(s0)")
        println(io, "u0\t$(join(u0, ','))")
        println(io, "Ttr\t$(LLE_TTR)")
        println(io, "T\t$(LLE_TOTAL_TIME)")
        println(io, "delta_t\t$(LLE_DELTA_T)")
        println(io, "d0\t$(LLE_D0)")
        println(io, "d0_lower\t$(LLE_D0_LOWER)")
        println(io, "d0_upper\t$(LLE_D0_UPPER)")
        println(io, "running_final_estimate\t$(trace.final_estimate)")
        println(io, "direct_chaostools_estimate\t$(λ_direct)")
        println(io, "difference\t$(trace.final_estimate - λ_direct)")
        println(io, "sample_count\t$(length(trace.times))")
        println(io, "trace_runtime_seconds\t$(trace_runtime)")
        println(io, "direct_runtime_seconds\t$(direct_runtime)")
        println(io, "save_runtime_seconds\t$(save_runtime)")
    end

    println("grid_delta_x=$(delta_x)")
    println("grid_delta_ca=$(delta_ca)")
    println("section_s0=$(s0)")
    println("running_final_estimate=$(trace.final_estimate)")
    println("direct_chaostools_estimate=$(λ_direct)")
    println("difference=$(trace.final_estimate - λ_direct)")
    println("trace_runtime_seconds=$(trace_runtime)")
    println("direct_runtime_seconds=$(direct_runtime)")
    println("save_runtime_seconds=$(save_runtime)")
    println("plot_path=$(plot_path)")
    println("data_path=$(data_path)")
    println("summary_path=$(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
