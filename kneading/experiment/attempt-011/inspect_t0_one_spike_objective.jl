using Pkg

const ATTEMPT11_ONE_SPIKE_ROOT = @__DIR__
const REPO_ROOT_011_ONE_SPIKE = normpath(joinpath(ATTEMPT11_ONE_SPIKE_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_011_ONE_SPIKE)

include(joinpath(ATTEMPT11_ONE_SPIKE_ROOT, "inspect_t0_diagnostics.jl"))

using CairoMakie
using Printf

const ONE_SPIKE_DELTA_X = env_float_local("ATTEMPT011_ONE_SPIKE_DELTA_X", -1.001002)
const ONE_SPIKE_DELTA_CAS = parse.(Float64, split(get(ENV, "ATTEMPT011_ONE_SPIKE_DELTA_CAS", "-38.286573,-38.386774,-38.486974"), ','))
const ONE_SPIKE_OUTPUT_TAG = get(ENV, "ATTEMPT011_ONE_SPIKE_TAG", "hook_one_spike_eval")
const ONE_SPIKE_CA_WINDOW = env_float_local("ATTEMPT011_ONE_SPIKE_CA_WINDOW", 0.0035)
const ONE_SPIKE_GRID_POINTS = parse(Int, get(ENV, "ATTEMPT011_ONE_SPIKE_GRID_POINTS", "401"))
const ONE_SPIKE_REFINE_STEPS = parse(Int, get(ENV, "ATTEMPT011_ONE_SPIKE_REFINE_STEPS", "16"))
const ONE_SPIKE_FIGURE_SIZE = (2200, 700)
const ONE_SPIKE_PX_PER_UNIT = env_float_local("ATTEMPT011_ONE_SPIKE_PX_PER_UNIT", 2.5)

mutable struct UpSpikeCounter
    count::Int
end

function make_upcrossing_callback(counter::UpSpikeCounter)
    function condition(u, t, integrator)
        return u[6]
    end

    function affect!(integrator)
        counter.count += 1
    end

    return ContinuousCallback(condition, affect!, affect_neg! = nothing, save_positions=(false, false))
end

function objective_with_upspikes(
    p,
    Ca0::Float64,
    x0::Float64,
    x_eq_SF::Float64;
    abstol::Float64=1e-8,
    reltol::Float64=1e-8,
)
    counter = UpSpikeCounter(0)
    callback = CallbackSet(make_upcrossing_callback(counter), make_ca_min_callback(x_eq_SF))
    u0 = EquilibriaSubset.dune(p, x0, Ca0)
    prob = ODEProblem(Plant.melibeNew!, u0, TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)
    return (
        returned_V = Float64(sol.u[end][6]),
        returned_Ca = Float64(sol.u[end][5]),
        upspikes = counter.count,
    )
end

function contiguous_intervals_by_value(values::Vector{Int})
    intervals = NamedTuple[]
    start_idx = 1
    current = values[1]
    for idx in 2:length(values)
        if values[idx] != current
            push!(intervals, (start_idx=start_idx, end_idx=idx - 1, value=current))
            start_idx = idx
            current = values[idx]
        end
    end
    push!(intervals, (start_idx=start_idx, end_idx=length(values), value=current))
    return intervals
end

function interval_bounds(xs::Vector{Float64}, start_idx::Int, end_idx::Int)
    left = start_idx == 1 ? xs[start_idx] : 0.5 * (xs[start_idx - 1] + xs[start_idx])
    right = end_idx == length(xs) ? xs[end_idx] : 0.5 * (xs[end_idx] + xs[end_idx + 1])
    return left, right
end

function refine_one_spike_maximum(
    p,
    x0::Float64,
    x_eq_SF::Float64,
    a::Float64,
    b::Float64;
    steps::Int,
)
    golden_ratio = (sqrt(5) - 1) / 2
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)
    fc = objective_with_upspikes(p, c, x0, x_eq_SF)
    fd = objective_with_upspikes(p, d, x0, x_eq_SF)
    if fc.upspikes != 1 || fd.upspikes != 1
        error("Refinement bracket escaped the 1-spike interval.")
    end

    for _ in 1:steps
        if fc.returned_V > fd.returned_V
            b = d
            d = c
            fd = fc
            c = b - golden_ratio * (b - a)
            fc = objective_with_upspikes(p, c, x0, x_eq_SF)
            fc.upspikes == 1 || error("Refinement left the 1-spike interval.")
        else
            a = c
            c = d
            fc = fd
            d = a + golden_ratio * (b - a)
            fd = objective_with_upspikes(p, d, x0, x_eq_SF)
            fd.upspikes == 1 || error("Refinement left the 1-spike interval.")
        end
    end

    return fc.returned_V >= fd.returned_V ? (c, fc) : (d, fd)
end

function evaluate_point(delta_x::Float64, delta_ca::Float64)
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    T0 = init_T0_highres(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0; mapres=DIAG_T0_MAPRES)
    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)

    ca_values = collect(range(T0[5] - ONE_SPIKE_CA_WINDOW, T0[5] + ONE_SPIKE_CA_WINDOW, length=ONE_SPIKE_GRID_POINTS))
    returned_Vs = Float64[]
    upspikes = Int[]
    sizehint!(returned_Vs, length(ca_values))
    sizehint!(upspikes, length(ca_values))
    for ca in ca_values
        result = objective_with_upspikes(p, ca, T0[1], saddle_data.x_eq_SF)
        push!(returned_Vs, result.returned_V)
        push!(upspikes, result.upspikes)
    end

    unrestricted_idx = argmax(returned_Vs)
    unrestricted_ca = ca_values[unrestricted_idx]
    unrestricted_V = returned_Vs[unrestricted_idx]
    unrestricted_upspikes = upspikes[unrestricted_idx]

    spike_intervals = contiguous_intervals_by_value(upspikes)
    one_spike_intervals = [interval for interval in spike_intervals if interval.value == 1]
    isempty(one_spike_intervals) && error("No 1-spike interval found in local objective window.")

    containing_interval = findfirst(interval -> ca_values[interval.start_idx] <= T0[5] <= ca_values[interval.end_idx], one_spike_intervals)
    target_interval = isnothing(containing_interval) ? one_spike_intervals[argmax([maximum(returned_Vs[interval.start_idx:interval.end_idx]) for interval in one_spike_intervals])] : one_spike_intervals[containing_interval]
    left, right = interval_bounds(ca_values, target_interval.start_idx, target_interval.end_idx)
    one_spike_ca, one_spike_result = refine_one_spike_maximum(
        p,
        T0[1],
        saddle_data.x_eq_SF,
        left,
        right;
        steps=ONE_SPIKE_REFINE_STEPS,
    )

    return (
        delta_x=delta_x,
        delta_ca=delta_ca,
        p=p,
        T0=T0,
        T_scs=T_scs,
        T0_objective=objective_with_upspikes(p, T0[5], T0[1], saddle_data.x_eq_SF),
        ca_values=ca_values,
        returned_Vs=returned_Vs,
        upspikes=upspikes,
        unrestricted_ca=unrestricted_ca,
        unrestricted_V=unrestricted_V,
        unrestricted_upspikes=unrestricted_upspikes,
        one_spike_interval=target_interval,
        one_spike_bounds=(left, right),
        one_spike_ca=one_spike_ca,
        one_spike_V=one_spike_result.returned_V,
        one_spike_upspikes=one_spike_result.upspikes,
    )
end

function main()
    evaluations = [evaluate_point(ONE_SPIKE_DELTA_X, delta_ca) for delta_ca in ONE_SPIKE_DELTA_CAS]

    fig = Figure(size=ONE_SPIKE_FIGURE_SIZE)
    for (panel_idx, eval) in enumerate(evaluations)
        ax = Axis(
            fig[1, panel_idx],
            xlabel="Ca₀",
            ylabel="returned V at first Ca-min",
            title=@sprintf("Δx = %.6f, ΔCa = %.6f", eval.delta_x, eval.delta_ca),
        )

        unique_counts = sort(unique(eval.upspikes))
        for count in unique_counts
            idxs = findall(==(count), eval.upspikes)
            scatter!(
                ax,
                eval.ca_values[idxs],
                eval.returned_Vs[idxs];
                color=spike_count_color(count; alpha=0.85f0),
                markersize=5,
            )
        end

        lines!(ax, eval.ca_values, eval.returned_Vs; color=RGBAf(0.1, 0.1, 0.1, 0.25), linewidth=1.2)
        vspan!(ax, eval.one_spike_bounds[1], eval.one_spike_bounds[2]; color=RGBAf(0.8, 0.1, 0.1, 0.08))
        scatter!(ax, [eval.T0[5]], [eval.T0_objective.returned_V]; color=:firebrick, markersize=16, marker=:star5)
        scatter!(ax, [eval.one_spike_ca], [eval.one_spike_V]; color=:royalblue, markersize=14, marker=:diamond)
    end

    plot_path = joinpath(ATTEMPT11_ONE_SPIKE_ROOT, "$(ONE_SPIKE_OUTPUT_TAG).png")
    summary_path = joinpath(ATTEMPT11_ONE_SPIKE_ROOT, "$(ONE_SPIKE_OUTPUT_TAG)_summary.txt")
    save(plot_path, fig; px_per_unit=ONE_SPIKE_PX_PER_UNIT)

    open(summary_path, "w") do io
        for eval in evaluations
            println(io, "delta_x\t$(eval.delta_x)")
            println(io, "delta_ca\t$(eval.delta_ca)")
            println(io, "T0_Ca\t$(eval.T0[5])")
            println(io, "T0_V\t$(eval.T0[6])")
            println(io, "T0_objective_V\t$(eval.T0_objective.returned_V)")
            println(io, "T0_objective_upspikes\t$(eval.T0_objective.upspikes)")
            println(io, "T_scs\t$(join(eval.T_scs, ','))")
            println(io, "unrestricted_argmax_Ca\t$(eval.unrestricted_ca)")
            println(io, "unrestricted_argmax_V\t$(eval.unrestricted_V)")
            println(io, "unrestricted_argmax_upspikes\t$(eval.unrestricted_upspikes)")
            println(io, "one_spike_interval_indices\t$(eval.one_spike_interval.start_idx):$(eval.one_spike_interval.end_idx)")
            println(io, "one_spike_interval_bounds\t$(eval.one_spike_bounds[1]),$(eval.one_spike_bounds[2])")
            println(io, "one_spike_refined_Ca\t$(eval.one_spike_ca)")
            println(io, "one_spike_refined_V\t$(eval.one_spike_V)")
            println(io, "one_spike_refined_upspikes\t$(eval.one_spike_upspikes)")
            println(io, "one_spike_minus_T0_Ca\t$(eval.one_spike_ca - eval.T0[5])")
            println(io, "one_spike_minus_T0_objective_V\t$(eval.one_spike_V - eval.T0_objective.returned_V)")
            println(io)
        end
        println(io, "plot_path\t$(plot_path)")
    end

    for eval in evaluations
        println("delta_ca=$(eval.delta_ca)")
        println("T0_Ca=$(eval.T0[5])")
        println("T0_V=$(eval.T0[6])")
        println("T_scs=$(join(eval.T_scs, ','))")
        println("unrestricted_argmax_Ca=$(eval.unrestricted_ca)")
        println("unrestricted_argmax_upspikes=$(eval.unrestricted_upspikes)")
        println("one_spike_refined_Ca=$(eval.one_spike_ca)")
        println("one_spike_refined_upspikes=$(eval.one_spike_upspikes)")
        println("one_spike_minus_T0_Ca=$(eval.one_spike_ca - eval.T0[5])")
        println("")
    end
    println("plot_path=$(plot_path)")
    println("summary_path=$(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
