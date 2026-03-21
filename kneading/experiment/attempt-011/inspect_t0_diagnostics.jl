using Pkg

const ATTEMPT11_DIAG_ROOT = @__DIR__
const REPO_ROOT_011_DIAG = normpath(joinpath(ATTEMPT11_DIAG_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_011_DIAG)

include(joinpath(ATTEMPT11_DIAG_ROOT, "inspect_return_section.jl"))

using Printf

const DIAG_DELTA_X = env_float_local("ATTEMPT011_DIAG_DELTA_X", -1.0)
const DIAG_DELTA_CA = env_float_local("ATTEMPT011_DIAG_DELTA_CA", -38.386774)
const DIAG_OUTPUT_TAG = get(ENV, "ATTEMPT011_DIAG_TAG", "hookdiag_dxm1_dca38p39")
const DIAG_SAMPLE_COUNT = parse(Int, get(ENV, "ATTEMPT011_DIAG_SAMPLE_COUNT", "240"))
const DIAG_MAP_POINTS = parse(Int, get(ENV, "ATTEMPT011_DIAG_MAP_POINTS", "121"))
const DIAG_CA_WINDOW = env_float_local("ATTEMPT011_DIAG_CA_WINDOW", 0.0035)
const DIAG_T0_MAPRES = parse(Int, get(ENV, "ATTEMPT011_DIAG_T0_MAPRES", "400"))
const DIAG_REFINED_POINTS_MIN = parse(Int, get(ENV, "ATTEMPT011_DIAG_REFINED_POINTS_MIN", "24"))
const DIAG_FIGURE_SIZE = (3000, 1120)
const DIAG_PX_PER_UNIT = env_float_local("ATTEMPT011_DIAG_PX_PER_UNIT", 2.5)

function init_T0_highres(p, x_eq_SF, gamma_sd_minus0; mapres::Int)
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
    first_max_index = nothing
    for (idx, u0) in enumerate(u0s)
        local_prob = ODEProblem(Plant.melibeNew, u0, TSPAN, p)
        local_sol = solve(local_prob, SOLVER_010; callback=callback, abstol=1e-8, reltol=1e-8, save_everystep=false)
        push!(return_ca_mins, Float64(local_sol.u[end][5]))
    end
    for idx in 2:length(return_ca_mins)-1
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
    T0 = SVector{6, Float64}(Tuple(Float64.(EquilibriaSubset.dune(p, reference_u0[1], T_Ca0))))
    return T0
end

function pullback_preimage(
    p,
    section,
    target_s::Float64,
    sample_ss::Vector{Float64},
    sample_return_ss::Vector{Float64};
    saveat::Float64,
    abstol::Float64,
    reltol::Float64,
)
    valid = [idx for idx in eachindex(sample_ss) if isfinite(sample_return_ss[idx])]
    isempty(valid) && error("No valid section-return samples available for pullback.")

    ordered = sort(valid; by=idx -> abs(sample_return_ss[idx] - target_s))
    best_idx = ordered[1]
    best_s = sample_ss[best_idx]
    best_sol, best_target = solve_section_from_s(p, section, best_s; saveat=saveat, abstol=abstol, reltol=reltol)
    best_err = abs(ray_coordinate(best_sol.u[end], section) - target_s)

    if length(ordered) >= 2
        s0 = sample_ss[ordered[1]]
        s1 = sample_ss[ordered[2]]
        f0 = sample_return_ss[ordered[1]] - target_s
        f1 = sample_return_ss[ordered[2]] - target_s
        for _ in 1:6
            if !isfinite(f0) || !isfinite(f1) || abs(f1 - f0) < 1.0e-12
                break
            end
            s2 = clamp(s1 - f1 * (s1 - s0) / (f1 - f0), SECTION_S_MIN, SECTION_S_MAX)
            sol2, target2 = solve_section_from_s(p, section, s2; saveat=saveat, abstol=abstol, reltol=reltol)
            if !returned_to_section(sol2, target2)
                break
            end
            f2 = ray_coordinate(sol2.u[end], section) - target_s
            if abs(f2) < best_err
                best_s = s2
                best_sol = sol2
                best_target = target2
                best_err = abs(f2)
            end
            s0, f0 = s1, f1
            s1, f1 = s2, f2
        end
    end

    returned_to_section(best_sol, best_target) || error("Pullback refinement did not return to the section.")
    return (
        start_s = best_s,
        end_s = ray_coordinate(best_sol.u[end], section),
        sol = best_sol,
        start_u = lift_section_point(p, section, best_s),
        end_u = state6(best_sol.u[end]),
        error = best_err,
    )
end

function local_t0_objective_curve(p, x_eq_SF::Float64, T0::SVector{6, Float64})
    callback = make_ca_min_callback(x_eq_SF)
    ca_values = collect(range(T0[5] - DIAG_CA_WINDOW, T0[5] + DIAG_CA_WINDOW, length=DIAG_MAP_POINTS))
    returned_vs = Float64[]
    sizehint!(returned_vs, length(ca_values))
    for ca in ca_values
        push!(returned_vs, return_voltage_at_ca_min(p, ca, T0[1], callback))
    end
    t0_returned_v = return_voltage_at_ca_min(p, T0[5], T0[1], callback)
    return ca_values, returned_vs, t0_returned_v
end

function contiguous_spike_intervals(
    sample_ss::Vector{Float64},
    sample_return_ss::Vector{Float64},
    sample_spike_counts::Vector{Int},
)
    valid = [idx for idx in eachindex(sample_ss) if isfinite(sample_return_ss[idx])]
    isempty(valid) && error("No valid section-return samples available for refined return-map sampling.")

    intervals = NamedTuple[]
    start_idx = valid[1]
    end_idx = valid[1]
    current_spike_count = sample_spike_counts[valid[1]]

    for idx in Iterators.drop(valid, 1)
        if idx == end_idx + 1 && sample_spike_counts[idx] == current_spike_count
            end_idx = idx
        else
            push!(intervals, (start_idx=start_idx, end_idx=end_idx, spike_count=current_spike_count))
            start_idx = idx
            end_idx = idx
            current_spike_count = sample_spike_counts[idx]
        end
    end

    push!(intervals, (start_idx=start_idx, end_idx=end_idx, spike_count=current_spike_count))
    return intervals
end

function interval_bounds(sample_ss::Vector{Float64}, start_idx::Int, end_idx::Int)
    left = start_idx == 1 ? sample_ss[start_idx] : 0.5 * (sample_ss[start_idx - 1] + sample_ss[start_idx])
    right = end_idx == length(sample_ss) ? sample_ss[end_idx] : 0.5 * (sample_ss[end_idx] + sample_ss[end_idx + 1])
    return (max(left, SECTION_S_MIN), min(right, SECTION_S_MAX))
end

function refine_return_map_samples(
    p,
    section,
    sample_ss::Vector{Float64},
    sample_return_ss::Vector{Float64},
    sample_spike_counts::Vector{Int};
    saveat::Float64,
    abstol::Float64,
    reltol::Float64,
)
    intervals = contiguous_spike_intervals(sample_ss, sample_return_ss, sample_spike_counts)
    points_per_interval = max(DIAG_REFINED_POINTS_MIN, cld(DIAG_SAMPLE_COUNT, length(intervals)))
    refined_groups = NamedTuple[]

    for interval in intervals
        left, right = interval_bounds(sample_ss, interval.start_idx, interval.end_idx)
        refined_ss = left == right ? fill(left, points_per_interval) : collect(range(left, right, length=points_per_interval))

        group_start_ss = Float64[]
        group_return_ss = Float64[]
        group_spike_counts = Int[]
        for s in refined_ss
            sol, target_phase = solve_section_from_s(p, section, s; saveat=saveat, abstol=abstol, reltol=reltol)
            if !returned_to_section(sol, target_phase)
                continue
            end
            push!(group_start_ss, s)
            push!(group_return_ss, ray_coordinate(sol.u[end], section))
            push!(group_spike_counts, spike_count_to_return(sol))
        end

        if !isempty(group_start_ss)
            push!(
                refined_groups,
                (
                    initial_spike_count=interval.spike_count,
                    start_idx=interval.start_idx,
                    end_idx=interval.end_idx,
                    start_ss=group_start_ss,
                    return_ss=group_return_ss,
                    spike_counts=group_spike_counts,
                ),
            )
        end
    end

    return intervals, points_per_interval, refined_groups
end

function main()
    rows = parse_results(INSPECT_RESULTS_PATH)
    saved_row = nearest_saved_row(rows, DIAG_DELTA_X, DIAG_DELTA_CA)

    delta_x = parse(Float64, saved_row["delta_x"])
    delta_ca = parse(Float64, saved_row["delta_ca"])
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    section = equilibrium_section_data(p)

    default_T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    default_T_scs = compute_sscs(p, default_T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)

    # High-resolution T0 solve from scratch for the hook probe.
    T0 = init_T0_highres(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0; mapres=DIAG_T0_MAPRES)
    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    gamma_scs = compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)

    sample_ss = collect(range(SECTION_S_MIN, SECTION_S_MAX, length=DIAG_SAMPLE_COUNT))
    sample_trajs = Any[]
    sample_return_ss = fill(NaN, length(sample_ss))
    start_points = Point2f[]
    return_points = Point2f[]
    sample_spike_counts = Int[]

    for (idx, s) in enumerate(sample_ss)
        u0 = lift_section_point(p, section, s)
        sol, target_phase = solve_section_from_s(p, section, s; saveat=SAMPLE_SAVEAT, abstol=1e-8, reltol=1e-8)
        push!(sample_trajs, sol)
        push!(start_points, Point2f(u0[5], u0[1]))
        push!(sample_spike_counts, spike_count_to_return(sol))
        if returned_to_section(sol, target_phase)
            u_end = sol.u[end]
            sample_return_ss[idx] = ray_coordinate(u_end, section)
            push!(return_points, Point2f(u_end[5], u_end[1]))
        end
    end

    T_first_return_sol, T_first_target = solve_section_return(
        p,
        section,
        T0;
        target_returns=1,
        saveat=SPECIAL_SAVEAT,
        abstol=3e-6,
        reltol=3e-6,
    )
    returned_to_section(T_first_return_sol, T_first_target) || error("Fresh T0 did not return to the section.")
    T_target_s = ray_coordinate(T_first_return_sol.u[end], section)
    T_pullback = pullback_preimage(
        p,
        section,
        T_target_s,
        sample_ss,
        sample_return_ss;
        saveat=SPECIAL_SAVEAT,
        abstol=3e-6,
        reltol=3e-6,
    )

    gamma_return_sol, gamma_target = solve_section_return(
        p,
        section,
        saddle_data.gamma_sd_minus0;
        target_returns=1,
        saveat=SPECIAL_SAVEAT,
        abstol=1e-8,
        reltol=1e-8,
    )
    returned_to_section(gamma_return_sol, gamma_target) || error("Gamma_SD^- did not return to the section.")
    gamma_target_s = ray_coordinate(gamma_return_sol.u[end], section)
    gamma_pullback = pullback_preimage(
        p,
        section,
        gamma_target_s,
        sample_ss,
        sample_return_ss;
        saveat=SPECIAL_SAVEAT,
        abstol=1e-8,
        reltol=1e-8,
    )

    ca_values, returned_vs, t0_returned_v = local_t0_objective_curve(p, saddle_data.x_eq_SF, T0)
    refined_intervals, refined_points_per_interval, refined_return_groups = refine_return_map_samples(
        p,
        section,
        sample_ss,
        sample_return_ss,
        sample_spike_counts;
        saveat=SAMPLE_SAVEAT,
        abstol=1e-8,
        reltol=1e-8,
    )

    max_ray_s = maximum(vcat(
        [1.0],
        [ray_coordinate(u, section) for u in (T_pullback.start_u, T_pullback.end_u, gamma_pullback.start_u, gamma_pullback.end_u)],
        [sample_return_ss[idx] for idx in eachindex(sample_return_ss) if isfinite(sample_return_ss[idx])],
    ))
    ray_endpoint = section.point_SF .+ (1.05 * max_ray_s) .* section.tangent

    plot_path = joinpath(ATTEMPT11_DIAG_ROOT, "$(DIAG_OUTPUT_TAG)_diagnostics.png")
    summary_path = joinpath(ATTEMPT11_DIAG_ROOT, "$(DIAG_OUTPUT_TAG)_summary.txt")

    fig = Figure(size=DIAG_FIGURE_SIZE)

    ax_phase = Axis(
        fig[1, 1],
        xlabel="Ca",
        ylabel="x",
        title=@sprintf("Hook Probe Phase Portrait, Δx = %.6f, ΔCa = %.6f", delta_x, delta_ca),
    )
    for (sol, spike_count) in zip(sample_trajs, sample_spike_counts)
        lines!(ax_phase, extract_ca(sol), extract_x(sol); color=spike_count_color(spike_count), linewidth=1.2)
    end
    lines!(ax_phase, [section.point_SF[1], ray_endpoint[1]], [section.point_SF[2], ray_endpoint[2]]; color=:black, linewidth=3, linestyle=:dash)
    scatter!(ax_phase, [section.point_SF[1]], [section.point_SF[2]]; color=:black, markersize=14, marker=:circle)
    scatter!(ax_phase, [section.point_SD[1]], [section.point_SD[2]]; color=:black, markersize=16, marker=:utriangle)
    returned_idxs = [idx for idx in eachindex(sample_ss) if isfinite(sample_return_ss[idx])]
    scatter!(
        ax_phase,
        [p[1] for p in return_points],
        [p[2] for p in return_points];
        color=[spike_count_color(sample_spike_counts[idx]; alpha=0.82f0) for idx in returned_idxs],
        markersize=5,
    )
    lines!(ax_phase, extract_ca(T_pullback.sol), extract_x(T_pullback.sol); color=:firebrick, linewidth=4.2, label="T0 ray-to-ray")
    lines!(ax_phase, extract_ca(gamma_return_sol), extract_x(gamma_return_sol); color=:royalblue, linewidth=4.2, label="Γ_SD^- to section")
    scatter!(ax_phase, [T_pullback.start_u[5]], [T_pullback.start_u[1]]; color=:firebrick, markersize=14, marker=:diamond)
    scatter!(ax_phase, [T_pullback.end_u[5]], [T_pullback.end_u[1]]; color=:firebrick, markersize=18, marker=:star5)
    scatter!(ax_phase, [T0[5]], [T0[1]]; color=:firebrick, markersize=16, marker=:xcross)
    scatter!(ax_phase, [gamma_pullback.start_u[5]], [gamma_pullback.start_u[1]]; color=:royalblue, markersize=14, marker=:diamond)
    scatter!(ax_phase, [state6(gamma_return_sol.u[end])[5]], [state6(gamma_return_sol.u[end])[1]]; color=:royalblue, markersize=18, marker=:star5)
    scatter!(ax_phase, [saddle_data.gamma_sd_minus0[5]], [saddle_data.gamma_sd_minus0[1]]; color=:royalblue, markersize=14, marker=:rect)
    axislegend(ax_phase, position=:rb)

    ax_return = Axis(
        fig[1, 2],
        xlabel="section start sₙ",
        ylabel="section return sₙ₊₁",
        title="Refined Return Map From Spike-Count Intervals",
    )
    for group in refined_return_groups
        branch_color = spike_count_color(group.initial_spike_count; alpha=0.9f0)
        lines!(ax_return, group.start_ss, group.return_ss; color=branch_color, linewidth=1.6)
        scatter!(ax_return, group.start_ss, group.return_ss; color=branch_color, markersize=6)
    end
    lines!(ax_return, [SECTION_S_MIN, SECTION_S_MAX], [SECTION_S_MIN, SECTION_S_MAX]; color=RGBAf(0.1, 0.1, 0.1, 0.35), linestyle=:dash, linewidth=2)
    scatter!(ax_return, [T_pullback.start_s], [T_pullback.end_s]; color=:firebrick, markersize=18, marker=:diamond)
    scatter!(ax_return, [gamma_pullback.start_s], [gamma_pullback.end_s]; color=:royalblue, markersize=18, marker=:diamond)
    xlims!(ax_return, SECTION_S_MIN, SECTION_S_MAX)
    ylims!(ax_return, SECTION_S_MIN, SECTION_S_MAX)

    ax_t0 = Axis(
        fig[1, 3],
        xlabel="Ca₀ along local T0 branch",
        ylabel="returned V at first Ca-min",
        title="Fresh T0 Solve Objective",
    )
    lines!(ax_t0, ca_values, returned_vs; color=:firebrick, linewidth=2.8)
    scatter!(ax_t0, ca_values, returned_vs; color=RGBAf(0.7, 0.15, 0.15, 0.35), markersize=5)
    vlines!(ax_t0, [T0[5]]; color=:firebrick, linewidth=2, linestyle=:dash)
    scatter!(ax_t0, [T0[5]], [t0_returned_v]; color=:firebrick, markersize=18, marker=:star5)

    save(plot_path, fig; px_per_unit=DIAG_PX_PER_UNIT)

    open(summary_path, "w") do io
        println(io, "requested_delta_x\t$(DIAG_DELTA_X)")
        println(io, "requested_delta_ca\t$(DIAG_DELTA_CA)")
        println(io, "grid_delta_x\t$(delta_x)")
        println(io, "grid_delta_ca\t$(delta_ca)")
        println(io, "saved_T_scs\t$(saved_row["T_scs"])")
        println(io, "saved_gamma_scs\t$(saved_row["gamma_scs"])")
        println(io, "default_T_scs\t$(join(default_T_scs, ','))")
        println(io, "fresh_T_scs\t$(join(T_scs, ','))")
        println(io, "fresh_gamma_scs\t$(join(gamma_scs, ','))")
        println(io, "default_T0_V\t$(default_T0[6])")
        println(io, "default_T0_Ca\t$(default_T0[5])")
        println(io, "fresh_T0_V\t$(T0[6])")
        println(io, "fresh_T0_Ca\t$(T0[5])")
        println(io, "T_pullback_start_s\t$(T_pullback.start_s)")
        println(io, "T_pullback_end_s\t$(T_pullback.end_s)")
        println(io, "T_pullback_error\t$(T_pullback.error)")
        println(io, "gamma_pullback_start_s\t$(gamma_pullback.start_s)")
        println(io, "gamma_pullback_end_s\t$(gamma_pullback.end_s)")
        println(io, "gamma_pullback_error\t$(gamma_pullback.error)")
        println(io, "returned_sample_trajectories\t$(length(returned_idxs))/$(length(sample_ss))")
        println(io, "refined_interval_count\t$(length(refined_intervals))")
        println(io, "refined_points_per_interval\t$(refined_points_per_interval)")
        println(io, "refined_returned_points\t$(sum(length(group.start_ss) for group in refined_return_groups))")
        println(io, "t0_objective_at_T0\t$(t0_returned_v)")
        println(io, "plot_path\t$(plot_path)")
    end

    println("grid_delta_x=$(delta_x)")
    println("grid_delta_ca=$(delta_ca)")
    println("saved_T_scs=$(saved_row["T_scs"])")
    println("saved_gamma_scs=$(saved_row["gamma_scs"])")
    println("default_T_scs=$(join(default_T_scs, ','))")
    println("fresh_T_scs=$(join(T_scs, ','))")
    println("fresh_gamma_scs=$(join(gamma_scs, ','))")
    println("returned_sample_trajectories=$(length(returned_idxs))/$(length(sample_ss))")
    println("refined_interval_count=$(length(refined_intervals))")
    println("refined_points_per_interval=$(refined_points_per_interval)")
    println("plot_path=$(plot_path)")
    println("summary_path=$(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
