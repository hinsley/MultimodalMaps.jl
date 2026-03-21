using Pkg

const ATTEMPT11_ROOT = @__DIR__
const REPO_ROOT_011 = normpath(joinpath(ATTEMPT11_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_011)

include(joinpath(ATTEMPT11_ROOT, "main.jl"))

using Printf

env_float_local(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

const INSPECT_DELTA_X = env_float_local("ATTEMPT011_INSPECT_DELTA_X", -1.0)
const INSPECT_DELTA_CA = env_float_local("ATTEMPT011_INSPECT_DELTA_CA", -32.5)
const INSPECT_RESULTS_PATH = get(
    ENV,
    "ATTEMPT011_INSPECT_RESULTS",
    joinpath(ATTEMPT11_ROOT, "grid500_seq7_prefixes_results.tsv"),
)
const INSPECT_OUTPUT_TAG = get(ENV, "ATTEMPT011_INSPECT_TAG", "returninspect_dxm1_dcam32p5")
const SECTION_SAMPLE_COUNT = parse(Int, get(ENV, "ATTEMPT011_SECTION_SAMPLE_COUNT", "240"))
const SECTION_TSPAN = (0.0, parse(Float64, get(ENV, "ATTEMPT011_SECTION_TMAX", "5.0e4")))
const SECTION_S_MIN = parse(Float64, get(ENV, "ATTEMPT011_SECTION_S_MIN", "0.02"))
const SECTION_S_MAX = parse(Float64, get(ENV, "ATTEMPT011_SECTION_S_MAX", "0.98"))
const SAMPLE_SAVEAT = parse(Float64, get(ENV, "ATTEMPT011_SECTION_SAVEAT", "0.25"))
const SPECIAL_SAVEAT = parse(Float64, get(ENV, "ATTEMPT011_SPECIAL_SAVEAT", "0.5"))
const T0_SECTION_ITERATE = parse(Int, get(ENV, "ATTEMPT011_T0_SECTION_ITERATE", "0"))
const INSPECT_FIGURE_SIZE = (1800, 1450)
const INSPECT_PX_PER_UNIT = env_float_local("ATTEMPT011_INSPECT_PX_PER_UNIT", 2.5)
const SPIKE_THRESHOLD = 0.0
const SPIKE_COLORMAP = to_colormap(:glasbey_bw_n256)

function parse_results(path::String)
    open(path, "r") do io
        header = split(readline(io), '\t')
        rows = Dict{String, String}[]
        for line in eachline(io)
            fields = split(line, '\t')
            push!(rows, Dict(header[idx] => fields[idx] for idx in eachindex(header)))
        end
        return rows
    end
end

function nearest_saved_row(rows::Vector{Dict{String, String}}, delta_x::Float64, delta_ca::Float64)
    best_idx = 0
    best_dist = Inf
    for (idx, row) in enumerate(rows)
        row_x = parse(Float64, row["delta_x"])
        row_ca = parse(Float64, row["delta_ca"])
        dist = (row_x - delta_x)^2 + (row_ca - delta_ca)^2
        if dist < best_dist
            best_dist = dist
            best_idx = idx
        end
    end
    return rows[best_idx]
end

function previous_successful_seed(
    rows::Vector{Dict{String, String}},
    current_row::Dict{String, String},
)::Union{Nothing, T0ContinuationSeed}
    current_x = parse(Float64, current_row["delta_x"])
    current_ca = parse(Float64, current_row["delta_ca"])
    previous_row = nothing
    previous_x = Inf
    for row in rows
        if row["status"] != "ok"
            continue
        end
        row_ca = parse(Float64, row["delta_ca"])
        row_x = parse(Float64, row["delta_x"])
        if !isapprox(row_ca, current_ca; atol=1e-9, rtol=0.0)
            continue
        end
        if row_x > current_x && row_x < previous_x
            previous_row = row
            previous_x = row_x
        end
    end

    if isnothing(previous_row)
        return nothing
    end

    return T0ContinuationSeed(parse(Float64, previous_row["T0_V"]), parse(Float64, previous_row["T0_Ca"]))
end

function equilibrium_section_data(p)
    V_eqs = find_equilibria(p)
    V_eq_SF = V_eqs[2]
    V_eq_SD = V_eqs[3]

    x_eq_SF = Plant.xinf(p, V_eq_SF)
    Ca_eq_SF = EquilibriaSubset.Ca_null_Ca(p, V_eq_SF)
    x_eq_SD = Plant.xinf(p, V_eq_SD)
    Ca_eq_SD = EquilibriaSubset.Ca_null_Ca(p, V_eq_SD)

    point_SF = @SVector [Ca_eq_SF, x_eq_SF]
    point_SD = @SVector [Ca_eq_SD, x_eq_SD]
    tangent = point_SD - point_SF
    normal = @SVector [-tangent[2], tangent[1]]
    tangent_norm_sq = dot(tangent, tangent)

    SF_eq = @SVector [x_eq_SF, 0.0, Plant.ninf(V_eq_SF), Plant.hinf(V_eq_SF), Ca_eq_SF, V_eq_SF]
    SD_eq = @SVector [x_eq_SD, 0.0, Plant.ninf(V_eq_SD), Plant.hinf(V_eq_SD), Ca_eq_SD, V_eq_SD]

    return (; point_SF, point_SD, tangent, normal, tangent_norm_sq, SF_eq, SD_eq)
end

section_projection(u, section) = @SVector [u[5], u[1]]
ray_coordinate(u, section) = dot(section_projection(u, section) - section.point_SF, section.tangent) / section.tangent_norm_sq
ray_angle(u, section) = atan(u[1] - section.point_SF[2], u[5] - section.point_SF[1])

function angular_phase_derivative(u::SVector{6, Float64}, p, section)
    f = Plant.melibeNew(u, p, 0.0)
    ca_rel = u[5] - section.point_SF[1]
    x_rel = u[1] - section.point_SF[2]
    denom = max(ca_rel^2 + x_rel^2, 1.0e-12)
    return (ca_rel * f[1] - x_rel * f[5]) / denom
end

rotation_sign(u0::SVector{6, Float64}, p, section) = begin
    sign0 = sign(angular_phase_derivative(u0, p, section))
    sign0 == 0 ? 1.0 : sign0
end

function phase_to_reference_ray(u0::SVector{6, Float64}, section, sign_turn::Float64)
    theta0 = ray_angle(u0, section)
    theta_ref = atan(section.tangent[2], section.tangent[1])
    if sign_turn > 0
        delta = mod(theta_ref - theta0, 2pi)
        return delta <= 1.0e-10 ? 2pi : delta
    else
        delta = mod(theta0 - theta_ref, 2pi)
        return -(delta <= 1.0e-10 ? 2pi : delta)
    end
end

function make_phase_return_callback(target_phase::Float64)
    function condition(u, t, integrator)
        return u[7] - target_phase
    end

    function affect!(integrator)
        terminate!(integrator)
    end

    return ContinuousCallback(condition, affect!, affect_neg! = affect!, save_positions=(false, true))
end

function make_v_upcross_counter(spike_count::Base.RefValue{Int})
    function condition(u, t, integrator)
        return u[6]
    end

    function affect!(integrator)
        spike_count[] += 1
    end

    return ContinuousCallback(condition, affect!, affect_neg! = nothing, save_positions=(false, false))
end

function lift_section_point(p, section, s::Float64)
    point = section.point_SF .+ s .* section.tangent
    return SVector{6, Float64}(Tuple(Float64.(EquilibriaSubset.dune(p, point[2], point[1]))))
end

function solve_section_return(
    p,
    section,
    u0::SVector{6, Float64};
    target_returns::Int,
    direction::Int=1,
    saveat::Float64,
    abstol::Float64,
    reltol::Float64,
)
    function augmented_rhs(u, _, t)
        state = @SVector [u[1], u[2], u[3], u[4], u[5], u[6]]
        f = Plant.melibeNew(state, p, t)

        ca_rel = u[5] - section.point_SF[1]
        x_rel = u[1] - section.point_SF[2]
        denom = max(ca_rel^2 + x_rel^2, 1.0e-12)
        phase_dot = (ca_rel * f[1] - x_rel * f[5]) / denom
        return SVector{7, Float64}(f[1], f[2], f[3], f[4], f[5], f[6], phase_dot)
    end

    turn_sign = direction * rotation_sign(u0, p, section)
    first_hit_phase = phase_to_reference_ray(u0, section, turn_sign)
    target_phase = first_hit_phase + turn_sign * (2pi * (target_returns - 1))
    u0_aug = SVector{7, Float64}(u0[1], u0[2], u0[3], u0[4], u0[5], u0[6], 0.0)
    tspan = direction > 0 ? SECTION_TSPAN : (SECTION_TSPAN[1], -SECTION_TSPAN[2])
    prob = ODEProblem{false}(augmented_rhs, u0_aug, tspan)
    spike_count = Ref(0)
    callback = CallbackSet(make_phase_return_callback(target_phase), make_v_upcross_counter(spike_count))
    sol = solve(
        prob,
        SOLVER_010;
        callback=callback,
        abstol=abstol,
        reltol=reltol,
        save_everystep=false,
        saveat=saveat,
    )
    return sol, target_phase, spike_count[]
end

returned_to_section(sol, target_phase::Float64) =
    !isempty(sol.u) &&
    abs(sol.t[end]) < SECTION_TSPAN[2] - 1.0e-6 &&
    abs(sol.u[end][7] - target_phase) <= 5.0e-4

function solve_special_trajectory(
    p,
    u0::SVector{6, Float64},
    V_sd::Float64;
    saveat::Float64,
    abstol::Float64,
    reltol::Float64,
)
    state_machine = make_state_machine(V_sd)
    callback = VectorContinuousCallback(
        sscs_condition!,
        make_affect_sscs!(state_machine),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(Plant.melibeNew, u0, SSCS_ODE_TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false, saveat=saveat)
    return sol, state_machine.scs
end

extract_ca(sol) = [u[5] for u in sol.u]
extract_x(sol) = [u[1] for u in sol.u]
state6(u) = SVector{6, Float64}(u[1], u[2], u[3], u[4], u[5], u[6])

function spike_count_to_return(sol; threshold::Float64=SPIKE_THRESHOLD)
    count = 0
    was_above = false
    for u in sol.u
        above = u[6] > threshold
        if above && !was_above
            count += 1
        end
        was_above = above
    end
    return count
end

function spike_count_color(count::Int; alpha::Float32=0.62f0)
    idx = mod1(19 * count + 11, length(SPIKE_COLORMAP))
    c = SPIKE_COLORMAP[idx]
    return RGBAf(c.r, c.g, c.b, alpha)
end

function solve_section_from_s(p, section, s::Float64; saveat::Float64, abstol::Float64, reltol::Float64)
    u0 = lift_section_point(p, section, s)
    sol, target_phase, spike_count = solve_section_return(
        p,
        section,
        u0;
        target_returns=1,
        saveat=saveat,
        abstol=abstol,
        reltol=reltol,
    )
    return sol, target_phase, spike_count
end

function recompute_T0(rows, saved_row, p, saddle_data)
    if startswith(saved_row["T0_method"], "continued")
        seed = previous_successful_seed(rows, saved_row)
        if isnothing(seed)
            error("Saved point used continuation, but no previous successful seed was found.")
        end
        return first(initialize_T_Ca0_from_seed(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0, seed))
    end
    return initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
end

function write_summary(path::String, saved_row, T_scs, gamma_scs, return_count::Int)
    open(path, "w") do io
        println(io, "requested_delta_x\t$(INSPECT_DELTA_X)")
        println(io, "requested_delta_ca\t$(INSPECT_DELTA_CA)")
        println(io, "grid_delta_x\t$(saved_row["delta_x"])")
        println(io, "grid_delta_ca\t$(saved_row["delta_ca"])")
        println(io, "saved_T0_method\t$(saved_row["T0_method"])")
        println(io, "saved_T_scs\t$(saved_row["T_scs"])")
        println(io, "saved_gamma_scs\t$(saved_row["gamma_scs"])")
        println(io, "recomputed_T_scs\t$(join(T_scs, ','))")
        println(io, "recomputed_gamma_scs\t$(join(gamma_scs, ','))")
        println(io, "T0_section_iterate\t$(T0_SECTION_ITERATE)")
        println(io, "returned_sample_trajectories\t$(return_count)")
        println(io, "sample_count\t$(SECTION_SAMPLE_COUNT)")
    end
end

function main()
    rows = parse_results(INSPECT_RESULTS_PATH)
    saved_row = nearest_saved_row(rows, INSPECT_DELTA_X, INSPECT_DELTA_CA)
    if saved_row["status"] != "ok"
        error("Nearest saved point is not successful: $(saved_row["status"])")
    end

    delta_x = parse(Float64, saved_row["delta_x"])
    delta_ca = parse(Float64, saved_row["delta_ca"])
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    section = equilibrium_section_data(p)
    T0 = recompute_T0(rows, saved_row, p, saddle_data)

    T_sol, T_scs = solve_special_trajectory(p, T0, saddle_data.V_eq_SD; saveat=SPECIAL_SAVEAT, abstol=3e-6, reltol=3e-6)
    gamma_sol, gamma_scs = solve_special_trajectory(
        p,
        saddle_data.gamma_sd_minus0,
        saddle_data.V_eq_SD;
        saveat=SPECIAL_SAVEAT,
        abstol=1e-8,
        reltol=1e-8,
    )

    sample_ss = collect(range(SECTION_S_MIN, SECTION_S_MAX, length=SECTION_SAMPLE_COUNT))
    sample_trajs = Any[]
    start_points = Point2f[]
    return_points = Point2f[]
    sample_return_ss = Float64[]
    sample_spike_counts = Int[]

    for s in sample_ss
        u0 = lift_section_point(p, section, s)
        sol, target_phase, spike_count = solve_section_from_s(p, section, s; saveat=SAMPLE_SAVEAT, abstol=1e-8, reltol=1e-8)
        push!(sample_trajs, sol)
        push!(start_points, Point2f(u0[5], u0[1]))
        push!(sample_spike_counts, spike_count)
        if returned_to_section(sol, target_phase)
            u_end = sol.u[end]
            push!(return_points, Point2f(u_end[5], u_end[1]))
            push!(sample_return_ss, ray_coordinate(u_end, section))
        else
            push!(sample_return_ss, NaN)
        end
    end

    T_first_return_sol, T_first_target_phase, _ = solve_section_return(
        p,
        section,
        T0;
        target_returns=1,
        saveat=SPECIAL_SAVEAT,
        abstol=3e-6,
        reltol=3e-6,
    )
    T_plot_ca = Float64[]
    T_plot_x = Float64[]
    T_plot_start = T0
    T_plot_end = T0
    T_plot_ready = false
    if returned_to_section(T_first_return_sol, T_first_target_phase)
        T_plot_end = state6(T_first_return_sol.u[end])
        target_s = ray_coordinate(T_plot_end, section)
        valid_idxs = findall(idx -> isfinite(sample_return_ss[idx]) && sample_spike_counts[idx] == 1, eachindex(sample_return_ss))
        if !isempty(valid_idxs)
            ordered = sort(valid_idxs; by=idx -> abs(sample_return_ss[idx] - target_s))
            best_sol = sample_trajs[ordered[1]]
            best_start = lift_section_point(p, section, sample_ss[ordered[1]])
            best_err = abs(sample_return_ss[ordered[1]] - target_s)

            if length(ordered) >= 2
                s0 = sample_ss[ordered[1]]
                s1 = sample_ss[ordered[2]]
                f0 = sample_return_ss[ordered[1]] - target_s
                f1 = sample_return_ss[ordered[2]] - target_s
                for _ in 1:5
                    if !isfinite(f0) || !isfinite(f1) || abs(f1 - f0) < 1.0e-10
                        break
                    end
                    s2 = clamp(s1 - f1 * (s1 - s0) / (f1 - f0), SECTION_S_MIN, SECTION_S_MAX)
                    sol2, target2, spike_count2 = solve_section_from_s(
                        p,
                        section,
                        s2;
                        saveat=SPECIAL_SAVEAT,
                        abstol=3e-6,
                        reltol=3e-6,
                    )
                    if !returned_to_section(sol2, target2) || spike_count2 != 1
                        break
                    end
                    f2 = ray_coordinate(sol2.u[end], section) - target_s
                    if abs(f2) < best_err
                        best_sol = sol2
                        best_start = lift_section_point(p, section, s2)
                        best_err = abs(f2)
                    end
                    s0, f0 = s1, f1
                    s1, f1 = s2, f2
                end
            end

            T_plot_start = best_start
            T_plot_ca = extract_ca(best_sol)
            T_plot_x = extract_x(best_sol)
            T_plot_ready = true
        else
            error("Could not locate a 1-spike section preimage for the plotted T0 return.")
        end
    else
        T_plot_ca = extract_ca(T_first_return_sol)
        T_plot_x = extract_x(T_first_return_sol)
    end

    if T_plot_ready && T0_SECTION_ITERATE > 0
        current_s = ray_coordinate(T_plot_end, section)
        current_start = T_plot_start
        current_end = T_plot_end
        current_ca = T_plot_ca
        current_x = T_plot_x
        for _ in 1:T0_SECTION_ITERATE
            current_start = lift_section_point(p, section, current_s)
            iterate_sol, iterate_target_phase, _ = solve_section_return(
                p,
                section,
                current_start;
                target_returns=1,
                saveat=SPECIAL_SAVEAT,
                abstol=3e-6,
                reltol=3e-6,
            )
            if !returned_to_section(iterate_sol, iterate_target_phase)
                error("Could not advance the plotted T0 branch by $(T0_SECTION_ITERATE) first-return iterates.")
            end
            current_end = state6(iterate_sol.u[end])
            current_s = ray_coordinate(current_end, section)
            current_ca = extract_ca(iterate_sol)
            current_x = extract_x(iterate_sol)
        end
        T_plot_start = current_start
        T_plot_end = current_end
        T_plot_ca = current_ca
        T_plot_x = current_x
    end

    gamma_return_sol, gamma_target_phase, _ = solve_section_return(
        p,
        section,
        saddle_data.gamma_sd_minus0;
        target_returns=1,
        saveat=SPECIAL_SAVEAT,
        abstol=1e-8,
        reltol=1e-8,
    )

    plot_path = joinpath(ATTEMPT11_ROOT, "$(INSPECT_OUTPUT_TAG)_return_segments.png")
    summary_path = joinpath(ATTEMPT11_ROOT, "$(INSPECT_OUTPUT_TAG)_return_section_summary.txt")

    max_ray_s = 1.0
    for (sol, _) in zip(sample_trajs, sample_ss)
        if sol.t[end] < SECTION_TSPAN[2] - 1.0e-6
            max_ray_s = max(max_ray_s, ray_coordinate(sol.u[end], section))
        end
    end
    if T_plot_ready
        max_ray_s = max(max_ray_s, ray_coordinate(T_plot_start, section))
        max_ray_s = max(max_ray_s, ray_coordinate(T_plot_end, section))
    end
    if returned_to_section(gamma_return_sol, gamma_target_phase)
        max_ray_s = max(max_ray_s, ray_coordinate(gamma_return_sol.u[end], section))
    end
    ray_endpoint = section.point_SF .+ (1.05 * max_ray_s) .* section.tangent

    fig = Figure(size=INSPECT_FIGURE_SIZE)
    ax_phase = Axis(
        fig[1, 1],
        xlabel="Ca",
        ylabel="x",
        title=@sprintf("Return-To-Section Trajectories, Δx = %.6f, ΔCa = %.6f", delta_x, delta_ca),
    )

    for (sol, spike_count) in zip(sample_trajs, sample_spike_counts)
        lines!(ax_phase, extract_ca(sol), extract_x(sol); color=spike_count_color(spike_count), linewidth=1.45)
    end

    section_cas = [section.point_SF[1], ray_endpoint[1]]
    section_xs = [section.point_SF[2], ray_endpoint[2]]
    lines!(ax_phase, section_cas, section_xs; color=:black, linewidth=3, linestyle=:dash)
    scatter!(ax_phase, [section.point_SF[1]], [section.point_SF[2]]; color=:black, markersize=14, marker=:circle)
    scatter!(ax_phase, [section.point_SD[1]], [section.point_SD[2]]; color=:black, markersize=16, marker=:utriangle)
    scatter!(ax_phase, [p[1] for p in start_points], [p[2] for p in start_points]; color=RGBAf(0.2, 0.2, 0.2, 0.45), markersize=6)
    if !isempty(return_points)
        returned_idxs = findall(isfinite, sample_return_ss)
        scatter!(
            ax_phase,
            [p[1] for p in return_points],
            [p[2] for p in return_points];
            color=[spike_count_color(sample_spike_counts[idx]; alpha=0.82f0) for idx in returned_idxs],
            markersize=7,
        )
    end

    lines!(ax_phase, T_plot_ca, T_plot_x; color=:firebrick, linewidth=4.5, label="T0 ray-to-ray")
    lines!(ax_phase, extract_ca(gamma_return_sol), extract_x(gamma_return_sol); color=:royalblue, linewidth=4.5, label="Γ_SD^- to section")
    scatter!(ax_phase, [T_plot_start[5]], [T_plot_start[1]]; color=:firebrick, markersize=16, marker=:diamond)
    scatter!(ax_phase, [saddle_data.gamma_sd_minus0[5]], [saddle_data.gamma_sd_minus0[1]]; color=:royalblue, markersize=16, marker=:rect)
    scatter!(ax_phase, [T_plot_end[5]], [T_plot_end[1]]; color=:firebrick, markersize=18, marker=:star5)
    if returned_to_section(gamma_return_sol, gamma_target_phase)
        u_end = gamma_return_sol.u[end]
        scatter!(ax_phase, [u_end[5]], [u_end[1]]; color=:royalblue, markersize=18, marker=:star5)
    end
    axislegend(ax_phase, position=:rb)
    save(plot_path, fig; px_per_unit=INSPECT_PX_PER_UNIT)

    write_summary(summary_path, saved_row, T_scs, gamma_scs, length(return_points))

    println("grid_delta_x=$(delta_x)")
    println("grid_delta_ca=$(delta_ca)")
    println("T0_method=$(saved_row["T0_method"])")
    println("T_scs=$(join(T_scs, ','))")
    println("gamma_scs=$(join(gamma_scs, ','))")
    println("T0_section_iterate=$(T0_SECTION_ITERATE)")
    println("returned_sample_trajectories=$(length(return_points))/$(SECTION_SAMPLE_COUNT)")
    println("T0_returned=$(T_plot_ready)")
    println("gamma_returned=$(returned_to_section(gamma_return_sol, gamma_target_phase))")
    println("plot_path=$(plot_path)")
    println("summary_path=$(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
