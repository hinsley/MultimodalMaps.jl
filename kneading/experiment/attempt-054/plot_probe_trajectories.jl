using Pkg

const ATTEMPT54_PROBE_ROOT = @__DIR__
const REPO_ROOT_054_PROBE = normpath(joinpath(ATTEMPT54_PROBE_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_054_PROBE)

include(joinpath(ATTEMPT54_PROBE_ROOT, "main.jl"))

using CairoMakie
using Printf
using StaticArrays

probe_env_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))
probe_env_int(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))

const PROBE_DELTA_CA = probe_env_float("ATTEMPT054_PROBE_DELTA_CA", -35.0)
const PROBE_DELTA_X = probe_env_float("ATTEMPT054_PROBE_DELTA_X", -1.0)
const PROBE_MAX_MINIMA = probe_env_int("ATTEMPT054_PROBE_MAX_MINIMA", 8)
const PROBE_TMAX = probe_env_float("ATTEMPT054_PROBE_TMAX", 1.0e5)
const PROBE_TAG = get(ENV, "ATTEMPT054_PROBE_TAG", "probe_dca35_dxm1_ystub_trajectories")

mutable struct CaMinRecorderProbe
    times::Vector{Float64}
    states::Vector{SVector{5, Float64}}
end

function make_ca_min_record_callback_probe(recorder::CaMinRecorderProbe)
    function condition(u, t, integrator)
        if t < MIN_EVENT_TIME_054
            return 1.0
        end
        return active_flow_054(state5_054(u), integrator.p, integrator.t)[4]
    end

    function affect!(integrator)
        push!(recorder.times, Float64(integrator.t))
        push!(recorder.states, state5_054(integrator.u))
        if length(recorder.times) >= PROBE_MAX_MINIMA
            terminate!(integrator)
        end
        return nothing
    end

    return ContinuousCallback(condition, affect!, affect_neg! = nothing; save_positions=(true, true))
end

function solve_active_probe_trajectory(p, u0_6::SVector{6, Float64}; abstol::Float64, reltol::Float64)
    u0 = collect(state5_from_state6_054(u0_6))
    recorder = CaMinRecorderProbe(Float64[], SVector{5, Float64}[])
    prob = ODEProblem((du, u, p, t) -> begin
        flow = active_flow_054(state5_054(u), p, t)
        for i in 1:5
            du[i] = flow[i]
        end
        return nothing
    end, u0, (0.0, PROBE_TMAX), p)
    sol = solve(
        prob,
        SOLVER_010;
        callback=make_ca_min_record_callback_probe(recorder),
        abstol=abstol,
        reltol=reltol,
        save_everystep=true,
    )
    return sol, recorder
end

extract_active_ca(sol) = [u[4] for u in sol.u]
extract_active_x(sol) = [u[1] for u in sol.u]
extract_marker_ca(recorder::CaMinRecorderProbe) = [u[4] for u in recorder.states]
extract_marker_x(recorder::CaMinRecorderProbe) = [u[1] for u in recorder.states]

function write_probe_summary(path::String, T0::SVector{6, Float64}, gamma0::SVector{6, Float64}, T_rec, gamma_rec)
    open(path, "w") do io
        println(io, "delta_ca\t$(PROBE_DELTA_CA)")
        println(io, "delta_x\t$(PROBE_DELTA_X)")
        println(io, "tmax\t$(PROBE_TMAX)")
        println(io, "max_minima\t$(PROBE_MAX_MINIMA)")
        println(io, "active_state_order\tx\tn\th\tCa\tV")
        println(io, @sprintf("T0_x\t%.12f", T0[1]))
        println(io, @sprintf("T0_Ca\t%.12f", T0[5]))
        println(io, @sprintf("T0_V\t%.12f", T0[6]))
        println(io, @sprintf("gamma0_x\t%.12f", gamma0[1]))
        println(io, @sprintf("gamma0_Ca\t%.12f", gamma0[5]))
        println(io, @sprintf("gamma0_V\t%.12f", gamma0[6]))
        println(io, "T_min_times\t$(join((@sprintf("%.10g", t) for t in T_rec.times), ","))")
        println(io, "gamma_min_times\t$(join((@sprintf("%.10g", t) for t in gamma_rec.times), ","))")
    end
end

function main()
    p = build_params(PROBE_DELTA_X, PROBE_DELTA_CA)
    saddle_data = compute_gamma_sd_minus0(p)
    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)

    T_sol, T_rec = solve_active_probe_trajectory(p, T0; abstol=3e-6, reltol=3e-6)
    gamma_sol, gamma_rec = solve_active_probe_trajectory(p, saddle_data.gamma_sd_minus0; abstol=1e-8, reltol=1e-8)

    plot_path = joinpath(ATTEMPT54_PROBE_ROOT, "$(PROBE_TAG).png")
    summary_path = joinpath(ATTEMPT54_PROBE_ROOT, "$(PROBE_TAG)_summary.txt")

    T_red = RGBAf(1.0, 0.0, 0.0, 0.88)
    gamma_blue = RGBAf(0.0, 0.23, 1.0, 0.84)

    fig = Figure(size=(1350, 1050), fontsize=22)
    ax = Axis(
        fig[1, 1],
        xlabel="Ca",
        ylabel="x",
        title=@sprintf("Active y-stubbed trajectories: ΔCa = %.3f, Δx = %.3f", PROBE_DELTA_CA, PROBE_DELTA_X),
        xlabelsize=28,
        ylabelsize=28,
        titlesize=28,
        xticklabelsize=20,
        yticklabelsize=20,
    )

    lines!(ax, extract_active_ca(T_sol), extract_active_x(T_sol); color=T_red, linewidth=2.6, label="T0 trajectory")
    lines!(ax, extract_active_ca(gamma_sol), extract_active_x(gamma_sol); color=gamma_blue, linewidth=2.6, label="Γ_SD^- trajectory")

    scatter!(
        ax,
        extract_marker_ca(T_rec),
        extract_marker_x(T_rec);
        color=gamma_blue,
        strokecolor=:white,
        strokewidth=1.0,
        markersize=13,
        label="T0 Ca minima",
    )
    scatter!(
        ax,
        extract_marker_ca(gamma_rec),
        extract_marker_x(gamma_rec);
        color=T_red,
        strokecolor=:white,
        strokewidth=1.0,
        markersize=13,
        marker=:diamond,
        label="Γ_SD^- Ca minima",
    )
    scatter!(ax, [T0[5]], [T0[1]]; color=T_red, markersize=18, strokecolor=:black, strokewidth=1.3)
    scatter!(ax, [saddle_data.gamma_sd_minus0[5]], [saddle_data.gamma_sd_minus0[1]]; color=gamma_blue, markersize=18, strokecolor=:black, strokewidth=1.3, marker=:diamond)
    axislegend(ax, position=:rb, labelsize=18)

    save(plot_path, fig; px_per_unit=2)
    write_probe_summary(summary_path, T0, saddle_data.gamma_sd_minus0, T_rec, gamma_rec)

    println("plot_path=$(plot_path)")
    println("summary_path=$(summary_path)")
    println("T_minima=$(length(T_rec.times))")
    println("gamma_minima=$(length(gamma_rec.times))")
    println("T_min_times=$(join((@sprintf("%.10g", t) for t in T_rec.times), ","))")
    println("gamma_min_times=$(join((@sprintf("%.10g", t) for t in gamma_rec.times), ","))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
