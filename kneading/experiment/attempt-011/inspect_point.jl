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
const INSPECT_OUTPUT_TAG = get(ENV, "ATTEMPT011_INSPECT_TAG", "inspect_dxm1_dcam32p5")

function parse_results(path::String)
    open(path, "r") do io
        header = split(readline(io), '\t')
        index = Dict(name => idx for (idx, name) in enumerate(header))
        rows = Dict{String, String}[]
        for line in eachline(io)
            fields = split(line, '\t')
            push!(rows, Dict(name => fields[idx] for (name, idx) in index))
        end
        return rows
    end
end

parse_sequence_field_local(field::String) = isempty(field) ? Int[] : parse.(Int, split(field, ","))

function nearest_saved_row(rows::Vector{Dict{String, String}}, delta_x::Float64, delta_ca::Float64)
    return argmin(row -> begin
        row_x = parse(Float64, row["delta_x"])
        row_ca = parse(Float64, row["delta_ca"])
        (row_x - delta_x)^2 + (row_ca - delta_ca)^2
    end, rows)
end

function previous_successful_seed(
    rows::Vector{Dict{String, String}},
    current_row::Dict{String, String},
)::Union{Nothing, T0ContinuationSeed}
    current_x = parse(Float64, current_row["delta_x"])
    current_ca = parse(Float64, current_row["delta_ca"])
    candidates = filter(rows) do row
        row["status"] == "ok" &&
            isapprox(parse(Float64, row["delta_ca"]), current_ca; atol=1e-9, rtol=0.0) &&
            parse(Float64, row["delta_x"]) > current_x
    end

    if isempty(candidates)
        return nothing
    end

    previous_row = argmin(row -> parse(Float64, row["delta_x"]) - current_x, candidates)
    return T0ContinuationSeed(parse(Float64, previous_row["T0_V"]), parse(Float64, previous_row["T0_Ca"]))
end

function solve_sscs_trajectory(
    p,
    u0::SVector{6, Float64},
    V_sd::Float64;
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
    sol = solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=true)
    return sol, state_machine.scs
end

extract_ca(sol) = [u[5] for u in sol.u]
extract_x(sol) = [u[1] for u in sol.u]

function write_summary(
    path::String,
    saved_row::Dict{String, String},
    T0::SVector{6, Float64},
    T_scs::Vector{Int},
    gamma_scs::Vector{Int},
)
    open(path, "w") do io
        println(io, "requested_delta_x\t$(INSPECT_DELTA_X)")
        println(io, "requested_delta_ca\t$(INSPECT_DELTA_CA)")
        println(io, "grid_delta_x\t$(saved_row["delta_x"])")
        println(io, "grid_delta_ca\t$(saved_row["delta_ca"])")
        println(io, "saved_T0_method\t$(saved_row["T0_method"])")
        println(io, "saved_T0_V\t$(saved_row["T0_V"])")
        println(io, "saved_T0_Ca\t$(saved_row["T0_Ca"])")
        println(io, @sprintf("recomputed_T0_x\t%.12f", T0[1]))
        println(io, @sprintf("recomputed_T0_Ca\t%.12f", T0[5]))
        println(io, @sprintf("recomputed_T0_V\t%.12f", T0[6]))
        println(io, "saved_T_scs\t$(saved_row["T_scs"])")
        println(io, "saved_gamma_scs\t$(saved_row["gamma_scs"])")
        println(io, "recomputed_T_scs\t$(join(T_scs, ','))")
        println(io, "recomputed_gamma_scs\t$(join(gamma_scs, ','))")
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

    T0 = if startswith(saved_row["T0_method"], "continued")
        seed = previous_successful_seed(rows, saved_row)
        if isnothing(seed)
            error("Saved point used continuation, but no previous successful seed was found.")
        end
        first(initialize_T_Ca0_from_seed(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0, seed))
    else
        initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    end

    T_sol, T_scs = solve_sscs_trajectory(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    gamma_sol, gamma_scs = solve_sscs_trajectory(
        p,
        saddle_data.gamma_sd_minus0,
        saddle_data.V_eq_SD;
        abstol=1e-8,
        reltol=1e-8,
    )

    plot_path = joinpath(ATTEMPT11_ROOT, "$(INSPECT_OUTPUT_TAG)_cax_phase.png")
    summary_path = joinpath(ATTEMPT11_ROOT, "$(INSPECT_OUTPUT_TAG)_summary.txt")

    fig = Figure(size=(1100, 850))
    ax = Axis(
        fig[1, 1],
        xlabel="Ca",
        ylabel="x",
        title=@sprintf("Δx = %.6f, ΔCa = %.6f", delta_x, delta_ca),
    )

    T_ca = extract_ca(T_sol)
    T_x = extract_x(T_sol)
    gamma_ca = extract_ca(gamma_sol)
    gamma_x = extract_x(gamma_sol)

    lines!(ax, T_ca, T_x; color=:firebrick, linewidth=2.5, label="T0 trajectory")
    lines!(ax, gamma_ca, gamma_x; color=:royalblue, linewidth=2.5, label="Γ_SD^- trajectory")
    scatter!(ax, [T_ca[1]], [T_x[1]]; color=:firebrick, markersize=12)
    scatter!(ax, [gamma_ca[1]], [gamma_x[1]]; color=:royalblue, markersize=12)
    axislegend(ax, position=:rb)
    save(plot_path, fig)

    write_summary(summary_path, saved_row, T0, T_scs, gamma_scs)

    println("grid_delta_x=$(delta_x)")
    println("grid_delta_ca=$(delta_ca)")
    println("T0_method=$(saved_row["T0_method"])")
    println("saved_T_scs=$(saved_row["T_scs"])")
    println("saved_gamma_scs=$(saved_row["gamma_scs"])")
    println("recomputed_T_scs=$(join(T_scs, ','))")
    println("recomputed_gamma_scs=$(join(gamma_scs, ','))")
    println("plot_path=$(plot_path)")
    println("summary_path=$(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
