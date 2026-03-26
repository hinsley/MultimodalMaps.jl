using Pkg

const ATTEMPT20_TRACE_ROOT = @__DIR__
const REPO_ROOT_020 = normpath(joinpath(ATTEMPT20_TRACE_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_020)

ENV["ATTEMPT020_OUTPUT_TAG"] = get(ENV, "ATTEMPT020_OUTPUT_TAG", "grid1200_seq10_tonly_sin_remap40")

include(joinpath(ATTEMPT20_TRACE_ROOT, "contours.jl"))

using Printf

const TRACE_DELTA_X = parse(Float64, get(ENV, "ATTEMPT020_TRACE_DELTA_X", "-0.500417"))
const TRACE_DELTA_CA = parse(Float64, get(ENV, "ATTEMPT020_TRACE_DELTA_CA", "-46.997498"))
const TRACE_ABSTOL = parse(Float64, get(ENV, "ATTEMPT020_TRACE_ABSTOL", "3e-6"))
const TRACE_RELTOL = parse(Float64, get(ENV, "ATTEMPT020_TRACE_RELTOL", "3e-6"))

struct ParsedTraceRow20
    delta_x::Float64
    delta_ca::Float64
    T0_V::Float64
    T0_Ca::Float64
    T0_method::String
    T_encoding::Union{BigInt, Nothing}
    T_scs::Vector{Int}
    error_message::Union{Nothing, String}
end

parse_bigint_or_nothing(field::AbstractString) = isempty(field) ? nothing : parse(BigInt, field)
parse_sequence_field(field::AbstractString) = isempty(field) ? Int[] : parse.(Int, split(field, ","))

function parse_trace_row(line::AbstractString)
    fields = split(line, '\t')
    status = fields[end]
    error_message = status == "ok" ? nothing : (startswith(status, "error: ") ? status[8:end] : status)
    return ParsedTraceRow20(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        isempty(fields[3]) ? NaN : parse(Float64, fields[3]),
        isempty(fields[4]) ? NaN : parse(Float64, fields[4]),
        fields[5],
        parse_bigint_or_nothing(fields[6]),
        parse_sequence_field(fields[7]),
        error_message,
    )
end

function load_column_rows(path::String)
    rows = ParsedTraceRow20[]
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            push!(rows, parse_trace_row(line))
        end
    end
    return rows
end

function find_target_indices()
    x_idx = findfirst(x -> isapprox(x, TRACE_DELTA_X; atol=1e-6, rtol=0.0), DELTA_XS_010)
    ca_idx = findfirst(x -> isapprox(x, TRACE_DELTA_CA; atol=1e-6, rtol=0.0), DELTA_CAS_010)
    isnothing(x_idx) && error("Trace Δx=$(TRACE_DELTA_X) is not on the attempt-020 grid.")
    isnothing(ca_idx) && error("Trace ΔCa=$(TRACE_DELTA_CA) is not on the attempt-020 grid.")
    return x_idx, ca_idx
end

function find_candidate_seed(rows::Vector{ParsedTraceRow20}, target_idx::Int)
    for idx in (target_idx + 1):length(rows)
        row = rows[idx]
        if isnothing(row.error_message)
            return T0ContinuationSeed(row.T0_V, row.T0_Ca)
        end
    end
    return nothing
end

function compute_trace_solution(
    p,
    T0::SVector{6, Float64},
    V_eq_SD::Float64;
    abstol::Float64=TRACE_ABSTOL,
    reltol::Float64=TRACE_RELTOL,
)
    state_machine = make_state_machine(V_eq_SD)
    callback = VectorContinuousCallback(
        sscs_condition!,
        make_affect_sscs!(state_machine),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(Plant.melibeNew, T0, SSCS_ODE_TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=true)
    return sol, state_machine.scs
end

function reproduce_T0(
    p,
    saddle_data,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)
    if !isnothing(candidate_seed)
        try
            T0, iterations = initialize_T_Ca0_from_seed(
                p,
                saddle_data.x_eq_SF,
                saddle_data.gamma_sd_minus0,
                candidate_seed,
            )
            return T0, "continued:$(iterations)"
        catch
            # Fall through to the full initializer.
        end
    end

    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    return T0, "full"
end

function trace_plot_path()
    dx_label = replace(@sprintf("%.6f", TRACE_DELTA_X), "-" => "m", "." => "p")
    dca_label = replace(@sprintf("%.6f", TRACE_DELTA_CA), "-" => "m", "." => "p")
    return joinpath(ATTEMPT20_ROOT, "trace_dx$(dx_label)_dca$(dca_label)_timeseries.png")
end

function render_trace_plot(sol, traced_sscs::Vector{Int}, reproduced::TContourScanResult20, target::ParsedTraceRow20)
    fig = Figure(size=(1800, 1600))
    axes = [
        Axis(fig[1, 1], ylabel="x"),
        Axis(fig[2, 1], ylabel="y"),
        Axis(fig[3, 1], ylabel="n"),
        Axis(fig[4, 1], ylabel="h"),
        Axis(fig[5, 1], ylabel="Ca"),
        Axis(fig[6, 1], xlabel="t", ylabel="V"),
    ]

    times = sol.t
    labels = ("x", "y", "n", "h", "Ca", "V")
    colors = (:firebrick, :darkgreen, :dodgerblue, :darkorange, :purple, :black)
    for (idx, ax) in enumerate(axes)
        series = [u[idx] for u in sol.u]
        lines!(ax, times, series; color=colors[idx], linewidth=1.2)
    end

    title = @sprintf(
        "attempt-020 trace at Δx=%.6f, ΔCa=%.6f\nmethod=%s, T0=(V=%s, Ca=%s)\nstored SSCS=[%s]\nreproduced SSCS=[%s]",
        target.delta_x,
        target.delta_ca,
        reproduced.T0_method,
        @sprintf("%.6f", reproduced.T0_V),
        @sprintf("%.6f", reproduced.T0_Ca),
        join(target.T_scs, ", "),
        join(traced_sscs, ", "),
    )
    Label(fig[0, 1], title; fontsize=18, tellwidth=false)

    save(trace_plot_path(), fig; px_per_unit=2.0)
    return trace_plot_path()
end

function main()
    target_x_idx, target_ca_idx = find_target_indices()
    path = joinpath(ATTEMPT20_ROOT, "$(CONTOUR_OUTPUT_TAG)_columns", @sprintf("column_%04d.tsv", target_ca_idx))
    rows = load_column_rows(path)
    target = rows[target_x_idx]
    isnothing(target.error_message) || error("Target point is not successful in repaired columns: $(target.error_message)")

    candidate_seed = find_candidate_seed(rows, target_x_idx)
    p = build_params(target.delta_x, target.delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    reproduced = run_T_point(target.delta_x, target.delta_ca, candidate_seed)
    T0, reproduced_method = reproduce_T0(p, saddle_data, candidate_seed)

    reproduced.T_scs == target.T_scs || error("Reproduced SSCS does not match stored SSCS.")
    reproduced_method == target.T0_method || error("Reproduced method does not match stored method.")
    abs(reproduced.T0_V - target.T0_V) <= 1e-5 || error("Reproduced T0_V differs from stored value.")
    abs(reproduced.T0_Ca - target.T0_Ca) <= 1e-5 || error("Reproduced T0_Ca differs from stored value.")

    sol, traced_sscs = compute_trace_solution(p, T0, saddle_data.V_eq_SD)
    traced_sscs == target.T_scs || error("Trace SSCS does not match stored SSCS.")

    output = render_trace_plot(sol, traced_sscs, reproduced, target)
    println(output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
