using Pkg

const ATTEMPT57_ROOT = @__DIR__
const ATTEMPT50_ROOT_FOR_057 = normpath(joinpath(ATTEMPT57_ROOT, "..", "attempt-050"))
const REPO_ROOT_057 = normpath(joinpath(ATTEMPT57_ROOT, "..", "..", ".."))

function alias_env_057!(dst::String, src::String, default::String)
    if !haskey(ENV, dst)
        ENV[dst] = get(ENV, src, default)
    end
end

alias_env_057!("ATTEMPT050_NX", "ATTEMPT057_NX", "500")
alias_env_057!("ATTEMPT050_NY", "ATTEMPT057_NY", "500")
alias_env_057!("ATTEMPT050_DELTA_X_MIN", "ATTEMPT057_DELTA_X_MIN", "-1.5")
alias_env_057!("ATTEMPT050_DELTA_X_MAX", "ATTEMPT057_DELTA_X_MAX", "-0.5")
alias_env_057!("ATTEMPT050_DELTA_CA_MIN", "ATTEMPT057_DELTA_CA_MIN", "-45.0")
alias_env_057!("ATTEMPT050_DELTA_CA_MAX", "ATTEMPT057_DELTA_CA_MAX", "-20.0")
alias_env_057!("ATTEMPT050_DELTA_X_TICK_STEP", "ATTEMPT057_DELTA_X_TICK_STEP", "0.1")
alias_env_057!("ATTEMPT050_DELTA_CA_TICK_STEP", "ATTEMPT057_DELTA_CA_TICK_STEP", "5.0")
alias_env_057!("ATTEMPT050_MAX_SEQ_LENGTH", "ATTEMPT057_MAX_SEQ_LENGTH", "20")
alias_env_057!("ATTEMPT050_SSCS_TMAX", "ATTEMPT057_SSCS_TMAX", "3.0e5")

Pkg.activate(REPO_ROOT_057)
include(joinpath(ATTEMPT50_ROOT_FOR_057, "main.jl"))

using Base.Threads
using Printf
using StaticArrays

const GH_057 = parse(Float64, get(ENV, "ATTEMPT057_GH", "0.0"))
const OUTPUT_TAG_057 = get(ENV, "ATTEMPT057_OUTPUT_TAG", "grid500_arbitrary_ic_gh0p000_seq20_tmax3e5")
const SWEEP_DIR_057 = joinpath(ATTEMPT57_ROOT, "$(OUTPUT_TAG_057)_columns")
const RESULTS_PATH_057 = joinpath(ATTEMPT57_ROOT, "$(OUTPUT_TAG_057)_results.tsv")
const SUMMARY_PATH_057 = joinpath(ATTEMPT57_ROOT, "$(OUTPUT_TAG_057)_summary.txt")
const ARBITRARY_U0_V_057 = parse(Float64, get(ENV, "ATTEMPT057_U0_V", "-30.0"))
const ARBITRARY_U0_X_OFFSET_057 = parse(Float64, get(ENV, "ATTEMPT057_U0_X_OFFSET", "-1.0e-4"))
const LOG_LOCK_057 = ReentrantLock()

mutable struct ArbitrarySSCSStateMachine057
    scs::Vector{Int}
    times::Vector{Float64}
    count::Int
    last_symbol::EventSymbol
    last2_symbol::EventSymbol
    V_sd::Float64
end

struct ArbitraryScanResult057
    delta_x::Float64
    delta_ca::Float64
    u0::SVector{6, Float64}
    scs::Vector{Int}
    event_times::Vector{Float64}
    retcode::String
    error_message::Union{Nothing, String}
end

function build_params_057(delta_x::Float64, delta_ca::Float64)
    p = MVector{18, Float64}(build_params(delta_x, delta_ca))
    p[4] = GH_057
    return SVector{18, Float64}(p)
end

function arbitrary_initial_condition_057(p)::SVector{6, Float64}
    V0 = ARBITRARY_U0_V_057
    return SVector{6, Float64}((
        Plant.xinf(p, V0) + ARBITRARY_U0_X_OFFSET_057,
        Plant.yinf(V0),
        Plant.ninf(V0),
        Plant.hinf(V0),
        EquilibriaSubset.Ca_null_Ca(p, V0),
        V0,
    ))
end

function make_arbitrary_state_machine_057(V_sd::Float64)
    return ArbitrarySSCSStateMachine057(Int[], Float64[], 0, Void, Void, V_sd)
end

function make_affect_arbitrary_sscs_057!(state_machine::ArbitrarySSCSStateMachine057)
    function affect_arbitrary_sscs!(integrator, idx)
        if idx == 1
            current_V = integrator.u[6]
            current_algorithmic_event = current_V > state_machine.V_sd ? Vplus : Vminus

            if current_algorithmic_event == Vminus
                symbol = if current_V > V_RETURN_THRESHOLD || state_machine.last2_symbol != Vplus
                    state_machine.count
                else
                    -state_machine.count
                end
                push!(state_machine.scs, symbol)
                push!(state_machine.times, Float64(integrator.t))
                state_machine.count = 0
            else
                state_machine.count += 1
            end

            state_machine.last2_symbol = state_machine.last_symbol
            state_machine.last_symbol = current_algorithmic_event

            if length(state_machine.scs) >= ATTEMPT10_MAX_SEQ_LENGTH
                terminate!(integrator)
            end
            if current_algorithmic_event == Vplus && state_machine.count > MAX_SPIKE_COUNT
                terminate!(integrator)
            end
        elseif idx == 2
            state_machine.last2_symbol = state_machine.last_symbol
            state_machine.last_symbol = I
        end
        return nothing
    end

    return affect_arbitrary_sscs!
end

function compute_arbitrary_sscs_057(p, u0::SVector{6, Float64}, V_sd::Float64)
    state_machine = make_arbitrary_state_machine_057(V_sd)
    callback = VectorContinuousCallback(
        sscs_condition!,
        make_affect_arbitrary_sscs_057!(state_machine),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(Plant.melibeNew, u0, ATTEMPT50_SSCS_ODE_TSPAN, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=3e-6, reltol=3e-6, save_everystep=false)
    return state_machine.scs, state_machine.times, string(sol.retcode)
end

function event_intervals_057(times::Vector{Float64})
    isempty(times) && return Float64[]
    intervals = similar(times)
    previous = 0.0
    for idx in eachindex(times)
        intervals[idx] = times[idx] - previous
        previous = times[idx]
    end
    return intervals
end

function run_arbitrary_point_057(delta_x::Float64, delta_ca::Float64)::ArbitraryScanResult057
    p = build_params_057(delta_x, delta_ca)
    V_eqs = find_equilibria(p)
    length(V_eqs) >= 3 || error("Expected at least three equilibria; got $(length(V_eqs)).")
    V_sd = V_eqs[3]
    u0 = arbitrary_initial_condition_057(p)
    scs, times, retcode = compute_arbitrary_sscs_057(p, u0, V_sd)
    return ArbitraryScanResult057(delta_x, delta_ca, u0, scs, times, retcode, nothing)
end

function run_arbitrary_point_safe_057(delta_x::Float64, delta_ca::Float64)::ArbitraryScanResult057
    try
        return run_arbitrary_point_057(delta_x, delta_ca)
    catch err
        return ArbitraryScanResult057(
            delta_x,
            delta_ca,
            SVector{6, Float64}(NaN, NaN, NaN, NaN, NaN, NaN),
            Int[],
            Float64[],
            "",
            sprint(showerror, err),
        )
    end
end

column_path_057(col_idx::Int) = joinpath(SWEEP_DIR_057, @sprintf("column_%04d.tsv", col_idx))

function row_is_complete_057(path::String, expected_points::Int)
    isfile(path) || return false
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

format_float_vector_057(values::Vector{Float64}) =
    join((@sprintf("%.10g", value) for value in values), ",")

parse_sequence_field_057(field::AbstractString) =
    isempty(field) ? Int[] : parse.(Int, split(field, ","))

function write_column_057(path::String, results::Vector{ArbitraryScanResult057})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tg_h\tu0_x\tu0_y\tu0_n\tu0_h\tu0_Ca\tu0_V\tarbitrary_scs\tevent_times\tevent_intervals\tretcode\tstatus")
        for result in results
            status = isnothing(result.error_message) ? "ok" : "error: " * result.error_message
            println(
                io,
                join([
                    @sprintf("%.8f", result.delta_x),
                    @sprintf("%.8f", result.delta_ca),
                    @sprintf("%.8g", GH_057),
                    isfinite(result.u0[1]) ? @sprintf("%.10g", result.u0[1]) : "",
                    isfinite(result.u0[2]) ? @sprintf("%.10g", result.u0[2]) : "",
                    isfinite(result.u0[3]) ? @sprintf("%.10g", result.u0[3]) : "",
                    isfinite(result.u0[4]) ? @sprintf("%.10g", result.u0[4]) : "",
                    isfinite(result.u0[5]) ? @sprintf("%.10g", result.u0[5]) : "",
                    isfinite(result.u0[6]) ? @sprintf("%.10g", result.u0[6]) : "",
                    join(result.scs, ","),
                    format_float_vector_057(result.event_times),
                    format_float_vector_057(event_intervals_057(result.event_times)),
                    result.retcode,
                    status,
                ], '\t'),
            )
        end
    end
end

function run_column_057(col_idx::Int, delta_ca::Float64, total_cols::Int, total_rows::Int)
    path = column_path_057(col_idx)
    if row_is_complete_057(path, total_rows)
        lock(LOG_LOCK_057)
        try
            @printf("Skipping completed column %d/%d (Delta Ca=%.6f)\n", col_idx, total_cols, delta_ca)
            flush(stdout)
        finally
            unlock(LOG_LOCK_057)
        end
        return
    end

    started = time()
    column_results = Vector{ArbitraryScanResult057}(undef, total_rows)
    for row_idx in eachindex(DELTA_XS_010)
        delta_x = DELTA_XS_010[row_idx]
        column_results[row_idx] = run_arbitrary_point_safe_057(delta_x, delta_ca)
    end
    write_column_057(path, column_results)
    ok_count = count(result -> isnothing(result.error_message), column_results)
    full_count = count(result -> isnothing(result.error_message) && length(result.scs) >= ATTEMPT10_MAX_SEQ_LENGTH, column_results)

    lock(LOG_LOCK_057)
    try
        @printf(
            "Saved column %d/%d (Delta Ca=%.6f) with %d/%d successful points, %d/%d full-length sequences in %.2f s\n",
            col_idx,
            total_cols,
            delta_ca,
            ok_count,
            total_rows,
            full_count,
            total_rows,
            time() - started,
        )
        flush(stdout)
    finally
        unlock(LOG_LOCK_057)
    end
end

function run_or_resume_columns_057()
    mkpath(SWEEP_DIR_057)
    total_cols = length(DELTA_CAS_010)
    total_rows = length(DELTA_XS_010)
    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_010)
        run_column_057(col_idx, DELTA_CAS_010[col_idx], total_cols, total_rows)
    end
end

function merge_columns_057()
    open(RESULTS_PATH_057, "w") do out
        wrote_header = false
        for col_idx in eachindex(DELTA_CAS_010)
            path = column_path_057(col_idx)
            row_is_complete_057(path, length(DELTA_XS_010)) || error("Missing or incomplete column file: $(path)")
            open(path, "r") do io
                header = readline(io)
                if !wrote_header
                    println(out, header)
                    wrote_header = true
                end
                for line in eachline(io)
                    println(out, line)
                end
            end
        end
    end
end

function write_summary_057(elapsed_seconds::Float64)
    total = 0
    ok = 0
    full = 0
    errors = 0
    open(RESULTS_PATH_057, "r") do io
        header = split(readline(io), '\t')
        lookup = Dict(name => idx for (idx, name) in pairs(header))
        for line in eachline(io)
            total += 1
            fields = split(line, '\t'; keepempty=true)
            if fields[lookup["status"]] == "ok"
                ok += 1
                scs = parse_sequence_field_057(fields[lookup["arbitrary_scs"]])
                full += length(scs) >= ATTEMPT10_MAX_SEQ_LENGTH ? 1 : 0
            else
                errors += 1
            end
        end
    end

    open(SUMMARY_PATH_057, "w") do io
        println(io, "output_tag\t$(OUTPUT_TAG_057)")
        println(io, "results_path\t$(RESULTS_PATH_057)")
        println(io, "columns_dir\t$(SWEEP_DIR_057)")
        println(io, "grid_delta_ca\t$(length(DELTA_CAS_010))")
        println(io, "grid_delta_x\t$(length(DELTA_XS_010))")
        println(io, "delta_ca_min\t$(first(DELTA_CAS_010))")
        println(io, "delta_ca_max\t$(last(DELTA_CAS_010))")
        println(io, "delta_x_min\t$(first(DELTA_XS_010))")
        println(io, "delta_x_max\t$(last(DELTA_XS_010))")
        println(io, "max_seq_length\t$(ATTEMPT10_MAX_SEQ_LENGTH)")
        println(io, "sscs_tmax\t$(ATTEMPT50_SSCS_TMAX)")
        println(io, "g_h\t$(GH_057)")
        println(io, "u0_rule\tV=$(ARBITRARY_U0_V_057), x=xinf(p,V)+$(ARBITRARY_U0_X_OFFSET_057), y=yinf(V), n=ninf(V), h=hinf(V), Ca=Ca_null_Ca(p,V)")
        println(io, "total_points\t$(total)")
        println(io, "ok_count\t$(ok)")
        println(io, "error_count\t$(errors)")
        println(io, "full_length_count\t$(full)")
        println(io, "elapsed_seconds\t$(elapsed_seconds)")
    end
end

function main()
    println("Running attempt-057 arbitrary-initial-condition SSCS scan.")
    println("Grid: $(length(DELTA_CAS_010)) Delta Ca x $(length(DELTA_XS_010)) Delta x")
    println("Parameter window: Delta Ca in [$(first(DELTA_CAS_010)), $(last(DELTA_CAS_010))], Delta x in [$(first(DELTA_XS_010)), $(last(DELTA_XS_010))]")
    println("Max symbols: $(ATTEMPT10_MAX_SEQ_LENGTH), tmax=$(ATTEMPT50_SSCS_TMAX), g_h=$(GH_057)")
    println("Output tag: $(OUTPUT_TAG_057)")
    flush(stdout)

    started = time()
    run_or_resume_columns_057()
    merge_columns_057()
    elapsed = time() - started
    write_summary_057(elapsed)
    println("Saved merged results to $(RESULTS_PATH_057)")
    println("Saved summary to $(SUMMARY_PATH_057)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
