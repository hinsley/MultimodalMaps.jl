using Pkg

ENV["ATTEMPT059_GH_SWEEP_TAG"] = get(ENV, "ATTEMPT059_GH_SWEEP_TAG", "gh_sweep_full6d_warmstart_nointertransient_smoke10_t2500s")

include(joinpath(@__DIR__, "run_gh_sweep_full6d_lle_lz76_smoke.jl"))

function run_gh_point_no_intertransient_059(idx::Int, g_h::Float64, post_transient_state::SVector{6, Float64}, original_initial_state::SVector{6, Float64})
    tag = gh_tag_059(idx, g_h)
    events_path = joinpath(GH_SWEEP_DIR_059, "$(tag)_events.tsv")
    metadata_path = joinpath(GH_SWEEP_DIR_059, "$(tag)_metadata.txt")
    started = time()
    try
        p = build_params_059(g_h)
        V_eqs = find_equilibria_059(p)
        length(V_eqs) >= 3 || error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
        V_sd = Float64(V_eqs[3])
        local result
        main_runtime = @elapsed result = run_main_059(p, post_transient_state, V_sd)
        write_events_059(events_path, result.recorder)
        write_metadata_059(metadata_path, idx, g_h, V_eqs, V_sd, original_initial_state, post_transient_state, result, 0.0, main_runtime)
        return (;
            idx,
            g_h,
            status="ok",
            retcode=result.retcode,
            final_state=result.final_state,
            lambda1_nats_per_second=result.λ_nats_per_second,
            lz76_nats_per_second=result.lz_nats,
            lambda1_bits_per_second=result.λ_bits_per_second,
            lz76_bits_per_second=result.lz_bits,
            lz76_alphabet_size=result.alphabet_size,
            n_symbols=length(result.recorder.scs),
            lz76_phrase_count=result.phrase_count,
            mean_event_interval_seconds=result.mean_interval,
            transient_runtime=0.0,
            main_runtime,
            total_runtime=time() - started,
            events_path,
            metadata_path,
            error_message="",
        )
    catch err
        return (;
            idx,
            g_h,
            status="error",
            retcode="error",
            final_state=nothing,
            lambda1_nats_per_second=NaN,
            lz76_nats_per_second=NaN,
            lambda1_bits_per_second=NaN,
            lz76_bits_per_second=NaN,
            lz76_alphabet_size=0,
            n_symbols=0,
            lz76_phrase_count=0,
            mean_event_interval_seconds=NaN,
            transient_runtime=0.0,
            main_runtime=NaN,
            total_runtime=time() - started,
            events_path,
            metadata_path,
            error_message=sprint(showerror, err),
        )
    end
end

function write_partial_outputs_059(results)
    completed = [result for result in results if result !== nothing]
    isempty(completed) && return
    write_summary_059(completed)
    plot_gh_sweep_059(completed)
end

function main_warmstart_no_intertransient_059()
    mkpath(GH_SWEEP_DIR_059)
    values = gh_values_059()
    original_initial_state = initial_state_059(build_params_059(first(values)))
    current_state = original_initial_state
    results = Vector{Any}(nothing, length(values))
    started = time()
    @info "Starting warm-start/no-intertransient full 6D g_h sweep" n=length(values) run_time=RUN_T_059 run_time_seconds=RUN_T_059 * SECONDS_PER_TIME_UNIT_059 first_transient_time=TRANSIENT_T_059 initial_state=current_state

    for idx in eachindex(values)
        g_h = Float64(values[idx])
        @info "Starting warm-start/no-intertransient full 6D g_h point" idx g_h current_state
        result = idx == 1 ?
            run_gh_point_059(idx, g_h, current_state) :
            run_gh_point_no_intertransient_059(idx, g_h, current_state, original_initial_state)
        results[idx] = result
        write_partial_outputs_059(results)
        @info "Finished warm-start/no-intertransient full 6D g_h point" idx g_h status=result.status main_runtime=result.main_runtime n_symbols=result.n_symbols lambda1_bits=result.lambda1_bits_per_second lz76_bits=result.lz76_bits_per_second
        if result.status == "ok" && result.final_state !== nothing
            current_state = result.final_state
        else
            @warn "Keeping previous warm-start state after failed point" idx g_h error=result.error_message
        end
    end

    total_runtime = time() - started
    write_partial_outputs_059(results)
    open(GH_SWEEP_TIMING_PATH_059, "w") do io
        println(io, "created_at\t$(Dates.now())")
        println(io, "n_threads\t$(Threads.nthreads())")
        println(io, "parallelism\tordered warm-start continuation")
        println(io, "n_g_h_values\t$(length(values))")
        println(io, "g_h_min\t$(minimum(values))")
        println(io, "g_h_max\t$(maximum(values))")
        println(io, "delta_x\t$(DELTA_X_059)")
        println(io, "delta_ca\t$(DELTA_CA_059)")
        println(io, "model\tfull 6D SiN; y integrated")
        println(io, "warm_start_rule\tfirst point uses transient; each later point starts directly from the previous g_h final state")
        println(io, "first_transient_time_model_units\t$(TRANSIENT_T_059)")
        println(io, "run_time_model_units\t$(RUN_T_059)")
        println(io, "run_time_seconds\t$(RUN_T_059 * SECONDS_PER_TIME_UNIT_059)")
        println(io, "total_wall_runtime_seconds\t$(total_runtime)")
        println(io, "summary_path\t$(GH_SWEEP_SUMMARY_PATH_059)")
        println(io, "plot_path\t$(GH_SWEEP_PLOT_PATH_059)")
    end
    @info "Wrote warm-start/no-intertransient full 6D g_h sweep outputs" plot=GH_SWEEP_PLOT_PATH_059 summary=GH_SWEEP_SUMMARY_PATH_059 timing=GH_SWEEP_TIMING_PATH_059 total_runtime
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_warmstart_no_intertransient_059()
end
