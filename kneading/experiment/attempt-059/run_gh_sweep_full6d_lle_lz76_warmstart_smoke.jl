using Pkg

ENV["ATTEMPT059_GH_SWEEP_TAG"] = get(ENV, "ATTEMPT059_GH_SWEEP_TAG", "gh_sweep_full6d_warmstart_smoke10_t2500s")

include(joinpath(@__DIR__, "run_gh_sweep_full6d_lle_lz76_smoke.jl"))

function main_warmstart_059()
    mkpath(GH_SWEEP_DIR_059)
    values = gh_values_059()
    current_state = initial_state_059(build_params_059(first(values)))
    results = Vector{Any}(undef, length(values))
    started = time()
    @info "Starting warm-start full 6D g_h sweep" n=length(values) run_time=RUN_T_059 run_time_seconds=RUN_T_059 * SECONDS_PER_TIME_UNIT_059 transient_time=TRANSIENT_T_059 initial_state=current_state

    for idx in eachindex(values)
        g_h = Float64(values[idx])
        @info "Starting warm-start full 6D g_h point" idx g_h current_state
        result = run_gh_point_059(idx, g_h, current_state)
        results[idx] = result
        @info "Finished warm-start full 6D g_h point" idx g_h status=result.status main_runtime=result.main_runtime n_symbols=result.n_symbols lambda1_bits=result.lambda1_bits_per_second lz76_bits=result.lz76_bits_per_second
        if result.status == "ok" && result.final_state !== nothing
            current_state = result.final_state
        else
            @warn "Keeping previous warm-start state after failed point" idx g_h error=result.error_message
        end
    end

    total_runtime = time() - started
    write_summary_059(results)
    plot_gh_sweep_059(results)
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
        println(io, "warm_start_rule\teach point starts from the final state of the previous g_h point")
        println(io, "transient_time_model_units\t$(TRANSIENT_T_059)")
        println(io, "run_time_model_units\t$(RUN_T_059)")
        println(io, "run_time_seconds\t$(RUN_T_059 * SECONDS_PER_TIME_UNIT_059)")
        println(io, "total_wall_runtime_seconds\t$(total_runtime)")
        println(io, "summary_path\t$(GH_SWEEP_SUMMARY_PATH_059)")
        println(io, "plot_path\t$(GH_SWEEP_PLOT_PATH_059)")
    end
    @info "Wrote warm-start full 6D g_h sweep outputs" plot=GH_SWEEP_PLOT_PATH_059 summary=GH_SWEEP_SUMMARY_PATH_059 timing=GH_SWEEP_TIMING_PATH_059 total_runtime
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_warmstart_059()
end
