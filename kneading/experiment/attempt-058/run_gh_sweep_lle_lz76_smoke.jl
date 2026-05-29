using Pkg

ENV["ATTEMPT058_RUN_T"] = "2.5e6"
ENV["ATTEMPT058_LYAP_K"] = "1"
ENV["ATTEMPT058_QR_DT"] = get(ENV, "ATTEMPT058_QR_DT", "10.0")
ENV["ATTEMPT058_RECORD_DT"] = "2.5e6"
ENV["ATTEMPT058_OUTPUT_TAG"] = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh_sweep_smoke10_t2500s")

include(joinpath(@__DIR__, "main.jl"))

using CairoMakie
using Dates
using Printf
using StaticArrays

const GH_SWEEP_N_058 = env_int_058("ATTEMPT058_GH_SWEEP_N", 10)
const GH_SWEEP_MIN_058 = env_float_058("ATTEMPT058_GH_SWEEP_MIN", 0.0)
const GH_SWEEP_MAX_058 = env_float_058("ATTEMPT058_GH_SWEEP_MAX", 0.01)
const GH_SWEEP_TAG_058 = get(ENV, "ATTEMPT058_GH_SWEEP_TAG", @sprintf("gh_sweep_smoke%02d_t2500s", GH_SWEEP_N_058))
const GH_SWEEP_DIR_058 = joinpath(ATTEMPT58_ROOT, GH_SWEEP_TAG_058)
const GH_SWEEP_SUMMARY_PATH_058 = joinpath(GH_SWEEP_DIR_058, "$(GH_SWEEP_TAG_058)_summary.tsv")
const GH_SWEEP_TIMING_PATH_058 = joinpath(GH_SWEEP_DIR_058, "$(GH_SWEEP_TAG_058)_timing.txt")
const GH_SWEEP_PLOT_PATH_058 = joinpath(GH_SWEEP_DIR_058, "$(GH_SWEEP_TAG_058)_lle_lz76_bits_per_second.png")

function gh_values_058()
    return collect(range(GH_SWEEP_MIN_058, GH_SWEEP_MAX_058; length=GH_SWEEP_N_058))
end

function build_params_for_gh_058(g_h::Float64)::SVector{18, Float64}
    p = Vector{Float64}(Plant058.default_params)
    p[4] = g_h
    p[17] = DELTA_X_058
    p[18] = DELTA_CA_058
    return SVector{18, Float64}(p)
end

function gh_tag_058(idx::Int, g_h::Float64)
    return @sprintf("gh%02d_%0.6f", idx, g_h)
end

function write_events_to_path_058(path::String, recorder::SSCSRecorder058)
    open(path, "w") do io
        println(io, "idx\ttime\ttime_seconds\tsymbol\tinterval\tinterval_seconds")
        for i in eachindex(recorder.scs)
            interval = i == 1 ? recorder.times[i] : recorder.times[i] - recorder.times[i - 1]
            println(io, join((
                i,
                @sprintf("%.12g", recorder.times[i]),
                @sprintf("%.12g", recorder.times[i] * SECONDS_PER_TIME_UNIT_058),
                recorder.scs[i],
                @sprintf("%.12g", interval),
                @sprintf("%.12g", interval * SECONDS_PER_TIME_UNIT_058),
            ), '\t'))
        end
    end
end

function write_metadata_to_path_058(path::String, idx::Int, g_h::Float64, V_eqs, V_sd, u0, post_transient_state, final_state, transient_runtime, main_runtime, retcode, rows, recorder)
    open(path, "w") do io
        println(io, "created_at\t$(Dates.now())")
        println(io, "sweep_index\t$(idx)")
        println(io, "delta_x\t$(DELTA_X_058)")
        println(io, "delta_ca\t$(DELTA_CA_058)")
        println(io, "g_h\t$(g_h)")
        println(io, "y_rule\ty identically zero; 5D state [x,n,h,Ca,V]")
        println(io, "transient_time\t$(TRANSIENT_T_058)")
        println(io, "run_time\t$(RUN_T_058)")
        println(io, "run_time_seconds\t$(RUN_T_058 * SECONDS_PER_TIME_UNIT_058)")
        println(io, "lyap_k\t$(LYAP_K_058)")
        println(io, "time_unit_seconds\t$(SECONDS_PER_TIME_UNIT_058)")
        println(io, "qr_dt\t$(QR_DT_058)")
        println(io, "record_dt\t$(RECORD_DT_058)")
        println(io, "abstol\t$(ABSTOL_058)")
        println(io, "reltol\t$(RELTOL_058)")
        println(io, "V_equilibria\t$(join(string.(V_eqs), ","))")
        println(io, "V_sd\t$(V_sd)")
        println(io, "initial_state\t$(join(string.(u0), ","))")
        println(io, "post_transient_state\t$(join(string.(post_transient_state), ","))")
        println(io, "final_state\t$(join(string.(final_state), ","))")
        println(io, "retcode\t$(retcode)")
        println(io, "transient_runtime_seconds\t$(transient_runtime)")
        println(io, "main_runtime_seconds\t$(main_runtime)")
        println(io, "n_symbols\t$(length(recorder.scs))")
        if !isempty(rows)
            row = rows[end]
            println(io, "final_lambda1_nats_per_second\t$(row.lambda1_nats_per_second)")
            println(io, "final_lambda1_bits_per_second\t$(row.lambda1_bits_per_second)")
            println(io, "final_lz76_nats_per_second\t$(row.lz76_nats_per_second)")
            println(io, "final_lz76_bits_per_second\t$(row.lz76_bits_per_second)")
            println(io, "final_lz76_alphabet_size\t$(row.lz76_alphabet_size)")
            println(io, "final_lz76_phrase_count\t$(row.lz76_phrase_count)")
        end
    end
end

function run_gh_point_058(idx::Int, g_h::Float64, u0::SVector{5, Float64})
    tag = gh_tag_058(idx, g_h)
    events_path = joinpath(GH_SWEEP_DIR_058, "$(tag)_events.tsv")
    metadata_path = joinpath(GH_SWEEP_DIR_058, "$(tag)_metadata.txt")

    started = time()
    try
        p = build_params_for_gh_058(g_h)
        V_eqs = find_equilibria_058(p)
        length(V_eqs) >= 3 || error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
        V_sd = Float64(V_eqs[3])
        u0_aug = augmented_state_058(u0, p, 0.0)
        transient_runtime = @elapsed post_transient_state, _ = run_transient_058(p, u0_aug)

        local recorder
        local rows
        local final_state
        local retcode
        main_runtime = @elapsed begin
            recorder, rows, final_state, retcode = run_main_058(p, post_transient_state, V_sd)
        end
        write_events_to_path_058(events_path, recorder)
        write_metadata_to_path_058(metadata_path, idx, g_h, V_eqs, V_sd, u0, post_transient_state, final_state, transient_runtime, main_runtime, retcode, rows, recorder)
        row = rows[end]
        return (;
            idx,
            g_h,
            status="ok",
            retcode,
            lambda1_nats_per_second=row.lambda1_nats_per_second,
            lz76_nats_per_second=row.lz76_nats_per_second,
            lambda1_bits_per_second=row.lambda1_bits_per_second,
            lz76_bits_per_second=row.lz76_bits_per_second,
            lz76_alphabet_size=row.lz76_alphabet_size,
            n_symbols=length(recorder.scs),
            lz76_phrase_count=row.lz76_phrase_count,
            transient_runtime,
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
            lambda1_nats_per_second=NaN,
            lz76_nats_per_second=NaN,
            lambda1_bits_per_second=NaN,
            lz76_bits_per_second=NaN,
            lz76_alphabet_size=0,
            n_symbols=0,
            lz76_phrase_count=0,
            transient_runtime=NaN,
            main_runtime=NaN,
            total_runtime=time() - started,
            events_path,
            metadata_path,
            error_message=sprint(showerror, err),
        )
    end
end

function write_summary_058(results)
    open(GH_SWEEP_SUMMARY_PATH_058, "w") do io
        println(io, "idx\tg_h\tstatus\tretcode\tlambda1_nats_per_second\tlz76_nats_per_second\tlambda1_bits_per_second\tlz76_bits_per_second\tlz76_alphabet_size\tn_symbols\tlz76_phrase_count\ttransient_runtime_seconds\tmain_runtime_seconds\ttotal_runtime_seconds\tevents_path\tmetadata_path\terror_message")
        for result in sort(results; by=r -> r.idx)
            println(io, join((
                result.idx,
                @sprintf("%.12g", result.g_h),
                result.status,
                result.retcode,
                isfinite(result.lambda1_nats_per_second) ? @sprintf("%.12g", result.lambda1_nats_per_second) : "NaN",
                isfinite(result.lz76_nats_per_second) ? @sprintf("%.12g", result.lz76_nats_per_second) : "NaN",
                isfinite(result.lambda1_bits_per_second) ? @sprintf("%.12g", result.lambda1_bits_per_second) : "NaN",
                isfinite(result.lz76_bits_per_second) ? @sprintf("%.12g", result.lz76_bits_per_second) : "NaN",
                result.lz76_alphabet_size,
                result.n_symbols,
                result.lz76_phrase_count,
                isfinite(result.transient_runtime) ? @sprintf("%.12g", result.transient_runtime) : "NaN",
                isfinite(result.main_runtime) ? @sprintf("%.12g", result.main_runtime) : "NaN",
                @sprintf("%.12g", result.total_runtime),
                result.events_path,
                result.metadata_path,
                result.error_message,
            ), '\t'))
        end
    end
end

function plot_gh_sweep_058(results)
    ok_results = sort(filter(result -> result.status == "ok", results); by=r -> r.g_h)
    ghs = [result.g_h for result in ok_results]
    λ_bits = [result.lambda1_bits_per_second for result in ok_results]
    lz_bits = [result.lz76_bits_per_second for result in ok_results]

    fig = Figure(size=(1260, 550), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="g_h",
        ylabel="Entropy rate estimate (bits/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    )
    λ_line = lines!(ax, ghs, λ_bits; color=:black, linewidth=3.2)
    scatter!(ax, ghs, λ_bits; color=:black, markersize=11)
    lz_line = lines!(ax, ghs, lz_bits; color=:firebrick3, linewidth=3.2)
    scatter!(ax, ghs, lz_bits; color=:firebrick3, markersize=11)
    Legend(
        fig[1, 1],
        [λ_line, lz_line],
        ["Maximal Lyapunov exponent", "SSCS Abramov LZ76"],
        labelsize=20,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )
    save(GH_SWEEP_PLOT_PATH_058, fig, px_per_unit=2)
end

function main()
    mkpath(GH_SWEEP_DIR_058)
    values = gh_values_058()
    base_params = build_params_for_gh_058(first(values))
    fixed_initial_state = initial_state_058(base_params)
    results = Vector{Any}(undef, length(values))

    started = time()
    @info "Starting g_h sweep" n=length(values) threads=Threads.nthreads() run_time=RUN_T_058 run_time_seconds=RUN_T_058 * SECONDS_PER_TIME_UNIT_058 transient_time=TRANSIENT_T_058 fixed_initial_state
    Threads.@threads :dynamic for idx in eachindex(values)
        g_h = Float64(values[idx])
        @info "Starting g_h point" idx g_h
        result = run_gh_point_058(idx, g_h, fixed_initial_state)
        results[idx] = result
        @info "Finished g_h point" idx g_h status=result.status main_runtime=result.main_runtime n_symbols=result.n_symbols lambda1_bits=result.lambda1_bits_per_second lz76_bits=result.lz76_bits_per_second
    end
    total_runtime = time() - started
    write_summary_058(results)
    plot_gh_sweep_058(results)
    open(GH_SWEEP_TIMING_PATH_058, "w") do io
        println(io, "created_at\t$(Dates.now())")
        println(io, "n_threads\t$(Threads.nthreads())")
        println(io, "n_g_h_values\t$(length(values))")
        println(io, "g_h_min\t$(minimum(values))")
        println(io, "g_h_max\t$(maximum(values))")
        println(io, "delta_x\t$(DELTA_X_058)")
        println(io, "delta_ca\t$(DELTA_CA_058)")
        println(io, "transient_time_model_units\t$(TRANSIENT_T_058)")
        println(io, "run_time_model_units\t$(RUN_T_058)")
        println(io, "run_time_seconds\t$(RUN_T_058 * SECONDS_PER_TIME_UNIT_058)")
        println(io, "total_wall_runtime_seconds\t$(total_runtime)")
        println(io, "summary_path\t$(GH_SWEEP_SUMMARY_PATH_058)")
        println(io, "plot_path\t$(GH_SWEEP_PLOT_PATH_058)")
    end
    @info "Wrote g_h sweep outputs" plot=GH_SWEEP_PLOT_PATH_058 summary=GH_SWEEP_SUMMARY_PATH_058 timing=GH_SWEEP_TIMING_PATH_058 total_runtime
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
