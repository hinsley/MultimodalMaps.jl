using Pkg

ENV["ATTEMPT058_RUN_T"] = get(ENV, "ATTEMPT058_RUN_T", "1.0e7")
ENV["ATTEMPT058_LYAP_K"] = get(ENV, "ATTEMPT058_LYAP_K", "1")
ENV["ATTEMPT058_QR_DT"] = get(ENV, "ATTEMPT058_QR_DT", "10.0")
ENV["ATTEMPT058_RECORD_DT"] = get(ENV, "ATTEMPT058_RECORD_DT", "1.0e4")
ENV["ATTEMPT058_OUTPUT_TAG"] = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")

include(joinpath(@__DIR__, "main.jl"))

using CairoMakie
using Printf
using Random
using StaticArrays

const ENSEMBLE_N_058 = env_int_058("ATTEMPT058_ENSEMBLE_N", 10)
const ENSEMBLE_SEED_058 = env_int_058("ATTEMPT058_ENSEMBLE_SEED", 58010)
const ENSEMBLE_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_trajectories")
const ENSEMBLE_SUMMARY_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_summary.tsv")
const ENSEMBLE_PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_convergence.png")

function ensemble_initial_states_058(p, count::Int)
    rng = MersenneTwister(ENSEMBLE_SEED_058)
    states = SVector{5, Float64}[]
    for idx in 1:count
        V0 = -58.0 + 32.0 * rand(rng)
        x0 = clamp(Plant058.xinf(p, V0) + 2.0e-3 * (rand(rng) - 0.5), 1.0e-6, 0.999999)
        n0 = clamp(Plant058.ninf(V0) + 2.0e-3 * (rand(rng) - 0.5), 1.0e-6, 0.999999)
        h0 = clamp(Plant058.hinf(V0) + 2.0e-3 * (rand(rng) - 0.5), 1.0e-6, 0.999999)
        ca_base = EquilibriaSubset.Ca_null_Ca(p, V0)
        ca0 = max(1.0e-6, ca_base * (1.0 + 0.02 * (rand(rng) - 0.5)))
        push!(states, SVector{5, Float64}(x0, n0, h0, ca0, V0))
    end
    return states
end

function trajectory_tag_058(idx::Int)
    return @sprintf("traj%02d", idx)
end

function trajectory_paths_058(idx::Int)
    tag = trajectory_tag_058(idx)
    return (;
        events=joinpath(ENSEMBLE_DIR_058, "$(tag)_events.tsv"),
        convergence=joinpath(ENSEMBLE_DIR_058, "$(tag)_convergence.tsv"),
        metadata=joinpath(ENSEMBLE_DIR_058, "$(tag)_metadata.txt"),
    )
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

function write_convergence_to_path_058(path::String, rows::Vector{ConvergenceRow058})
    open(path, "w") do io
        println(io, "time\ttime_seconds\tlambda1_nat_per_timeunit\tlambda1_nats_per_second\tlambda1_bits_per_second\tlz76_nats_per_second\tlz76_bits_per_second\tlz76_alphabet_units_per_second\tlz76_alphabet_size\tn_symbols\tlast_symbol_time\tlast_symbol_time_seconds\tlz76_phrase_count")
        for row in rows
            println(io, join((
                @sprintf("%.12g", row.t),
                @sprintf("%.12g", row.t * SECONDS_PER_TIME_UNIT_058),
                @sprintf("%.12g", row.lambda1_nat_per_timeunit),
                @sprintf("%.12g", row.lambda1_nats_per_second),
                @sprintf("%.12g", row.lambda1_bits_per_second),
                isfinite(row.lz76_nats_per_second) ? @sprintf("%.12g", row.lz76_nats_per_second) : "NaN",
                isfinite(row.lz76_bits_per_second) ? @sprintf("%.12g", row.lz76_bits_per_second) : "NaN",
                isfinite(row.lz76_alphabet_units_per_second) ? @sprintf("%.12g", row.lz76_alphabet_units_per_second) : "NaN",
                row.lz76_alphabet_size,
                row.n_symbols,
                isfinite(row.last_symbol_time) ? @sprintf("%.12g", row.last_symbol_time) : "NaN",
                isfinite(row.last_symbol_time) ? @sprintf("%.12g", row.last_symbol_time * SECONDS_PER_TIME_UNIT_058) : "NaN",
                row.lz76_phrase_count,
            ), '\t'))
        end
    end
end

function write_metadata_to_path_058(path::String, idx::Int, u0, post_transient_state, final_state, transient_runtime, main_runtime, retcode, rows, recorder)
    open(path, "w") do io
        println(io, "trajectory_index\t$(idx)")
        println(io, "delta_x\t$(DELTA_X_058)")
        println(io, "delta_ca\t$(DELTA_CA_058)")
        println(io, "g_h\t$(G_H_058)")
        println(io, "transient_time\t$(TRANSIENT_T_058)")
        println(io, "run_time\t$(RUN_T_058)")
        println(io, "lyap_k\t$(LYAP_K_058)")
        println(io, "qr_dt\t$(QR_DT_058)")
        println(io, "record_dt\t$(RECORD_DT_058)")
        println(io, "abstol\t$(ABSTOL_058)")
        println(io, "reltol\t$(RELTOL_058)")
        println(io, "initial_state\t$(join(string.(u0), ","))")
        println(io, "post_transient_state\t$(join(string.(post_transient_state), ","))")
        println(io, "final_state\t$(join(string.(final_state), ","))")
        println(io, "retcode\t$(retcode)")
        println(io, "transient_runtime_seconds\t$(transient_runtime)")
        println(io, "main_runtime_seconds\t$(main_runtime)")
        println(io, "n_symbols\t$(length(recorder.scs))")
        if !isempty(rows)
            last_row = rows[end]
            println(io, "final_lambda1_nats_per_second\t$(last_row.lambda1_nats_per_second)")
            println(io, "final_lambda1_bits_per_second\t$(last_row.lambda1_bits_per_second)")
            println(io, "final_lz76_nats_per_second\t$(last_row.lz76_nats_per_second)")
            println(io, "final_lz76_bits_per_second\t$(last_row.lz76_bits_per_second)")
            println(io, "final_lz76_alphabet_units_per_second\t$(last_row.lz76_alphabet_units_per_second)")
            println(io, "final_lz76_alphabet_size\t$(last_row.lz76_alphabet_size)")
            println(io, "final_lz76_phrase_count\t$(last_row.lz76_phrase_count)")
        end
    end
end

function plot_ensemble_058(results)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="time (s)",
        ylabel="entropy rate (nats/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
    )
    lle_handle = nothing
    lz_handle = nothing
    for result in results
        times_s = [row.t * SECONDS_PER_TIME_UNIT_058 for row in result.rows]
        λ1_nats = [row.lambda1_nats_per_second for row in result.rows]
        lz_nats = [row.lz76_nats_per_second for row in result.rows]
        local a = lines!(ax, times_s, λ1_nats; color=(:black, 0.35), linewidth=2.0)
        finite_lz = isfinite.(lz_nats)
        local b = lines!(ax, times_s[finite_lz], lz_nats[finite_lz]; color=(:red, 0.35), linewidth=2.0)
        lle_handle === nothing && (lle_handle = a)
        lz_handle === nothing && (lz_handle = b)
    end
    Legend(
        fig[1, 1],
        [lle_handle, lz_handle],
        ["LLE", "Abramov LZ76"],
        labelsize=22,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )
    save(ENSEMBLE_PLOT_PATH_058, fig, px_per_unit=2)
end

function main_ensemble_058()
    mkpath(ENSEMBLE_DIR_058)
    p = build_params_058()
    V_eqs = find_equilibria_058(p)
    length(V_eqs) >= 3 || error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
    V_sd = Float64(V_eqs[3])
    initial_states = ensemble_initial_states_058(p, ENSEMBLE_N_058)
    results = NamedTuple[]

    open(ENSEMBLE_SUMMARY_PATH_058, "w") do io
        println(io, "trajectory\tinitial_state\tpost_transient_state\tfinal_state\tretcode\tn_symbols\tfinal_lambda1_nats_per_second\tfinal_lz76_nats_per_second\tfinal_lambda1_bits_per_second\tfinal_lz76_bits_per_second\tfinal_lz76_alphabet_units_per_second\tfinal_lz76_alphabet_size\tfinal_lz76_phrase_count\tevents_path\tconvergence_path\tmetadata_path")
        for (idx, u0) in enumerate(initial_states)
            paths = trajectory_paths_058(idx)
            @info "Starting ensemble trajectory" idx count=ENSEMBLE_N_058 u0
            u0_aug = augmented_state_058(u0, p, 0.0)
            transient_runtime = @elapsed post_transient_state, _ = run_transient_058(p, u0_aug)
            local recorder
            local rows
            local final_state
            local retcode
            main_runtime = @elapsed begin
                recorder, rows, final_state, retcode = run_main_058(p, post_transient_state, V_sd)
            end
            write_events_to_path_058(paths.events, recorder)
            write_convergence_to_path_058(paths.convergence, rows)
            write_metadata_to_path_058(paths.metadata, idx, u0, post_transient_state, final_state, transient_runtime, main_runtime, retcode, rows, recorder)
            push!(results, (; idx, rows, recorder))
            last_row = rows[end]
            println(io, join((
                idx,
                join(string.(u0), ","),
                join(string.(post_transient_state), ","),
                join(string.(final_state), ","),
                retcode,
                length(recorder.scs),
                @sprintf("%.12g", last_row.lambda1_nats_per_second),
                @sprintf("%.12g", last_row.lz76_nats_per_second),
                @sprintf("%.12g", last_row.lambda1_bits_per_second),
                @sprintf("%.12g", last_row.lz76_bits_per_second),
                @sprintf("%.12g", last_row.lz76_alphabet_units_per_second),
                last_row.lz76_alphabet_size,
                last_row.lz76_phrase_count,
                paths.events,
                paths.convergence,
                paths.metadata,
            ), '\t'))
            flush(io)
            @info "Finished ensemble trajectory" idx main_runtime n_symbols=length(recorder.scs) final_lle_nats=last_row.lambda1_nats_per_second final_lz76_nats=last_row.lz76_nats_per_second
        end
    end

    plot_ensemble_058(results)
    @info "Wrote ensemble outputs" plot=ENSEMBLE_PLOT_PATH_058 summary=ENSEMBLE_SUMMARY_PATH_058 trajectories=ENSEMBLE_DIR_058
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_ensemble_058()
end
