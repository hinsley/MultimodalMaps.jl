using Pkg

const ATTEMPT59_ROOT = @__DIR__
const REPO_ROOT_059 = normpath(joinpath(ATTEMPT59_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_059)

using CairoMakie
using Dates
using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using Printf
using Roots
using StaticArrays

include(joinpath(REPO_ROOT_059, "models", "SiN.jl"))
include(joinpath(REPO_ROOT_059, "kneading", "experiment", "attempt-009", "vendor", "equilibria_subset.jl"))

using .SiN
using .EquilibriaSubset

const Plant059 = SiN

env_float_059(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))
env_int_059(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))

const DELTA_X_059 = env_float_059("ATTEMPT059_DELTA_X", -1.0)
const DELTA_CA_059 = env_float_059("ATTEMPT059_DELTA_CA", -35.0)
const GH_SWEEP_N_059 = env_int_059("ATTEMPT059_GH_SWEEP_N", 10)
const GH_SWEEP_MIN_059 = env_float_059("ATTEMPT059_GH_SWEEP_MIN", 0.0)
const GH_SWEEP_MAX_059 = env_float_059("ATTEMPT059_GH_SWEEP_MAX", 0.01)
const TRANSIENT_T_059 = env_float_059("ATTEMPT059_TRANSIENT_T", 1.0e4)
const RUN_T_059 = env_float_059("ATTEMPT059_RUN_T", 2.5e6)
const QR_DT_059 = env_float_059("ATTEMPT059_QR_DT", 10.0)
const ABSTOL_059 = env_float_059("ATTEMPT059_ABSTOL", 1.0e-8)
const RELTOL_059 = env_float_059("ATTEMPT059_RELTOL", 1.0e-8)
const MAXITERS_059 = env_int_059("ATTEMPT059_MAXITERS", 10_000_000_000)
const VDDOT_DT_059 = env_float_059("ATTEMPT059_VDDOT_DT", 1.0e-4)
const MIN_EVENT_TIME_059 = env_float_059("ATTEMPT059_MIN_EVENT_TIME", 1.0e-6)
const V_RETURN_THRESHOLD_059 = env_float_059("ATTEMPT059_V_RETURN_THRESHOLD", -46.0)
const INITIAL_V_059 = env_float_059("ATTEMPT059_INITIAL_V", -30.0)
const INITIAL_X_OFFSET_059 = env_float_059("ATTEMPT059_INITIAL_X_OFFSET", -1.0e-4)
const SECONDS_PER_TIME_UNIT_059 = 1.0e-3
const GH_SWEEP_TAG_059 = get(ENV, "ATTEMPT059_GH_SWEEP_TAG", @sprintf("gh_sweep_full6d_smoke%02d_t2500s", GH_SWEEP_N_059))
const GH_SWEEP_DIR_059 = joinpath(ATTEMPT59_ROOT, GH_SWEEP_TAG_059)
const GH_SWEEP_SUMMARY_PATH_059 = joinpath(GH_SWEEP_DIR_059, "$(GH_SWEEP_TAG_059)_summary.tsv")
const GH_SWEEP_TIMING_PATH_059 = joinpath(GH_SWEEP_DIR_059, "$(GH_SWEEP_TAG_059)_timing.txt")
const GH_SWEEP_PLOT_PATH_059 = joinpath(GH_SWEEP_DIR_059, "$(GH_SWEEP_TAG_059)_lle_lz76_bits_per_second.png")

@enum EventSymbol059 begin
    Void059
    I059
    Vplus059
    Vminus059
end

mutable struct SSCSRecorder059
    scs::Vector{Int}
    times::Vector{Float64}
    count::Int
    last_symbol::EventSymbol059
    last2_symbol::EventSymbol059
    V_sd::Float64
end

function gh_values_059()
    return collect(range(GH_SWEEP_MIN_059, GH_SWEEP_MAX_059; length=GH_SWEEP_N_059))
end

function build_params_059(g_h::Float64)::SVector{18, Float64}
    p = Vector{Float64}(Plant059.default_params)
    p[4] = g_h
    p[17] = DELTA_X_059
    p[18] = DELTA_CA_059
    return SVector{18, Float64}(p)
end

function find_equilibria_059(p)
    return find_zeros(
        v -> EquilibriaSubset.Ca_difference(p, v),
        Plant059.xinfinv(p, 0.99),
        Plant059.xinfinv(p, 0.01),
    )
end

function initial_state_059(p)::SVector{6, Float64}
    V0 = INITIAL_V_059
    return SVector{6, Float64}(
        Plant059.xinf(p, V0) + INITIAL_X_OFFSET_059,
        Plant059.yinf(V0),
        Plant059.ninf(V0),
        Plant059.hinf(V0),
        EquilibriaSubset.Ca_null_Ca(p, V0),
        V0,
    )
end

state6_059(u) = SVector{6, Float64}(ntuple(i -> Float64(u[i]), 6))

function active_flow_059(state::SVector{6, T}, p, t) where {T}
    return Plant059.melibe_h(state, p, t)
end

function jvp_059(state::SVector{6, Float64}, tangent::SVector{6, Float64}, p, t)
    dual_zero = ForwardDiff.Dual(0.0, 1.0)
    dual_state = state .+ dual_zero .* tangent
    fdual = active_flow_059(dual_state, p, t)
    return SVector{6, Float64}(ntuple(i -> ForwardDiff.partials(fdual[i])[1], 6))
end

function augmented_rhs_059!(du, u, p, t)
    state = state6_059(u)
    flow = active_flow_059(state, p, t)
    tangent = SVector{6, Float64}(ntuple(i -> Float64(u[6 + i]), 6))
    tangent_dot = jvp_059(state, tangent, p, t)
    for i in 1:6
        du[i] = flow[i]
        du[6 + i] = tangent_dot[i]
    end
    return nothing
end

function augmented_state_059(state::SVector{6, Float64})::Vector{Float64}
    tangent = zeros(Float64, 6)
    tangent[1] = 1.0
    return vcat(collect(state), tangent)
end

function normalize_tangent_059!(u)
    norm_value = norm(@view u[7:12])
    isfinite(norm_value) && norm_value > 0 || error("Non-finite tangent norm.")
    for i in 7:12
        u[i] /= norm_value
    end
    return log(norm_value)
end

function vddot_059(state::SVector{6, Float64}, p, t)::Float64
    f0 = active_flow_059(state, p, t)
    state1 = state .+ VDDOT_DT_059 .* f0
    f1 = active_flow_059(state1, p, t + VDDOT_DT_059)
    return (f1[6] - f0[6]) / VDDOT_DT_059
end

function make_sscs_condition_059()
    function sscs_condition_059!(out, u, t, integrator)
        if t < MIN_EVENT_TIME_059
            out[1] = 1.0
            out[2] = 1.0
            return nothing
        end
        state = state6_059(u)
        out[1] = -active_flow_059(state, integrator.p, integrator.t)[6]
        out[2] = -vddot_059(state, integrator.p, integrator.t)
        return nothing
    end
    return sscs_condition_059!
end

function make_affect_sscs_059!(recorder::SSCSRecorder059)
    function affect_sscs_059!(integrator, idx)
        if idx == 1
            current_V = Float64(integrator.u[6])
            current_event = current_V > recorder.V_sd ? Vplus059 : Vminus059
            if current_event == Vminus059
                symbol = current_V > V_RETURN_THRESHOLD_059 || recorder.last2_symbol != Vplus059 ?
                    recorder.count : -recorder.count
                push!(recorder.scs, symbol)
                push!(recorder.times, Float64(integrator.t))
                recorder.count = 0
            else
                recorder.count += 1
            end
            recorder.last2_symbol = recorder.last_symbol
            recorder.last_symbol = current_event
        elseif idx == 2
            recorder.last2_symbol = recorder.last_symbol
            recorder.last_symbol = I059
        end
        return nothing
    end
    return affect_sscs_059!
end

function lz76_phrase_count_059(xs::AbstractVector{<:Integer})::Int
    n = length(xs)
    n == 0 && return 0
    phrases = Set{Tuple{Vararg{Int}}}()
    count = 0
    i = 1
    while i <= n
        j = i
        phrase = Tuple(Int[xs[k] for k in i:j])
        while phrase in phrases && j < n
            j += 1
            phrase = Tuple(Int[xs[k] for k in i:j])
        end
        push!(phrases, phrase)
        count += 1
        i = j + 1
    end
    return count
end

function lz76_rates_059(symbols::Vector{Int}, times::Vector{Float64})
    n = length(symbols)
    n <= 1 && return NaN, NaN, 0, 0, NaN
    mean_interval_seconds = ((times[end] - times[1]) * SECONDS_PER_TIME_UNIT_059) / (n - 1)
    mean_interval_seconds > 0 || return NaN, NaN, 0, 0, NaN
    alphabet_size = length(unique(symbols))
    alphabet_size > 1 || return NaN, NaN, alphabet_size, 0, mean_interval_seconds
    phrase_count = lz76_phrase_count_059(symbols)
    nats_per_symbol = phrase_count * log(n) / n
    bits_per_symbol = nats_per_symbol / log(2)
    return nats_per_symbol / mean_interval_seconds, bits_per_symbol / mean_interval_seconds, alphabet_size, phrase_count, mean_interval_seconds
end

function run_transient_059(p, u0_aug::Vector{Float64})
    prob = ODEProblem(augmented_rhs_059!, copy(u0_aug), (0.0, TRANSIENT_T_059), p)
    integrator = init(prob, Tsit5(); abstol=ABSTOL_059, reltol=RELTOL_059, maxiters=MAXITERS_059, save_everystep=false)
    while integrator.t < TRANSIENT_T_059 - eps(Float64)
        dt = min(QR_DT_059, TRANSIENT_T_059 - integrator.t)
        t_before = Float64(integrator.t)
        step!(integrator, dt, true)
        integrator.t > t_before || error("Transient integration stalled at t=$(integrator.t), retcode=$(integrator.sol.retcode).")
        normalize_tangent_059!(integrator.u)
    end
    return state6_059(integrator.u)
end

function run_main_059(p, post_transient_state::SVector{6, Float64}, V_sd::Float64)
    u0_aug = augmented_state_059(post_transient_state)
    recorder = SSCSRecorder059(Int[], Float64[], 0, Void059, Void059, V_sd)
    callback = VectorContinuousCallback(
        make_sscs_condition_059(),
        make_affect_sscs_059!(recorder),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(augmented_rhs_059!, u0_aug, (0.0, RUN_T_059), p)
    integrator = init(prob, Tsit5(); callback=callback, abstol=ABSTOL_059, reltol=RELTOL_059, maxiters=MAXITERS_059, save_everystep=false)
    λsum = 0.0
    while integrator.t < RUN_T_059 - eps(Float64)
        dt = min(QR_DT_059, RUN_T_059 - integrator.t)
        t_before = Float64(integrator.t)
        step!(integrator, dt, true)
        integrator.t > t_before || error("Main integration stalled at t=$(integrator.t), retcode=$(integrator.sol.retcode).")
        λsum += normalize_tangent_059!(integrator.u)
    end
    λ_nat_per_timeunit = λsum / RUN_T_059
    λ_nats_per_second = λ_nat_per_timeunit / SECONDS_PER_TIME_UNIT_059
    λ_bits_per_second = λ_nat_per_timeunit / log(2) / SECONDS_PER_TIME_UNIT_059
    lz_nats, lz_bits, alphabet_size, phrase_count, mean_interval = lz76_rates_059(recorder.scs, recorder.times)
    return (; recorder, final_state=state6_059(integrator.u), retcode=string(integrator.sol.retcode), λ_nats_per_second, λ_bits_per_second, lz_nats, lz_bits, alphabet_size, phrase_count, mean_interval)
end

function gh_tag_059(idx::Int, g_h::Float64)
    return @sprintf("gh%02d_%0.6f", idx, g_h)
end

function write_events_059(path::String, recorder::SSCSRecorder059)
    open(path, "w") do io
        println(io, "idx\ttime\ttime_seconds\tsymbol\tinterval\tinterval_seconds")
        for i in eachindex(recorder.scs)
            interval = i == 1 ? recorder.times[i] : recorder.times[i] - recorder.times[i - 1]
            println(io, join((
                i,
                @sprintf("%.12g", recorder.times[i]),
                @sprintf("%.12g", recorder.times[i] * SECONDS_PER_TIME_UNIT_059),
                recorder.scs[i],
                @sprintf("%.12g", interval),
                @sprintf("%.12g", interval * SECONDS_PER_TIME_UNIT_059),
            ), '\t'))
        end
    end
end

function write_metadata_059(path::String, idx::Int, g_h::Float64, V_eqs, V_sd, u0, post_transient_state, result, transient_runtime, main_runtime)
    open(path, "w") do io
        println(io, "created_at\t$(Dates.now())")
        println(io, "sweep_index\t$(idx)")
        println(io, "delta_x\t$(DELTA_X_059)")
        println(io, "delta_ca\t$(DELTA_CA_059)")
        println(io, "g_h\t$(g_h)")
        println(io, "model\tfull 6D SiN; y integrated")
        println(io, "transient_time_model_units\t$(TRANSIENT_T_059)")
        println(io, "run_time_model_units\t$(RUN_T_059)")
        println(io, "run_time_seconds\t$(RUN_T_059 * SECONDS_PER_TIME_UNIT_059)")
        println(io, "time_unit_seconds\t$(SECONDS_PER_TIME_UNIT_059)")
        println(io, "qr_dt\t$(QR_DT_059)")
        println(io, "abstol\t$(ABSTOL_059)")
        println(io, "reltol\t$(RELTOL_059)")
        println(io, "V_equilibria\t$(join(string.(V_eqs), ","))")
        println(io, "V_sd\t$(V_sd)")
        println(io, "initial_state\t$(join(string.(u0), ","))")
        println(io, "post_transient_state\t$(join(string.(post_transient_state), ","))")
        println(io, "final_state\t$(join(string.(result.final_state), ","))")
        println(io, "retcode\t$(result.retcode)")
        println(io, "transient_runtime_seconds\t$(transient_runtime)")
        println(io, "main_runtime_seconds\t$(main_runtime)")
        println(io, "n_symbols\t$(length(result.recorder.scs))")
        println(io, "final_lambda1_nats_per_second\t$(result.λ_nats_per_second)")
        println(io, "final_lambda1_bits_per_second\t$(result.λ_bits_per_second)")
        println(io, "final_lz76_nats_per_second\t$(result.lz_nats)")
        println(io, "final_lz76_bits_per_second\t$(result.lz_bits)")
        println(io, "final_lz76_alphabet_size\t$(result.alphabet_size)")
        println(io, "final_lz76_phrase_count\t$(result.phrase_count)")
        println(io, "mean_event_interval_seconds\t$(result.mean_interval)")
    end
end

function run_gh_point_059(idx::Int, g_h::Float64, u0::SVector{6, Float64})
    tag = gh_tag_059(idx, g_h)
    events_path = joinpath(GH_SWEEP_DIR_059, "$(tag)_events.tsv")
    metadata_path = joinpath(GH_SWEEP_DIR_059, "$(tag)_metadata.txt")
    started = time()
    try
        p = build_params_059(g_h)
        V_eqs = find_equilibria_059(p)
        length(V_eqs) >= 3 || error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
        V_sd = Float64(V_eqs[3])
        transient_runtime = @elapsed post_transient_state = run_transient_059(p, augmented_state_059(u0))
        local result
        main_runtime = @elapsed result = run_main_059(p, post_transient_state, V_sd)
        write_events_059(events_path, result.recorder)
        write_metadata_059(metadata_path, idx, g_h, V_eqs, V_sd, u0, post_transient_state, result, transient_runtime, main_runtime)
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
            final_state=nothing,
            lambda1_nats_per_second=NaN,
            lz76_nats_per_second=NaN,
            lambda1_bits_per_second=NaN,
            lz76_bits_per_second=NaN,
            lz76_alphabet_size=0,
            n_symbols=0,
            lz76_phrase_count=0,
            mean_event_interval_seconds=NaN,
            transient_runtime=NaN,
            main_runtime=NaN,
            total_runtime=time() - started,
            events_path,
            metadata_path,
            error_message=sprint(showerror, err),
        )
    end
end

function write_summary_059(results)
    open(GH_SWEEP_SUMMARY_PATH_059, "w") do io
        println(io, "idx\tg_h\tstatus\tretcode\tlambda1_nats_per_second\tlz76_nats_per_second\tlambda1_bits_per_second\tlz76_bits_per_second\tlz76_alphabet_size\tn_symbols\tlz76_phrase_count\tmean_event_interval_seconds\ttransient_runtime_seconds\tmain_runtime_seconds\ttotal_runtime_seconds\tevents_path\tmetadata_path\terror_message")
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
                isfinite(result.mean_event_interval_seconds) ? @sprintf("%.12g", result.mean_event_interval_seconds) : "NaN",
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

function plot_gh_sweep_059(results)
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
    finite_lz = isfinite.(lz_bits)
    lz_line = lines!(ax, ghs[finite_lz], lz_bits[finite_lz]; color=:firebrick3, linewidth=3.2)
    scatter!(ax, ghs[finite_lz], lz_bits[finite_lz]; color=:firebrick3, markersize=11)
    xlims!(ax, GH_SWEEP_MIN_059, GH_SWEEP_MAX_059)
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
    save(GH_SWEEP_PLOT_PATH_059, fig, px_per_unit=2)
end

function main()
    mkpath(GH_SWEEP_DIR_059)
    values = gh_values_059()
    fixed_initial_state = initial_state_059(build_params_059(first(values)))
    results = Vector{Any}(undef, length(values))
    started = time()
    @info "Starting full 6D g_h sweep" n=length(values) threads=Threads.nthreads() run_time=RUN_T_059 run_time_seconds=RUN_T_059 * SECONDS_PER_TIME_UNIT_059 transient_time=TRANSIENT_T_059 fixed_initial_state
    Threads.@threads :dynamic for idx in eachindex(values)
        g_h = Float64(values[idx])
        @info "Starting full 6D g_h point" idx g_h
        result = run_gh_point_059(idx, g_h, fixed_initial_state)
        results[idx] = result
        @info "Finished full 6D g_h point" idx g_h status=result.status main_runtime=result.main_runtime n_symbols=result.n_symbols lambda1_bits=result.lambda1_bits_per_second lz76_bits=result.lz76_bits_per_second
    end
    total_runtime = time() - started
    write_summary_059(results)
    plot_gh_sweep_059(results)
    open(GH_SWEEP_TIMING_PATH_059, "w") do io
        println(io, "created_at\t$(Dates.now())")
        println(io, "n_threads\t$(Threads.nthreads())")
        println(io, "n_g_h_values\t$(length(values))")
        println(io, "g_h_min\t$(minimum(values))")
        println(io, "g_h_max\t$(maximum(values))")
        println(io, "delta_x\t$(DELTA_X_059)")
        println(io, "delta_ca\t$(DELTA_CA_059)")
        println(io, "model\tfull 6D SiN; y integrated")
        println(io, "transient_time_model_units\t$(TRANSIENT_T_059)")
        println(io, "run_time_model_units\t$(RUN_T_059)")
        println(io, "run_time_seconds\t$(RUN_T_059 * SECONDS_PER_TIME_UNIT_059)")
        println(io, "total_wall_runtime_seconds\t$(total_runtime)")
        println(io, "summary_path\t$(GH_SWEEP_SUMMARY_PATH_059)")
        println(io, "plot_path\t$(GH_SWEEP_PLOT_PATH_059)")
    end
    @info "Wrote full 6D g_h sweep outputs" plot=GH_SWEEP_PLOT_PATH_059 summary=GH_SWEEP_SUMMARY_PATH_059 timing=GH_SWEEP_TIMING_PATH_059 total_runtime
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
