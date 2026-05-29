using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie
using Dates
using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using Printf
using Roots
using StaticArrays

include(joinpath(REPO_ROOT_058, "models", "SiN.jl"))
include(joinpath(REPO_ROOT_058, "kneading", "experiment", "attempt-009", "vendor", "equilibria_subset.jl"))

using .SiN
using .EquilibriaSubset

const Plant058 = SiN

env_float_058(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))
env_int_058(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))

const DELTA_X_058 = env_float_058("ATTEMPT058_DELTA_X", -1.0)
const DELTA_CA_058 = env_float_058("ATTEMPT058_DELTA_CA", -35.0)
const G_H_058 = env_float_058("ATTEMPT058_G_H", 0.0)
const TRANSIENT_T_058 = env_float_058("ATTEMPT058_TRANSIENT_T", 1.0e4)
const RUN_T_058 = env_float_058("ATTEMPT058_RUN_T", 3.0e7)
const LYAP_K_058 = env_int_058("ATTEMPT058_LYAP_K", 2)
const QR_DT_058 = env_float_058("ATTEMPT058_QR_DT", 1.0e2)
const RECORD_DT_058 = env_float_058("ATTEMPT058_RECORD_DT", 1.0e4)
const ABSTOL_058 = env_float_058("ATTEMPT058_ABSTOL", 1.0e-8)
const RELTOL_058 = env_float_058("ATTEMPT058_RELTOL", 1.0e-8)
const MAXITERS_058 = env_int_058("ATTEMPT058_MAXITERS", 10_000_000_000)
const VDDOT_DT_058 = env_float_058("ATTEMPT058_VDDOT_DT", 1.0e-4)
const MIN_EVENT_TIME_058 = env_float_058("ATTEMPT058_MIN_EVENT_TIME", 1.0e-6)
const V_RETURN_THRESHOLD_058 = env_float_058("ATTEMPT058_V_RETURN_THRESHOLD", -46.0)
const INITIAL_V_058 = env_float_058("ATTEMPT058_INITIAL_V", -30.0)
const INITIAL_X_OFFSET_058 = env_float_058("ATTEMPT058_INITIAL_X_OFFSET", -1.0e-4)
const PLOT_WIDTH_058 = env_int_058("ATTEMPT058_PLOT_WIDTH", 1600)
const PLOT_HEIGHT_058 = env_int_058("ATTEMPT058_PLOT_HEIGHT", 1000)

const SECONDS_PER_TIME_UNIT_058 = 1.0e-3
const OUTPUT_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax3e7_k2")
const EVENTS_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_events.tsv")
const CONVERGENCE_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_convergence.tsv")
const METADATA_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_metadata.txt")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(OUTPUT_TAG_058)_convergence.png")

@enum EventSymbol058 begin
    Void058
    I058
    Vplus058
    Vminus058
end

mutable struct SSCSRecorder058
    scs::Vector{Int}
    times::Vector{Float64}
    count::Int
    last_symbol::EventSymbol058
    last2_symbol::EventSymbol058
    V_sd::Float64
end

struct ConvergenceRow058
    t::Float64
    lambda1_nat_per_timeunit::Float64
    lambda2_nat_per_timeunit::Float64
    lambda1_nats_per_second::Float64
    lambda2_nats_per_second::Float64
    positive_lyap_sum_nats_per_second::Float64
    lambda1_bits_per_second::Float64
    lambda2_bits_per_second::Float64
    positive_lyap_sum_bits_per_second::Float64
    lz76_nats_per_second::Float64
    lz76_bits_per_second::Float64
    lz76_alphabet_units_per_second::Float64
    lz76_alphabet_size::Int
    n_symbols::Int
    last_symbol_time::Float64
    lz76_phrase_count::Int
end

function build_params_058()::SVector{18, Float64}
    p = Vector{Float64}(Plant058.default_params)
    p[4] = G_H_058
    p[17] = DELTA_X_058
    p[18] = DELTA_CA_058
    return SVector{18, Float64}(p)
end

function find_equilibria_058(p)
    return find_zeros(
        v -> EquilibriaSubset.Ca_difference(p, v),
        Plant058.xinfinv(p, 0.99),
        Plant058.xinfinv(p, 0.01),
    )
end

function initial_state_058(p)::SVector{5, Float64}
    V0 = INITIAL_V_058
    return SVector{5, Float64}(
        Plant058.xinf(p, V0) + INITIAL_X_OFFSET_058,
        Plant058.ninf(V0),
        Plant058.hinf(V0),
        EquilibriaSubset.Ca_null_Ca(p, V0),
        V0,
    )
end

state5_058(u) = SVector{5, Float64}(ntuple(i -> Float64(u[i]), 5))

function tangent_matrix_058(u, k::Int=LYAP_K_058)
    matrix = Matrix{Float64}(undef, 5, k)
    for j in 1:k, i in 1:5
        matrix[i, j] = Float64(u[5 + 5 * (j - 1) + i])
    end
    return matrix
end

function set_tangent_matrix_058!(u, matrix::AbstractMatrix{<:Real})
    k = size(matrix, 2)
    for j in 1:k, i in 1:5
        u[5 + 5 * (j - 1) + i] = matrix[i, j]
    end
    return nothing
end

function active_flow_058(state::SVector{5, T}, p, t) where {T}
    x, n, h, Ca, V = state
    return SVector{5, T}(
        Plant058.dx(p, x, V),
        Plant058.dn(n, V),
        Plant058.dh(h, V),
        Plant058.dCa(p, Ca, x, V),
        Plant058.dV(p, x, zero(T), n, h, Ca, V),
    )
end

function jvp_058(state::SVector{5, Float64}, tangent::SVector{5, Float64}, p, t)
    dual_zero = ForwardDiff.Dual(0.0, 1.0)
    dual_state = state .+ dual_zero .* tangent
    fdual = active_flow_058(dual_state, p, t)
    return SVector{5, Float64}(ntuple(i -> ForwardDiff.partials(fdual[i])[1], 5))
end

function tangent_augmented_rhs_058!(du, u, p, t)
    state = state5_058(u)
    flow = active_flow_058(state, p, t)
    for i in 1:5
        du[i] = flow[i]
    end
    for j in 1:LYAP_K_058
        tangent = SVector{5, Float64}(ntuple(i -> Float64(u[5 + 5 * (j - 1) + i]), 5))
        tangent_dot = jvp_058(state, tangent, p, t)
        for i in 1:5
            du[5 + 5 * (j - 1) + i] = tangent_dot[i]
        end
    end
    return nothing
end

function qr_tangent_step_058!(u, k::Int=LYAP_K_058)
    tangent = tangent_matrix_058(u, k)
    factorization = qr(tangent)
    Q = Matrix(factorization.Q)[:, 1:k]
    R = factorization.R
    logs = Vector{Float64}(undef, k)
    for i in 1:k
        value = abs(R[i, i])
        isfinite(value) && value > 0 || error("Non-finite tangent QR diagonal at index $(i).")
        logs[i] = log(value)
    end
    set_tangent_matrix_058!(u, Q)
    return logs
end

function augmented_state_058(state::SVector{5, Float64}, p, t)::Vector{Float64}
    LYAP_K_058 <= 5 || error("LYAP_K_058=$(LYAP_K_058) exceeds state dimension 5.")
    tangent = Matrix{Float64}(I, 5, LYAP_K_058)
    return vcat(collect(state), vec(tangent))
end

function vddot_058(state::SVector{5, Float64}, p, t)::Float64
    f0 = active_flow_058(state, p, t)
    state1 = state .+ VDDOT_DT_058 .* f0
    f1 = active_flow_058(state1, p, t + VDDOT_DT_058)
    return (f1[5] - f0[5]) / VDDOT_DT_058
end

function make_sscs_condition_058()
    function sscs_condition_058!(out, u, t, integrator)
        if t < MIN_EVENT_TIME_058
            out[1] = 1.0
            out[2] = 1.0
            return nothing
        end
        state = state5_058(u)
        out[1] = -active_flow_058(state, integrator.p, integrator.t)[5]
        out[2] = -vddot_058(state, integrator.p, integrator.t)
        return nothing
    end
    return sscs_condition_058!
end

function make_affect_sscs_058!(recorder::SSCSRecorder058)
    function affect_sscs_058!(integrator, idx)
        if idx == 1
            current_V = Float64(integrator.u[5])
            current_event = current_V > recorder.V_sd ? Vplus058 : Vminus058
            if current_event == Vminus058
                symbol = current_V > V_RETURN_THRESHOLD_058 || recorder.last2_symbol != Vplus058 ?
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
            recorder.last_symbol = I058
        end
        return nothing
    end
    return affect_sscs_058!
end

function lz76_phrase_count_058(xs::AbstractVector{<:Integer})::Int
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

function lz76_rates_058(symbols::Vector{Int}, times::Vector{Float64}, n::Int)
    if n <= 1
        return NaN, NaN, NaN, 0, 0
    end
    mean_interval_seconds = ((times[n] - times[1]) * SECONDS_PER_TIME_UNIT_058) / (n - 1)
    mean_interval_seconds > 0 || return NaN, NaN, NaN, 0, 0
    alphabet_size = length(unique(@view symbols[1:n]))
    alphabet_size > 1 || return NaN, NaN, NaN, alphabet_size, 0
    phrase_count = lz76_phrase_count_058(@view symbols[1:n])
    entropy_alphabet_units_per_symbol = phrase_count * (log(n) / log(alphabet_size)) / n
    alphabet_units_per_second = entropy_alphabet_units_per_symbol / mean_interval_seconds
    nats_per_second = alphabet_units_per_second * log(alphabet_size)
    bits_per_second = alphabet_units_per_second * log2(alphabet_size)
    return nats_per_second, bits_per_second, alphabet_units_per_second, alphabet_size, phrase_count
end

function run_transient_058(p, u0_aug::Vector{Float64})
    transient_t = 0.0
    prob = ODEProblem(tangent_augmented_rhs_058!, copy(u0_aug), (0.0, TRANSIENT_T_058), p)
    integrator = init(prob, Tsit5(); abstol=ABSTOL_058, reltol=RELTOL_058, maxiters=MAXITERS_058, save_everystep=false)
    while integrator.t < TRANSIENT_T_058 - eps(Float64)
        dt = min(QR_DT_058, TRANSIENT_T_058 - integrator.t)
        t_before = Float64(integrator.t)
        step!(integrator, dt, true)
        integrator.t > t_before || error("Transient integration stalled at t=$(integrator.t), retcode=$(integrator.sol.retcode).")
        qr_tangent_step_058!(integrator.u)
        transient_t = Float64(integrator.t)
    end
    state = state5_058(integrator.u)
    return state, transient_t
end

function run_main_058(p, post_transient_state::SVector{5, Float64}, V_sd::Float64)
    u0_aug = augmented_state_058(post_transient_state, p, 0.0)
    recorder = SSCSRecorder058(Int[], Float64[], 0, Void058, Void058, V_sd)
    callback = VectorContinuousCallback(
        make_sscs_condition_058(),
        make_affect_sscs_058!(recorder),
        nothing,
        2,
        save_positions=(false, false),
    )
    prob = ODEProblem(tangent_augmented_rhs_058!, u0_aug, (0.0, RUN_T_058), p)
    integrator = init(prob, Tsit5(); callback=callback, abstol=ABSTOL_058, reltol=RELTOL_058, maxiters=MAXITERS_058, save_everystep=false)

    rows = ConvergenceRow058[]
    λsum = zeros(Float64, LYAP_K_058)
    next_record = RECORD_DT_058

    while integrator.t < RUN_T_058 - eps(Float64)
        dt = min(QR_DT_058, RUN_T_058 - integrator.t)
        t_before = Float64(integrator.t)
        step!(integrator, dt, true)
        integrator.t > t_before || error("Main integration stalled at t=$(integrator.t), retcode=$(integrator.sol.retcode).")

        λsum .+= qr_tangent_step_058!(integrator.u)

        if integrator.t >= next_record - eps(Float64) || integrator.t >= RUN_T_058 - eps(Float64)
            n = searchsortedlast(recorder.times, Float64(integrator.t))
            lz_nats, lz_bits, lz_alphabet_units, alphabet_size, phrases = lz76_rates_058(recorder.scs, recorder.times, n)
            λ_nat = sort(λsum ./ Float64(integrator.t); rev=true)
            λ_nats_per_second = λ_nat ./ SECONDS_PER_TIME_UNIT_058
            λ_bits = λ_nat ./ log(2) ./ SECONDS_PER_TIME_UNIT_058
            λ1_nat = length(λ_nat) >= 1 ? λ_nat[1] : NaN
            λ2_nat = length(λ_nat) >= 2 ? λ_nat[2] : NaN
            λ1_nats = length(λ_nats_per_second) >= 1 ? λ_nats_per_second[1] : NaN
            λ2_nats = length(λ_nats_per_second) >= 2 ? λ_nats_per_second[2] : NaN
            positive_sum_nats = sum(max(x, 0.0) for x in λ_nats_per_second)
            λ1_bits = length(λ_bits) >= 1 ? λ_bits[1] : NaN
            λ2_bits = length(λ_bits) >= 2 ? λ_bits[2] : NaN
            positive_sum_bits = sum(max(x, 0.0) for x in λ_bits)
            last_symbol_time = n > 0 ? recorder.times[n] : NaN
            push!(rows, ConvergenceRow058(Float64(integrator.t), λ1_nat, λ2_nat, λ1_nats, λ2_nats, positive_sum_nats, λ1_bits, λ2_bits, positive_sum_bits, lz_nats, lz_bits, lz_alphabet_units, alphabet_size, n, last_symbol_time, phrases))
            next_record += RECORD_DT_058
        end
    end

    return recorder, rows, state5_058(integrator.u), string(integrator.sol.retcode)
end

function write_events_058(recorder::SSCSRecorder058)
    open(EVENTS_PATH_058, "w") do io
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

function write_convergence_058(rows::Vector{ConvergenceRow058})
    open(CONVERGENCE_PATH_058, "w") do io
        println(io, "time\ttime_seconds\tlambda1_nat_per_timeunit\tlambda2_nat_per_timeunit\tlambda1_nats_per_second\tlambda2_nats_per_second\tpositive_lyap_sum_nats_per_second\tlambda1_bits_per_second\tlambda2_bits_per_second\tpositive_lyap_sum_bits_per_second\tlz76_nats_per_second\tlz76_bits_per_second\tlz76_alphabet_units_per_second\tlz76_alphabet_size\tn_symbols\tlast_symbol_time\tlast_symbol_time_seconds\tlz76_phrase_count")
        for row in rows
            println(io, join((
                @sprintf("%.12g", row.t),
                @sprintf("%.12g", row.t * SECONDS_PER_TIME_UNIT_058),
                @sprintf("%.12g", row.lambda1_nat_per_timeunit),
                @sprintf("%.12g", row.lambda2_nat_per_timeunit),
                @sprintf("%.12g", row.lambda1_nats_per_second),
                @sprintf("%.12g", row.lambda2_nats_per_second),
                @sprintf("%.12g", row.positive_lyap_sum_nats_per_second),
                @sprintf("%.12g", row.lambda1_bits_per_second),
                @sprintf("%.12g", row.lambda2_bits_per_second),
                @sprintf("%.12g", row.positive_lyap_sum_bits_per_second),
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

function write_metadata_058(p, V_eqs, V_sd, u0, post_transient_state, final_state, transient_runtime, main_runtime, retcode, rows, recorder)
    open(METADATA_PATH_058, "w") do io
        println(io, "created_at\t$(Dates.now())")
        println(io, "repo_root\t$(REPO_ROOT_058)")
        println(io, "attempt_root\t$(ATTEMPT58_ROOT)")
        println(io, "delta_x\t$(DELTA_X_058)")
        println(io, "delta_ca\t$(DELTA_CA_058)")
        println(io, "g_h\t$(G_H_058)")
        println(io, "y_rule\ty identically zero; 5D state [x,n,h,Ca,V]")
        println(io, "transient_time\t$(TRANSIENT_T_058)")
        println(io, "run_time\t$(RUN_T_058)")
        println(io, "lyap_k\t$(LYAP_K_058)")
        println(io, "time_unit_seconds\t$(SECONDS_PER_TIME_UNIT_058)")
        println(io, "qr_dt\t$(QR_DT_058)")
        println(io, "record_dt\t$(RECORD_DT_058)")
        println(io, "abstol\t$(ABSTOL_058)")
        println(io, "reltol\t$(RELTOL_058)")
        println(io, "maxiters\t$(MAXITERS_058)")
        println(io, "Vddot_dt\t$(VDDOT_DT_058)")
        println(io, "V_return_threshold\t$(V_RETURN_THRESHOLD_058)")
        println(io, "V_equilibria\t$(join(string.(V_eqs), ","))")
        println(io, "V_sd\t$(V_sd)")
        println(io, "initial_state\t$(join(string.(u0), ","))")
        println(io, "post_transient_state\t$(join(string.(post_transient_state), ","))")
        println(io, "final_state\t$(join(string.(final_state), ","))")
        println(io, "retcode\t$(retcode)")
        println(io, "transient_runtime_seconds\t$(transient_runtime)")
        println(io, "main_runtime_seconds\t$(main_runtime)")
        println(io, "n_convergence_rows\t$(length(rows))")
        println(io, "n_symbols\t$(length(recorder.scs))")
        if !isempty(rows)
            last_row = rows[end]
            println(io, "final_lambda1_nats_per_second\t$(last_row.lambda1_nats_per_second)")
            println(io, "final_lambda2_nats_per_second\t$(last_row.lambda2_nats_per_second)")
            println(io, "final_positive_lyap_sum_nats_per_second\t$(last_row.positive_lyap_sum_nats_per_second)")
            println(io, "final_lambda1_bits_per_second\t$(last_row.lambda1_bits_per_second)")
            println(io, "final_lambda2_bits_per_second\t$(last_row.lambda2_bits_per_second)")
            println(io, "final_positive_lyap_sum_bits_per_second\t$(last_row.positive_lyap_sum_bits_per_second)")
            println(io, "final_lz76_nats_per_second\t$(last_row.lz76_nats_per_second)")
            println(io, "final_lz76_bits_per_second\t$(last_row.lz76_bits_per_second)")
            println(io, "final_lz76_alphabet_units_per_second\t$(last_row.lz76_alphabet_units_per_second)")
            println(io, "final_lz76_alphabet_size\t$(last_row.lz76_alphabet_size)")
            println(io, "final_lz76_phrase_count\t$(last_row.lz76_phrase_count)")
        end
    end
end

function plot_convergence_058(rows::Vector{ConvergenceRow058})
    times_s = [row.t * SECONDS_PER_TIME_UNIT_058 for row in rows]
    λ1_nats = [row.lambda1_nats_per_second for row in rows]
    λ2_nats = [row.lambda2_nats_per_second for row in rows]
    positive_sum_nats = [row.positive_lyap_sum_nats_per_second for row in rows]
    lz_nats = [row.lz76_nats_per_second for row in rows]
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
    lines!(ax, times_s, λ1_nats; color=:black, linewidth=3.0, label="lambda1")
    if LYAP_K_058 >= 2
        lines!(ax, times_s, λ2_nats; color=:gray35, linewidth=2.6, label="lambda2")
        lines!(ax, times_s, positive_sum_nats; color=:purple4, linewidth=3.2, label="positive Lyapunov sum")
    end
    finite_lz = isfinite.(lz_nats)
    if any(finite_lz)
        lines!(ax, times_s[finite_lz], lz_nats[finite_lz]; color=:red, linewidth=3.2, label="Abramov LZ76")
    end
    axislegend(ax; position=:rt, labelsize=22, framevisible=false)
    save(PLOT_PATH_058, fig, px_per_unit=2)
    return PLOT_PATH_058
end

function main_058()
    p = build_params_058()
    V_eqs = find_equilibria_058(p)
    length(V_eqs) >= 3 || error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
    V_sd = Float64(V_eqs[3])
    u0 = initial_state_058(p)
    u0_aug = augmented_state_058(u0, p, 0.0)

    @info "Starting transient" transient_time=TRANSIENT_T_058 run_time=RUN_T_058 delta_x=DELTA_X_058 delta_ca=DELTA_CA_058 g_h=G_H_058
    transient_runtime = @elapsed post_transient_state, transient_reached = run_transient_058(p, u0_aug)
    @info "Transient complete" transient_reached transient_runtime post_transient_state

    @info "Starting main run" run_time=RUN_T_058 qr_dt=QR_DT_058 record_dt=RECORD_DT_058
    local recorder
    local rows
    local final_state
    local retcode
    main_runtime = @elapsed begin
        recorder, rows, final_state, retcode = run_main_058(p, post_transient_state, V_sd)
    end
    @info "Main run complete" retcode main_runtime n_symbols=length(recorder.scs) n_rows=length(rows)

    write_events_058(recorder)
    write_convergence_058(rows)
    write_metadata_058(p, V_eqs, V_sd, u0, post_transient_state, final_state, transient_runtime, main_runtime, retcode, rows, recorder)
    plot_convergence_058(rows)

    @info "Wrote outputs" events=EVENTS_PATH_058 convergence=CONVERGENCE_PATH_058 metadata=METADATA_PATH_058 plot=PLOT_PATH_058
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_058()
end
