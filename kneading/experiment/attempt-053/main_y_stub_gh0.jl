using Pkg

const ATTEMPT53_ROOT = @__DIR__
const REPO_ROOT_053 = normpath(joinpath(ATTEMPT53_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_053)

using CairoMakie
using ChaosTools
using Colors
using Dates
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using StaticArrays

include(joinpath(REPO_ROOT_053, "models", "SiN.jl"))
const Plant053 = SiN

env_int_053(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))
env_float_053(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

const NX_053 = env_int_053("ATTEMPT053_NX", 200)
const NY_053 = env_int_053("ATTEMPT053_NY", 200)
const DELTA_X_MIN_053 = env_float_053("ATTEMPT053_DELTA_X_MIN", -1.5)
const DELTA_X_MAX_053 = env_float_053("ATTEMPT053_DELTA_X_MAX", -0.5)
const DELTA_CA_MIN_053 = env_float_053("ATTEMPT053_DELTA_CA_MIN", -45.0)
const DELTA_CA_MAX_053 = env_float_053("ATTEMPT053_DELTA_CA_MAX", -20.0)
const DELTA_XS_053 = collect(range(DELTA_X_MIN_053, DELTA_X_MAX_053; length=NX_053))
const DELTA_CAS_053 = collect(range(DELTA_CA_MIN_053, DELTA_CA_MAX_053; length=NY_053))

const TAU_Y_053 = env_float_053("ATTEMPT053_TAU_Y", 2.0e4)
const LYAP_K_053 = env_int_053("ATTEMPT053_LYAP_K", 5)
const LYAP_TMAX_053 = env_float_053("ATTEMPT053_LYAP_TMAX", 1.0e5)
const LYAP_TTR_053 = env_float_053("ATTEMPT053_LYAP_TTR", 5.0e3)
const LYAP_DT_053 = env_float_053("ATTEMPT053_LYAP_DT", 10.0)
const LYAP_MIN_TIME_053 = env_float_053("ATTEMPT053_LYAP_MIN_TIME", 3.0e4)
const LYAP_CHECK_INTERVAL_053 = env_float_053("ATTEMPT053_LYAP_CHECK_INTERVAL", 5.0e3)
const LYAP_CONV_ATOL_053 = env_float_053("ATTEMPT053_LYAP_CONV_ATOL", 1.0e-4)
const LYAP_CONV_RTOL_053 = env_float_053("ATTEMPT053_LYAP_CONV_RTOL", 1.0e-2)
const LYAP_DIM_CONV_ATOL_053 = env_float_053("ATTEMPT053_LYAP_DIM_CONV_ATOL", 5.0e-3)
const LYAP_STABLE_CHECKS_053 = env_int_053("ATTEMPT053_LYAP_STABLE_CHECKS", 3)
const LYAP_ABSTOL_053 = env_float_053("ATTEMPT053_LYAP_ABSTOL", 1.0e-8)
const LYAP_RELTOL_053 = env_float_053("ATTEMPT053_LYAP_RELTOL", 1.0e-8)
const LYAP_ALG_053 = Tsit5()

const OUTPUT_TAG_053 = get(ENV, "ATTEMPT053_OUTPUT_TAG", "grid200_lyapdim_tmax1e5_gh0p000_y_stub")
const OUTPUT_DIR_053 = get(ENV, "ATTEMPT053_OUTPUT_DIR", ATTEMPT53_ROOT)
const COLUMN_DIR_053 = joinpath(OUTPUT_DIR_053, "$(OUTPUT_TAG_053)_columns")
const RESULTS_PATH_053 = joinpath(OUTPUT_DIR_053, "$(OUTPUT_TAG_053)_results.tsv")
const PLOT_PATH_053 = joinpath(OUTPUT_DIR_053, "$(OUTPUT_TAG_053)_dimension.png")
const SUMMARY_PATH_053 = joinpath(OUTPUT_DIR_053, "$(OUTPUT_TAG_053)_summary.txt")

const PLOT_WIDTH_053 = env_int_053("ATTEMPT053_PLOT_WIDTH", 1800)
const PLOT_HEIGHT_053 = env_int_053("ATTEMPT053_PLOT_HEIGHT", 1300)
const PLOT_PX_PER_UNIT_053 = env_float_053("ATTEMPT053_PLOT_PX_PER_UNIT", 2.0)
const AXIS_LABEL_SIZE_053 = env_int_053("ATTEMPT053_AXIS_LABEL_SIZE", 34)
const AXIS_TITLE_SIZE_053 = env_int_053("ATTEMPT053_AXIS_TITLE_SIZE", 38)
const TICK_LABEL_SIZE_053 = env_int_053("ATTEMPT053_TICK_LABEL_SIZE", 24)
const DELTA_CA_TICK_STEP_053 = env_float_053("ATTEMPT053_DELTA_CA_TICK_STEP", 5.0)
const DELTA_X_TICK_STEP_053 = env_float_053("ATTEMPT053_DELTA_X_TICK_STEP", 0.1)
const DIM_COLOR_MODE_053 = get(ENV, "ATTEMPT053_DIM_COLOR_MODE", "linear")
const DIM_COLOR_LOGIT_ATAN_SCALE_053 = env_float_053("ATTEMPT053_DIM_COLOR_LOGIT_ATAN_SCALE", 3.0)

struct LyapResult053
    delta_x::Float64
    delta_ca::Float64
    dimension::Float64
    lambda::Vector{Float64}
    integrated_time::Float64
    transient_time::Float64
    converged::Bool
    checks::Int
    spectrum_delta_last::Float64
    dimension_delta_last::Float64
    runtime::Float64
    status::String
    error::String
end

function build_params_053(delta_x::Float64, delta_ca::Float64)::SVector{18, Float64}
    p = Vector{Float64}(Plant053.default_params)
    p[4] = 0.0
    p[15] = TAU_Y_053
    p[17] = delta_x
    p[18] = delta_ca
    return SVector{18, Float64}(p)
end

function initial_state_reduced_053()
    u = Plant053.default_state
    return Float64[u[1], u[3], u[4], u[5], u[6]]
end

function melibe_gh0_y_stub!(du, u, p, t)
    x, n, h, Ca, V = u
    du[1] = Plant053.dx(p, x, V)
    du[2] = Plant053.dn(n, V)
    du[3] = Plant053.dh(h, V)
    du[4] = Plant053.dCa(p, Ca, x, V)
    du[5] = Plant053.dV(p, x, 0.0, n, h, Ca, V)
    return nothing
end

function kaplan_yorke_dimension_053(λs::AbstractVector{<:Real})::Float64
    if isempty(λs) || any(!isfinite, λs)
        return NaN
    end

    λ = sort(collect(Float64, λs); rev=true)
    total = 0.0
    for j in eachindex(λ)
        next_total = total + λ[j]
        if next_total < 0
            if j == 1
                return max(0.0, total / abs(λ[j]))
            end
            return (j - 1) + total / abs(λ[j])
        end
        total = next_total
    end
    return Float64(length(λ))
end

function qr_step_053!(tands::TangentDynamicalSystem, k::Int)
    deviations = Matrix(current_deviations(tands))
    factorization = qr(deviations)
    Q = Matrix(factorization.Q)
    R = factorization.R
    set_deviations!(tands, Q[:, 1:k])

    logs = Vector{Float64}(undef, k)
    @inbounds for i in 1:k
        logs[i] = log(abs(R[i, i]))
    end
    return logs
end

function orient_deviations_053!(tands::TangentDynamicalSystem, k::Int)
    LYAP_TTR_053 <= 0 && return
    start_time = current_time(tands)
    while current_time(tands) < start_time + LYAP_TTR_053 - eps(Float64)
        dt = min(LYAP_DT_053, start_time + LYAP_TTR_053 - current_time(tands))
        step!(tands, dt)
        successful_step(tands) || error("Transient Lyapunov tangent integration failed.")
        qr_step_053!(tands, k)
    end
end

function lyapunov_dimension_053(delta_x::Float64, delta_ca::Float64)::LyapResult053
    local raw_result::LyapResult053
    runtime = @elapsed begin
        p = build_params_053(delta_x, delta_ca)
        u0 = initial_state_reduced_053()
        ds = CoupledODEs(
            melibe_gh0_y_stub!,
            copy(u0),
            p;
            diffeq=(alg=LYAP_ALG_053, abstol=LYAP_ABSTOL_053, reltol=LYAP_RELTOL_053),
        )
        k = min(LYAP_K_053, dimension(ds))
        tands = TangentDynamicalSystem(ds; k)
        reinit!(tands, u0)
        orient_deviations_053!(tands, k)

        λsum = zeros(Float64, k)
        previous_λ = nothing
        previous_dim = NaN
        stable_checks = 0
        checks = 0
        converged = false
        spectrum_delta_last = NaN
        dimension_delta_last = NaN
        t0 = current_time(tands)
        next_check = LYAP_CHECK_INTERVAL_053

        while current_time(tands) < t0 + LYAP_TMAX_053 - eps(Float64)
            dt = min(LYAP_DT_053, t0 + LYAP_TMAX_053 - current_time(tands))
            step!(tands, dt)
            successful_step(tands) || error("Lyapunov tangent integration failed.")
            λsum .+= qr_step_053!(tands, k)

            elapsed = current_time(tands) - t0
            if elapsed >= next_check - eps(Float64) || elapsed >= LYAP_TMAX_053 - eps(Float64)
                checks += 1
                λnow = sort(λsum ./ elapsed; rev=true)
                dim_now = kaplan_yorke_dimension_053(λnow)
                if elapsed >= LYAP_MIN_TIME_053 && !isnothing(previous_λ)
                    scale = max(maximum(abs.(λnow)), maximum(abs.(previous_λ)), 1.0e-12)
                    spectrum_delta_last = maximum(abs.(λnow .- previous_λ))
                    dimension_delta_last = abs(dim_now - previous_dim)
                    spectrum_tol = max(LYAP_CONV_ATOL_053, LYAP_CONV_RTOL_053 * scale)
                    if spectrum_delta_last <= spectrum_tol && dimension_delta_last <= LYAP_DIM_CONV_ATOL_053
                        stable_checks += 1
                    else
                        stable_checks = 0
                    end
                    if stable_checks >= LYAP_STABLE_CHECKS_053
                        converged = true
                        break
                    end
                end
                previous_λ = λnow
                previous_dim = dim_now
                next_check += LYAP_CHECK_INTERVAL_053
            end
        end

        elapsed = current_time(tands) - t0
        λ = sort(λsum ./ elapsed; rev=true)
        raw_result = LyapResult053(delta_x, delta_ca, kaplan_yorke_dimension_053(λ), λ, elapsed, LYAP_TTR_053, converged, checks, spectrum_delta_last, dimension_delta_last, 0.0, "ok", "")
    end

    return LyapResult053(
        raw_result.delta_x,
        raw_result.delta_ca,
        raw_result.dimension,
        raw_result.lambda,
        raw_result.integrated_time,
        raw_result.transient_time,
        raw_result.converged,
        raw_result.checks,
        raw_result.spectrum_delta_last,
        raw_result.dimension_delta_last,
        runtime,
        raw_result.status,
        raw_result.error,
    )
end

function lyapunov_dimension_safe_053(delta_x::Float64, delta_ca::Float64)::LyapResult053
    try
        return lyapunov_dimension_053(delta_x, delta_ca)
    catch err
        return LyapResult053(delta_x, delta_ca, NaN, Float64[], 0.0, LYAP_TTR_053, false, 0, NaN, NaN, 0.0, "error", sprint(showerror, err))
    end
end

function complete_column_053(path::String)::Bool
    isfile(path) || return false
    rows = countlines(path) - 1
    return rows == length(DELTA_XS_053)
end

function write_column_053(col_idx::Int, delta_ca::Float64)
    mkpath(COLUMN_DIR_053)
    path = joinpath(COLUMN_DIR_053, @sprintf("column_%04d.tsv", col_idx))
    if complete_column_053(path)
        @printf("Skipping completed column %d/%d (Delta Ca=%.6f)\n", col_idx, length(DELTA_CAS_053), delta_ca)
        return
    end

    temp_path = path * ".tmp"
    open(temp_path, "w") do io
        println(io, "delta_x\tdelta_ca\tlyap_dim\tlambda1\tlambda2\tlambda3\tlambda4\tlambda5\tlambda6\tintegrated_time\ttransient_time\tconverged\tchecks\tspectrum_delta_last\tdimension_delta_last\truntime_seconds\tstatus\terror")
        column_runtime = @elapsed begin
            ok_count = 0
            converged_count = 0
            for delta_x in DELTA_XS_053
                result = lyapunov_dimension_safe_053(delta_x, delta_ca)
                result.status == "ok" && (ok_count += 1)
                result.converged && (converged_count += 1)
                λ = fill(NaN, 6)
                for i in 1:min(5, length(result.lambda))
                    λ[i] = result.lambda[i]
                end
                error_text = isempty(result.error) ? "none" : replace(result.error, '\t' => ' ', '\n' => ' ')
                println(io, join((
                    @sprintf("%.9f", result.delta_x),
                    @sprintf("%.9f", result.delta_ca),
                    @sprintf("%.12g", result.dimension),
                    (@sprintf("%.12g", λ[1])),
                    (@sprintf("%.12g", λ[2])),
                    (@sprintf("%.12g", λ[3])),
                    (@sprintf("%.12g", λ[4])),
                    (@sprintf("%.12g", λ[5])),
                    (@sprintf("%.12g", λ[6])),
                    @sprintf("%.6f", result.integrated_time),
                    @sprintf("%.6f", result.transient_time),
                    result.converged ? "true" : "false",
                    string(result.checks),
                    @sprintf("%.12g", result.spectrum_delta_last),
                    @sprintf("%.12g", result.dimension_delta_last),
                    @sprintf("%.6f", result.runtime),
                    result.status,
                    error_text,
                ), '\t'))
            end
            @printf("Saved column %d/%d (Delta Ca=%.6f) with %d/%d ok, %d converged\n", col_idx, length(DELTA_CAS_053), delta_ca, ok_count, length(DELTA_XS_053), converged_count)
        end
        @printf("Column %d runtime: %.2f s\n", col_idx, column_runtime)
    end
    mv(temp_path, path; force=true)
end

function run_scan_053()
    mkpath(OUTPUT_DIR_053)
    mkpath(COLUMN_DIR_053)
    @printf("Running attempt-053 reduced gh=0 Lyapunov dimension scan with y stubbed out.\n")
    @printf("Grid: %d Delta Ca x %d Delta x\n", length(DELTA_CAS_053), length(DELTA_XS_053))
    @printf("Delta Ca range: [%.6f, %.6f]\n", first(DELTA_CAS_053), last(DELTA_CAS_053))
    @printf("Delta x range: [%.6f, %.6f]\n", first(DELTA_XS_053), last(DELTA_XS_053))
    @printf("g_h=0, tau_y ignored by active tangent, stored tau_y=%.12g\n", TAU_Y_053)
    @printf("Lyapunov: Ttr=%.3f, Tmax=%.3f, dt=%.3f, min_time=%.3f, check_interval=%.3f\n", LYAP_TTR_053, LYAP_TMAX_053, LYAP_DT_053, LYAP_MIN_TIME_053, LYAP_CHECK_INTERVAL_053)
    @printf("Threads: %d\n", Threads.nthreads())
    flush(stdout)

    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_053)
        write_column_053(col_idx, DELTA_CAS_053[col_idx])
    end
end

function merge_columns_053()
    mkpath(OUTPUT_DIR_053)
    open(RESULTS_PATH_053, "w") do out
        header_written = false
        for col_idx in eachindex(DELTA_CAS_053)
            path = joinpath(COLUMN_DIR_053, @sprintf("column_%04d.tsv", col_idx))
            isfile(path) || error("Missing column file: $(path)")
            open(path, "r") do input
                header = readline(input)
                if !header_written
                    println(out, header)
                    header_written = true
                end
                for line in eachline(input)
                    println(out, line)
                end
            end
        end
    end
    return RESULTS_PATH_053
end

function fixed_ticks_053(values::AbstractVector{<:Real}, fmt::String, step::Real)
    lo = ceil(minimum(values) / step) * step
    hi = floor(maximum(values) / step) * step
    ticks = collect(lo:step:hi)
    formatter = Printf.Format(fmt)
    return (ticks, [Printf.format(formatter, tick) for tick in ticks])
end

function dimension_color_053(dim::Float64)
    if !isfinite(dim)
        return RGBAf(0.88, 0.88, 0.88, 1.0)
    end

    bases = RGBf[
        RGBf(0.0, 0.0, 0.0),
        RGBf(0.02, 0.20, 1.0),
        RGBf(1.0, 0.48, 0.0),
        RGBf(0.0, 0.62, 0.20),
        RGBf(0.55, 0.0, 0.78),
    ]
    level = clamp(floor(Int, dim), 0, length(bases) - 1)
    frac = clamp(dim - level, 0.0, 1.0)
    if DIM_COLOR_MODE_053 == "logit_integer" && level >= 2
        frac = clamp(frac, eps(Float64), 1.0 - eps(Float64))
        logit_frac = log(frac / (1.0 - frac))
        frac = clamp(0.5 + atan(logit_frac / DIM_COLOR_LOGIT_ATAN_SCALE_053) / pi, 0.0, 1.0)
    end
    strength = 0.18 + 0.82 * frac
    base = bases[level + 1]
    rgb = RGBf((1 - strength) + strength * base.r, (1 - strength) + strength * base.g, (1 - strength) + strength * base.b)
    return RGBAf(rgb.r, rgb.g, rgb.b, 1.0)
end

function read_dimension_grid_053(results_path::String)
    grid = fill(RGBAf(0.88, 0.88, 0.88, 1.0), length(DELTA_CAS_053), length(DELTA_XS_053))
    dimensions = fill(NaN, length(DELTA_CAS_053), length(DELTA_XS_053))
    ok = falses(length(DELTA_CAS_053), length(DELTA_XS_053))
    converged = falses(length(DELTA_CAS_053), length(DELTA_XS_053))

    open(results_path, "r") do io
        header = split(readline(io), '\t')
        lookup = Dict(name => idx for (idx, name) in enumerate(header))
        for line in eachline(io)
            isempty(line) && continue
            fields = split(line, '\t')
            fields[lookup["status"]] == "ok" || continue
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            ca_idx = argmin(abs.(DELTA_CAS_053 .- delta_ca))
            x_idx = argmin(abs.(DELTA_XS_053 .- delta_x))
            dim = parse(Float64, fields[lookup["lyap_dim"]])
            dimensions[ca_idx, x_idx] = dim
            grid[ca_idx, x_idx] = dimension_color_053(dim)
            ok[ca_idx, x_idx] = true
            converged[ca_idx, x_idx] = fields[lookup["converged"]] == "true"
        end
    end

    return (; grid, dimensions, ok, converged)
end

function plot_dimension_053(results_path::String=RESULTS_PATH_053)
    data = read_dimension_grid_053(results_path)
    fig = Figure(size=(PLOT_WIDTH_053, PLOT_HEIGHT_053), backgroundcolor=:white)
    ax = Axis(
        fig[1, 1],
        xlabel="Delta Ca",
        ylabel="Delta x",
        title="Lyapunov dimension, g_h = 0, y stubbed",
        xlabelsize=AXIS_LABEL_SIZE_053,
        ylabelsize=AXIS_LABEL_SIZE_053,
        titlesize=AXIS_TITLE_SIZE_053,
        xticklabelsize=TICK_LABEL_SIZE_053,
        yticklabelsize=TICK_LABEL_SIZE_053,
    )
    heatmap!(ax, DELTA_CAS_053, DELTA_XS_053, data.grid)
    ax.xticks = fixed_ticks_053(DELTA_CAS_053, "%.0f", DELTA_CA_TICK_STEP_053)
    ax.yticks = fixed_ticks_053(DELTA_XS_053, "%.1f", DELTA_X_TICK_STEP_053)
    xlims!(ax, minimum(DELTA_CAS_053), maximum(DELTA_CAS_053))
    ylims!(ax, minimum(DELTA_XS_053), maximum(DELTA_XS_053))

    labels = ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5]"]
    colors = [dimension_color_053(i + 0.95) for i in 0:4]
    elements = [MarkerElement(marker=:rect, color=colors[i], markersize=22) for i in eachindex(colors)]
    Legend(fig[1, 2], elements, labels, "dim interval"; tellheight=false, tellwidth=true)

    save(PLOT_PATH_053, fig; px_per_unit=PLOT_PX_PER_UNIT_053)

    ok_count = count(data.ok)
    converged_count = count(data.converged)
    dims = data.dimensions[data.ok]
    open(SUMMARY_PATH_053, "w") do io
        println(io, "output_tag\t$(OUTPUT_TAG_053)")
        println(io, "generated_at\t$(Dates.now())")
        println(io, "grid\t$(length(DELTA_CAS_053))x$(length(DELTA_XS_053))")
        println(io, "delta_ca_range\t$(first(DELTA_CAS_053))\t$(last(DELTA_CAS_053))")
        println(io, "delta_x_range\t$(first(DELTA_XS_053))\t$(last(DELTA_XS_053))")
        println(io, "g_h\t0.0")
        println(io, "y_stubbed\ttrue")
        println(io, "active_state_order\tx\tn\th\tCa\tV")
        println(io, "tau_y\t$(TAU_Y_053)")
        println(io, "lyap_ttr\t$(LYAP_TTR_053)")
        println(io, "lyap_tmax\t$(LYAP_TMAX_053)")
        println(io, "lyap_dt\t$(LYAP_DT_053)")
        println(io, "lyap_min_time\t$(LYAP_MIN_TIME_053)")
        println(io, "lyap_check_interval\t$(LYAP_CHECK_INTERVAL_053)")
        println(io, "lyap_conv_atol\t$(LYAP_CONV_ATOL_053)")
        println(io, "lyap_conv_rtol\t$(LYAP_CONV_RTOL_053)")
        println(io, "lyap_dim_conv_atol\t$(LYAP_DIM_CONV_ATOL_053)")
        println(io, "lyap_stable_checks\t$(LYAP_STABLE_CHECKS_053)")
        println(io, "dimension_color_mode\t$(DIM_COLOR_MODE_053)")
        println(io, "dimension_color_logit_atan_scale\t$(DIM_COLOR_LOGIT_ATAN_SCALE_053)")
        println(io, "ok_count\t$(ok_count)")
        println(io, "converged_count\t$(converged_count)")
        println(io, "dimension_min\t$(isempty(dims) ? NaN : minimum(dims))")
        println(io, "dimension_max\t$(isempty(dims) ? NaN : maximum(dims))")
        println(io, "dimension_mean\t$(isempty(dims) ? NaN : sum(dims) / length(dims))")
        println(io, "plot_path\t$(PLOT_PATH_053)")
        println(io, "results_path\t$(results_path)")
    end

    println("Saved plot to $(PLOT_PATH_053)")
    println("Saved summary to $(SUMMARY_PATH_053)")
    println("ok_count=$(ok_count)")
    println("converged_count=$(converged_count)")
    isempty(dims) || println(@sprintf("dimension range: %.6f to %.6f", minimum(dims), maximum(dims)))
end

function main_053()
    if haskey(ENV, "ATTEMPT053_PLOT_ONLY_RESULTS")
        plot_dimension_053(ENV["ATTEMPT053_PLOT_ONLY_RESULTS"])
        return
    end

    run_runtime = @elapsed run_scan_053()
    merge_runtime = @elapsed merge_columns_053()
    plot_runtime = @elapsed plot_dimension_053(RESULTS_PATH_053)

    open(SUMMARY_PATH_053, "a") do io
        println(io, "scan_runtime_seconds\t$(run_runtime)")
        println(io, "merge_runtime_seconds\t$(merge_runtime)")
        println(io, "plot_runtime_seconds\t$(plot_runtime)")
    end
    println(@sprintf("scan_runtime_seconds=%.3f", run_runtime))
    println(@sprintf("merge_runtime_seconds=%.3f", merge_runtime))
    println(@sprintf("plot_runtime_seconds=%.3f", plot_runtime))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_053()
end
