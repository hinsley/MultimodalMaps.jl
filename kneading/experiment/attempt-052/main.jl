using Pkg

const ATTEMPT52_ROOT = @__DIR__
const REPO_ROOT_052 = normpath(joinpath(ATTEMPT52_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_052)

using CairoMakie
using ChaosTools
using Colors
using Dates
using LinearAlgebra
using OrdinaryDiffEq
using Printf
using StaticArrays

include(joinpath(REPO_ROOT_052, "models", "SiN.jl"))
const Plant = SiN

env_int(name::String, default::Int) = parse(Int, get(ENV, name, string(default)))
env_float(name::String, default::Float64) = parse(Float64, get(ENV, name, string(default)))

const NX_052 = env_int("ATTEMPT052_NX", 1000)
const NY_052 = env_int("ATTEMPT052_NY", 1000)
const DELTA_X_MIN_052 = env_float("ATTEMPT052_DELTA_X_MIN", -1.5)
const DELTA_X_MAX_052 = env_float("ATTEMPT052_DELTA_X_MAX", -0.5)
const DELTA_CA_MIN_052 = env_float("ATTEMPT052_DELTA_CA_MIN", -45.0)
const DELTA_CA_MAX_052 = env_float("ATTEMPT052_DELTA_CA_MAX", -20.0)
const DELTA_XS_052 = collect(range(DELTA_X_MIN_052, DELTA_X_MAX_052; length=NX_052))
const DELTA_CAS_052 = collect(range(DELTA_CA_MIN_052, DELTA_CA_MAX_052; length=NY_052))

const G_H_052 = env_float("ATTEMPT052_G_H", 1.0e-3)
const TAU_Y_052 = env_float("ATTEMPT052_TAU_Y", 2.0e4)
const LYAP_K_052 = env_int("ATTEMPT052_LYAP_K", 6)
const LYAP_TMAX_052 = env_float("ATTEMPT052_LYAP_TMAX", 1.0e5)
const LYAP_TTR_052 = env_float("ATTEMPT052_LYAP_TTR", 5.0e3)
const LYAP_DT_052 = env_float("ATTEMPT052_LYAP_DT", 10.0)
const LYAP_MIN_TIME_052 = env_float("ATTEMPT052_LYAP_MIN_TIME", 3.0e4)
const LYAP_CHECK_INTERVAL_052 = env_float("ATTEMPT052_LYAP_CHECK_INTERVAL", 5.0e3)
const LYAP_CONV_ATOL_052 = env_float("ATTEMPT052_LYAP_CONV_ATOL", 5.0e-4)
const LYAP_CONV_RTOL_052 = env_float("ATTEMPT052_LYAP_CONV_RTOL", 2.0e-2)
const LYAP_STABLE_CHECKS_052 = env_int("ATTEMPT052_LYAP_STABLE_CHECKS", 1)
const LYAP_ABSTOL_052 = env_float("ATTEMPT052_LYAP_ABSTOL", 1.0e-8)
const LYAP_RELTOL_052 = env_float("ATTEMPT052_LYAP_RELTOL", 1.0e-8)
const LYAP_ALG_052 = Tsit5()

const OUTPUT_TAG_052 = get(ENV, "ATTEMPT052_OUTPUT_TAG", "grid1000_lyapdim_tmax1e5_gh0p001")
const OUTPUT_DIR_052 = get(ENV, "ATTEMPT052_OUTPUT_DIR", ATTEMPT52_ROOT)
const COLUMN_DIR_052 = joinpath(OUTPUT_DIR_052, "$(OUTPUT_TAG_052)_columns")
const RESULTS_PATH_052 = joinpath(OUTPUT_DIR_052, "$(OUTPUT_TAG_052)_results.tsv")
const PLOT_PATH_052 = joinpath(OUTPUT_DIR_052, "$(OUTPUT_TAG_052)_dimension.png")
const SUMMARY_PATH_052 = joinpath(OUTPUT_DIR_052, "$(OUTPUT_TAG_052)_summary.txt")

const PLOT_WIDTH_052 = env_int("ATTEMPT052_PLOT_WIDTH", 1600)
const PLOT_HEIGHT_052 = env_int("ATTEMPT052_PLOT_HEIGHT", 1200)
const PLOT_PX_PER_UNIT_052 = env_float("ATTEMPT052_PLOT_PX_PER_UNIT", 2.0)
const AXIS_LABEL_SIZE_052 = env_int("ATTEMPT052_AXIS_LABEL_SIZE", 34)
const AXIS_TITLE_SIZE_052 = env_int("ATTEMPT052_AXIS_TITLE_SIZE", 38)
const TICK_LABEL_SIZE_052 = env_int("ATTEMPT052_TICK_LABEL_SIZE", 24)
const DELTA_CA_TICK_STEP_052 = env_float("ATTEMPT052_DELTA_CA_TICK_STEP", 5.0)
const DELTA_X_TICK_STEP_052 = env_float("ATTEMPT052_DELTA_X_TICK_STEP", 0.1)

struct LyapResult052
    delta_x::Float64
    delta_ca::Float64
    dimension::Float64
    lambda::Vector{Float64}
    integrated_time::Float64
    transient_time::Float64
    converged::Bool
    checks::Int
    runtime::Float64
    status::String
    error::String
end

function build_params_052(delta_x::Float64, delta_ca::Float64)::SVector{18, Float64}
    p = Vector{Float64}(Plant.default_params)
    p[4] = G_H_052
    p[15] = TAU_Y_052
    p[17] = delta_x
    p[18] = delta_ca
    return SVector{18, Float64}(p)
end

function initial_state_052()
    return Vector{Float64}(Plant.default_state)
end

function kaplan_yorke_dimension_052(λs::AbstractVector{<:Real})::Float64
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

function qr_step_052!(tands::TangentDynamicalSystem, k::Int)
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

function orient_deviations_052!(tands::TangentDynamicalSystem, k::Int)
    LYAP_TTR_052 <= 0 && return
    start_time = current_time(tands)
    while current_time(tands) < start_time + LYAP_TTR_052 - eps(Float64)
        dt = min(LYAP_DT_052, start_time + LYAP_TTR_052 - current_time(tands))
        step!(tands, dt)
        successful_step(tands) || error("Transient Lyapunov tangent integration failed.")
        qr_step_052!(tands, k)
    end
end

function lyapunov_dimension_052(delta_x::Float64, delta_ca::Float64)::LyapResult052
    local raw_result::LyapResult052
    runtime = @elapsed begin
        p = build_params_052(delta_x, delta_ca)
        u0 = initial_state_052()
        ds = CoupledODEs(
            Plant.melibeNew!,
            copy(u0),
            p;
            diffeq=(alg=LYAP_ALG_052, abstol=LYAP_ABSTOL_052, reltol=LYAP_RELTOL_052),
        )
        k = min(LYAP_K_052, dimension(ds))
        tands = TangentDynamicalSystem(ds; k)
        reinit!(tands, u0)
        orient_deviations_052!(tands, k)

        λsum = zeros(Float64, k)
        previous_λ = nothing
        stable_checks = 0
        checks = 0
        converged = false
        t0 = current_time(tands)
        next_check = LYAP_CHECK_INTERVAL_052

        while current_time(tands) < t0 + LYAP_TMAX_052 - eps(Float64)
            dt = min(LYAP_DT_052, t0 + LYAP_TMAX_052 - current_time(tands))
            step!(tands, dt)
            successful_step(tands) || error("Lyapunov tangent integration failed.")
            λsum .+= qr_step_052!(tands, k)

            elapsed = current_time(tands) - t0
            if elapsed >= next_check - eps(Float64) || elapsed >= LYAP_TMAX_052 - eps(Float64)
                checks += 1
                λnow = sort(λsum ./ elapsed; rev=true)
                if elapsed >= LYAP_MIN_TIME_052 && !isnothing(previous_λ)
                    scale = max(maximum(abs.(λnow)), maximum(abs.(previous_λ)), 1.0e-12)
                    delta = maximum(abs.(λnow .- previous_λ))
                    tol = max(LYAP_CONV_ATOL_052, LYAP_CONV_RTOL_052 * scale)
                    if delta <= tol
                        stable_checks += 1
                    else
                        stable_checks = 0
                    end
                    if stable_checks >= LYAP_STABLE_CHECKS_052
                        converged = true
                        break
                    end
                end
                previous_λ = λnow
                next_check += LYAP_CHECK_INTERVAL_052
            end
        end

        elapsed = current_time(tands) - t0
        λ = sort(λsum ./ elapsed; rev=true)
        dim = kaplan_yorke_dimension_052(λ)
        raw_result = LyapResult052(delta_x, delta_ca, dim, λ, elapsed, LYAP_TTR_052, converged, checks, 0.0, "ok", "")
    end

    return LyapResult052(
        raw_result.delta_x,
        raw_result.delta_ca,
        raw_result.dimension,
        raw_result.lambda,
        raw_result.integrated_time,
        raw_result.transient_time,
        raw_result.converged,
        raw_result.checks,
        runtime,
        raw_result.status,
        raw_result.error,
    )
end

function lyapunov_dimension_safe_052(delta_x::Float64, delta_ca::Float64)::LyapResult052
    try
        return lyapunov_dimension_052(delta_x, delta_ca)
    catch err
        return LyapResult052(delta_x, delta_ca, NaN, Float64[], 0.0, LYAP_TTR_052, false, 0, 0.0, "error", sprint(showerror, err))
    end
end

function complete_column_052(path::String)::Bool
    isfile(path) || return false
    rows = countlines(path) - 1
    return rows == length(DELTA_XS_052)
end

function write_column_052(col_idx::Int, delta_ca::Float64)
    mkpath(COLUMN_DIR_052)
    path = joinpath(COLUMN_DIR_052, @sprintf("column_%04d.tsv", col_idx))
    if complete_column_052(path)
        @printf("Skipping completed column %d/%d (Delta Ca=%.6f)\n", col_idx, length(DELTA_CAS_052), delta_ca)
        return
    end

    temp_path = path * ".tmp"
    open(temp_path, "w") do io
        println(io, "delta_x\tdelta_ca\tlyap_dim\tlambda1\tlambda2\tlambda3\tlambda4\tlambda5\tlambda6\tintegrated_time\ttransient_time\tconverged\tchecks\truntime_seconds\tstatus\terror")
        column_runtime = @elapsed begin
            ok_count = 0
            converged_count = 0
            for delta_x in DELTA_XS_052
                result = lyapunov_dimension_safe_052(delta_x, delta_ca)
                result.status == "ok" && (ok_count += 1)
                result.converged && (converged_count += 1)
                λ = fill(NaN, 6)
                for i in 1:min(6, length(result.lambda))
                    λ[i] = result.lambda[i]
                end
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
                    @sprintf("%.6f", result.runtime),
                    result.status,
                    replace(result.error, '\t' => ' ', '\n' => ' '),
                ), '\t'))
            end
            @printf(
                "Saved column %d/%d (Delta Ca=%.6f) with %d/%d ok, %d converged\n",
                col_idx,
                length(DELTA_CAS_052),
                delta_ca,
                ok_count,
                length(DELTA_XS_052),
                converged_count,
            )
        end
        @printf("Column %d runtime: %.2f s\n", col_idx, column_runtime)
    end
    mv(temp_path, path; force=true)
end

function run_scan_052()
    mkpath(OUTPUT_DIR_052)
    mkpath(COLUMN_DIR_052)
    @printf("Running attempt-052 Lyapunov dimension scan.\n")
    @printf("Grid: %d Delta Ca x %d Delta x\n", length(DELTA_CAS_052), length(DELTA_XS_052))
    @printf("Delta Ca range: [%.6f, %.6f]\n", first(DELTA_CAS_052), last(DELTA_CAS_052))
    @printf("Delta x range: [%.6f, %.6f]\n", first(DELTA_XS_052), last(DELTA_XS_052))
    @printf("g_h=%.12g, tau_y=%.12g\n", G_H_052, TAU_Y_052)
    @printf("Lyapunov: Ttr=%.3f, Tmax=%.3f, dt=%.3f, min_time=%.3f, check_interval=%.3f\n", LYAP_TTR_052, LYAP_TMAX_052, LYAP_DT_052, LYAP_MIN_TIME_052, LYAP_CHECK_INTERVAL_052)
    @printf("Threads: %d\n", Threads.nthreads())
    flush(stdout)

    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_052)
        write_column_052(col_idx, DELTA_CAS_052[col_idx])
    end
end

function merge_columns_052()
    mkpath(OUTPUT_DIR_052)
    open(RESULTS_PATH_052, "w") do out
        header_written = false
        for col_idx in eachindex(DELTA_CAS_052)
            path = joinpath(COLUMN_DIR_052, @sprintf("column_%04d.tsv", col_idx))
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
    return RESULTS_PATH_052
end

function fixed_ticks_052(values::AbstractVector{<:Real}, fmt::String, step::Real)
    lo = ceil(minimum(values) / step) * step
    hi = floor(maximum(values) / step) * step
    ticks = collect(lo:step:hi)
    formatter = Printf.Format(fmt)
    return (ticks, [Printf.format(formatter, tick) for tick in ticks])
end

function dimension_color_052(dim::Float64)
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
    strength = 0.18 + 0.82 * frac
    base = bases[level + 1]
    rgb = RGBf(
        (1 - strength) + strength * base.r,
        (1 - strength) + strength * base.g,
        (1 - strength) + strength * base.b,
    )
    return RGBAf(rgb.r, rgb.g, rgb.b, 1.0)
end

function read_dimension_grid_052(results_path::String)
    grid = fill(RGBAf(0.88, 0.88, 0.88, 1.0), length(DELTA_CAS_052), length(DELTA_XS_052))
    dimensions = fill(NaN, length(DELTA_CAS_052), length(DELTA_XS_052))
    ok = falses(length(DELTA_CAS_052), length(DELTA_XS_052))
    converged = falses(length(DELTA_CAS_052), length(DELTA_XS_052))

    open(results_path, "r") do io
        header = split(readline(io), '\t')
        lookup = Dict(name => idx for (idx, name) in enumerate(header))
        required = ("delta_x", "delta_ca", "lyap_dim", "converged", "status")
        for name in required
            haskey(lookup, name) || error("Missing column $(name) in $(results_path)")
        end
        for line in eachline(io)
            isempty(line) && continue
            fields = split(line, '\t')
            status = fields[lookup["status"]]
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            ca_idx = argmin(abs.(DELTA_CAS_052 .- delta_ca))
            x_idx = argmin(abs.(DELTA_XS_052 .- delta_x))
            status == "ok" || continue
            dim = parse(Float64, fields[lookup["lyap_dim"]])
            dimensions[ca_idx, x_idx] = dim
            grid[ca_idx, x_idx] = dimension_color_052(dim)
            ok[ca_idx, x_idx] = true
            converged[ca_idx, x_idx] = fields[lookup["converged"]] == "true"
        end
    end

    return (; grid, dimensions, ok, converged)
end

function plot_dimension_052(results_path::String=RESULTS_PATH_052)
    data = read_dimension_grid_052(results_path)
    fig = Figure(size=(PLOT_WIDTH_052, PLOT_HEIGHT_052), backgroundcolor=:white)
    ax = Axis(
        fig[1, 1],
        xlabel="Delta Ca",
        ylabel="Delta x",
        title=@sprintf("Lyapunov dimension, g_h = %.4g", G_H_052),
        xlabelsize=AXIS_LABEL_SIZE_052,
        ylabelsize=AXIS_LABEL_SIZE_052,
        titlesize=AXIS_TITLE_SIZE_052,
        xticklabelsize=TICK_LABEL_SIZE_052,
        yticklabelsize=TICK_LABEL_SIZE_052,
    )
    heatmap!(ax, DELTA_CAS_052, DELTA_XS_052, data.grid)
    ax.xticks = fixed_ticks_052(DELTA_CAS_052, "%.0f", DELTA_CA_TICK_STEP_052)
    ax.yticks = fixed_ticks_052(DELTA_XS_052, "%.1f", DELTA_X_TICK_STEP_052)
    xlims!(ax, minimum(DELTA_CAS_052), maximum(DELTA_CAS_052))
    ylims!(ax, minimum(DELTA_XS_052), maximum(DELTA_XS_052))

    labels = ["[0,1)", "[1,2)", "[2,3)", "[3,4)", "[4,5]"]
    colors = [dimension_color_052(i + 0.95) for i in 0:4]
    elements = [MarkerElement(marker=:rect, color=colors[i], markersize=22) for i in eachindex(colors)]
    Legend(fig[1, 2], elements, labels, "dim interval"; tellheight=false, tellwidth=true)

    save(PLOT_PATH_052, fig; px_per_unit=PLOT_PX_PER_UNIT_052)

    ok_count = count(data.ok)
    converged_count = count(data.converged)
    dims = data.dimensions[data.ok]
    open(SUMMARY_PATH_052, "w") do io
        println(io, "output_tag\t$(OUTPUT_TAG_052)")
        println(io, "generated_at\t$(Dates.now())")
        println(io, "grid\t$(length(DELTA_CAS_052))x$(length(DELTA_XS_052))")
        println(io, "delta_ca_range\t$(first(DELTA_CAS_052))\t$(last(DELTA_CAS_052))")
        println(io, "delta_x_range\t$(first(DELTA_XS_052))\t$(last(DELTA_XS_052))")
        println(io, "g_h\t$(G_H_052)")
        println(io, "tau_y\t$(TAU_Y_052)")
        println(io, "lyap_ttr\t$(LYAP_TTR_052)")
        println(io, "lyap_tmax\t$(LYAP_TMAX_052)")
        println(io, "lyap_dt\t$(LYAP_DT_052)")
        println(io, "lyap_min_time\t$(LYAP_MIN_TIME_052)")
        println(io, "lyap_check_interval\t$(LYAP_CHECK_INTERVAL_052)")
        println(io, "ok_count\t$(ok_count)")
        println(io, "converged_count\t$(converged_count)")
        println(io, "dimension_min\t$(isempty(dims) ? NaN : minimum(dims))")
        println(io, "dimension_max\t$(isempty(dims) ? NaN : maximum(dims))")
        println(io, "dimension_mean\t$(isempty(dims) ? NaN : sum(dims) / length(dims))")
        println(io, "plot_path\t$(PLOT_PATH_052)")
        println(io, "results_path\t$(results_path)")
    end

    println("Saved plot to $(PLOT_PATH_052)")
    println("Saved summary to $(SUMMARY_PATH_052)")
    println("ok_count=$(ok_count)")
    println("converged_count=$(converged_count)")
    isempty(dims) || println(@sprintf("dimension range: %.6f to %.6f", minimum(dims), maximum(dims)))
end

function main_052()
    if haskey(ENV, "ATTEMPT052_PLOT_ONLY_RESULTS")
        plot_dimension_052(ENV["ATTEMPT052_PLOT_ONLY_RESULTS"])
        return
    end

    run_runtime = @elapsed run_scan_052()
    merge_runtime = @elapsed merge_columns_052()
    plot_runtime = @elapsed plot_dimension_052(RESULTS_PATH_052)

    open(SUMMARY_PATH_052, "a") do io
        println(io, "scan_runtime_seconds\t$(run_runtime)")
        println(io, "merge_runtime_seconds\t$(merge_runtime)")
        println(io, "plot_runtime_seconds\t$(plot_runtime)")
    end
    println(@sprintf("scan_runtime_seconds=%.3f", run_runtime))
    println(@sprintf("merge_runtime_seconds=%.3f", merge_runtime))
    println(@sprintf("plot_runtime_seconds=%.3f", plot_runtime))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_052()
end
