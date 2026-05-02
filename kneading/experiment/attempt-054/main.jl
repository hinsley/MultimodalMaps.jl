using Pkg

const ATTEMPT54_ROOT = @__DIR__
const ATTEMPT50_ROOT_FOR_054 = normpath(joinpath(ATTEMPT54_ROOT, "..", "attempt-050"))
const REPO_ROOT_054 = normpath(joinpath(ATTEMPT54_ROOT, "..", "..", ".."))

function alias_env_054!(dst::String, src::String, default::String)
    if !haskey(ENV, dst)
        ENV[dst] = get(ENV, src, default)
    end
end

alias_env_054!("ATTEMPT050_NX", "ATTEMPT054_NX", "200")
alias_env_054!("ATTEMPT050_NY", "ATTEMPT054_NY", "200")
alias_env_054!("ATTEMPT050_DELTA_X_MIN", "ATTEMPT054_DELTA_X_MIN", "-1.5")
alias_env_054!("ATTEMPT050_DELTA_X_MAX", "ATTEMPT054_DELTA_X_MAX", "-0.5")
alias_env_054!("ATTEMPT050_DELTA_CA_MIN", "ATTEMPT054_DELTA_CA_MIN", "-45.0")
alias_env_054!("ATTEMPT050_DELTA_CA_MAX", "ATTEMPT054_DELTA_CA_MAX", "-20.0")
alias_env_054!("ATTEMPT050_DELTA_X_TICK_STEP", "ATTEMPT054_DELTA_X_TICK_STEP", "0.1")
alias_env_054!("ATTEMPT050_DELTA_CA_TICK_STEP", "ATTEMPT054_DELTA_CA_TICK_STEP", "5.0")
alias_env_054!("ATTEMPT050_MAX_SEQ_LENGTH", "ATTEMPT054_MAX_ITER", "8")
alias_env_054!("ATTEMPT050_MAP_RESOLUTION", "ATTEMPT054_MAP_RESOLUTION", "40")

Pkg.activate(REPO_ROOT_054)
include(joinpath(ATTEMPT50_ROOT_FOR_054, "main.jl"))

using Base.Threads
using CairoMakie
using Colors
using ForwardDiff
using LinearAlgebra
using Printf
using StaticArrays

const OUTPUT_TAG_054 = get(ENV, "ATTEMPT054_OUTPUT_TAG", "grid200_tangent_ca_dotzero_tmax1e5_iter8_ystub")
const SWEEP_DIR_054 = joinpath(ATTEMPT54_ROOT, "$(OUTPUT_TAG_054)_columns")
const TANGENT_TMAX_054 = parse(Float64, get(ENV, "ATTEMPT054_TMAX", "1.0e5"))
const TANGENT_TSPAN_054 = (0.0, TANGENT_TMAX_054)
const MAX_ITER_054 = parse(Int, get(ENV, "ATTEMPT054_MAX_ITER", "8"))
const MIN_EVENT_TIME_054 = parse(Float64, get(ENV, "ATTEMPT054_MIN_EVENT_TIME", "1.0e-6"))
const REORTH_EVERY_STEP_054 = get(ENV, "ATTEMPT054_REORTH_EVERY_STEP", "1") != "0"
const TANGENT_ABSTOL_054 = parse(Float64, get(ENV, "ATTEMPT054_TANGENT_ABSTOL", "3.0e-7"))
const TANGENT_RELTOL_054 = parse(Float64, get(ENV, "ATTEMPT054_TANGENT_RELTOL", "3.0e-7"))
const CA_MIN_V_MAX_054 = parse(Float64, get(ENV, "ATTEMPT054_CA_MIN_V_MAX", "0.0"))
const T_COLOR_054 = RGBAf(1.0, 0.0, 0.0, 0.82)
const GAMMA_COLOR_054 = RGBAf(0.0, 0.23, 1.0, 0.78)
const LINEWIDTH_054 = parse(Float64, get(ENV, "ATTEMPT054_CONTOUR_LINEWIDTH", "0.45"))
const PLOT_WIDTH_054 = parse(Int, get(ENV, "ATTEMPT054_PLOT_WIDTH", "1600"))
const PLOT_HEIGHT_054 = parse(Int, get(ENV, "ATTEMPT054_PLOT_HEIGHT", "1200"))
const PLOT_PX_PER_UNIT_054 = parse(Float64, get(ENV, "ATTEMPT054_PLOT_PX_PER_UNIT", "2.0"))
const LOG_LOCK_054 = ReentrantLock()
const ACTIVE_BASIS_054 = (
    SVector{5, Float64}(0.0, 0.0, 0.0, 1.0, 0.0), # Ca
    SVector{5, Float64}(0.0, 0.0, 0.0, 0.0, 1.0), # V
    SVector{5, Float64}(1.0, 0.0, 0.0, 0.0, 0.0), # x
    SVector{5, Float64}(0.0, 1.0, 0.0, 0.0, 0.0), # n
)

mutable struct TangentRecorder054
    signs::Vector{Int}
    times::Vector{Float64}
    ca_components::Vector{Float64}
end

struct TangentScanResult054
    delta_x::Float64
    delta_ca::Float64
    T_signs::Vector{Int}
    gamma_signs::Vector{Int}
    T_times::Vector{Float64}
    gamma_times::Vector{Float64}
    T_ca_components::Vector{Float64}
    gamma_ca_components::Vector{Float64}
    T0_V::Float64
    T0_Ca::Float64
    T0_method::String
    error_message::Union{Nothing, String}
end

state5_054(u) = SVector{5, Float64}(ntuple(i -> Float64(u[i]), 5))
tangent5_054(u) = SVector{5, Float64}(ntuple(i -> Float64(u[i + 5]), 5))
state5_from_state6_054(u::SVector{6, Float64}) = SVector{5, Float64}(u[1], u[3], u[4], u[5], u[6])

function active_flow_054(state::SVector{5, T}, p, t) where {T}
    x, n, h, Ca, V = state
    return SVector{5, T}(
        Plant.dx(p, x, V),
        Plant.dn(n, V),
        Plant.dh(h, V),
        Plant.dCa(p, Ca, x, V),
        Plant.dV(p, x, zero(T), n, h, Ca, V),
    )
end

function projected_unit_tangent_054(
    state::SVector{5, Float64},
    tangent::SVector{5, Float64},
    p,
    t::Float64,
)::SVector{5, Float64}
    flow = SVector{5, Float64}(active_flow_054(state, p, t))
    flow_norm2 = dot(flow, flow)
    v = tangent
    if isfinite(flow_norm2) && flow_norm2 > 1.0e-24
        v = v - (dot(v, flow) / flow_norm2) * flow
    end
    v_norm = norm(v)
    if isfinite(v_norm) && v_norm > 1.0e-12
        return v / v_norm
    end

    for basis in ACTIVE_BASIS_054
        v = basis
        if isfinite(flow_norm2) && flow_norm2 > 1.0e-24
            v = v - (dot(v, flow) / flow_norm2) * flow
        end
        v_norm = norm(v)
        if isfinite(v_norm) && v_norm > 1.0e-12
            return v / v_norm
        end
    end

    error("Could not construct a nonzero tangent orthogonal to the flow.")
end

function reorthonormalize_augmented_054!(u, p, t)
    state = state5_054(u)
    tangent = tangent5_054(u)
    tangent_new = projected_unit_tangent_054(state, tangent, p, Float64(t))
    for i in 1:5
        u[i + 5] = tangent_new[i]
    end
    return nothing
end

function jvp_054(state::SVector{5, Float64}, tangent::SVector{5, Float64}, p, t)
    dual_zero = ForwardDiff.Dual(0.0, 1.0)
    dual_state = state .+ dual_zero .* tangent
    fdual = active_flow_054(dual_state, p, t)
    return SVector{5, Float64}(ntuple(i -> ForwardDiff.partials(fdual[i])[1], 5))
end

function tangent_augmented_rhs_054!(du, u, p, t)
    state = state5_054(u)
    tangent = tangent5_054(u)
    flow = active_flow_054(state, p, t)
    tangent_dot = jvp_054(state, tangent, p, t)
    for i in 1:5
        du[i] = flow[i]
        du[i + 5] = tangent_dot[i]
    end
    return nothing
end

function make_reorth_callback_054()
    condition(u, t, integrator) = REORTH_EVERY_STEP_054
    affect!(integrator) = reorthonormalize_augmented_054!(integrator.u, integrator.p, integrator.t)
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end

function make_ca_min_tangent_callback_054(recorder::TangentRecorder054)
    function condition(u, t, integrator)
        if t < MIN_EVENT_TIME_054
            return 1.0
        end
        state = state5_054(u)
        return active_flow_054(state, integrator.p, integrator.t)[4]
    end

    function affect!(integrator)
        state = state5_054(integrator.u)
        if state[5] > CA_MIN_V_MAX_054
            return nothing
        end
        reorthonormalize_augmented_054!(integrator.u, integrator.p, integrator.t)
        ca_component = Float64(integrator.u[9])
        push!(recorder.signs, ca_component > 0 ? 1 : (ca_component < 0 ? -1 : 0))
        push!(recorder.times, Float64(integrator.t))
        push!(recorder.ca_components, ca_component)
        if length(recorder.signs) >= MAX_ITER_054
            terminate!(integrator)
        end
        return nothing
    end

    return ContinuousCallback(condition, affect!, affect_neg! = nothing, save_positions=(false, false))
end

function tangent_minima_signs_054(
    p,
    u0::SVector{6, Float64},
    tangent0::SVector{5, Float64};
    abstol::Float64=TANGENT_ABSTOL_054,
    reltol::Float64=TANGENT_RELTOL_054,
)
    active_u0 = state5_from_state6_054(u0)
    tangent = projected_unit_tangent_054(active_u0, tangent0, p, 0.0)
    u0_aug = vcat(collect(active_u0), collect(tangent))
    recorder = TangentRecorder054(Int[], Float64[], Float64[])
    callback = CallbackSet(make_ca_min_tangent_callback_054(recorder), make_reorth_callback_054())
    prob = ODEProblem(tangent_augmented_rhs_054!, u0_aug, TANGENT_TSPAN_054, p)
    sol = solve(prob, SOLVER_010; callback=callback, abstol=abstol, reltol=reltol, save_everystep=false)
    return recorder.signs, recorder.times, recorder.ca_components, string(sol.retcode)
end

function upper_saddle_weak_stable_tangent_054(p)::SVector{5, Float64}
    V_eqs = find_equilibria(p)
    length(V_eqs) >= 3 || error("Expected at least three slow-subsystem equilibria, got $(length(V_eqs)).")
    V_eq_SD = V_eqs[3]
    Ca_eq_SD = EquilibriaSubset.Ca_null_Ca(p, V_eq_SD)
    x_eq_SD = Plant.xinf(p, V_eq_SD)
    SD_eq = @SVector [
        x_eq_SD,
        Plant.ninf(V_eq_SD),
        Plant.hinf(V_eq_SD),
        Ca_eq_SD,
        V_eq_SD,
    ]
    jac = ForwardDiff.jacobian(u -> active_flow_054(SVector{5}(u), p, 0.0), SD_eq)
    vals, vecs = eigen(Matrix(jac))
    real_stable = findall(i -> real(vals[i]) < 0 && abs(imag(vals[i])) < 1.0e-8, eachindex(vals))
    candidates = isempty(real_stable) ? findall(i -> real(vals[i]) < 0, eachindex(vals)) : real_stable
    isempty(candidates) && error("Could not find a stable eigendirection at the upper saddle.")
    ordered = sort(candidates; by=i -> real(vals[i]), rev=true)
    weak_idx = first(ordered)

    raw_vec = real.(vecs[:, weak_idx])
    tangent = SVector{5, Float64}(Tuple(Float64.(raw_vec)))
    tangent_norm = norm(tangent)
    tangent_norm > 0 || error("Weak stable eigenvector has zero norm.")
    tangent = tangent / tangent_norm

    active_values = abs.(collect(tangent))
    orient_component = argmax(active_values)
    if tangent[orient_component] < 0
        tangent = -tangent
    end
    return tangent
end

initial_T_tangent_054(p, T0::SVector{6, Float64}) =
    projected_unit_tangent_054(state5_from_state6_054(T0), ACTIVE_BASIS_054[1], p, 0.0)

function finalize_tangent_point_054(
    delta_x::Float64,
    delta_ca::Float64,
    p,
    saddle_data,
    T0::SVector{6, Float64},
    T0_method::String,
)::TangentScanResult054
    T_signs, T_times, T_ca_components, T_retcode =
        tangent_minima_signs_054(p, T0, initial_T_tangent_054(p, T0); abstol=3e-6, reltol=3e-6)
    gamma_tangent0 = upper_saddle_weak_stable_tangent_054(p)
    gamma_signs, gamma_times, gamma_ca_components, gamma_retcode =
        tangent_minima_signs_054(p, saddle_data.gamma_sd_minus0, gamma_tangent0; abstol=1e-8, reltol=1e-8)

    if length(T_signs) < MAX_ITER_054
        error("T tangent signs only reached $(length(T_signs)) / $(MAX_ITER_054) minima; retcode=$(T_retcode)")
    end
    if length(gamma_signs) < MAX_ITER_054
        error("Gamma tangent signs only reached $(length(gamma_signs)) / $(MAX_ITER_054) minima; retcode=$(gamma_retcode)")
    end

    return TangentScanResult054(
        delta_x,
        delta_ca,
        T_signs,
        gamma_signs,
        T_times,
        gamma_times,
        T_ca_components,
        gamma_ca_components,
        Float64(T0[6]),
        Float64(T0[5]),
        T0_method,
        nothing,
    )
end

function run_tangent_point_054(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::TangentScanResult054
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)

    if !isnothing(candidate_seed)
        try
            T0, iterations = initialize_T_Ca0_from_seed(
                p,
                saddle_data.x_eq_SF,
                saddle_data.gamma_sd_minus0,
                candidate_seed,
            )
            method = @sprintf("continued:%d", iterations)
            return finalize_tangent_point_054(delta_x, delta_ca, p, saddle_data, T0, method)
        catch
            # Fall through to the full remap initializer.
        end
    end

    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    return finalize_tangent_point_054(delta_x, delta_ca, p, saddle_data, T0, "full")
end

function error_result_054(delta_x::Float64, delta_ca::Float64, err)
    return TangentScanResult054(
        delta_x,
        delta_ca,
        Int[],
        Int[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        NaN,
        NaN,
        "",
        sprint(showerror, err),
    )
end

function run_tangent_point_safe_054(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::TangentScanResult054
    try
        return run_tangent_point_054(delta_x, delta_ca, candidate_seed)
    catch err
        return error_result_054(delta_x, delta_ca, err)
    end
end

make_candidate_seed_054(previous_successful::Union{Nothing, TangentScanResult054}) =
    isnothing(previous_successful) ? nothing : T0ContinuationSeed(previous_successful.T0_V, previous_successful.T0_Ca)

column_path_054(col_idx::Int) = joinpath(SWEEP_DIR_054, @sprintf("column_%04d.tsv", col_idx))

function row_is_complete_054(path::String, expected_points::Int)
    if !isfile(path)
        return false
    end
    count = 0
    open(path, "r") do io
        for _ in eachline(io)
            count += 1
        end
    end
    return count == expected_points + 1
end

format_float_vector_054(values::Vector{Float64}) =
    join((@sprintf("%.10g", value) for value in values), ",")

function write_column_054(path::String, results::Vector{TangentScanResult054})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tT_signs\tgamma_signs\tT_min_times\tgamma_min_times\tT_tangent_Ca\tgamma_tangent_Ca\tstatus")
        for result in results
            status = isnothing(result.error_message) ? "ok" : "error: " * result.error_message
            println(
                io,
                join([
                    @sprintf("%.8f", result.delta_x),
                    @sprintf("%.8f", result.delta_ca),
                    isfinite(result.T0_V) ? @sprintf("%.8f", result.T0_V) : "",
                    isfinite(result.T0_Ca) ? @sprintf("%.10f", result.T0_Ca) : "",
                    result.T0_method,
                    join(result.T_signs, ","),
                    join(result.gamma_signs, ","),
                    format_float_vector_054(result.T_times),
                    format_float_vector_054(result.gamma_times),
                    format_float_vector_054(result.T_ca_components),
                    format_float_vector_054(result.gamma_ca_components),
                    status,
                ], '\t'),
            )
        end
    end
end

function run_column_054(col_idx::Int, delta_ca::Float64, total_cols::Int, total_rows::Int)
    path = column_path_054(col_idx)
    if row_is_complete_054(path, total_rows)
        lock(LOG_LOCK_054)
        try
            @printf("Skipping completed column %d/%d (Delta Ca=%.6f)\n", col_idx, total_cols, delta_ca)
            flush(stdout)
        finally
            unlock(LOG_LOCK_054)
        end
        return
    end

    started = time()
    column_results = Vector{TangentScanResult054}(undef, total_rows)
    previous_successful = nothing
    for row_idx in length(DELTA_XS_010):-1:1
        delta_x = DELTA_XS_010[row_idx]
        result = run_tangent_point_safe_054(delta_x, delta_ca, make_candidate_seed_054(previous_successful))
        column_results[row_idx] = result
        if isnothing(result.error_message)
            previous_successful = result
        end
    end
    write_column_054(path, column_results)
    ok_count = count(result -> isnothing(result.error_message), column_results)

    lock(LOG_LOCK_054)
    try
        @printf(
            "Saved column %d/%d (Delta Ca=%.6f) with %d/%d successful points in %.2f s\n",
            col_idx,
            total_cols,
            delta_ca,
            ok_count,
            total_rows,
            time() - started,
        )
        flush(stdout)
    finally
        unlock(LOG_LOCK_054)
    end
end

function run_or_resume_columns_054()
    mkpath(SWEEP_DIR_054)
    total_cols = length(DELTA_CAS_010)
    total_rows = length(DELTA_XS_010)
    Threads.@threads :dynamic for col_idx in eachindex(DELTA_CAS_010)
        run_column_054(col_idx, DELTA_CAS_010[col_idx], total_cols, total_rows)
    end
end

function scan_column_files_054(pass_fn)
    for col_idx in eachindex(DELTA_CAS_010)
        path = column_path_054(col_idx)
        if !row_is_complete_054(path, length(DELTA_XS_010))
            error("Missing or incomplete column file: $(path)")
        end
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(split(line, '\t'))
            end
        end
    end
end

function write_merged_results_054(path::String)
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tT0_V\tT0_Ca\tT0_method\tT_signs\tgamma_signs\tT_min_times\tgamma_min_times\tT_tangent_Ca\tgamma_tangent_Ca\tstatus")
        scan_column_files_054() do fields
            println(io, join(fields, '\t'))
        end
    end
end

function parse_int_vector_054(field::AbstractString)
    return isempty(field) ? Int[] : parse.(Int, split(field, ","))
end

function nearest_index_054(values::Vector{Float64}, target::Float64, label::String)
    idx = findmin(abs.(values .- target))[2]
    if !isapprox(values[idx], target; atol=5e-5, rtol=0.0)
        error("$(label)=$(target) does not align with plotting grid.")
    end
    return idx
end

sign_category_054(sign_value::Int) = sign_value < 0 ? 1 : (sign_value == 0 ? 2 : 3)

function build_iterate_grids_054()
    T_grids = [fill(0, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in 1:MAX_ITER_054]
    gamma_grids = [fill(0, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in 1:MAX_ITER_054]
    T_scalar_grids = [fill(NaN, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in 1:MAX_ITER_054]
    gamma_scalar_grids = [fill(NaN, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in 1:MAX_ITER_054]
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    error_count = 0

    scan_column_files_054() do fields
        delta_x = parse(Float64, fields[1])
        delta_ca = parse(Float64, fields[2])
        status = fields[12]
        x_idx = nearest_index_054(DELTA_CAS_010, delta_ca, "Delta Ca")
        y_idx = nearest_index_054(DELTA_XS_010, delta_x, "Delta x")
        filled[x_idx, y_idx] = true
        if status != "ok"
            error_count += 1
            return
        end
        T_signs = parse_int_vector_054(fields[6])
        gamma_signs = parse_int_vector_054(fields[7])
        T_ca = parse_float_vector_054(fields[10])
        gamma_ca = parse_float_vector_054(fields[11])
        for k in 1:MAX_ITER_054
            if length(T_signs) >= k
                T_grids[k][x_idx, y_idx] = sign_category_054(T_signs[k])
            end
            if length(gamma_signs) >= k
                gamma_grids[k][x_idx, y_idx] = sign_category_054(gamma_signs[k])
            end
            if length(T_ca) >= k
                T_scalar_grids[k][x_idx, y_idx] = T_ca[k]
            end
            if length(gamma_ca) >= k
                gamma_scalar_grids[k][x_idx, y_idx] = gamma_ca[k]
            end
        end
    end

    all(filled) || error("One or more tangent-sign grid entries were not filled.")
    return T_grids, gamma_grids, T_scalar_grids, gamma_scalar_grids, error_count
end

function edge_point_054(edge::Int, x0::Float64, x1::Float64, y0::Float64, y1::Float64)
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    if edge == 0
        return x0, ym
    elseif edge == 1
        return xm, y1
    elseif edge == 2
        return x1, ym
    else
        return xm, y0
    end
end

function case_segments_054(mask_case::Int)
    if mask_case == 1
        return ((3, 0),)
    elseif mask_case == 2
        return ((0, 1),)
    elseif mask_case == 3
        return ((3, 1),)
    elseif mask_case == 4
        return ((1, 2),)
    elseif mask_case == 5
        return ((3, 0), (1, 2))
    elseif mask_case == 6
        return ((0, 2),)
    elseif mask_case == 7
        return ((3, 2),)
    elseif mask_case == 8
        return ((2, 3),)
    elseif mask_case == 9
        return ((0, 2),)
    elseif mask_case == 10
        return ((2, 3), (0, 1))
    elseif mask_case == 11
        return ((1, 2),)
    elseif mask_case == 12
        return ((1, 3),)
    elseif mask_case == 13
        return ((0, 1),)
    elseif mask_case == 14
        return ((3, 0),)
    else
        return ()
    end
end

function normalize_segment_054(x1::Float64, y1::Float64, x2::Float64, y2::Float64)
    if x1 < x2 || (x1 == x2 && y1 <= y2)
        return (x1, y1, x2, y2)
    else
        return (x2, y2, x1, y1)
    end
end

function push_unique_segment_054!(segments::Vector{NTuple{4, Float64}}, segment::NTuple{4, Float64})
    for existing in segments
        if existing == segment
            return
        end
    end
    push!(segments, segment)
end

function categorical_marching_squares_054(grid::Matrix{Int}, x_values::Vector{Float64}, y_values::Vector{Float64})
    xs = Float32[]
    ys = Float32[]
    sizehint!(xs, 3 * (length(x_values) - 1) * (length(y_values) - 1))
    sizehint!(ys, 3 * (length(x_values) - 1) * (length(y_values) - 1))
    local_segments = NTuple{4, Float64}[]
    sizehint!(local_segments, 8)

    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
            y0 = y_values[y_idx]
            y1 = y_values[y_idx + 1]
            bottom_left = grid[x_idx, y_idx]
            bottom_right = grid[x_idx + 1, y_idx]
            top_right = grid[x_idx + 1, y_idx + 1]
            top_left = grid[x_idx, y_idx + 1]

            if bottom_left == 0 || bottom_right == 0 || top_right == 0 || top_left == 0
                continue
            end
            if bottom_left == bottom_right == top_right == top_left
                continue
            end

            empty!(local_segments)
            categories = unique((bottom_left, bottom_right, top_right, top_left))
            for category in categories
                mask_case =
                    (bottom_left == category ? 1 : 0) +
                    (bottom_right == category ? 2 : 0) +
                    (top_right == category ? 4 : 0) +
                    (top_left == category ? 8 : 0)
                for (edge_a, edge_b) in case_segments_054(mask_case)
                    x_a, y_a = edge_point_054(edge_a, x0, x1, y0, y1)
                    x_b, y_b = edge_point_054(edge_b, x0, x1, y0, y1)
                    push_unique_segment_054!(local_segments, normalize_segment_054(x_a, y_a, x_b, y_b))
                end
            end

            for (x_a, y_a, x_b, y_b) in local_segments
                push!(xs, Float32(x_a), Float32(x_b), NaN32)
                push!(ys, Float32(y_a), Float32(y_b), NaN32)
            end
        end
    end
    return xs, ys
end

function parse_float_vector_054(field::AbstractString)
    return isempty(field) ? Float64[] : parse.(Float64, split(field, ","))
end

function zero_cross_point_054(
    v0::Float64,
    v1::Float64,
    x0::Float64,
    y0::Float64,
    x1::Float64,
    y1::Float64,
)
    denom = v0 - v1
    alpha = abs(denom) <= eps(Float64) ? 0.5 : clamp(v0 / denom, 0.0, 1.0)
    return x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)
end

function scalar_zero_marching_squares_054(grid::Matrix{Float64}, x_values::Vector{Float64}, y_values::Vector{Float64})
    xs = Float32[]
    ys = Float32[]
    sizehint!(xs, 3 * (length(x_values) - 1) * (length(y_values) - 1))
    sizehint!(ys, 3 * (length(x_values) - 1) * (length(y_values) - 1))

    for x_idx in 1:(length(x_values) - 1)
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in 1:(length(y_values) - 1)
            y0 = y_values[y_idx]
            y1 = y_values[y_idx + 1]
            values = (
                grid[x_idx, y_idx],
                grid[x_idx + 1, y_idx],
                grid[x_idx + 1, y_idx + 1],
                grid[x_idx, y_idx + 1],
            )
            any(!isfinite, values) && continue
            if all(>(0.0), values) || all(<(0.0), values)
                continue
            end

            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            edge_pairs = ((1, 2), (2, 3), (3, 4), (4, 1))
            points = Tuple{Float64, Float64}[]
            for (a, b) in edge_pairs
                va = values[a]
                vb = values[b]
                if va == 0.0 && vb == 0.0
                    push!(points, corners[a], corners[b])
                elseif va == 0.0
                    push!(points, corners[a])
                elseif vb == 0.0
                    push!(points, corners[b])
                elseif signbit(va) != signbit(vb)
                    push!(points, zero_cross_point_054(va, vb, corners[a][1], corners[a][2], corners[b][1], corners[b][2]))
                end
            end

            unique!(points)
            if length(points) == 2
                push!(xs, Float32(points[1][1]), Float32(points[2][1]), NaN32)
                push!(ys, Float32(points[1][2]), Float32(points[2][2]), NaN32)
            elseif length(points) == 4
                push!(xs, Float32(points[1][1]), Float32(points[2][1]), NaN32)
                push!(ys, Float32(points[1][2]), Float32(points[2][2]), NaN32)
                push!(xs, Float32(points[3][1]), Float32(points[4][1]), NaN32)
                push!(ys, Float32(points[3][2]), Float32(points[4][2]), NaN32)
            end
        end
    end
    return xs, ys
end

function iterate_color_054(base::RGBAf, k::Int)
    alpha_scale = k == 1 ? 1.0 : (1.0 / (k ^ 0.3))
    return RGBAf(red(base), green(base), blue(base), alpha(base) * alpha_scale)
end

function save_tangent_contour_plot_054(path::String, T_scalar_grids::Vector{Matrix{Float64}}, gamma_scalar_grids::Vector{Matrix{Float64}})
    fig = Figure(size=(PLOT_WIDTH_054, PLOT_HEIGHT_054))
    ax = Axis(
        fig[1, 1],
        title="Ca-min tangent-Ca zero contours (8 iterates)",
        xlabel="Delta Ca",
        ylabel="Delta x",
        titlesize=40,
        xlabelsize=34,
        ylabelsize=34,
        xticklabelsize=24,
        yticklabelsize=24,
    )

    for k in 1:MAX_ITER_054
        T_xs, T_ys = scalar_zero_marching_squares_054(T_scalar_grids[k], DELTA_CAS_010, DELTA_XS_010)
        gamma_xs, gamma_ys = scalar_zero_marching_squares_054(gamma_scalar_grids[k], DELTA_CAS_010, DELTA_XS_010)
        lines!(ax, T_xs, T_ys; color=iterate_color_054(T_COLOR_054, k), linewidth=LINEWIDTH_054)
        lines!(ax, gamma_xs, gamma_ys; color=iterate_color_054(GAMMA_COLOR_054, k), linewidth=LINEWIDTH_054)
    end

    ax.xticks = fixed_ticks(DELTA_CAS_010, "%.0f", DELTA_CA_TICK_STEP_010)
    ax.yticks = fixed_ticks(DELTA_XS_010, "%.1f", DELTA_X_TICK_STEP_010)
    save(path, fig; px_per_unit=PLOT_PX_PER_UNIT_054)
end

function write_summary_054(path::String, error_count::Int, elapsed::Float64)
    total_points = length(DELTA_CAS_010) * length(DELTA_XS_010)
    open(path, "w") do io
        println(io, "output_tag\t$(OUTPUT_TAG_054)")
        println(io, "grid_delta_ca\t$(length(DELTA_CAS_010))")
        println(io, "grid_delta_x\t$(length(DELTA_XS_010))")
        println(io, "delta_ca_min\t$(minimum(DELTA_CAS_010))")
        println(io, "delta_ca_max\t$(maximum(DELTA_CAS_010))")
        println(io, "delta_x_min\t$(minimum(DELTA_XS_010))")
        println(io, "delta_x_max\t$(maximum(DELTA_XS_010))")
        println(io, "max_iter\t$(MAX_ITER_054)")
        println(io, "tmax\t$(TANGENT_TMAX_054)")
        println(io, "reorth_every_step\t$(REORTH_EVERY_STEP_054)")
        println(io, "ca_min_v_max\t$(CA_MIN_V_MAX_054)")
        println(io, "y_stubbed\ttrue")
        println(io, "active_state_order\tx\tn\th\tCa\tV")
        println(io, "total_points\t$(total_points)")
        println(io, "successful_points\t$(total_points - error_count)")
        println(io, "error_points\t$(error_count)")
        println(io, "elapsed_seconds\t$(elapsed)")
    end
end

function main()
    started = time()
    println("Running attempt-054 tangent-sign Ca-minimum contour scan.")
    println("Grid: $(length(DELTA_CAS_010)) Delta Ca points x $(length(DELTA_XS_010)) Delta x points")
    println("Output tag: $(OUTPUT_TAG_054)")
    println("Tangent tmax: $(TANGENT_TMAX_054), iterates: $(MAX_ITER_054), Julia threads: $(nthreads())")
    println("Column checkpoint directory: $(SWEEP_DIR_054)")
    flush(stdout)

    run_or_resume_columns_054()

    results_path = joinpath(ATTEMPT54_ROOT, "$(OUTPUT_TAG_054)_results.tsv")
    plot_path = joinpath(ATTEMPT54_ROOT, "$(OUTPUT_TAG_054)_contours.png")
    summary_path = joinpath(ATTEMPT54_ROOT, "$(OUTPUT_TAG_054)_summary.txt")
    write_merged_results_054(results_path)
    _, _, T_scalar_grids, gamma_scalar_grids, error_count = build_iterate_grids_054()
    save_tangent_contour_plot_054(plot_path, T_scalar_grids, gamma_scalar_grids)
    elapsed = time() - started
    write_summary_054(summary_path, error_count, elapsed)

    total_points = length(DELTA_CAS_010) * length(DELTA_XS_010)
    println("Successful points: $(total_points - error_count) / $(total_points)")
    println("Saved merged results to $(results_path)")
    println("Saved contour plot to $(plot_path)")
    println("Saved summary to $(summary_path)")
    println(@sprintf("Elapsed %.2f s", elapsed))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
