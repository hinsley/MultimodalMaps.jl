using Pkg

const ATTEMPT15_ROOT = @__DIR__
const ATTEMPT11_ROOT = normpath(joinpath(ATTEMPT15_ROOT, "..", "attempt-011"))
const REPO_ROOT_015 = normpath(joinpath(ATTEMPT15_ROOT, "..", "..", ".."))

if !haskey(ENV, "ATTEMPT010_MAX_SEQ_LENGTH")
    ENV["ATTEMPT010_MAX_SEQ_LENGTH"] = get(ENV, "ATTEMPT015_MAX_SEQ_LENGTH", "7")
end

Pkg.activate(REPO_ROOT_015)

include(joinpath(ATTEMPT11_ROOT, "main.jl"))

using Printf

struct TContourScanResult
    delta_x::Float64
    delta_ca::Float64
    T_scs::Vector{Int}
    T0_V::Float64
    T0_Ca::Float64
    T0_method::String
    T_encoding::Union{BigInt, Nothing}
    error_message::Union{Nothing, String}
end

function finalize_T_point(
    delta_x::Float64,
    delta_ca::Float64,
    p,
    saddle_data,
    T0::SVector{6, Float64},
    T0_method::String,
)::TContourScanResult
    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    if isempty(T_scs) || abs(T_scs[1]) != 1
        error("T SSCS does not start with ±1: $(T_scs)")
    end

    return TContourScanResult(
        delta_x,
        delta_ca,
        T_scs,
        Float64(T0[6]),
        Float64(T0[5]),
        T0_method,
        encode_sequence(T_scs),
        nothing,
    )
end

function run_T_point(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::TContourScanResult
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
            return finalize_T_point(delta_x, delta_ca, p, saddle_data, T0, method)
        catch
            # Fall through to the full initializer.
        end
    end

    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)
    return finalize_T_point(delta_x, delta_ca, p, saddle_data, T0, "full")
end

function run_T_point_safe(
    delta_x::Float64,
    delta_ca::Float64,
    candidate_seed::Union{Nothing, T0ContinuationSeed},
)::TContourScanResult
    try
        return run_T_point(delta_x, delta_ca, candidate_seed)
    catch err
        return TContourScanResult(delta_x, delta_ca, Int[], NaN, NaN, "", nothing, sprint(showerror, err))
    end
end

make_candidate_seed(previous_successful::Union{Nothing, TContourScanResult}) =
    isnothing(previous_successful) ? nothing : T0ContinuationSeed(previous_successful.T0_V, previous_successful.T0_Ca)
