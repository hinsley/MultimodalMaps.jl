using Pkg

const ATTEMPT10_ROOT = @__DIR__
const ATTEMPT09_ROOT = normpath(joinpath(ATTEMPT10_ROOT, "..", "attempt-009"))
const REPO_ROOT_010 = normpath(joinpath(ATTEMPT10_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_010)

include(joinpath(ATTEMPT09_ROOT, "main.jl"))

using Colors
using Printf
using Random

const GRID_NX = parse(Int, get(ENV, "ATTEMPT010_NX", "4"))
const GRID_NY = parse(Int, get(ENV, "ATTEMPT010_NY", "5"))
const DELTA_XS_010 = collect(range(-1.5, -0.5, length=GRID_NX))
const DELTA_CAS_010 = collect(range(-33.0, -20.0, length=GRID_NY))
const COLOR_SEED = parse(Int, get(ENV, "ATTEMPT010_COLOR_SEED", "20260306"))

struct SSCSScanResult
    delta_x::Float64
    delta_ca::Float64
    T_scs::Vector{Int}
    gamma_scs::Vector{Int}
    encoding::Union{BigInt, Nothing}
    category_id::Union{Int, Nothing}
    error_message::Union{Nothing, String}
end

zigzag_encode(n::Int)::BigInt = n >= 0 ? 2 * BigInt(n) : -2 * BigInt(n) - 1

function cantor_pair(a::BigInt, b::BigInt)::BigInt
    s = a + b
    return (s * (s + 1)) ÷ 2 + b
end

function encode_sequence(seq::Vector{Int})::BigInt
    code = BigInt(0)
    for value in reverse(seq)
        code = cantor_pair(zigzag_encode(value), code) + 1
    end
    return code
end

function encode_sscs_pair(T_scs::Vector{Int}, gamma_scs::Vector{Int})::BigInt
    return cantor_pair(encode_sequence(T_scs), encode_sequence(gamma_scs))
end

function run_sscs_pair(delta_x::Float64, delta_ca::Float64)::SSCSScanResult
    p = build_params(delta_x, delta_ca)
    saddle_data = compute_gamma_sd_minus0(p)
    T0 = initialize_T_Ca0(p, saddle_data.x_eq_SF, saddle_data.gamma_sd_minus0)

    T_scs = compute_sscs(p, T0, saddle_data.V_eq_SD; abstol=3e-6, reltol=3e-6)
    if isempty(T_scs) || abs(T_scs[1]) != 1
        error("T SSCS does not start with ±1: $(T_scs)")
    end

    gamma_scs = compute_sscs(p, saddle_data.gamma_sd_minus0, saddle_data.V_eq_SD; abstol=1e-8, reltol=1e-8)
    encoding = encode_sscs_pair(T_scs, gamma_scs)

    return SSCSScanResult(delta_x, delta_ca, T_scs, gamma_scs, encoding, nothing, nothing)
end

function assign_category_ids(results::Vector{SSCSScanResult})
    encodings = sort(unique(result.encoding for result in results if !isnothing(result.encoding)))
    category_lookup = Dict(encoding => idx for (idx, encoding) in enumerate(encodings))

    categorized = SSCSScanResult[]
    for result in results
        category_id = isnothing(result.encoding) ? nothing : category_lookup[result.encoding]
        push!(
            categorized,
            SSCSScanResult(
                result.delta_x,
                result.delta_ca,
                result.T_scs,
                result.gamma_scs,
                result.encoding,
                category_id,
                result.error_message,
            ),
        )
    end

    return categorized, encodings
end

function random_palette(n::Int)
    rng = MersenneTwister(COLOR_SEED)
    return [
        HSV(rand(rng), 0.65 + 0.35 * rand(rng), 0.75 + 0.25 * rand(rng))
        for _ in 1:n
    ]
end

function build_category_matrix(results::Vector{SSCSScanResult})
    matrix = fill(NaN, length(DELTA_CAS_010), length(DELTA_XS_010))
    label_matrix = fill("", length(DELTA_CAS_010), length(DELTA_XS_010))

    x_index = Dict(x => idx for (idx, x) in enumerate(DELTA_XS_010))
    y_index = Dict(y => idx for (idx, y) in enumerate(DELTA_CAS_010))

    for result in results
        i = y_index[result.delta_ca]
        j = x_index[result.delta_x]
        if isnothing(result.category_id)
            matrix[i, j] = NaN
            label_matrix[i, j] = "X"
        else
            matrix[i, j] = Float64(result.category_id)
            label_matrix[i, j] = string(result.category_id)
        end
    end

    return matrix, label_matrix
end

function save_heatmap(path::String, results::Vector{SSCSScanResult}, unique_encodings::Vector{BigInt})
    matrix, labels = build_category_matrix(results)
    fig = Figure(size=(1100, 850))
    ax = Axis(
        fig[1, 1],
        xlabel="Δx",
        ylabel="ΔCa",
        title="Attempt 010: Low-Resolution SSCS Pair Encoding Heatmap",
    )

    if isempty(unique_encodings)
        error("No successful encodings available for plotting.")
    end

    palette = random_palette(length(unique_encodings))
    hm = heatmap!(
        ax,
        1:length(DELTA_XS_010),
        1:length(DELTA_CAS_010),
        matrix;
        colormap=cgrad(palette, categorical=true),
        colorrange=(1, length(unique_encodings)),
        interpolate=false,
        nan_color=RGBf(0.15, 0.15, 0.15),
    )

    ax.xticks = (1:length(DELTA_XS_010), [@sprintf("%.2f", x) for x in DELTA_XS_010])
    ax.yticks = (1:length(DELTA_CAS_010), [@sprintf("%.1f", y) for y in DELTA_CAS_010])

    for i in 1:length(DELTA_CAS_010), j in 1:length(DELTA_XS_010)
        label = labels[i, j]
        color = label == "X" ? :white : :black
        text!(ax, j, i, text=label, align=(:center, :center), color=color, fontsize=18)
    end

    Colorbar(fig[1, 2], hm, label="Encoding Category ID")
    save(path, fig)
end

function write_results(path::String, results::Vector{SSCSScanResult})
    open(path, "w") do io
        println(io, "delta_x\tdelta_ca\tcategory_id\tencoding\tT_scs\tgamma_scs\tstatus")
        for result in results
            status = isnothing(result.error_message) ? "ok" : "error: " * result.error_message
            println(
                io,
                join([
                    @sprintf("%.6f", result.delta_x),
                    @sprintf("%.6f", result.delta_ca),
                    isnothing(result.category_id) ? "" : string(result.category_id),
                    isnothing(result.encoding) ? "" : string(result.encoding),
                    join(result.T_scs, ","),
                    join(result.gamma_scs, ","),
                    status,
                ], '\t'),
            )
        end
    end
end

function write_legend(path::String, results::Vector{SSCSScanResult})
    seen = Dict{Int, SSCSScanResult}()
    counts = Dict{Int, Int}()
    for result in results
        if !isnothing(result.category_id)
            seen[result.category_id] = result
            counts[result.category_id] = get(counts, result.category_id, 0) + 1
        end
    end

    open(path, "w") do io
        println(io, "category_id\tcount\tencoding\tT_scs\tgamma_scs")
        for category_id in sort(collect(keys(seen)))
            result = seen[category_id]
            println(
                io,
                join([
                    string(category_id),
                    string(counts[category_id]),
                    string(result.encoding),
                    join(result.T_scs, ","),
                    join(result.gamma_scs, ","),
                ], '\t'),
            )
        end
    end
end

function main()
    println("Running low-resolution 2D SSCS encoding scan for attempt-010.")
    println("Grid: $(length(DELTA_XS_010)) Δx points × $(length(DELTA_CAS_010)) ΔCa points")

    raw_results = SSCSScanResult[]
    for (i, delta_ca) in enumerate(DELTA_CAS_010)
        for (j, delta_x) in enumerate(DELTA_XS_010)
            @printf(
                "Sweeping [%d/%d, %d/%d] Δx=%.4f ΔCa=%.4f\n",
                j,
                length(DELTA_XS_010),
                i,
                length(DELTA_CAS_010),
                delta_x,
                delta_ca,
            )
            try
                result = run_sscs_pair(delta_x, delta_ca)
                push!(raw_results, result)
                @printf(
                    "  encoding=%s T_scs=%s gamma_scs=%s\n",
                    string(result.encoding),
                    string(result.T_scs),
                    string(result.gamma_scs),
                )
            catch err
                message = sprint(showerror, err)
                push!(raw_results, SSCSScanResult(delta_x, delta_ca, Int[], Int[], nothing, nothing, message))
                @printf("  failed: %s\n", message)
            end
        end
    end

    categorized_results, unique_encodings = assign_category_ids(raw_results)
    success_count = count(result -> !isnothing(result.category_id), categorized_results)
    println("Successful points: $(success_count) / $(length(categorized_results))")
    println("Unique SSCS-pair encodings: $(length(unique_encodings))")

    results_path = joinpath(ATTEMPT10_ROOT, "lowres_encoding_results.tsv")
    legend_path = joinpath(ATTEMPT10_ROOT, "lowres_encoding_legend.tsv")
    plot_path = joinpath(ATTEMPT10_ROOT, "lowres_encoding_heatmap.png")

    write_results(results_path, categorized_results)
    write_legend(legend_path, categorized_results)
    save_heatmap(plot_path, categorized_results, unique_encodings)

    println("Saved pointwise results to $(results_path)")
    println("Saved encoding legend to $(legend_path)")
    println("Saved heatmap to $(plot_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
