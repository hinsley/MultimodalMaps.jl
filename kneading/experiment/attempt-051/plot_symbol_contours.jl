using Pkg

const ATTEMPT51_SYMBOL_ROOT = @__DIR__
const REPO_ROOT_051_SYMBOL = normpath(joinpath(ATTEMPT51_SYMBOL_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_051_SYMBOL)

include(joinpath(ATTEMPT51_SYMBOL_ROOT, "contours.jl"))

using Printf

const SYMBOL_RESULTS_PATH = get(
    ENV,
    "ATTEMPT051_SYMBOL_RESULTS",
    joinpath(ATTEMPT51_SYMBOL_ROOT, "grid1000_seq12_tmax1e5_gh0p000_prefixes_remap40_newmodel_results.tsv"),
)
const SYMBOL_OUTPUT_DIR = get(ENV, "ATTEMPT051_SYMBOL_OUTPUT_DIR", dirname(SYMBOL_RESULTS_PATH))
const SYMBOL_OUTPUT_TAG = get(ENV, "ATTEMPT051_SYMBOL_OUTPUT_TAG", "grid1000_seq12_tmax1e5_gh0p000_symbol")
const SYMBOL_LENGTHS = collect(1:min(ATTEMPT10_MAX_SEQ_LENGTH, parse(Int, get(ENV, "ATTEMPT051_SYMBOL_MAX_LENGTH", "12"))))

function header_lookup(header::AbstractString)
    names = split(header, '\t'; keepempty=true)
    return Dict(name => idx for (idx, name) in pairs(names))
end

function symbol_key(seq::Vector{Int}, symbol_idx::Int)
    return symbol_idx <= length(seq) ? string(seq[symbol_idx]) : "missing_k"
end

function category_id!(lookup::Dict{String, Int}, key::String)
    return get!(lookup, key) do
        length(lookup) + 1
    end
end

function nearest_grid_indices(delta_x::Float64, delta_ca::Float64)
    ca_idx = nearest_index(DELTA_CAS_010, delta_ca, "ΔCa")
    x_idx = nearest_index(DELTA_XS_010, delta_x, "Δx")
    return ca_idx, x_idx
end

function build_symbol_grids(results_path::String, symbol_lengths::Vector{Int})
    T_grids = [fill(0, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in symbol_lengths]
    gamma_grids = [fill(0, length(DELTA_CAS_010), length(DELTA_XS_010)) for _ in symbol_lengths]
    T_lookups = [Dict{String, Int}() for _ in symbol_lengths]
    gamma_lookups = [Dict{String, Int}() for _ in symbol_lengths]
    T_counts = [Dict{String, Int}() for _ in symbol_lengths]
    gamma_counts = [Dict{String, Int}() for _ in symbol_lengths]
    filled = falses(length(DELTA_CAS_010), length(DELTA_XS_010))
    ok_count = 0
    error_count = 0

    open(results_path, "r") do io
        lookup = header_lookup(readline(io))
        required = ("delta_x", "delta_ca", "T_scs", "gamma_scs", "status")
        for name in required
            haskey(lookup, name) || error("Missing required TSV column: $(name)")
        end

        for line in eachline(io)
            fields = split(line, '\t'; keepempty=true)
            delta_x = parse(Float64, fields[lookup["delta_x"]])
            delta_ca = parse(Float64, fields[lookup["delta_ca"]])
            ca_idx, x_idx = nearest_grid_indices(delta_x, delta_ca)

            if filled[ca_idx, x_idx]
                error("Duplicate grid entry for Δx=$(delta_x), ΔCa=$(delta_ca).")
            end
            filled[ca_idx, x_idx] = true

            status = fields[lookup["status"]]
            if status != "ok"
                error_count += 1
                continue
            end
            ok_count += 1

            T_scs = parse_sequence_field(fields[lookup["T_scs"]])
            gamma_scs = parse_sequence_field(fields[lookup["gamma_scs"]])

            for (idx, symbol_idx) in enumerate(symbol_lengths)
                T_key = symbol_key(T_scs, symbol_idx)
                gamma_key = symbol_key(gamma_scs, symbol_idx)

                T_grids[idx][ca_idx, x_idx] = category_id!(T_lookups[idx], T_key)
                gamma_grids[idx][ca_idx, x_idx] = category_id!(gamma_lookups[idx], gamma_key)
                T_counts[idx][T_key] = get(T_counts[idx], T_key, 0) + 1
                gamma_counts[idx][gamma_key] = get(gamma_counts[idx], gamma_key, 0) + 1
            end
        end
    end

    all(filled) || error("One or more symbol contour grid entries were not filled.")
    return T_grids, gamma_grids, T_lookups, gamma_lookups, T_counts, gamma_counts, ok_count, error_count
end

function write_symbol_legend(path::String, lookup::Dict{String, Int}, counts::Dict{String, Int})
    inverse = Dict(id => key for (key, id) in lookup)
    open(path, "w") do io
        println(io, "category_id\tcount\tsymbol")
        for id in sort(collect(keys(inverse)))
            key = inverse[id]
            println(io, join([string(id), string(get(counts, key, 0)), key], '\t'))
        end
    end
end

symbol_plot_path(symbol_idx::Int) =
    joinpath(SYMBOL_OUTPUT_DIR, @sprintf("%s%02d_contours.png", SYMBOL_OUTPUT_TAG, symbol_idx))

symbol_T_legend_path(symbol_idx::Int) =
    joinpath(SYMBOL_OUTPUT_DIR, @sprintf("%s%02d_T_legend.tsv", SYMBOL_OUTPUT_TAG, symbol_idx))

symbol_gamma_legend_path(symbol_idx::Int) =
    joinpath(SYMBOL_OUTPUT_DIR, @sprintf("%s%02d_gamma_legend.tsv", SYMBOL_OUTPUT_TAG, symbol_idx))

function main()
    mkpath(SYMBOL_OUTPUT_DIR)
    println("Running attempt-051 per-symbol contour replay.")
    println("Results source: $(SYMBOL_RESULTS_PATH)")
    println("Output directory: $(SYMBOL_OUTPUT_DIR)")
    println("Symbol lengths: $(join(SYMBOL_LENGTHS, ", "))")
    flush(stdout)

    T_grids, gamma_grids, T_lookups, gamma_lookups, T_counts, gamma_counts, ok_count, error_count =
        build_symbol_grids(SYMBOL_RESULTS_PATH, SYMBOL_LENGTHS)

    summary_path = joinpath(SYMBOL_OUTPUT_DIR, "$(SYMBOL_OUTPUT_TAG)_summary.tsv")
    open(summary_path, "w") do io
        println(io, "symbol_index\tT_unique\tgamma_unique\tplot_path\tT_legend_path\tgamma_legend_path")
        for (idx, symbol_idx) in enumerate(SYMBOL_LENGTHS)
            plot_path = symbol_plot_path(symbol_idx)
            T_legend = symbol_T_legend_path(symbol_idx)
            gamma_legend = symbol_gamma_legend_path(symbol_idx)

            write_symbol_legend(T_legend, T_lookups[idx], T_counts[idx])
            write_symbol_legend(gamma_legend, gamma_lookups[idx], gamma_counts[idx])
            save_contour_plot(
                plot_path,
                T_grids[idx],
                gamma_grids[idx];
                title=@sprintf("SSCS symbol %02d contour overlay", symbol_idx),
            )

            println(io, join([
                string(symbol_idx),
                string(length(T_lookups[idx])),
                string(length(gamma_lookups[idx])),
                plot_path,
                T_legend,
                gamma_legend,
            ], '\t'))
            println("Saved symbol $(symbol_idx) plot to $(plot_path) (unique T=$(length(T_lookups[idx])), unique gamma=$(length(gamma_lookups[idx])))")
            flush(stdout)
        end
    end

    println("Successful rows: $(ok_count)")
    println("Error rows: $(error_count)")
    println("Saved summary to $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
