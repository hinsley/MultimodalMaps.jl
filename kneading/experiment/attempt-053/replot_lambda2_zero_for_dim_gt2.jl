using Printf

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, "..", "..", ".."))
const RAW_TAG = "grid200_lyapdim_tmax1e5_gh0p000"
const ADJUSTED_TAG = "grid200_lyapdim_tmax1e5_gh0p000_lambda2zero_dim_gt2"
const RAW_PATH = joinpath(SCRIPT_DIR, "$(RAW_TAG)_results.tsv")
const ADJUSTED_PATH = joinpath(SCRIPT_DIR, "$(ADJUSTED_TAG)_results.tsv")

function kaplan_yorke_dimension_adjusted(λs::AbstractVector{<:Real})::Float64
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

function adjust_results()
    adjusted_count = 0
    total_count = 0
    open(RAW_PATH, "r") do input
        header = split(readline(input), '\t')
        lookup = Dict(name => idx for (idx, name) in enumerate(header))
        required = ["lyap_dim"; ["lambda$(i)" for i in 1:6]]
        for name in required
            haskey(lookup, name) || error("Missing column $(name) in $(RAW_PATH)")
        end

        open(ADJUSTED_PATH, "w") do output
            println(output, join(header, '\t'))
            for line in eachline(input)
                isempty(line) && continue
                total_count += 1
                fields = split(line, '\t')
                original_dim = parse(Float64, fields[lookup["lyap_dim"]])
                if isfinite(original_dim) && original_dim > 2.0
                    fields[lookup["lambda2"]] = "0"
                    λ = [parse(Float64, fields[lookup["lambda$(i)"]]) for i in 1:6]
                    fields[lookup["lyap_dim"]] = @sprintf("%.12g", kaplan_yorke_dimension_adjusted(λ))
                    adjusted_count += 1
                end
                println(output, join(fields, '\t'))
            end
        end
    end
    return (; total_count, adjusted_count)
end

stats = adjust_results()
println("Wrote adjusted TSV: $(ADJUSTED_PATH)")
println("Adjusted $(stats.adjusted_count) / $(stats.total_count) rows")

ENV["ATTEMPT052_NX"] = "200"
ENV["ATTEMPT052_NY"] = "200"
ENV["ATTEMPT052_DELTA_X_MIN"] = "-1.5"
ENV["ATTEMPT052_DELTA_X_MAX"] = "-0.5"
ENV["ATTEMPT052_DELTA_CA_MIN"] = "-45.0"
ENV["ATTEMPT052_DELTA_CA_MAX"] = "-20.0"
ENV["ATTEMPT052_G_H"] = "0.0"
ENV["ATTEMPT052_TAU_Y"] = "2.0e4"
ENV["ATTEMPT052_OUTPUT_DIR"] = SCRIPT_DIR
ENV["ATTEMPT052_OUTPUT_TAG"] = ADJUSTED_TAG
ENV["ATTEMPT052_PLOT_ONLY_RESULTS"] = ADJUSTED_PATH

include(joinpath(REPO_ROOT, "kneading", "experiment", "attempt-052", "main.jl"))
main_052()
