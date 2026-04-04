using Pkg

const ATTEMPT25_ROOT = @__DIR__
const REPO_ROOT_025 = normpath(joinpath(ATTEMPT25_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025)

using Printf
using StaticArrays

const ATTEMPT025_ALPHA_MIN = parse(Float64, get(ENV, "ATTEMPT025_ALPHA_MIN", "0.0"))
const ATTEMPT025_ALPHA_MAX = parse(Float64, get(ENV, "ATTEMPT025_ALPHA_MAX", "0.7"))
const ATTEMPT025_LAMBDA_MIN = parse(Float64, get(ENV, "ATTEMPT025_LAMBDA_MIN", "0.2"))
const ATTEMPT025_LAMBDA_MAX = parse(Float64, get(ENV, "ATTEMPT025_LAMBDA_MAX", "1.6"))

const ATTEMPT025_ATTEMPT023_RESULTS = get(
    ENV,
    "ATTEMPT025_ATTEMPT023_RESULTS",
    joinpath(
        REPO_ROOT_025,
        "kneading",
        "experiment",
        "attempt-023",
        "grid5000_branch16_criticality_shimizu_morioka_cpu_results.tsv",
    ),
)

const ATTEMPT025_ATTEMPT024_COLUMNS = get(
    ENV,
    "ATTEMPT025_ATTEMPT024_COLUMNS",
    joinpath(
        REPO_ROOT_025,
        "kneading",
        "experiment",
        "attempt-024",
        "grid1200_branch8_floworth_state_slip_shimizu_morioka_cpu_columns",
    ),
)

struct Attempt023Result25
    alpha::Float64
    lambda::Float64
    maxima_count::Int
    red_mask::UInt16
    zmax_values::Vector{Float64}
    return_times::Vector{Float64}
    status::String
end

struct Attempt024Result25
    alpha::Float64
    lambda::Float64
    zmax_count::Int
    zmax_dot_values::Vector{Float64}
    zmax_states::Vector{SVector{3, Float64}}
    absxmax_count::Int
    absxmax_dot_values::Vector{Float64}
    absxmax_states::Vector{SVector{3, Float64}}
    status::String
end

join_float_list_025(values::Vector{Float64}) = join((@sprintf("%.9f", v) for v in values), ",")

function parse_float_list_025(field::AbstractString)
    isempty(field) && return Float64[]
    return parse.(Float64, split(field, ","))
end

function join_state_list_025(states::Vector{SVector{3, Float64}})
    return join((@sprintf("%.9f,%.9f,%.9f", state[1], state[2], state[3]) for state in states), ";")
end

function parse_state_list_025(field::AbstractString)
    isempty(field) && return SVector{3, Float64}[]
    return map(split(field, ';')) do chunk
        coords = parse.(Float64, split(chunk, ','))
        length(coords) == 3 || error("Expected 3 state coordinates, got $(length(coords)) in: $(chunk)")
        return SVector{3, Float64}(coords[1], coords[2], coords[3])
    end
end

function parse_attempt023_result_025(fields::Vector{SubString{String}})
    length(fields) == 7 || error("Expected 7 fields for attempt-023 result row, got $(length(fields))")
    return Attempt023Result25(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        parse(UInt16, fields[4]),
        parse_float_list_025(fields[5]),
        parse_float_list_025(fields[6]),
        String(fields[7]),
    )
end

function parse_attempt024_result_025(fields::Vector{SubString{String}})
    if length(fields) == 7
        return Attempt024Result25(
            parse(Float64, fields[1]),
            parse(Float64, fields[2]),
            parse(Int, fields[3]),
            parse_float_list_025(fields[4]),
            SVector{3, Float64}[],
            parse(Int, fields[5]),
            parse_float_list_025(fields[6]),
            SVector{3, Float64}[],
            String(fields[7]),
        )
    end

    length(fields) == 9 || error("Expected 7 or 9 fields for attempt-024 result row, got $(length(fields))")
    return Attempt024Result25(
        parse(Float64, fields[1]),
        parse(Float64, fields[2]),
        parse(Int, fields[3]),
        parse_float_list_025(fields[4]),
        parse_state_list_025(fields[5]),
        parse(Int, fields[6]),
        parse_float_list_025(fields[7]),
        parse_state_list_025(fields[8]),
        String(fields[9]),
    )
end

function scan_attempt023_results_025(pass_fn; path::String=ATTEMPT025_ATTEMPT023_RESULTS)
    isfile(path) || error("attempt-023 results TSV not found: $(path)")
    open(path, "r") do io
        readline(io)
        for line in eachline(io)
            pass_fn(parse_attempt023_result_025(split(line, '\t')))
        end
    end
end

function attempt024_column_paths_025(; dir::String=ATTEMPT025_ATTEMPT024_COLUMNS)
    isdir(dir) || error("attempt-024 columns directory not found: $(dir)")
    paths = sort(filter(path -> endswith(path, ".tsv"), joinpath.(dir, readdir(dir))))
    isempty(paths) && error("No attempt-024 column TSVs found in $(dir)")
    return paths
end

function scan_attempt024_results_025(pass_fn; dir::String=ATTEMPT025_ATTEMPT024_COLUMNS)
    for path in attempt024_column_paths_025(dir=dir)
        open(path, "r") do io
            readline(io)
            for line in eachline(io)
                pass_fn(parse_attempt024_result_025(split(line, '\t')))
            end
        end
    end
end

attempt023_dt_values_025(result::Attempt023Result25) = diff(result.return_times)
attempt024_zmax_x_values_025(result::Attempt024Result25) = [state[1] for state in result.zmax_states]
attempt024_zmax_z_values_025(result::Attempt024Result25) = [state[3] for state in result.zmax_states]
attempt024_absxmax_x_values_025(result::Attempt024Result25) = [state[1] for state in result.absxmax_states]
attempt024_absxmax_z_values_025(result::Attempt024Result25) = [state[3] for state in result.absxmax_states]

function summarize_sources_025()
    attempt023_count = Ref(0)
    scan_attempt023_results_025() do _
        attempt023_count[] += 1
    end

    attempt024_count = Ref(0)
    scan_attempt024_results_025() do _
        attempt024_count[] += 1
    end

    return (
        attempt023_results=attempt023_count[],
        attempt024_results=attempt024_count[],
        attempt023_path=ATTEMPT025_ATTEMPT023_RESULTS,
        attempt024_dir=ATTEMPT025_ATTEMPT024_COLUMNS,
    )
end
