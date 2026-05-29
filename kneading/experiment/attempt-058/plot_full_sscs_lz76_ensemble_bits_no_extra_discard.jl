using Pkg

const ATTEMPT58_ROOT = @__DIR__
const REPO_ROOT_058 = normpath(joinpath(ATTEMPT58_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_058)

using CairoMakie

const BASE_TAG_058 = get(ENV, "ATTEMPT058_OUTPUT_TAG", "gh0_dCa-35_dx-1_ystub_ttr1e4_tmax1e7_ensemble10")
const TRAJECTORY_DIR_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_trajectories")
const PLOT_PATH_058 = joinpath(ATTEMPT58_ROOT, "$(BASE_TAG_058)_full_sscs_lz76_no_extra_discard_convergence_bits_per_second.png")
const PLOT_WIDTH_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_WIDTH", "1260"))
const PLOT_HEIGHT_058 = parse(Int, get(ENV, "ATTEMPT058_PLOT_HEIGHT", "550"))

function read_tsv_058(path::String)
    lines = readlines(path)
    header = split(first(lines), '\t')
    rows = [split(line, '\t') for line in lines[2:end] if !isempty(strip(line))]
    return header, rows
end

function column_index_058(header, name::String)
    idx = findfirst(==(name), header)
    idx === nothing && error("Column $(name) not found.")
    return idx
end

function trajectory_indices_058()
    paths = sort(filter(path -> occursin(r"traj\d+_convergence\.tsv$", basename(path)), readdir(TRAJECTORY_DIR_058; join=true)))
    return [parse(Int, match(r"traj(\d+)_convergence\.tsv$", basename(path)).captures[1]) for path in paths]
end

convergence_path_058(idx::Int) = joinpath(TRAJECTORY_DIR_058, "traj$(lpad(string(idx), 2, '0'))_convergence.tsv")

function read_convergence_058(idx::Int)
    header, rows = read_tsv_058(convergence_path_058(idx))
    time_seconds_idx = column_index_058(header, "time_seconds")
    lle_bits_idx = column_index_058(header, "lambda1_bits_per_second")
    lz_bits_idx = column_index_058(header, "lz76_bits_per_second")
    return (;
        idx,
        time_seconds=Float64[parse(Float64, row[time_seconds_idx]) for row in rows],
        lle_bits=Float64[parse(Float64, row[lle_bits_idx]) for row in rows],
        lz_bits=Float64[parse(Float64, row[lz_bits_idx]) for row in rows],
    )
end

function plot_ensemble_bits_058(results)
    fig = Figure(size=(PLOT_WIDTH_058, PLOT_HEIGHT_058), fontsize=24)
    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel="Entropy rate estimate (bits/s)",
        xlabelsize=30,
        ylabelsize=30,
        xticklabelsize=22,
        yticklabelsize=22,
        xticks=0:2500:10000,
        limits=(nothing, (0.1, 0.55)),
    )
    lle_handle = nothing
    lz_handle = nothing
    for result in results
        local a = lines!(ax, result.time_seconds, result.lle_bits; color=(:black, 0.35), linewidth=2.0)
        finite_lz = isfinite.(result.lz_bits)
        local b = lines!(ax, result.time_seconds[finite_lz], result.lz_bits[finite_lz]; color=(:firebrick3, 0.45), linewidth=2.0)
        lle_handle === nothing && (lle_handle = a)
        lz_handle === nothing && (lz_handle = b)
    end
    Legend(
        fig[1, 1],
        [lle_handle, lz_handle],
        ["Maximal Lyapunov exponent", "SSCS Abramov LZ76"],
        labelsize=22,
        framevisible=false,
        tellwidth=false,
        tellheight=false,
        halign=:right,
        valign=:top,
    )
    save(PLOT_PATH_058, fig, px_per_unit=2)
end

function main()
    indices = trajectory_indices_058()
    isempty(indices) && error("No trajectory convergence files found in $(TRAJECTORY_DIR_058).")
    results = [read_convergence_058(idx) for idx in indices]
    plot_ensemble_bits_058(results)
    mean_lle_bits = sum(result.lle_bits[end] for result in results) / length(results)
    mean_lz_bits = sum(result.lz_bits[end] for result in results) / length(results)
    @info "Wrote full SSCS LZ76 no-extra-discard ensemble in bits/s" plot=PLOT_PATH_058 n=length(results) mean_lle_bits mean_lz_bits
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
