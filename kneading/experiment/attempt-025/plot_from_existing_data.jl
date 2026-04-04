using Pkg

const ATTEMPT25_PLOT_ROOT = @__DIR__
const REPO_ROOT_025_PLOT = normpath(joinpath(ATTEMPT25_PLOT_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025_PLOT)

include(joinpath(ATTEMPT25_PLOT_ROOT, "main.jl"))

function main()
    summary = summarize_sources_025()
    println("attempt-025 read-only plotting scaffold")
    println("attempt-023 rows: $(summary.attempt023_results)")
    println("attempt-024 rows: $(summary.attempt024_results)")
    println("attempt-023 source: $(summary.attempt023_path)")
    println("attempt-024 columns: $(summary.attempt024_dir)")
    println("Implement the next plotting algorithm here, reusing the existing disk data only.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
