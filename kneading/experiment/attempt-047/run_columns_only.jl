using Pkg

const ATTEMPT047_ROOT = @__DIR__
const REPO_ROOT_047 = normpath(joinpath(ATTEMPT047_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_047)

include(joinpath(ATTEMPT047_ROOT, "contours.jl"))

function verify_columns_047()
    total_cols = length(ALPHAS_025)
    total_rows = length(LAMBDAS_025)
    for col_idx in eachindex(ALPHAS_025)
        path = column_path_025(col_idx)
        row_is_complete_025(path, total_rows) || error("Column $(col_idx) incomplete after recompute: $(path)")
    end
    println("Verified $(total_cols) complete column files in $(SWEEP_DIR_025)")
end

function main()
    println("Running attempt-047 column recompute only.")
    println("Grid: $(length(ALPHAS_025)) alpha points x $(length(LAMBDAS_025)) lambda points")
    println("Threads: $(Threads.nthreads()), stored |x|-max iterates: $(ATTEMPT025_MAX_EVENT_ITERATES)")
    println("Sweep dir: $(SWEEP_DIR_025)")
    flush(stdout)

    mkpath(SWEEP_DIR_025)
    run_or_resume_columns_025()
    verify_columns_047()

    usable = 0
    total = length(ALPHAS_025) * length(LAMBDAS_025)
    scan_column_files_025() do result
        result_usable_025(result) && (usable += 1)
    end
    println("Usable points after recompute: $(usable) / $(total)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
