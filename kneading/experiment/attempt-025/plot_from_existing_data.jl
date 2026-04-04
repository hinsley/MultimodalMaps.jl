using Pkg

const ATTEMPT25_PLOT_ROOT = @__DIR__
const REPO_ROOT_025_PLOT = normpath(joinpath(ATTEMPT25_PLOT_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_025_PLOT)

include(joinpath(ATTEMPT25_PLOT_ROOT, "main.jl"))

function main()
    println("attempt-025 no longer plots from previously computed files by default.")
    println("Use kneading/experiment/attempt-025/contours.jl to recompute the |x|-max dataset and render the skip-adjusted contour plot.")
    println("Configured plot cap: $(ATTEMPT025_PLOT_ITERATE_CAP)")
    println("Configured stored |x|-max iterate budget: $(ATTEMPT025_MAX_EVENT_ITERATES)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
