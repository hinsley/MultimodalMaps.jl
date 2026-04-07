using Pkg

const ATTEMPT028_XY_ROOT = @__DIR__
include(joinpath(ATTEMPT028_XY_ROOT, "main.jl"))

using CairoMakie
using Printf
using StaticArrays

const ATTEMPT028_XY_RESULTS_PATH = joinpath(
    ATTEMPT028_XY_ROOT,
    "alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv",
)
const ATTEMPT028_XY_OUTPUT_TAG = get(
    ENV,
    "ATTEMPT028_XY_OUTPUT_TAG",
    "alpha_slice_critical_neighborhood_xy_overlay",
)
const ATTEMPT028_XY_PNG_PATH = joinpath(ATTEMPT028_XY_ROOT, "$(ATTEMPT028_XY_OUTPUT_TAG).png")

const ATTEMPT028_XY_GOOD_ALPHA = parse(Float64, get(ENV, "ATTEMPT028_XY_GOOD_ALPHA", "0.235667833917"))
const ATTEMPT028_XY_BAD_ALPHA = parse(Float64, get(ENV, "ATTEMPT028_XY_BAD_ALPHA", "0.226913456728"))
const ATTEMPT028_XY_NEIGHBOR_RADIUS = parse(Float64, get(ENV, "ATTEMPT028_XY_NEIGHBOR_RADIUS", "0.02"))
const ATTEMPT028_XY_NEIGHBOR_COUNT = parse(Int, get(ENV, "ATTEMPT028_XY_NEIGHBOR_COUNT", "9"))
const ATTEMPT028_XY_SAVE_DT = parse(Float64, get(ENV, "ATTEMPT028_XY_SAVE_DT", "0.002"))
const ATTEMPT028_XY_WIDTH = parse(Int, get(ENV, "ATTEMPT028_XY_WIDTH", "1600"))
const ATTEMPT028_XY_HEIGHT = parse(Int, get(ENV, "ATTEMPT028_XY_HEIGHT", "1200"))
const ATTEMPT028_XY_LW = parse(Float64, get(ENV, "ATTEMPT028_XY_LW", "1.25"))
const ATTEMPT028_XY_CRIT_LW = parse(Float64, get(ENV, "ATTEMPT028_XY_CRIT_LW", "4.0"))

struct ContinuationPoint028
    alpha::Float64
    status::String
    current_state::SVector{3, Float64}
end

function load_continuation_point_028(path::String, target_alpha::Float64)
    open(path, "r") do io
        header = split(chomp(readline(io)), '\t')
        colidx = Dict(name => i for (i, name) in enumerate(header))
        best = nothing
        best_diff = Inf
        for line in eachline(io)
            fields = split(chomp(line), '\t')
            alpha = parse(Float64, fields[colidx["alpha"]])
            diff = abs(alpha - target_alpha)
            if diff < best_diff
                best_diff = diff
                best = ContinuationPoint028(
                    alpha,
                    fields[colidx["status"]],
                    SVector(
                        parse(Float64, fields[colidx["current_x"]]),
                        parse(Float64, fields[colidx["current_y"]]),
                        parse(Float64, fields[colidx["current_z"]])
                    ),
                )
            end
        end
        best === nothing && error("No continuation rows found in $path")
        best_diff <= 1e-9 || error("Could not find alpha=$target_alpha in $path")
        return best
    end
end

function integrate_to_first_absxmax_path_028(alpha::Float64, lambda::Float64, u0::SVector{3, Float64}; target_next_sign::Int=ATTEMPT028_TARGET_NEXT_SIGN)
    p = [alpha, lambda]
    armed = Ref(false)
    hit = Ref(false)

    condition(u, t, integrator) = armed[] ? u[2] : 1.0

    function affect!(integrator)
        x = integrator.u[1]
        z = integrator.u[3]
        sign_ok = target_next_sign < 0 ? x < -ATTEMPT028_MIN_SECTION_X : x > ATTEMPT028_MIN_SECTION_X
        if sign_ok && z > 1.0
            hit[] = true
            terminate!(integrator)
        end
    end

    arm_cb = PresetTimeCallback([ATTEMPT028_MIN_EVENT_TIME], integrator -> begin
        armed[] = true
    end)
    event_cb = ContinuousCallback(
        condition,
        affect!;
        rootfind=true,
        save_positions=(false, true),
    )
    cb = CallbackSet(arm_cb, event_cb)

    prob = ODEProblem(
        shimizu_morioka_state_028!,
        collect(u0),
        (0.0, ATTEMPT028_EVENT_T_END),
        p,
    )
    sol = solve(
        prob,
        Vern9();
        callback=cb,
        adaptive=true,
        dt=ATTEMPT028_EVENT_DT,
        dtmax=ATTEMPT028_EVENT_DT,
        abstol=ATTEMPT028_EVENT_ABSTOL,
        reltol=ATTEMPT028_EVENT_RELTOL,
        maxiters=ATTEMPT028_EVENT_MAXITERS,
        saveat=ATTEMPT028_XY_SAVE_DT,
        save_start=true,
        save_end=true,
        save_everystep=false,
    )

    hit[] || error("No first-return |x|-maximum found for alpha=$alpha and initial state=$(Tuple(u0))")
    xs = Float64[]
    ys = Float64[]
    sizehint!(xs, length(sol.u))
    sizehint!(ys, length(sol.u))
    for u in sol.u
        push!(xs, u[1])
        push!(ys, u[2])
    end
    return xs, ys
end

function local_neighborhood_states_028(center::SVector{3, Float64}, radius::Float64, count::Int)
    count >= 3 || error("Neighborhood count must be at least 3")
    offsets = collect(range(-radius, radius, length=count))
    states = SVector{3, Float64}[]
    sizehint!(states, count)
    for δx in offsets
        x = center[1] + δx
        x > ATTEMPT028_MIN_SECTION_X || error("Neighborhood reaches invalid x <= 0; reduce ATTEMPT028_XY_NEIGHBOR_RADIUS")
        push!(states, @SVector [x, center[2], center[3]])
    end
    return offsets, states
end

function add_family_028!(ax, alpha::Float64, center::SVector{3, Float64}, color, label::String)
    offsets, states = local_neighborhood_states_028(center, ATTEMPT028_XY_NEIGHBOR_RADIUS, ATTEMPT028_XY_NEIGHBOR_COUNT)
    critical_index = findfirst(iszero, offsets)
    critical_index === nothing && error("Neighborhood offsets must include 0")
    rgb = RGBf(color)

    for (i, state) in enumerate(states)
        xs, ys = integrate_to_first_absxmax_path_028(alpha, ATTEMPT028_LAMBDA, state)
        is_critical = i == critical_index
        lw = is_critical ? ATTEMPT028_XY_CRIT_LW : ATTEMPT028_XY_LW
        alpha_line = is_critical ? 1.0 : 0.28
        linecolor = is_critical ? rgb : RGBAf(rgb.r, rgb.g, rgb.b, alpha_line)
        lines!(ax, xs, ys; color=linecolor, linewidth=lw)
        scatter!(ax, [xs[1]], [ys[1]]; color=linecolor, markersize=is_critical ? 18 : 7)
        scatter!(ax, [xs[end]], [ys[end]]; color=linecolor, markersize=is_critical ? 14 : 5, marker=:diamond)
    end

    scatter!(
        ax,
        [center[1]],
        [center[2]];
        color=:white,
        strokecolor=rgb,
        strokewidth=3,
        marker=:star5,
        markersize=28,
        label=label,
    )
end

function main()
    good = load_continuation_point_028(ATTEMPT028_XY_RESULTS_PATH, ATTEMPT028_XY_GOOD_ALPHA)
    bad = load_continuation_point_028(ATTEMPT028_XY_RESULTS_PATH, ATTEMPT028_XY_BAD_ALPHA)

    fig = Figure(size=(ATTEMPT028_XY_WIDTH, ATTEMPT028_XY_HEIGHT), backgroundcolor=:white)
    ax = Axis(
        fig[1, 1];
        xlabel="x",
        ylabel="y",
        title="Attempt-028 critical neighborhoods to first |x|-maximum return",
        aspect=DataAspect(),
    )

    add_family_028!(
        ax,
        good.alpha,
        good.current_state,
        RGBf(0.1, 0.45, 0.95),
        @sprintf("alpha = %.12f critical IC", good.alpha),
    )
    add_family_028!(
        ax,
        bad.alpha,
        bad.current_state,
        RGBf(0.95, 0.4, 0.1),
        @sprintf("alpha = %.12f critical IC", bad.alpha),
    )

    axislegend(ax; position=:rb, framevisible=true, backgroundcolor=RGBAf(1, 1, 1, 0.85))
    xlims!(ax, nothing, nothing)
    ylims!(ax, nothing, nothing)
    save(ATTEMPT028_XY_PNG_PATH, fig, px_per_unit=2)
    println("Saved $(ATTEMPT028_XY_PNG_PATH)")
end

main()
