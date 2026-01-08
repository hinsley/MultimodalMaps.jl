# Shimizu-Morioka Return Map Construction
# Constructs a Poincaré return map from a slice near a saddle-focus equilibrium.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using OrdinaryDiffEq
using StaticArrays
using LinearAlgebra
using GLMakie

# ============================================================
# Parameters
# ============================================================
const α = 0.55f0
const λ = 0.75f0
const B = 0.0f0
const p = @SVector [α, λ, B]

# ============================================================
# Shimizu-Morioka System
# dx/dt = y
# dy/dt = x - λy - xz
# dz/dt = -αz + x²
# ============================================================
function shimizu_morioka(u, p, t)
    x, y, z = u
    α, λ, _ = p
    dx = y
    dy = x - λ * y - x * z
    dz = -α * z + x^2
    return SVector{3}(dx, dy, dz)
end

# ============================================================
# Simulation Configuration
# ============================================================
const eq_branch = :plus  # :plus or :minus saddle-focus equilibrium
const slice_mode = :unstable_xz  # :z for z-axis slice, :unstable_xz for unstable eigendirection in xz-plane
const return_mode = :x_maxima  # :x_maxima for return map of successive x maxima
const slice_span = (0.0f0, 5.0f0) # Distance along slice direction from equilibrium.
const slice_points = 800
const return_tspan = (0.0f0, 5000.0f0)
const min_return_time = 5.0f0
const initial_section_offset = -1.0f-3
const require_x_positive = true
const trajectory_dt = 0.05f0
const trajectory_stride = 10
const return_label = return_mode == :x_maxima ? "x" : (slice_mode == :z ? "z" : "s")

# ============================================================
# Equilibrium, Section, and Slice Geometry
# ============================================================
function equilibrium_point(α, branch)
    x = sqrt(α)
    x = branch == :minus ? -x : x
    return SVector{3,Float32}(x, 0.0f0, 1.0f0)
end

function jacobian(u)
    x, _, z = u
    return @SMatrix [
        0.0f0 1.0f0 0.0f0;
        (1.0f0-z) (-λ) (-x);
        (2.0f0*x) 0.0f0 (-α)
    ]
end

function normalize_vec(v)
    n = norm(v)
    return n == 0 ? v : v / n
end

function unstable_eigenvector(equilibrium)
    J = Float64.(jacobian(equilibrium))
    ev = eigen(J)
    idx = argmax(real(ev.values))
    return real(ev.vectors[:, idx])
end

function slice_direction_xz(equilibrium)
    unstable = unstable_eigenvector(equilibrium)
    vx = unstable[1]
    vz = unstable[3]
    if abs(vx) < 1.0e-8
        dir = SVector(0.0, 0.0, 1.0)
    else
        slope = vz / vx
        dir = SVector(1.0, 0.0, slope)
    end
    dir = normalize_vec(dir)
    return SVector{3,Float32}(Float32(dir[1]), 0.0f0, Float32(dir[3]))
end

const equilibrium = equilibrium_point(α, eq_branch)
const section_point = equilibrium
const section_normal = @SVector [0.0f0, 1.0f0, 0.0f0] # y = 0 plane
const slice_dir = slice_mode == :unstable_xz ? slice_direction_xz(equilibrium) : @SVector [0.0f0, 0.0f0, 1.0f0]

return_coordinate(u) = return_mode == :x_maxima ? u[1] : (slice_mode == :z ? u[3] : dot(u - section_point, slice_dir))

function first_return_coordinate(
    u0,
    section_point,
    section_normal,
    return_coordinate,
    tspan,
    min_return_time,
    require_x_positive,
    trajectory_dt,
    capture_trajectory,
    capture_minima
)
    hit = Ref(false)
    s_return = Ref(0.0f0)
    zmin_points = Point3f[]

    function return_condition(u, t, integrator)
        return u[2]
    end

    function return_affect!(integrator)
        if integrator.t <= min_return_time
            return
        end
        x, y, z = integrator.u
        ydot = x - λ * y - x * z
        is_max = ydot < 0
        if is_max && (!require_x_positive || x > 0)
            hit[] = true
            s_return[] = return_coordinate(integrator.u)
            terminate!(integrator)
        end
    end

    function zmin_condition(u, t, integrator)
        x, _, z = u
        return -α * z + x^2
    end

    function zmin_affect!(integrator)
        if !capture_minima
            return
        end
        x, y, z = integrator.u
        d2z = 2f0 * x * y
        if d2z > 0
            push!(zmin_points, Point3f(x, y, z))
        end
    end

    prob = ODEProblem{false}(shimizu_morioka, u0, tspan, p)
    cb_return = ContinuousCallback(
        return_condition,
        return_affect!,
        return_affect!;
        save_positions=(false, false)
    )
    cb_zmin = ContinuousCallback(
        zmin_condition,
        zmin_affect!,
        zmin_affect!;
        save_positions=(false, false)
    )
    cb = CallbackSet(cb_return, cb_zmin)
    traj_points = Point3f[]
    if capture_trajectory
        sol = solve(
            prob,
            Tsit5();
            callback=cb,
            saveat=trajectory_dt,
            save_everystep=false
        )
        traj_points = Point3f.(sol.u)
    else
        solve(
            prob,
            Tsit5();
            callback=cb,
            save_everystep=false,
            save_start=false,
            save_end=false
        )
    end

    return hit[] ? s_return[] : nothing, traj_points, zmin_points
end

# ============================================================
# Run Simulation
# ============================================================
function run_return_map()
    println("Shimizu-Morioka Return Map")
    println("  α = $α")
    println("  λ = $λ")
    println("  Equilibrium branch: $eq_branch")
    println("  Equilibrium: $equilibrium")
    println("  Return event: x maxima (y = 0, ydot < 0)")
    println("  Require x > 0: $require_x_positive")
    println("  Slice mode: $slice_mode")
    println("  Return mode: $return_mode")
    println("  Slice direction: $slice_dir")
    if slice_mode == :unstable_xz && abs(slice_dir[1]) > 1.0f-6
        println("  Slice dz/dx (unstable): $(slice_dir[3] / slice_dir[1])")
    end
    println("  Slice span: $slice_span")
    println("  Slice points: $slice_points")
    println("  Initial section offset: $initial_section_offset")
    println("  Return time span: $return_tspan")
    println("  Min return time: $min_return_time")
    println("  Trajectory dt: $trajectory_dt")
    println("  Trajectory stride: $trajectory_stride")
    println("  u0(s) = equilibrium + s * slice_dir + initial_section_offset * section_normal")
    println()

    println("Running return map sweep...")
    s_vals = range(slice_span[1], slice_span[2], length=slice_points)
    s0s = Float32[]
    s1s = Float32[]
    trajectories = Vector{Vector{Point3f}}()
    minima_sets = Vector{Vector{Point3f}}()
    failed = 0
    @time for (i, s0) in enumerate(s_vals)
        u0 = section_point + s0 * slice_dir + initial_section_offset * section_normal
        capture_trajectory = (i - 1) % trajectory_stride == 0
        s1, traj_points, zmin_points = first_return_coordinate(
            u0,
            section_point,
            section_normal,
            return_coordinate,
            return_tspan,
            min_return_time,
            require_x_positive,
            trajectory_dt,
            capture_trajectory,
            capture_trajectory
        )
        if capture_trajectory
            push!(trajectories, traj_points)
            push!(minima_sets, zmin_points)
        end
        if s1 === nothing
            failed += 1
        else
            push!(s0s, Float32(return_coordinate(u0)))
            push!(s1s, s1)
        end
    end

    println("\nCollected $(length(s0s)) return points (failed $failed)")

    # ============================================================
    # Construct Return Map
    # ============================================================
    if length(s0s) >= 2
        s_n = s0s
        s_np1 = s1s

        println("\nReturn map statistics:")
        println("  s_n range: [$(minimum(s_n)), $(maximum(s_n))]")
        println("  s_{n+1} range: [$(minimum(s_np1)), $(maximum(s_np1))]")

        # ============================================================
        # Plot Return Map
        # ============================================================
        println("\nPlotting return map...")

        fig = Figure(size=(1200, 800))
        ax = Axis(fig[1, 1],
            xlabel="$(return_label)_n",
            ylabel="$(return_label)_{n+1}",
            title="Shimizu-Morioka Return Map (α=$α, λ=$λ)",
            aspect=1
        )

        # Plot identity line for reference
        xmin = min(minimum(s_n), minimum(s_np1)) * 0.9
        xmax = max(maximum(s_n), maximum(s_np1)) * 1.1
        lines!(ax, [xmin, xmax], [xmin, xmax], color=:gray, linestyle=:dash, label="Identity")

        # Plot return map points
        scatter!(ax, s_n, s_np1, markersize=6, color=:blue, alpha=0.7)

        ax3 = Axis3(fig[1, 2],
            xlabel="x",
            ylabel="y",
            zlabel="z",
            title="Trajectories from Slice"
        )
        colormap = cgrad(:turbo, max(length(trajectories), 1), categorical=true)
        for (i, (traj, zmins)) in enumerate(zip(trajectories, minima_sets))
            if isempty(traj)
                continue
            end
            color = colormap[i]
            lines!(ax3, traj, color=color, linewidth=1.0)
            scatter!(ax3, [traj[1]], color=:red, markersize=6)
            scatter!(ax3, [traj[end]], color=:blue, markersize=6)
            if !isempty(zmins)
                scatter!(ax3, zmins, color=:black, markersize=4)
            end
        end

        display(fig)
        println("Return map displayed. Close the window to exit.")

        # Wait for the window to be closed
        wait(GLMakie.Screen(fig.scene))
    else
        println("\nInsufficient returns to construct return map")
    end

    println("\nReturn map construction complete.")
end

run_return_map()
