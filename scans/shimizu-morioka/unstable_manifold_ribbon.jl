# Shimizu-Morioka Unstable Manifold Visualization (Forward-Only)
# Uses a single forward pass with a transient period for tangent vector alignment.

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

function shimizu_morioka(u, p, t)
    x, y, z = u
    α, λ, _ = p
    dx = y
    dy = x - λ * y - x * z
    dz = -α * z + x^2
    return SVector{3}(dx, dy, dz)
end

function jacobian(u, p)
    x, y, z = u
    α, λ, _ = p
    return @SMatrix [
        0.0f0 1.0f0 0.0f0;
        (1.0f0-z) (-λ) (-x);
        (2.0f0*x) 0.0f0 (-α)
    ]
end

function combined_dynamics(U, p, t)
    u = SVector{3}(U[1], U[2], U[3])
    v = SVector{3}(U[4], U[5], U[6])
    du = shimizu_morioka(u, p, t)
    dv = jacobian(u, p) * v
    return SVector{6}(du[1], du[2], du[3], dv[1], dv[2], dv[3])
end

# Normalization callback
const last_normalize_time = Ref(0.0f0)
const normalize_interval = 0.05f0

function should_normalize(u, t, integrator)
    return t - last_normalize_time[] >= normalize_interval
end

function normalize_callback!(integrator)
    v = SVector{3}(integrator.u[4], integrator.u[5], integrator.u[6])
    n = norm(v)
    if n > 0
        integrator.u = SVector{6}(integrator.u[1], integrator.u[2], integrator.u[3],
            v[1] / n, v[2] / n, v[3] / n)
    end
    last_normalize_time[] = integrator.t
end

function unstable_direction_at_origin(p)
    α, λ, _ = p
    J = [0.0 1.0 0.0; 1.0 -λ 0.0; 0.0 0.0 -α]
    ev = eigen(J)
    idx = argmax(real.(ev.values))
    v = real.(ev.vectors[:, idx])
    return SVector{3,Float32}(v / norm(v))
end

# ============================================================
# Configuration
# ============================================================
const Ttr = 50.0f0                   # Alignment transient (let tangent vector settle)
const Tvis = 100.0f0                 # Visual duration
const Δt = 0.03f0                    # Visual density
const segment_length = 0.075f0

# Initial condition
const origin = @SVector [0.0f0, 0.0f0, 0.0f0]
const unstable_dir = unstable_direction_at_origin(p)
const epsilon = 1f-4                # Start even closer to origin for more transient room
const u0 = origin + epsilon * unstable_dir
const v0 = unstable_dir

const U0 = SVector{6}(u0[1], u0[2], u0[3], v0[1], v0[2], v0[3])

# ============================================================
# Execution
# ============================================================
println("Shimizu-Morioka Forward Visualization")
println("  Transient: $Ttr, Visual: $Tvis, Density: $Δt")

prob = ODEProblem{false}(combined_dynamics, U0, (0.0f0, Ttr + Tvis), p)
cb = DiscreteCallback(should_normalize, normalize_callback!)

println("Integrating...")
@time sol = solve(prob, Tsit5(); callback=cb, saveat=(Ttr:Δt:(Ttr+Tvis)), save_start=false, save_everystep=false)

# Filter points to be absolutely sure we only have the visual range
visual_mask = sol.t .>= Ttr - 1e-6
visual_u = sol.u[visual_mask]

println("Processing $(length(visual_u)) visual points...")
trajectory = [Point3f(u[1], u[2], u[3]) for u in visual_u]
# Direct forward vectors (already normalized by callback, but re-normalize for safety)
vectors = [normalize(SVector{3}(u[4], u[5], u[6])) for u in visual_u]
flows = [normalize(shimizu_morioka(SVector{3}(u[1], u[2], u[3]), p, 0.0f0)) for u in visual_u]

function orthogonalize(v, flow)
    v_orth = v - dot(v, flow) * flow
    n = norm(v_orth)
    return n > 1e-8 ? v_orth / n : v
end

orth_vectors = [orthogonalize(vectors[i], flows[i]) for i in 1:length(vectors)]

# Build segments
segment_points = Vector{Point3f}(undef, 2 * length(trajectory))
for i in 1:length(trajectory)
    pt = trajectory[i]
    v = orth_vectors[i]
    segment_points[2i-1] = pt
    segment_points[2i] = Point3f(pt[1] + segment_length * v[1],
        pt[2] + segment_length * v[2],
        pt[3] + segment_length * v[3])
end

println("Plotting...")
fig = Figure(size=(1200, 900))
ax = Axis3(fig[1, 1], aspect=:data, title="Shimizu-Morioka: Forward Tangent (Aligned)")
linesegments!(ax, segment_points, color=repeat(1:length(trajectory), inner=2), colormap=:viridis)
lines!(ax, trajectory, color=:red, linewidth=2)
scatter!(ax, [Point3f(0, 0, 0)], color=:black, markersize=10, marker=:cross)

display(fig)
wait(GLMakie.Screen(fig.scene))
