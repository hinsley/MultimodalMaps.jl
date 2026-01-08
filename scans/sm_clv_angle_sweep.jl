# Shimizu-Morioka CLV Angle Sweep (Wolfe-Samelson Algorithm)
# GPU-accelerated computation of covariant Lyapunov vectors
# Computes minimal angle between CLV₃ and plane(CLV₁, CLV₂)
#
# Uses Wolfe-Samelson (2007) method which avoids storing R matrix history,
# computing CLVs as intersection of forward and backward subspaces.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using DiffEqGPU
using OrdinaryDiffEq
using StaticArrays
using Metal
using LinearAlgebra
using CairoMakie
using Statistics

# ============================================================
# Sweep Configuration
# ============================================================
const N_ALPHA = 50    # Number of points in alpha dimension
const N_LAMBDA = 50   # Number of points in lambda dimension

# Parameter ranges
const ALPHA_MIN = 0.15f0
const ALPHA_MAX = 0.65f0
const LAMBDA_MIN = 0.5f0
const LAMBDA_MAX = 0.95f0

# CLV computation parameters
const DT_QR = 0.5f0           # Time between QR orthonormalizations (for stability)
const N_QR_PER_OUTPUT = 10    # QR steps per CLV output point (coarse output)
const N_TRANSIENT = 100       # Forward QR steps for transient (discarded)
const N_TRANSIENT_BKW = 50    # Backward transient outputs (for FSV convergence)
const N_CLV_OUTPUTS = 50      # Number of CLV output points (coarse)
const DT_INTEGRATOR = 0.01f0  # Integrator timestep
const N_RK4_SUBSTEPS = 50     # Substeps for RK4 adjoint integration per QR step

# Derived: total time per output = DT_QR * N_QR_PER_OUTPUT
# Total QR steps for CLV = N_CLV_OUTPUTS * N_QR_PER_OUTPUT

# ============================================================
# Augmented Shimizu-Morioka System (12D) - Forward dynamics
# State: (x, y, z, M₁₁, M₂₁, M₃₁, M₁₂, M₂₂, M₃₂, M₁₃, M₂₃, M₃₃)
# Tangent dynamics: Ṁ = J * M
# ============================================================
function shimizu_morioka_tangent(u, p, t)
    x, y, z = u[1], u[2], u[3]
    α, λ = p[1], p[2]

    # State dynamics
    dx = y
    dy = x - λ * y - x * z
    dz = -α * z + x^2

    # Jacobian J = [0, 1, 0; 1-z, -λ, -x; 2x, 0, -α]
    m11, m21, m31 = u[4], u[5], u[6]
    m12, m22, m32 = u[7], u[8], u[9]
    m13, m23, m33 = u[10], u[11], u[12]

    # Ṁ = J * M
    dm11 = m21
    dm21 = (1 - z) * m11 - λ * m21 - x * m31
    dm31 = 2x * m11 - α * m31

    dm12 = m22
    dm22 = (1 - z) * m12 - λ * m22 - x * m32
    dm32 = 2x * m12 - α * m32

    dm13 = m23
    dm23 = (1 - z) * m13 - λ * m23 - x * m33
    dm33 = 2x * m13 - α * m33

    return SVector{12,Float32}(dx, dy, dz, dm11, dm21, dm31, dm12, dm22, dm32, dm13, dm23, dm33)
end

# ============================================================
# 3×3 QR via Modified Gram-Schmidt
# ============================================================
function qr_3x3_mgs(M::AbstractMatrix{T}) where T
    Q = similar(M)

    # Column 1
    r11 = sqrt(M[1, 1]^2 + M[2, 1]^2 + M[3, 1]^2)
    Q[1, 1] = M[1, 1] / r11
    Q[2, 1] = M[2, 1] / r11
    Q[3, 1] = M[3, 1] / r11

    # Column 2
    r12 = Q[1, 1] * M[1, 2] + Q[2, 1] * M[2, 2] + Q[3, 1] * M[3, 2]
    v2_1 = M[1, 2] - r12 * Q[1, 1]
    v2_2 = M[2, 2] - r12 * Q[2, 1]
    v2_3 = M[3, 2] - r12 * Q[3, 1]
    r22 = sqrt(v2_1^2 + v2_2^2 + v2_3^2)
    Q[1, 2] = v2_1 / r22
    Q[2, 2] = v2_2 / r22
    Q[3, 2] = v2_3 / r22

    # Column 3
    r13 = Q[1, 1] * M[1, 3] + Q[2, 1] * M[2, 3] + Q[3, 1] * M[3, 3]
    r23 = Q[1, 2] * M[1, 3] + Q[2, 2] * M[2, 3] + Q[3, 2] * M[3, 3]
    v3_1 = M[1, 3] - r13 * Q[1, 1] - r23 * Q[1, 2]
    v3_2 = M[2, 3] - r13 * Q[2, 1] - r23 * Q[2, 2]
    v3_3 = M[3, 3] - r13 * Q[3, 1] - r23 * Q[3, 2]
    r33 = sqrt(v3_1^2 + v3_2^2 + v3_3^2)
    Q[1, 3] = v3_1 / r33
    Q[2, 3] = v3_2 / r33
    Q[3, 3] = v3_3 / r33

    return Q
end

# ============================================================
# Compute CLVs using Wolfe-Samelson intersection method
# η = backward SVs (from forward evolution), ξ = forward SVs (from adjoint)
# ============================================================
function compute_clvs_wolfe_samelson(η::AbstractMatrix{T}, ξ::AbstractMatrix{T}) where T
    tol = T(1e-10)

    # CLV₁ is the first backward vector
    clv1 = η[:, 1]

    # CLV₂: null vector of 1×2 matrix [<ξ₁, η₁> <ξ₁, η₂>]
    c11 = ξ[1, 1] * η[1, 1] + ξ[2, 1] * η[2, 1] + ξ[3, 1] * η[3, 1]
    c12 = ξ[1, 1] * η[1, 2] + ξ[2, 1] * η[2, 2] + ξ[3, 1] * η[3, 2]
    y1 = -c12
    y2 = c11
    n2 = sqrt(y1^2 + y2^2)
    if n2 < tol
        clv2 = η[:, 2]
    else
        clv2 = (y1 * η[:, 1] + y2 * η[:, 2]) / n2
    end

    # CLV₃: null vector of 2×3 matrix C (cross of rows)
    r11 = c11
    r12 = c12
    r13 = ξ[1, 1] * η[1, 3] + ξ[2, 1] * η[2, 3] + ξ[3, 1] * η[3, 3]

    r21 = ξ[1, 2] * η[1, 1] + ξ[2, 2] * η[2, 1] + ξ[3, 2] * η[3, 1]
    r22 = ξ[1, 2] * η[1, 2] + ξ[2, 2] * η[2, 2] + ξ[3, 2] * η[3, 2]
    r23 = ξ[1, 2] * η[1, 3] + ξ[2, 2] * η[2, 3] + ξ[3, 2] * η[3, 3]

    y3_1 = r12 * r23 - r13 * r22
    y3_2 = r13 * r21 - r11 * r23
    y3_3 = r11 * r22 - r12 * r21
    n3 = sqrt(y3_1^2 + y3_2^2 + y3_3^2)
    if n3 < tol
        clv3 = η[:, 3]
    else
        clv3 = (y3_1 * η[:, 1] + y3_2 * η[:, 2] + y3_3 * η[:, 3]) / n3
    end

    return clv1, clv2, clv3
end

# ============================================================
# Adjoint evolution: dξ/dt = -Jᵀ ξ, integrated backward in time
# ============================================================
function mul_neg_jt!(out::AbstractMatrix{T}, x::T, z::T, α::T, λ::T, M::AbstractMatrix{T}) where T
    @inbounds for j in 1:3
        m1 = M[1, j]
        m2 = M[2, j]
        m3 = M[3, j]
        out[1, j] = -(one(T) - z) * m2 - (T(2) * x) * m3
        out[2, j] = -m1 + λ * m2
        out[3, j] = x * m2 + α * m3
    end
    return nothing
end

function evolve_adjoint_backward_rk4(ξ::Matrix{T}, x_start::AbstractVector{T}, x_end::AbstractVector{T}, p, dt::T) where T
    α, λ = p[1], p[2]
    h = -dt / T(N_RK4_SUBSTEPS)

    ξ_work = copy(ξ)
    k1 = similar(ξ_work)
    k2 = similar(ξ_work)
    k3 = similar(ξ_work)
    k4 = similar(ξ_work)
    tmp = similar(ξ_work)

    # Linear interpolation between stored states approximates the Jacobian over the QR step.
    dx = x_end[1] - x_start[1]
    dz = x_end[3] - x_start[3]

    @inbounds for step in 1:N_RK4_SUBSTEPS
        s0 = T(step - 1) / T(N_RK4_SUBSTEPS)
        s1 = T(step) / T(N_RK4_SUBSTEPS) - T(0.5) / T(N_RK4_SUBSTEPS)
        s2 = T(step) / T(N_RK4_SUBSTEPS)

        x0 = x_start[1] + dx * s0
        z0 = x_start[3] + dz * s0
        x1 = x_start[1] + dx * s1
        z1 = x_start[3] + dz * s1
        x2 = x_start[1] + dx * s2
        z2 = x_start[3] + dz * s2

        mul_neg_jt!(k1, x0, z0, α, λ, ξ_work)
        @. tmp = ξ_work + (h / 2) * k1
        mul_neg_jt!(k2, x1, z1, α, λ, tmp)
        @. tmp = ξ_work + (h / 2) * k2
        mul_neg_jt!(k3, x1, z1, α, λ, tmp)
        @. tmp = ξ_work + h * k3
        mul_neg_jt!(k4, x2, z2, α, λ, tmp)

        @. ξ_work = ξ_work + (h / 6) * (k1 + 2k2 + 2k3 + k4)
    end

    return qr_3x3_mgs(ξ_work)
end

# ============================================================
# Angle between vector and plane
# ============================================================
function angle_to_plane(v1, v2, v3)
    n = cross(v1, v2)
    n_norm = norm(n)
    if n_norm < 1f-10
        return 0f0
    end
    n_hat = n / n_norm
    return asin(min(1f0, abs(dot(v3, n_hat))))
end

# ============================================================
# Main CLV Angle Sweep using Wolfe-Samelson Algorithm
# ============================================================
function run_clv_angle_sweep()
    println("="^60)
    println("Shimizu-Morioka CLV Angle Sweep (Wolfe-Samelson)")
    println("="^60)

    # Generate parameter grid
    alphas = range(ALPHA_MIN, ALPHA_MAX, length=N_ALPHA)
    lambdas = range(LAMBDA_MIN, LAMBDA_MAX, length=N_LAMBDA)

    params = [@SVector [Float32(α), Float32(λ)] for α in alphas, λ in lambdas]
    params_flat = vec(params)
    n_trajectories = length(params_flat)
    output_qr_steps = N_CLV_OUTPUTS * N_QR_PER_OUTPUT
    bkw_qr_steps = N_TRANSIENT_BKW * N_QR_PER_OUTPUT
    total_qr_steps = output_qr_steps + bkw_qr_steps

    println("Alpha: [$ALPHA_MIN, $ALPHA_MAX] × $N_ALPHA")
    println("Lambda: [$LAMBDA_MIN, $LAMBDA_MAX] × $N_LAMBDA")
    println("Trajectories: $n_trajectories")
    println("Transient: $N_TRANSIENT QR steps")
    println("CLV phase: $N_CLV_OUTPUTS outputs × $N_QR_PER_OUTPUT QR/output")
    println("Backward transient: $N_TRANSIENT_BKW outputs ($bkw_qr_steps QR)")
    println("Time per output: $(DT_QR * N_QR_PER_OUTPUT)s")
    println()

    # Memory estimate (Wolfe-Samelson: no R history!)
    mem_states = n_trajectories * 12 * 4  # Current states
    mem_state_history = n_trajectories * total_qr_steps * 3 * 4   # State at QR resolution
    mem_bsv = n_trajectories * N_CLV_OUTPUTS * 9 * 4              # BSV per output
    println("Memory Estimates:")
    println("  Current states: $(round(mem_states / 1024, digits=1)) KB")
    println("  State history (QR): $(round(mem_state_history / 1024^2, digits=1)) MB")
    println("  BSV snapshots: $(round(mem_bsv / 1024^2, digits=1)) MB")
    println("  (No R matrix history needed!)")
    println()

    # ========== Phase 1: Forward evolution ==========
    println("Phase 1: Forward evolution...")
    t_forward_start = time()

    # Initial condition
    u0 = @SVector Float32[0.1, 0.1, 0.1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    tspan = (0f0, DT_QR)

    prob = ODEProblem{false}(shimizu_morioka_tangent, u0, tspan, params_flat[1])

    # Storage for QR-resolution states (output window + backward transient)
    state_history = [Matrix{Float32}(undef, 3, total_qr_steps) for _ in 1:n_trajectories]
    bsv_snapshots = [Array{Float32}(undef, 3, 3, N_CLV_OUTPUTS) for _ in 1:n_trajectories]

    # Current states
    current_states = [copy(u0) for _ in 1:n_trajectories]

    # Build initial problems
    probs = map(params_flat) do p
        DiffEqGPU.make_prob_compatible(remake(prob, p=p))
    end
    probs_gpu = MtlArray(probs)

    # Helper: run one QR step, optionally storing the state at this QR index
    function run_qr_step!(probs_gpu, store_idx::Int)
        Metal.@sync _, us = DiffEqGPU.vectorized_asolve(
            probs_gpu, prob, GPUTsit5();
            save_everystep=false, dt=DT_INTEGRATOR
        )

        us_cpu = Array(us)
        store_state = store_idx > 0
        for traj in 1:n_trajectories
            u_final = us_cpu[traj]
            M = Float32[u_final[4] u_final[7] u_final[10];
                u_final[5] u_final[8] u_final[11];
                u_final[6] u_final[9] u_final[12]]
            Q = qr_3x3_mgs(M)
            current_states[traj] = @SVector Float32[
                u_final[1], u_final[2], u_final[3],
                Q[1, 1], Q[2, 1], Q[3, 1],
                Q[1, 2], Q[2, 2], Q[3, 2],
                Q[1, 3], Q[2, 3], Q[3, 3]
            ]

            if store_state
                state_history[traj][1, store_idx] = u_final[1]
                state_history[traj][2, store_idx] = u_final[2]
                state_history[traj][3, store_idx] = u_final[3]
            end
        end

        probs = map(enumerate(params_flat)) do (i, p)
            DiffEqGPU.make_prob_compatible(remake(prob, u0=current_states[i], p=p))
        end
        return MtlArray(probs)
    end

    # Transient phase (N_TRANSIENT QR steps)
    print("  Transient ($N_TRANSIENT QR steps): ")
    for qr_step in 1:N_TRANSIENT
        probs_gpu = run_qr_step!(probs_gpu, 0)
        if qr_step % 20 == 0
            print("$qr_step ")
        end
    end
    println("Done!")

    # CLV window + backward transient (QR resolution)
    print("  CLV window ($output_qr_steps QR) + future transient ($bkw_qr_steps QR): ")
    for store_idx in 1:total_qr_steps
        probs_gpu = run_qr_step!(probs_gpu, store_idx)

        if store_idx % N_QR_PER_OUTPUT == 0 && store_idx <= output_qr_steps
            output_idx = div(store_idx, N_QR_PER_OUTPUT)
            for traj in 1:n_trajectories
                s = current_states[traj]
                # Store current BSVs (columns 4-12 of state are the Q matrix)
                bsv_snapshots[traj][1, 1, output_idx] = s[4]
                bsv_snapshots[traj][2, 1, output_idx] = s[5]
                bsv_snapshots[traj][3, 1, output_idx] = s[6]
                bsv_snapshots[traj][1, 2, output_idx] = s[7]
                bsv_snapshots[traj][2, 2, output_idx] = s[8]
                bsv_snapshots[traj][3, 2, output_idx] = s[9]
                bsv_snapshots[traj][1, 3, output_idx] = s[10]
                bsv_snapshots[traj][2, 3, output_idx] = s[11]
                bsv_snapshots[traj][3, 3, output_idx] = s[12]
            end

            if output_idx % 10 == 0
                print("$output_idx ")
            end
        end
    end
    t_forward_end = time()
    println("Done! ($(round(t_forward_end - t_forward_start, digits=1))s)")

    # ========== Phase 2: Backward pass with CLV computation ==========
    println("Phase 2: Backward pass (computing CLVs)...")
    t_backward_start = time()

    min_angles = zeros(Float32, n_trajectories)

    # Process each trajectory
    Threads.@threads for traj in 1:n_trajectories
        p = params_flat[traj]
        angles = Vector{Float32}(undef, N_CLV_OUTPUTS)

        # Initialize adjoint tangent matrix as identity at final time
        ξ = Matrix{Float32}(I, 3, 3)

        # Backward transient: let ξ converge before recording CLVs
        for step in total_qr_steps:-1:(output_qr_steps + 1)
            @views x_start = state_history[traj][:, step]
            @views x_end = state_history[traj][:, step - 1]
            ξ = evolve_adjoint_backward_rk4(ξ, x_start, x_end, p, DT_QR)
        end

        # Backward sweep through QR steps, recording at output indices
        for step in output_qr_steps:-1:1
            if step % N_QR_PER_OUTPUT == 0
                output_idx = div(step, N_QR_PER_OUTPUT)
                η = bsv_snapshots[traj][:, :, output_idx]
                clv1, clv2, clv3 = compute_clvs_wolfe_samelson(η, ξ)
                angles[output_idx] = angle_to_plane(clv1, clv2, clv3)
            end

            if step > 1
                @views x_start = state_history[traj][:, step]
                @views x_end = state_history[traj][:, step - 1]
                ξ = evolve_adjoint_backward_rk4(ξ, x_start, x_end, p, DT_QR)
            end
        end

        min_angles[traj] = minimum(angles)
    end

    t_backward_end = time()
    println("Done! ($(round(t_backward_end - t_backward_start, digits=1))s)")

    # Reshape to grid
    min_angles_grid = reshape(min_angles, N_ALPHA, N_LAMBDA)

    println()
    println("Timing Summary:")
    println("  Forward pass:  $(round(t_forward_end - t_forward_start, digits=1))s")
    println("  Backward pass: $(round(t_backward_end - t_backward_start, digits=1))s")
    println("  Total:         $(round(t_backward_end - t_forward_start, digits=1))s")
    println()

    # ========== Phase 3: Plot heatmap ==========
    println("Generating heatmap...")

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel="α",
        ylabel="λ",
        title="Min angle between CLV₃ and plane(CLV₁, CLV₂)"
    )

    hm = heatmap!(ax,
        collect(alphas), collect(lambdas),
        min_angles_grid,
        colormap=:viridis
    )

    Colorbar(fig[1, 2], hm, label="Angle (radians)")

    output_path = joinpath(@__DIR__, "sm_clv_angle_heatmap.png")
    save(output_path, fig)
    println("Saved heatmap to: $output_path")

    return min_angles_grid, alphas, lambdas
end

# Run the sweep
@time min_angles, alphas, lambdas = run_clv_angle_sweep()

println("\nAngle statistics:")
println("  Min: $(minimum(min_angles)) rad ($(rad2deg(minimum(min_angles)))°)")
println("  Max: $(maximum(min_angles)) rad ($(rad2deg(maximum(min_angles)))°)")
println("  Mean: $(mean(min_angles)) rad ($(rad2deg(mean(min_angles)))°)")
