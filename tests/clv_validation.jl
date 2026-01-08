# CLV Validation Test
# Compares Wolfe-Samelson CLV computation against lyapunovvectors.jl (Ginelli)
# for the Shimizu-Morioka system

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using LinearAlgebra
using OrdinaryDiffEq
using DynamicalSystems
include(joinpath(@__DIR__, "..", "lyapunovvectors.jl"))

println("="^60)
println("CLV Validation: Wolfe-Samelson vs Ginelli")
println("="^60)

# ============================================================
# Shimizu-Morioka System
# ============================================================
function shimizu_morioka!(du, u, p, t)
    x, y, z = u
    α, λ = p
    du[1] = y
    du[2] = x - λ * y - x * z
    du[3] = -α * z + x^2
end

# Hand-coded Jacobian
function shimizu_morioka_jacobian!(J, u, p, t)
    x, y, z = u
    α, λ = p
    J[1, 1] = 0.0
    J[1, 2] = 1.0
    J[1, 3] = 0.0
    J[2, 1] = 1.0 - z
    J[2, 2] = -λ
    J[2, 3] = -x
    J[3, 1] = 2x
    J[3, 2] = 0.0
    J[3, 3] = -α
end

# ============================================================
# Reference: Ginelli via lyapunovvectors.jl
# ============================================================
function compute_clv_ginelli(u0, p; N=100, Ttr=50.0, Ttr_bkw=100, Δt=0.5)
    ds = CoupledODEs(shimizu_morioka!, u0, p;
        diffeq=(alg=Tsit5(), abstol=1e-9, reltol=1e-9))
    tands = TangentDynamicalSystem(ds; J=shimizu_morioka_jacobian!)

    result = clv(tands, N; Δt=Δt, Ttr=Ttr, Ttr_bkw=Ttr_bkw, show_progress=true)
    return result
end

# ============================================================
# Wolfe-Samelson Implementation (simplified, single trajectory)
# ============================================================

# 3×3 Modified Gram-Schmidt QR
function qr_3x3_mgs(M)
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

# Compute CLVs via Wolfe-Samelson intersection
function compute_clvs_wolfe_samelson(η, ξ)
    tol = 1e-10

    # CLV₁ is the first backward vector
    clv1 = η[:, 1]

    # CLV₂: null vector of 1×2 matrix [<ξ₁, η₁> <ξ₁, η₂>]
    c11 = dot(ξ[:, 1], η[:, 1])
    c12 = dot(ξ[:, 1], η[:, 2])
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
    r13 = dot(ξ[:, 1], η[:, 3])
    r21 = dot(ξ[:, 2], η[:, 1])
    r22 = dot(ξ[:, 2], η[:, 2])
    r23 = dot(ξ[:, 2], η[:, 3])
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

# Adjoint evolution: dξ/dt = -Jᵀ ξ, integrated backward in time
function mul_neg_jt!(out, x, z, α, λ, M)
    for j in 1:3
        m1 = M[1, j]
        m2 = M[2, j]
        m3 = M[3, j]
        out[1, j] = -(1 - z) * m2 - 2x * m3
        out[2, j] = -m1 + λ * m2
        out[3, j] = x * m2 + α * m3
    end
    return nothing
end

function evolve_adjoint_backward_rk4(M, x_start, x_end, p, dt, substeps)
    α, λ = p
    h = -dt / substeps

    M_work = copy(M)
    k1 = similar(M_work)
    k2 = similar(M_work)
    k3 = similar(M_work)
    k4 = similar(M_work)
    tmp = similar(M_work)

    dx = x_end[1] - x_start[1]
    dz = x_end[3] - x_start[3]

    for step in 1:substeps
        s0 = (step - 1) / substeps
        s1 = (step - 0.5) / substeps
        s2 = step / substeps

        x0 = x_start[1] + dx * s0
        z0 = x_start[3] + dz * s0
        x1 = x_start[1] + dx * s1
        z1 = x_start[3] + dz * s1
        x2 = x_start[1] + dx * s2
        z2 = x_start[3] + dz * s2

        mul_neg_jt!(k1, x0, z0, α, λ, M_work)
        @. tmp = M_work + (h / 2) * k1
        mul_neg_jt!(k2, x1, z1, α, λ, tmp)
        @. tmp = M_work + (h / 2) * k2
        mul_neg_jt!(k3, x1, z1, α, λ, tmp)
        @. tmp = M_work + h * k3
        mul_neg_jt!(k4, x2, z2, α, λ, tmp)

        @. M_work = M_work + (h / 6) * (k1 + 2k2 + 2k3 + k4)
    end

    return qr_3x3_mgs(M_work)
end

function compute_clv_wolfe_samelson(u0, p; N=100, N_transient=100, N_transient_bkw=100,
    dt_qr=0.5, dt_integrate=0.01, diffeq=(alg=Tsit5(), abstol=1e-9, reltol=1e-9))
    # Storage for forward pass
    total_steps = N + N_transient_bkw
    states = Vector{Vector{Float64}}(undef, total_steps)
    bsvs = Vector{Matrix{Float64}}(undef, N)

    ds = CoupledODEs(shimizu_morioka!, u0, p; diffeq=diffeq)
    tands = TangentDynamicalSystem(ds; J=shimizu_morioka_jacobian!)
    reinit!(tands, u0)
    set_deviations!(tands, Matrix{Float64}(I, 3, 3))

    steps_per_qr = round(Int, dt_qr / dt_integrate)
    Q_buffer = Matrix{Float64}(undef, 3, 3)
    R_buffer = Matrix{Float64}(undef, 3, 3)

    println("Forward pass (transient)...")
    # Forward transient
    for step in 1:N_transient
        step!(tands, dt_qr)
        _thin_qr_positive_diagonal!(Q_buffer, R_buffer, current_deviations(tands), 3)
        set_deviations!(tands, Q_buffer)
    end

    println("Forward pass (recording)...")
    # Forward pass with recording (including future steps for backward transient)
    for step in 1:total_steps
        step!(tands, dt_qr)
        _thin_qr_positive_diagonal!(Q_buffer, R_buffer, current_deviations(tands), 3)
        set_deviations!(tands, Q_buffer)

        states[step] = collect(current_state(tands))
        if step <= N
            bsvs[step] = copy(Q_buffer)
        end
    end

    println("Backward pass...")
    # Backward pass with FSV evolution
    clvs = Vector{Matrix{Float64}}(undef, N)
    M_fsv = Matrix{Float64}(I, 3, 3)

    # Backward transient through the extra future window
    for step in total_steps:-1:(N+1)
        M_fsv = evolve_adjoint_backward_rk4(M_fsv, states[step], states[step-1], p, dt_qr, steps_per_qr)
    end

    # Backward pass computing CLVs
    for idx in N:-1:1
        η = bsvs[idx]
        ξ = M_fsv

        clv1, clv2, clv3 = compute_clvs_wolfe_samelson(η, ξ)

        clvs[idx] = hcat(clv1, clv2, clv3)

        # Evolve FSV backward one QR step
        if idx > 1
            M_fsv = evolve_adjoint_backward_rk4(M_fsv, states[idx], states[idx-1], p, dt_qr, steps_per_qr)
        end
    end

    return (V=clvs, x=states[1:N])
end

# ============================================================
# Run Validation
# ============================================================
# Parameters
u0 = [0.1, 0.1, 0.1]
p = (0.375, 0.75)  # α, λ - chaotic regime

println("\n--- Computing CLVs with Ginelli (reference) ---")
@time ginelli_result = compute_clv_ginelli(u0, p; N=50, Ttr=25.0, Ttr_bkw=50, Δt=0.5)

println("\n--- Computing CLVs with Wolfe-Samelson ---")
@time ws_result = compute_clv_wolfe_samelson(u0, p; N=50, N_transient=50, N_transient_bkw=50,
    dt_qr=0.5, dt_integrate=0.01)

# ============================================================
# Compare Results
# ============================================================
println("\n" * "="^60)
println("COMPARISON")
println("="^60)

# Compare CLV₃ at several time points
println("\nCLV₃ comparison (should be parallel, |dot| ≈ 1):")
for idx in [1, 10, 25, 40, 50]
    clv3_ginelli = ginelli_result.V[idx][:, 3]
    clv3_ws = ws_result.V[idx][:, 3]

    # Normalize both
    clv3_ginelli ./= norm(clv3_ginelli)
    clv3_ws ./= norm(clv3_ws)

    dot_product = abs(dot(clv3_ginelli, clv3_ws))
    angle_deg = acos(min(1.0, dot_product)) * 180 / π

    println("  Step $idx: |dot| = $(round(dot_product, digits=6)), angle = $(round(angle_deg, digits=2))°")
end

# Compare CLV₁ and CLV₂
println("\nCLV₁ comparison (should be parallel, |dot| ≈ 1):")
for idx in [1, 25, 50]
    clv1_ginelli = ginelli_result.V[idx][:, 1]
    clv1_ws = ws_result.V[idx][:, 1]

    clv1_ginelli ./= norm(clv1_ginelli)
    clv1_ws ./= norm(clv1_ws)

    dot_product = abs(dot(clv1_ginelli, clv1_ws))
    angle_deg = acos(min(1.0, dot_product)) * 180 / π

    println("  Step $idx: |dot| = $(round(dot_product, digits=6)), angle = $(round(angle_deg, digits=2))°")
end

println("\nCLV₂ comparison (should be parallel, |dot| ≈ 1):")
for idx in [1, 25, 50]
    clv2_ginelli = ginelli_result.V[idx][:, 2]
    clv2_ws = ws_result.V[idx][:, 2]

    clv2_ginelli ./= norm(clv2_ginelli)
    clv2_ws ./= norm(clv2_ws)

    dot_product = abs(dot(clv2_ginelli, clv2_ws))
    angle_deg = acos(min(1.0, dot_product)) * 180 / π

    println("  Step $idx: |dot| = $(round(dot_product, digits=6)), angle = $(round(angle_deg, digits=2))°")
end

println("\n" * "="^60)
println("If angles are < 5°, the algorithms agree well.")
println("If angles are > 10°, there's likely a bug.")
println("="^60)
