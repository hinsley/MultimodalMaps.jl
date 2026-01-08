import LinearAlgebra
import ProgressMeter
import Random
export clv, clv_wolfe_samelson

"""
    clv(ds::DynamicalSystem, N, k = dimension(ds); kwargs...) → (V, λ, x, t)

Compute the Covariant Lyapunov Vectors (CLVs) of `ds` using the Ginelli algorithm [^Ginelli2007].
This is a convenience wrapper that initializes a [`TangentDynamicalSystem`](@ref) with
automatic Jacobian computation and then calls `clv(tands, N; kwargs...)`.

To use a hand-coded Jacobian, create a [`TangentDynamicalSystem`](@ref) manually and call
the lower-level method directly.

See [`clv(::TangentDynamicalSystem, ...)`](@ref) for the full documentation, keyword arguments,
and algorithm description.

See also [`lyapunovspectrum`](@ref), [`lyapunov`](@ref).

[^Ginelli2007]: F. Ginelli *et al.*, Phys. Rev. Lett. **99**, 130601 (2007)
"""
function clv(ds::CoreDynamicalSystem, N, k=dimension(ds); kwargs...)
    tands = TangentDynamicalSystem(ds; k)
    return clv(tands, N; kwargs...)
end

"""
    clv(tands::TangentDynamicalSystem, N::Int; kwargs...) → (V, λ, x, t)

Compute the Covariant Lyapunov Vectors (CLVs) using the Ginelli algorithm [^Ginelli2007].
CLVs are intrinsic, norm-independent vectors that characterize the local stability directions
of a dynamical system, in contrast to Gram-Schmidt vectors which depend on the chosen norm.

Use this method for looping over different initial conditions or parameters by
calling [`reinit!`](@ref) on `tands`, or when you have a hand-coded Jacobian.

## Returns

A named tuple `(V, λ, x, t)` where:
- `V::Vector{Matrix}`: CLVs at each stored time step. `V[i]` is a `D×k` matrix whose columns
  are the `k` CLVs at time `t[i]`, ordered from most expanding to most contracting.
- `λ::Vector`: Lyapunov exponents (growth rates along CLVs), ordered from largest to smallest.
- `x::Vector`: State vectors at each stored time step.
- `t::Vector`: Time stamps corresponding to each stored CLV snapshot.

## Keyword arguments

* `u0 = current_state(tands)`: Initial state.
* `Ttr = 0`: Forward transient time to evolve before storing CLVs.
  Both the system and deviation vectors are evolved during this time.
* `Ttr_bkw = N`: Backward transient steps. More steps improve convergence of
  the backward pass but increase computation time.
* `Δt = 1`: Time between successive orthonormalization steps.
  For continuous systems this is approximate.
* `show_progress = false`: Display a progress bar.

## Algorithm

The Ginelli algorithm computes CLVs via a two-pass procedure:

1. **Forward pass**: Evolve the tangent dynamics, applying QR decomposition at each step
   to obtain the Gram-Schmidt (GS) orthonormal basis Q and upper triangular R matrices.
   To ensure uniqueness, sign corrections are applied so that R has positive diagonal
   entries: if `R[j,j] < 0`, the j-th *row* of R is negated, and correspondingly the
   j-th *column* of Q is negated. Store Q and R for `N` steps (plus `Ttr_bkw` additional
   steps for backward convergence).

2. **Backward pass**: Initialize a coefficient matrix C = I. Iterate backward through
   the stored R matrices, computing `C ← R⁻¹ C` and normalizing columns at each step.
   The physical CLVs are then `V = Q C`.

The CLVs have the key property of being *covariant*: if V(t) is a CLV at time t,
then after evolution by the tangent dynamics, the resulting vector is parallel to V(t+Δt).

[^Ginelli2007]: F. Ginelli *et al.*, Phys. Rev. Lett. **99**, 130601 (2007)
"""
function clv(tands::TangentDynamicalSystem, N::Int;
    Δt::Real=1, Ttr::Real=0, Ttr_bkw::Int=N,
    show_progress=false, u0=current_state(tands),
)
    reinit!(tands, u0)
    k = size(current_deviations(tands), 2)

    # Set up progress bar
    total_steps = (Ttr > 0 ? ceil(Int, Ttr / Δt) : 0) + N + Ttr_bkw + N
    progress = ProgressMeter.Progress(total_steps;
        desc="CLV computation: ", dt=1.0, enabled=show_progress
    )
    step_count = 0

    # --- Phase 1: Forward transient (converge GS directions, discard R) ---
    D = size(current_deviations(tands), 1)
    T = eltype(current_deviations(tands))
    Q_transient = Matrix{T}(undef, D, k)
    R_transient = Matrix{T}(undef, k, k)
    if Ttr > 0
        t0 = current_time(tands)
        while current_time(tands) < t0 + Ttr
            step!(tands, Δt)
            _thin_qr_positive_diagonal!(Q_transient, R_transient, current_deviations(tands), k)
            set_deviations!(tands, Q_transient)
            step_count += 1
            ProgressMeter.update!(progress, step_count)
        end
    end

    # --- Phase 2: Forward pass (store Q and R for N + Ttr_bkw steps) ---
    total_store = N + Ttr_bkw
    T = eltype(current_deviations(tands))
    StateType = Vector{T}
    D = size(current_deviations(tands), 1)
    Q_history = Vector{Matrix{T}}(undef, N)           # Only store Q for kept window
    R_history = Vector{Matrix{T}}(undef, total_store) # Store R for all
    x_history = Vector{StateType}(undef, N)           # Store states for kept window
    t_history = Vector{Float64}(undef, N)

    # Pre-allocate buffers for QR decomposition
    Q_buffer = Matrix{T}(undef, D, k)
    R_buffer = Matrix{T}(undef, k, k)

    t_start = current_time(tands)
    for i in 1:total_store
        step!(tands, Δt)
        # Compute thin QR with positive diagonal (sign-corrected Q and R)
        _thin_qr!(Q_buffer, R_buffer, current_deviations(tands), k)

        # Set the sign-corrected Q as the new deviations
        set_deviations!(tands, Q_buffer)

        # Store R for backward pass (must copy since we reuse buffer)
        R_history[i] = copy(R_buffer)

        # Store Q, state, and time for the kept window (first N steps)
        if i <= N
            Q_history[i] = copy(Q_buffer)
            x_history[i] = current_state(tands)
            t_history[i] = current_time(tands)
        end

        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # Compute Lyapunov exponents from R diagonal elements
    λ = zeros(T, k)
    for i in 1:total_store
        for j in 1:k
            @inbounds λ[j] += log(abs(R_history[i][j, j]))
        end
    end
    t_total = current_time(tands) - t_start
    λ ./= t_total

    # --- Phase 3: Backward pass (compute CLVs as V = Q * C) ---
    C = Matrix{T}(LinearAlgebra.I, k, k)  # Coefficient matrix
    V_history = Vector{Matrix{T}}(undef, N)

    # First, traverse the backward transient (discard, just evolve C)
    for i in total_store:-1:(N+1)
        C = LinearAlgebra.UpperTriangular(R_history[i]) \ C
        _normalize_columns!(C)
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # Now traverse the kept window, computing physical CLVs
    for i in N:-1:1
        V_history[i] = Q_history[i] * C

        # Prepare C for the next (earlier) time step
        C = LinearAlgebra.UpperTriangular(R_history[i]) \ C
        _normalize_columns!(C)
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    return (V=V_history, λ=λ, x=x_history, t=t_history)
end

# === Kuptsov-Parlitz (forward-backward QR) ===

"""
    clv_wolfe_samelson(ds::DynamicalSystem, N, k = dimension(ds); kwargs...)

Compute Covariant Lyapunov Vectors (CLVs) using the Kuptsov-Parlitz
forward-backward QR method. This is a convenience wrapper that initializes a
`TangentDynamicalSystem` and calls the lower-level method.
"""
function clv_wolfe_samelson(ds::CoreDynamicalSystem, N, k=dimension(ds); kwargs...)
    tands = TangentDynamicalSystem(ds; k)
    return clv_wolfe_samelson(tands, N; kwargs...)
end

"""
    clv_wolfe_samelson(tands::TangentDynamicalSystem, N::Int; kwargs...)

Compute the Covariant Lyapunov Vectors (CLVs) using the Kuptsov-Parlitz algorithm,
which is a forward-backward QR method that reconstructs CLVs from stored
Gram-Schmidt bases and upper-triangular factors.

Note: the function name is retained for API compatibility, but the
implementation follows Kuptsov-Parlitz.

## Returns

Return value matches the Python implementation:
- If `traj == false`, returns `V::Vector{Matrix}`.
- If `traj == true`, returns `(V, history)` where `history` is a length `N+1`
  vector of state vectors at stored times.
- If `check == true`, also returns a copy of the dynamical system as the last
  element of the return tuple.

## Keyword arguments

* `u0 = current_state(tands)`: Initial state.
* `Ttr = 0`: Forward transient time before recording.
* `Ttr_fsv = N`: Forward steps to converge the coefficient matrix (A⁻).
* `Ttr_bsv = Ttr_fsv`: Forward steps to converge backward Lyapunov vectors (Φ⁻).
* `Δt = 1`: Time between orthonormalization steps.
* `traj = false`: Return the stored trajectory history.
* `check = false`: Also return a copy of the dynamical system (debugging aid).
* `adjoint_substeps = 10`: Ignored (kept for API compatibility).
* `show_progress = false`: Display a progress bar.

The stored CLVs start after the forward Φ⁻ convergence period (`Ttr_bsv` steps)
in addition to the initial transient `Ttr`, and include the initial snapshot
before the first `Δt` step of the storage window.

QR factorizations use the raw `qr` output without sign correction to match
the NumPy implementation.
"""
function clv_wolfe_samelson(tands::TangentDynamicalSystem, N::Int;
    Δt::Real=1, Ttr::Real=0, Ttr_fsv::Int=N, Ttr_bsv::Int=Ttr_fsv,
    traj::Bool=false, check::Bool=false, adjoint_substeps::Int=10,
    show_progress=false, u0=current_state(tands),
)
    reinit!(tands, u0)
    k = size(current_deviations(tands), 2)
    D = size(current_deviations(tands), 1)
    T = eltype(current_deviations(tands))

    # Set up progress bar
    transient_steps = Ttr > 0 ? ceil(Int, Ttr / Δt) : 0
    nspend_bsv = max(0, Ttr_bsv)
    nspend_bkw = max(0, Ttr_fsv)
    total_steps = transient_steps + nspend_bsv + N + nspend_bkw + nspend_bkw + N
    progress = ProgressMeter.Progress(total_steps;
        desc="CLV (Kuptsov-Parlitz): ", dt=1.0, enabled=show_progress
    )
    step_count = 0

    # --- Phase 0: Forward transient (base only) ---
    Q_buffer = Matrix{T}(undef, D, k)
    R_buffer = Matrix{T}(undef, k, k)
    if Ttr > 0
        t0 = current_time(tands)
        while current_time(tands) < t0 + Ttr
            step!(tands.ds, Δt)
            step_count += 1
            ProgressMeter.update!(progress, step_count)
        end
    end
    _set_identity!(Q_buffer)
    set_deviations!(tands, Q_buffer)

    # --- Phase 1: Converge backward Lyapunov vectors (Φ⁻) ---
    for _ in 1:nspend_bsv
        step!(tands, Δt)
        _thin_qr!(Q_buffer, R_buffer, current_deviations(tands), k)
        set_deviations!(tands, Q_buffer)
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # --- Phase 2: Stage A→B (store Φ⁻ and R) ---
    StateType = Vector{T}
    PhiMns = Vector{Matrix{T}}(undef, N + 1)
    R_list1 = Vector{Matrix{T}}(undef, N)
    x_history = traj ? Vector{StateType}(undef, N + 1) : StateType[]

    PhiMns[1] = copy(current_deviations(tands))
    if traj
        x_history[1] = collect(current_state(tands))
    end

    for i in 1:N
        step!(tands, Δt)
        _thin_qr!(Q_buffer, R_buffer, current_deviations(tands), k)
        set_deviations!(tands, Q_buffer)

        PhiMns[i + 1] = copy(Q_buffer)
        R_list1[i] = copy(R_buffer)
        if traj
            x_history[i + 1] = collect(current_state(tands))
        end

        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # --- Phase 3: Stage B→C (advance base, store R) ---
    R_list2 = Vector{Matrix{T}}(undef, nspend_bkw)
    for i in 1:nspend_bkw
        step!(tands, Δt)
        _thin_qr!(Q_buffer, R_buffer, current_deviations(tands), k)
        set_deviations!(tands, Q_buffer)

        R_list2[i] = copy(R_buffer)

        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end
    # No Lyapunov exponent computation in Kuptsov-Parlitz CLV routine.

    # --- Phase 4: Stage C→B (converge coefficient matrix A) ---
    A = rand(T, k, k)
    LinearAlgebra.triu!(A)
    B = Matrix{T}(undef, k, k)
    for R in Iterators.reverse(R_list2)
        @inbounds for j in 1:k
            col = @view A[:, j]
            s = LinearAlgebra.norm(col)
            @views B[:, j] .= col ./ s
        end
        LinearAlgebra.ldiv!(A, LinearAlgebra.UpperTriangular(R), B)
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end

    # --- Phase 5: Stage B→A (compute CLVs at stored times) ---
    V_history = Vector{Matrix{T}}(undef, N + 1)
    V_history[1] = PhiMns[end] * A
    write_idx = 2
    for (Q, R) in zip(Iterators.reverse(PhiMns[1:end-1]), Iterators.reverse(R_list1))
        @inbounds for j in 1:k
            col = @view A[:, j]
            s = LinearAlgebra.norm(col)
            @views B[:, j] .= col ./ s
        end
        LinearAlgebra.ldiv!(A, LinearAlgebra.UpperTriangular(R), B)
        V_history[write_idx] = Q * A
        _normalize_columns!(V_history[write_idx])
        write_idx += 1
        step_count += 1
        ProgressMeter.update!(progress, step_count)
    end
    reverse!(V_history)

    if check
        syscopy = deepcopy(tands.ds)
        if traj
            return (V_history, x_history, syscopy)
        end
        return (V_history, syscopy)
    end

    if traj
        return (V_history, x_history)
    end
    return V_history
end

# --- Helper functions ---

"""
Compute thin QR decomposition (no sign correction), storing results in pre-allocated buffers.
Q_out is m×k and R_out is k×k.
"""
function _thin_qr!(Q_out::AbstractMatrix, R_out::AbstractMatrix, Z::AbstractMatrix, k::Int)
    F = LinearAlgebra.qr(Z)
    m = size(Z, 1)
    Q_full = F.Q
    R_full = F.R
    @inbounds for j in 1:k
        for i in 1:m
            Q_out[i, j] = Q_full[i, j]
        end
        for i in 1:k
            R_out[i, j] = R_full[i, j]
        end
    end
    return nothing
end

"""
Compute thin QR decomposition with positive diagonal on R, storing results in pre-allocated buffers.
This ensures uniqueness of the QR decomposition for CLV computation.
Q_out is m×k and R_out is k×k.
Sign flips are applied to both Q columns and R rows to ensure R[j,j] > 0.
"""
function _thin_qr_positive_diagonal!(Q_out::AbstractMatrix, R_out::AbstractMatrix, Z::AbstractMatrix, k::Int)
    F = LinearAlgebra.qr(Z)
    m = size(Z, 1)
    # Copy into pre-allocated buffers
    Q_full = F.Q
    R_full = F.R
    @inbounds for j in 1:k
        for i in 1:m
            Q_out[i, j] = Q_full[i, j]
        end
        for i in 1:k
            R_out[i, j] = R_full[i, j]
        end
    end
    # Apply sign corrections
    @inbounds for j in 1:k
        if real(R_out[j, j]) < 0
            @views R_out[j, :] .*= -1
            @views Q_out[:, j] .*= -1
        end
    end
    return nothing
end

"""
Normalize each column of matrix C to unit length.
"""
function _normalize_columns!(C::AbstractMatrix)
    @inbounds for j in axes(C, 2)
        col = @view C[:, j]
        s = LinearAlgebra.norm(col)
        col ./= s
    end
    return nothing
end

function _eval_jacobian!(J_out::AbstractMatrix, tands::TangentDynamicalSystem, u, t)
    J = tands.J
    if isnothing(J)
        error("Jacobian function is required for adjoint CLV integration.")
    end
    if isinplace(tands)
        J(J_out, u, current_parameters(tands), t)
    else
        J_out .= J(u, current_parameters(tands), t)
    end
    return J_out
end

function _set_identity!(A::AbstractMatrix)
    fill!(A, zero(eltype(A)))
    @inbounds for i in 1:min(size(A, 1), size(A, 2))
        A[i, i] = one(eltype(A))
    end
    return nothing
end

function _mul_j!(out::AbstractMatrix, J::AbstractMatrix, X::AbstractMatrix)
    LinearAlgebra.mul!(out, J, X)
    return nothing
end

function _tangent_step_forward!(
    X::AbstractMatrix, tands::TangentDynamicalSystem,
    x_start::AbstractVector, x_end::AbstractVector, t_start::Real, t_end::Real,
    substeps::Int, J_buffer::AbstractMatrix, u_buffer::AbstractVector,
    dx_buffer::AbstractVector, k1::AbstractMatrix, k2::AbstractMatrix,
    k3::AbstractMatrix, k4::AbstractMatrix, tmp::AbstractMatrix,
)
    dt = (t_end - t_start) / substeps
    @. dx_buffer = x_end - x_start
    for step in 1:substeps
        s0 = (step - 1) / substeps
        s1 = (step - 0.5) / substeps
        s2 = step / substeps

        @. u_buffer = x_start + dx_buffer * s0
        t0 = t_start + (t_end - t_start) * s0
        _eval_jacobian!(J_buffer, tands, u_buffer, t0)
        _mul_j!(k1, J_buffer, X)

        @. tmp = X + (dt / 2) * k1
        @. u_buffer = x_start + dx_buffer * s1
        t1 = t_start + (t_end - t_start) * s1
        _eval_jacobian!(J_buffer, tands, u_buffer, t1)
        _mul_j!(k2, J_buffer, tmp)

        @. tmp = X + (dt / 2) * k2
        _eval_jacobian!(J_buffer, tands, u_buffer, t1)
        _mul_j!(k3, J_buffer, tmp)

        @. tmp = X + dt * k3
        @. u_buffer = x_start + dx_buffer * s2
        t2 = t_start + (t_end - t_start) * s2
        _eval_jacobian!(J_buffer, tands, u_buffer, t2)
        _mul_j!(k4, J_buffer, tmp)

        @. X = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    return X
end

function _mul_neg_jt!(out::AbstractMatrix, J::AbstractMatrix, X::AbstractMatrix)
    LinearAlgebra.mul!(out, transpose(J), X)
    @. out = -out
    return nothing
end

function _adjoint_step_backward!(
    X::AbstractMatrix, tands::TangentDynamicalSystem,
    x_start::AbstractVector, x_end::AbstractVector, t_start::Real, t_end::Real,
    substeps::Int, J_buffer::AbstractMatrix, u_buffer::AbstractVector,
    dx_buffer::AbstractVector, k1::AbstractMatrix, k2::AbstractMatrix,
    k3::AbstractMatrix, k4::AbstractMatrix, tmp::AbstractMatrix,
)
    dt = (t_end - t_start) / substeps
    @. dx_buffer = x_end - x_start
    for step in 1:substeps
        s0 = (step - 1) / substeps
        s1 = (step - 0.5) / substeps
        s2 = step / substeps

        @. u_buffer = x_start + dx_buffer * s0
        t0 = t_start + (t_end - t_start) * s0
        _eval_jacobian!(J_buffer, tands, u_buffer, t0)
        _mul_neg_jt!(k1, J_buffer, X)

        @. tmp = X + (dt / 2) * k1
        @. u_buffer = x_start + dx_buffer * s1
        t1 = t_start + (t_end - t_start) * s1
        _eval_jacobian!(J_buffer, tands, u_buffer, t1)
        _mul_neg_jt!(k2, J_buffer, tmp)

        @. tmp = X + (dt / 2) * k2
        _eval_jacobian!(J_buffer, tands, u_buffer, t1)
        _mul_neg_jt!(k3, J_buffer, tmp)

        @. tmp = X + dt * k3
        @. u_buffer = x_start + dx_buffer * s2
        t2 = t_start + (t_end - t_start) * s2
        _eval_jacobian!(J_buffer, tands, u_buffer, t2)
        _mul_neg_jt!(k4, J_buffer, tmp)

        @. X = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    end
    return X
end

function _wolfe_samelson_clvs(bsv::AbstractMatrix, fsv::AbstractMatrix, k::Int)
    D = size(bsv, 1)
    V = Matrix{eltype(bsv)}(undef, D, k)
    @views V[:, 1] .= bsv[:, 1]
    V[:, 1] ./= LinearAlgebra.norm(@view V[:, 1])

    for n in 2:k
        η = @view bsv[:, 1:n]
        ξ = @view fsv[:, 1:(n - 1)]
        M = transpose(η) * ξ
        F = LinearAlgebra.svd(transpose(M))
        y = F.Vt[end, :]
        v = η * y
        v ./= LinearAlgebra.norm(v)
        @views V[:, n] .= v
    end
    return V
end
