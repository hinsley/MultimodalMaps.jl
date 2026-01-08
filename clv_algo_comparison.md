# Covariant Lyapunov Vector Algorithms: A Comprehensive Implementation Guide

This document provides a detailed, implementation-focused comparison of two algorithms for computing **Covariant Lyapunov Vectors (CLVs)** in smooth, continuous-time dynamical systems:

1.  **Ginelli et al. (2007)**: The "Forward-Backward" algorithm based on storing the orthonormalized Gram-Schmidt basis.
2.  **Wolfe & Samelson (2007)**: The "Subspace Intersection" algorithm which minimizes memory by finding CLVs as the intersection of forward and backward spanning subspaces.

---

## 1. Introduction and Mathematical Background

### 1.1. Why Covariant Lyapunov Vectors?

For a dynamical system $\dot{\mathbf{x}} = \mathbf{F}(\mathbf{x})$, the standard Lyapunov exponents $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_D$ quantify the average rates of exponential growth or decay of infinitesimal perturbations. However, the standard **Gram-Schmidt (GS) vectors** (obtained during the QR-based Lyapunov exponent calculation) are a byproduct of an orthonormalization procedure and depend on the chosen inner product; they are not intrinsic to the dynamics.

**Covariant Lyapunov Vectors (CLVs)** are intrinsic, norm-independent vectors that are covariant with the dynamics: if $\mathbf{v}_j(t)$ is the $j$-th CLV at time $t$, evolving it forward in tangent space yields a vector parallel to the $j$-th CLV at time $t + \Delta t$.

### 1.2. The Fundamental Covariance Property

Consider the tangent linear propagator $\mathbf{M}(t_0, t_1)$, which maps an infinitesimal perturbation $\delta\mathbf{x}(t_0)$ to $\delta\mathbf{x}(t_1)$:
$$\delta\mathbf{x}(t_1) = \mathbf{M}(t_0, t_1) \delta\mathbf{x}(t_0)$$

The matrix $\mathbf{M}$ is obtained by integrating the variational equation:
$$\dot{\mathbf{M}} = D\mathbf{F}(\mathbf{x}(t)) \mathbf{M}, \quad \mathbf{M}(t_0, t_0) = \mathbf{I}_D$$
where $D\mathbf{F}(\mathbf{x})$ is the $D \times D$ Jacobian of the vector field $\mathbf{F}$ evaluated at $\mathbf{x}$.

The **Oseledets Multiplicative Ergodic Theorem** guarantees the existence of an invariant splitting of the tangent space into subspaces associated with each Lyapunov exponent. The CLVs are the basis vectors for this splitting.

**Definition (Covariant Lyapunov Vector):** The $j$-th CLV $\mathbf{v}_j(t)$ satisfies:
$$\mathbf{M}(t, t+\Delta t) \mathbf{v}_j(t) = \gamma_j(t, \Delta t) \mathbf{v}_j(t + \Delta t)$$
where $\gamma_j(t, \Delta t)$ is a time-dependent growth factor. The long-time average growth rate is the $j$-th Lyapunov exponent:
$$\lambda_j = \lim_{T \to \infty} \frac{1}{T} \int_0^T \ln |\gamma_j(t, dt)| \, dt$$

### 1.3. Oseledets Subspaces

A key property is that the CLVs define a nested set of subspaces.

*   **$\mathcal{V}_j(t)$**: The subspace spanned by the first $j$ CLVs (the most expanding directions).
    $$\mathcal{V}_j(t) = \text{span}\{\mathbf{v}_1(t), \mathbf{v}_2(t), \dots, \mathbf{v}_j(t)\}$$
    This is the same subspace spanned by the first $j$ Gram-Schmidt vectors $\{\mathbf{e}_1(t), \dots, \mathbf{e}_j(t)\}$ from the standard QR algorithm.

*   **$\mathcal{W}_j(t)$**: The subspace spanned by the $j$-th to $D$-th CLVs (the more contracting directions).
    $$\mathcal{W}_j(t) = \text{span}\{\mathbf{v}_j(t), \mathbf{v}_{j+1}(t), \dots, \mathbf{v}_D(t)\}$$

A crucial observation is:
$$\mathbf{v}_j(t) = \mathcal{V}_j(t) \cap \mathcal{W}_j(t)$$
The $j$-th CLV is the unique direction (up to scaling) that lies in both the "expanding" subspace up to $j$ and the "contracting" subspace from $j$ onward.

---

## 2. Ginelli Algorithm (Forward-Backward Method)

This algorithm exploits the fact that CLVs can be expressed as linear combinations of the Gram-Schmidt vectors, with coefficients living in an upper-triangular matrix.

### 2.1. Overview

1.  **Forward Pass**: Evolve the system and its tangent vectors. At each step, perform a QR decomposition to orthonormalize the tangent vectors. Store the $Q$ matrices (Gram-Schmidt vectors) and $R$ matrices (scaling/mixing factors).

2.  **Backward Pass**: Starting from the end of the trajectory, iterate backward through the stored $R$ matrices to find the coefficient matrix $C$ that expresses CLVs in terms of GS vectors.

3.  **Reconstruction**: At each time step, the CLV matrix is $V = QC$.

### 2.2. Notation

| Symbol | Meaning |
|--------|---------|
| $D$ | System dimension (number of state variables). |
| $k$ | Number of Lyapunov vectors to compute ($1 \le k \le D$). |
| $\Delta t$ | Time interval between orthonormalization steps. |
| $N$ | Number of time steps to store CLVs for output. |
| $N_{tr,fwd}$ | Forward transient steps (discarded, for GS convergence). |
| $N_{tr,bkw}$ | Backward transient steps (for CLV coefficient convergence). |
| $\mathbf{x}_n$ | State vector at step $n$. |
| $\mathbf{Z}_n$ | $D \times k$ matrix of (non-orthonormal) tangent vectors at step $n$. |
| $\mathbf{Q}_n$ | $D \times k$ orthonormal matrix (columns are GS vectors $\mathbf{e}_1, \dots, \mathbf{e}_k$). |
| $\mathbf{R}_n$ | $k \times k$ upper-triangular matrix from QR decomposition. |
| $\mathbf{C}_n$ | $k \times k$ upper-triangular coefficient matrix (CLVs in GS basis). |
| $\mathbf{V}_n$ | $D \times k$ matrix of CLVs at step $n$. |

### 2.3. Algorithm: Forward Pass (Phase 1 & 2)

**Goal:** Generate and store the GS basis $\mathbf{Q}_n$ and the upper-triangular matrices $\mathbf{R}_n$.

**Phase 1: Forward Transient (Discarded)**
This phase allows the tangent vectors to align with the Oseledets subspaces.

1.  **Initialize:**
    *   Set initial state $\mathbf{x}_0$.
    *   Set initial deviation matrix $\mathbf{Z}_0 = \mathbf{I}_{D \times k}$ (first $k$ columns of identity).

2.  **Iterate for $n = 1$ to $N_{tr,fwd}$:**
    a.  **Evolve trajectory:** Integrate $\dot{\mathbf{x}} = \mathbf{F}(\mathbf{x})$ from $t_{n-1}$ to $t_n = t_{n-1} + \Delta t$ to get $\mathbf{x}_n$.
    b.  **Evolve tangent vectors:** Integrate $\dot{\mathbf{Z}} = D\mathbf{F}(\mathbf{x}(t)) \mathbf{Z}$ concurrently, using $\mathbf{Q}_{n-1}$ as the initial condition, to get $\tilde{\mathbf{Z}}_n$.
    c.  **QR Decomposition with Positive Diagonal:**
        Compute $\tilde{\mathbf{Z}}_n = \mathbf{Q}_n \mathbf{R}_n$.
        *   For numerical uniqueness, ensure $R_{jj} > 0$ for all $j$. If $R_{jj} < 0$:
            *   Negate the $j$-th row of $\mathbf{R}_n$: $\mathbf{R}_n[j, :] \leftarrow -\mathbf{R}_n[j, :]$.
            *   Negate the $j$-th column of $\mathbf{Q}_n$: $\mathbf{Q}_n[:, j] \leftarrow -\mathbf{Q}_n[:, j]$.
    d.  **Discard:** Do not store $\mathbf{Q}_n$ or $\mathbf{R}_n$. The orthonormalized $\mathbf{Q}_n$ becomes the new deviation matrix for the next step.

**Phase 2: Forward Pass (Stored)**
This phase generates the data needed for the backward pass.

3.  **Initialize storage:**
    *   `Q_history`: Vector of $N$ matrices, each $D \times k$.
    *   `R_history`: Vector of $N + N_{tr,bkw}$ matrices, each $k \times k$.
    *   `x_history`: Vector of $N$ state vectors.
    *   `t_history`: Vector of $N$ time values.

4.  **Iterate for $n = 1$ to $N + N_{tr,bkw}$:**

    a.  **Evolve trajectory and tangent vectors** (as in Phase 1).

    b.  **QR Decomposition with Positive Diagonal** (as in Phase 1).

    c.  **Store:**

        *   `R_history[n]` $\leftarrow \mathbf{R}_n$ (always store $\mathbf{R}$).

        *   If $n \le N$:
        
            *   `Q_history[n]` $\leftarrow \mathbf{Q}_n$.
        
            *   `x_history[n]` $\leftarrow \mathbf{x}_n$.
        
            *   `t_history[n]` $\leftarrow t_n$.
    
    d.  **Update deviations:** Set $\mathbf{Q}_n$ as the new tangent vector matrix.

5.  **Compute Lyapunov Exponents:**
    
    $$\lambda_j = \frac{1}{T_{total}} \sum_{n=1}^{N + N_{tr,bkw}} \ln(R_{n,jj})$$
    
    where $T_{total} = (N + N_{tr,bkw}) \cdot \Delta t$.

### 2.4. Algorithm: Backward Pass (Phase 3)

**Goal:** Find the coefficient matrix $\mathbf{C}_n$ such that $\mathbf{V}_n = \mathbf{Q}_n \mathbf{C}_n$.

The key insight is that $\mathbf{C}$ is upper-triangular because the $j$-th CLV lies in the subspace $\mathcal{V}_j$ spanned by the first $j$ GS vectors.

1.  **Initialize:**
    
    *   $\mathbf{C} = \mathbf{I}_{k \times k}$ (or any random upper-triangular matrix with non-zero diagonal).

2.  **Backward Transient (Discarded CLVs):**
    
    Iterate backward from the end of the stored $\mathbf{R}$ matrices to converge $\mathbf{C}$.
    For $n = (N + N_{tr,bkw})$ down to $(N + 1)$:
    
    a.  **Backward propagation of coefficients:**

        The evolution of tangent vectors is $\tilde{\mathbf{Z}}_n = \mathbf{Q}_{n-1} \mathbf{R}_n \mathbf{Q}_{n-1}^T \mathbf{Q}_n$... but working in the coefficient space, the relation simplifies. If $\mathbf{V}_n = \mathbf{Q}_n \mathbf{C}_n$, then covariance implies:
        $$\tilde{\mathbf{C}}_{n-1} = \mathbf{R}_n^{-1} \mathbf{C}_n$$
        Since $\mathbf{R}_n$ is upper-triangular, $\mathbf{R}_n^{-1}$ is also upper-triangular, so $\tilde{\mathbf{C}}_{n-1}$ remains upper-triangular.
    
    b.  **Column normalization:**

        To prevent numerical overflow/underflow (columns grow/shrink at rate $e^{\pm \lambda_j \Delta t}$), normalize each column:
        $$\mathbf{C}_{n-1}[:, j] = \frac{\tilde{\mathbf{C}}_{n-1}[:, j]}{\|\tilde{\mathbf{C}}_{n-1}[:, j]\|}$$
        for $j = 1, \dots, k$.

3.  **Kept Window (CLVs stored):**

    For $n = N$ down to $1$:

    a.  **Reconstruct CLV matrix:**

        $$\mathbf{V}_n = \mathbf{Q}_n \mathbf{C}_n$$

        Store `V_history[n]` $\leftarrow \mathbf{V}_n$.

    b.  **Backward propagation and normalization** (as above):

        $$\tilde{\mathbf{C}}_{n-1} = \mathbf{R}_n^{-1} \mathbf{C}_n$$

        $$\mathbf{C}_{n-1}[:, j] = \frac{\tilde{\mathbf{C}}_{n-1}[:, j]}{\|\tilde{\mathbf{C}}_{n-1}[:, j]\|}$$

### 2.5. Detailed Matrix/Vector Operations

Here is the core loop logic, emphasizing every operation:

```
# --- Forward Pass (one step) ---
# Input: Q_prev (D x k), current state x_prev
# Output: Q_curr (D x k), R_curr (k x k), x_curr

x_curr, Z_tilde = integrate_tangent(x_prev, Q_prev, Δt)
# Z_tilde is the evolved D x k tangent matrix

# Thin QR decomposition: Z_tilde = Q_curr * R_curr
Q_full, R_full = qr(Z_tilde)  # Q_full is D x D (or D x k thin), R_full is k x k

# Extract thin factors
Q_curr = Q_full[:, 1:k]  # D x k
R_curr = R_full[1:k, 1:k]  # k x k (already k x k if thin QR)

# Ensure positive diagonal
for j in 1:k
    if R_curr[j, j] < 0
        R_curr[j, :] *= -1  # Negate row j of R
        Q_curr[:, j] *= -1  # Negate column j of Q
    end
end
```

```
# --- Backward Pass (one step) ---
# Input: R_curr (k x k), C_curr (k x k)
# Output: C_prev (k x k)

# Solve upper-triangular system R_curr * C_prev = C_curr for C_prev
# Equivalently: C_prev = R_curr \ C_curr (backslash for triangular solve)
C_prev = UpperTriangular(R_curr) \ C_curr

# Normalize columns
for j in 1:k
    C_prev[:, j] /= norm(C_prev[:, j])
end
```

```
# --- CLV Reconstruction ---
# Input: Q_n (D x k), C_n (k x k)
# Output: V_n (D x k), columns are CLVs

V_n = Q_n * C_n
```

### 2.6. Memory Requirements

| Data Structure | Size | Notes |
|----------------|------|-------|
| `Q_history` | $N \times D \times k$ | Only for the output window. |
| `R_history` | $(N + N_{tr,bkw}) \times k \times k$ | Needed for backward pass. |
| `x_history`, `t_history` | $N \times D$, $N \times 1$ | States and times. |
| `C` (current) | $k \times k$ | Reused in backward loop. |
| `V_history` | $N \times D \times k$ | Output CLVs. |

**Total:** $O((N + N_{tr,bkw}) \cdot k^2 + N \cdot D \cdot k)$.

For large systems or long trajectories, storage of `Q_history` can be prohibitive. An alternative is to only store `x_history` and `R_history`, then re-integrate the tangent equations during the backward pass to reconstruct $\mathbf{Q}_n$ on-the-fly. This trades memory for computation.

---

## 3. Wolfe & Samelson Algorithm (Subspace Intersection Method)

This algorithm avoids storing the entire history by exploiting the geometric interpretation of CLVs as the intersection of forward and backward subspaces.

### 3.1. Overview

The $j$-th CLV $\mathbf{v}_j(t)$ is the unique vector lying in both:
1.  $\mathcal{V}_j(t)$: spanned by the $j$ most expanding directions (characterized by backward-time evolution).
2.  $\mathcal{W}_j(t)$: spanned by directions with growth rates $\lambda_j, \lambda_{j+1}, \dots$ (characterized by forward-time evolution).

The algorithm computes orthonormal bases for these subspaces and finds their intersection.

### 3.2. Notation

| Symbol | Meaning |
|--------|---------|
| $\tau$ | Optimization interval (time for singular vectors to converge). |
| $\hat{\boldsymbol{\eta}}_1, \dots, \hat{\boldsymbol{\eta}}_n$ | Backward singular vectors (or GS vectors from backward integration). These span $\mathcal{V}_n(t)$. |
| $\hat{\boldsymbol{\xi}}_1, \dots, \hat{\boldsymbol{\xi}}_{n-1}$ | Forward singular vectors (or GS vectors from forward integration). These span the orthogonal complement to $\mathcal{W}_n(t)$. |
| $\mathbf{M}$ | $n \times (n-1)$ projection matrix. |
| $\mathbf{D}$ | $n \times n$ matrix $\mathbf{D} = \mathbf{M}\mathbf{M}^T$. |
| $\mathbf{y}^{(n)}$ | Null vector of $\mathbf{M}^T$ (coefficients for $\mathbf{v}_n$). |

### 3.3. Algorithm: Computing the $n$-th CLV at Time $t$

**Step 1: Compute Backward Spanning Vectors ($\hat{\boldsymbol{\eta}}_j$)**

These vectors span $\mathcal{V}_n(t)$. They can be obtained as:
*   **Singular Vectors:** Left singular vectors of the propagator $\mathbf{M}(t, t-\tau)$ (backward in time).
*   **Gram-Schmidt Vectors:** Orthonormalized tangent vectors from a forward QR integration starting from $t - \tau$ and ending at $t$.

**Procedure (using GS vectors):**
1.  Set a checkpoint at time $t - \tau$.
2.  Initialize $\mathbf{Z}(t-\tau) = \mathbf{I}_{D \times n}$.
3.  Integrate forward from $t - \tau$ to $t$, performing periodic QR orthonormalization.
4.  The final $\mathbf{Q}(t)$ has columns $\hat{\boldsymbol{\eta}}_1, \dots, \hat{\boldsymbol{\eta}}_n$.

These span the same subspace as $\{\mathbf{v}_1(t), \dots, \mathbf{v}_n(t)\}$.

**Step 2: Compute Forward Spanning Vectors ($\hat{\boldsymbol{\xi}}_i$)**

These vectors characterize the subspace $\mathcal{W}_n(t)$. Specifically, the first $n-1$ right singular vectors (or forward GS vectors) of the propagator $\mathbf{M}(t+\tau, t)$ span the **complement** of $\mathcal{W}_n(t)$. Equivalently, any vector orthogonal to all $\hat{\boldsymbol{\xi}}_1, \dots, \hat{\boldsymbol{\xi}}_{n-1}$ lies in $\mathcal{W}_n(t)$.

**Procedure (using GS vectors):**
1.  Set a checkpoint at time $t$.
2.  Initialize $\mathbf{Z}(t) = \mathbf{I}_{D \times (n-1)}$.
3.  Integrate forward from $t$ to $t + \tau$, performing periodic QR orthonormalization.
4.  The final $\mathbf{Q}(t+\tau)$ has columns that are the forward GS vectors.
5.  **Critical:** We need the vectors *at time $t$*, not $t + \tau$. The right singular vectors of $\mathbf{M}(t+\tau, t)$ are the orthonormalized initial conditions that produce the forward GS vectors. Since we used orthonormalized initial conditions ($\mathbf{Z}(t) = \mathbf{I}$), we can propagate them backward or, more simply, use the relation:
    $$\hat{\boldsymbol{\xi}}_i(t) = \text{columns of } \mathbf{Q}(t)$$
    where $\mathbf{Q}(t)$ is obtained by QR-orthonormalizing at time $t$ *before* forward evolution. In practice, you can track which initial direction corresponds to which final GS vector.

**Simplified approach:** Use the *same* orthonormalization procedure at time $t$. After evolving for time $\tau$, the first $n-1$ GS vectors at the *start* of the interval (which are the canonical basis vectors if starting from $\mathbf{I}$) correspond to the forward spanning directions.

**Step 3: Compute Projection Matrix $\mathbf{M}$**

The matrix $\mathbf{M}$ is $n \times (n-1)$ with elements:
$$M_{ji} = \langle \hat{\boldsymbol{\eta}}_j, \hat{\boldsymbol{\xi}}_i \rangle$$
This is the inner product (dot product for real systems) between the $j$-th backward vector and the $i$-th forward vector.

**Matrix operation:**
$$\mathbf{M} = \boldsymbol{\eta}^T \boldsymbol{\xi}$$
where $\boldsymbol{\eta}$ is $D \times n$ (columns are $\hat{\boldsymbol{\eta}}_j$) and $\boldsymbol{\xi}$ is $D \times (n-1)$ (columns are $\hat{\boldsymbol{\xi}}_i$).

**Step 4: Solve the Null Space Problem**

The $n$-th CLV lies in $\mathcal{V}_n$ and is orthogonal to all $\hat{\boldsymbol{\xi}}_i$. Expressing $\mathbf{v}_n$ as a linear combination of the backward vectors:
$$\mathbf{v}_n = \sum_{j=1}^n y_j \hat{\boldsymbol{\eta}}_j = \boldsymbol{\eta} \mathbf{y}$$

The orthogonality condition $\langle \mathbf{v}_n, \hat{\boldsymbol{\xi}}_i \rangle = 0$ for all $i$ gives:
$$\sum_{j=1}^n y_j \langle \hat{\boldsymbol{\eta}}_j, \hat{\boldsymbol{\xi}}_i \rangle = 0 \quad \Rightarrow \quad \mathbf{M}^T \mathbf{y} = \mathbf{0}$$

Since $\mathbf{M}^T$ is $(n-1) \times n$, its null space is (generically) 1-dimensional. The solution $\mathbf{y}^{(n)}$ is unique up to scaling.

**Numerical procedure:**
*   Compute $\mathbf{D} = \mathbf{M} \mathbf{M}^T$ (this is $n \times n$, symmetric, and has rank $n-1$).
*   Find the eigenvector of $\mathbf{D}$ corresponding to eigenvalue 0 (or the smallest eigenvalue numerically).
*   Alternatively, compute the SVD of $\mathbf{M}^T$ and take the right singular vector corresponding to singular value 0.

**Step 5: Reconstruct the CLV**

$$\mathbf{v}_n(t) = \boldsymbol{\eta} \mathbf{y}^{(n)} = \sum_{j=1}^n y_j^{(n)} \hat{\boldsymbol{\eta}}_j(t)$$

Normalize if needed: $\mathbf{v}_n \leftarrow \mathbf{v}_n / \|\mathbf{v}_n\|$.

### 3.4. Computing All $k$ CLVs

To compute all $k$ CLVs simultaneously:

1.  Compute backward spanning vectors $\hat{\boldsymbol{\eta}}_1, \dots, \hat{\boldsymbol{\eta}}_k$ (span $\mathcal{V}_k$).
2.  Compute forward spanning vectors $\hat{\boldsymbol{\xi}}_1, \dots, \hat{\boldsymbol{\xi}}_{k-1}$ (define $\mathcal{W}$ subspaces).

For each $n = 1, \dots, k$:
*   Use $\hat{\boldsymbol{\eta}}_1, \dots, \hat{\boldsymbol{\eta}}_n$ and $\hat{\boldsymbol{\xi}}_1, \dots, \hat{\boldsymbol{\xi}}_{n-1}$.
*   Form $\mathbf{M}^{(n)}$ and solve the null space problem.
*   Reconstruct $\mathbf{v}_n$.

**Special case: $n = 1$**
For the first CLV, there are no forward spanning vectors to exclude, so $\mathbf{v}_1 = \hat{\boldsymbol{\eta}}_1$ directly.

### 3.5. Detailed Matrix/Vector Operations

```
# --- Compute Backward Spanning Vectors (at time t) ---
# Integrate forward from t - τ to t with QR orthonormalization

x_check = integrate_trajectory(x_t, -τ)  # Checkpoint at t - τ (requires reverse-time integration or prior storage)
Z = I[1:D, 1:k]  # D x k

x_curr = x_check
for step in 1:num_steps
    x_curr, Z = integrate_tangent(x_curr, Z, Δt)
    Q, R = qr(Z)
    Z = Q[:, 1:k]
end
η = Z  # D x k matrix, columns are backward spanning vectors at time t
```

```
# --- Compute Forward Spanning Vectors (at time t) ---
# We need the initial vectors that, after forward evolution, give the leading GS vectors.
# If we start from orthonormal vectors at t, those are the forward spanning vectors.

ξ = I[1:D, 1:(k-1)]  # D x (k-1) matrix
# These are the forward spanning vectors at time t (assuming immediate orthonormalization)
```

```
# --- Compute n-th CLV ---
# Input: η (D x n), ξ (D x (n-1))
# Output: v_n (D x 1)

M = η' * ξ  # n x (n-1) matrix

# Find null vector of M^T
# Option 1: SVD
U, S, Vt = svd(M')  # M' is (n-1) x n
y = Vt[end, :]'  # Last row of V^T, which is the last column of V (n x 1)

# Option 2: Eigendecomposition of D = M * M'
D_mat = M * M'  # n x n
eigenvalues, eigenvectors = eigen(D_mat)
idx = argmin(eigenvalues)
y = eigenvectors[:, idx]

# Reconstruct CLV
v_n = η * y  # D x 1
v_n /= norm(v_n)
```

### 3.6. Memory Requirements

| Data Structure | Size | Notes |
|----------------|------|-------|
| $\boldsymbol{\eta}$ | $D \times k$ | Backward spanning vectors at current time. |
| $\boldsymbol{\xi}$ | $D \times (k-1)$ | Forward spanning vectors at current time. |
| $\mathbf{M}$ | $k \times (k-1)$ | Projection matrix. |
| $\mathbf{y}$ vectors | $k$ vectors of size $\le k$ | Null vectors for each CLV. |

**Total:** $O(D \cdot k + k^2)$ at any single time point.

**Key advantage:** No need to store the entire trajectory history. CLVs are computed "on-the-fly" at each time of interest.

**Trade-off:** Requires integrating both forward and backward (or from a past checkpoint) for each time point where CLVs are needed.

---

## 4. Comparative Summary

| Aspect | Ginelli (Forward-Backward) | Wolfe-Samelson (Intersection) |
|--------|----------------------------|-------------------------------|
| **Memory** | $O(T \cdot D \cdot k)$ (stores $Q$ history) | $O(D \cdot k)$ per time point |
| **Computation** | One forward pass + one backward pass | Two integrations per time point (forward from $t-\tau$, forward from $t$) |
| **Use Case** | Compute CLVs along entire trajectory | Compute CLVs at specific times |
| **Parallelization** | Limited (sequential backward pass) | Each time point is independent |
| **Implementation** | QR + triangular solve | QR + null space (SVD/eigen) |

---

## 5. Tuning Parameters and Practical Tips

### 5.1. For the Ginelli Algorithm

*   **Forward Transient ($N_{tr,fwd}$ or $T_{tr}$):**
    *   Purpose: Allow the tangent vectors to align with the Oseledets subspaces.
    *   Heuristic: Several times the inverse of the smallest Lyapunov exponent gap, $\sim 5/|\lambda_k - \lambda_{k+1}|$ or $\sim 10/|\lambda_{\min}|$.
    *   For chaotic systems, often 100–1000 orthonormalization steps are sufficient.

*   **Backward Transient ($N_{tr,bkw}$):**
    *   Purpose: Converge the coefficient matrix $\mathbf{C}$ to the true CLV representation.
    *   Heuristic: Similar to forward transient; $N_{tr,bkw} \approx N$ is a common conservative choice.
    *   The backwards iteration contracts errors at rate $\exp((\lambda_j - \lambda_{j-1}) \Delta t)$ for column $j$.

*   **Orthonormalization Interval ($\Delta t$):**
    *   Too small: Wasted computation.
    *   Too large: Tangent vectors collapse toward the leading direction; loss of subspace information.
    *   Heuristic: Choose $\Delta t$ such that $\|\mathbf{M}(t, t+\Delta t)\| \lesssim 10^3$–$10^6$ (factors of $R_{11}$ stay within numerical range).

*   **Positive Diagonal Convention:**
    *   Essential for sign consistency. Without it, CLV directions may spuriously flip.

### 5.2. For the Wolfe-Samelson Algorithm

*   **Optimization Interval ($\tau$):**
    *   Purpose: Allow singular vectors to converge to Lyapunov subspaces.
    *   Heuristic: $\tau \gg 1 / |\lambda_n - \lambda_{n+1}|$ for the $n$-th CLV.
    *   If $\tau$ is too short, the spanning vectors haven't converged and the intersection is incorrect.
    *   If $\tau$ is too long, leading vectors may dominate due to finite precision.

*   **Restarting / Periodic Orthonormalization:**
    *   Even within the $\tau$ interval, orthonormalize periodically (e.g., every 5 time units) to prevent numerical collapse.

*   **Coarse/Fine Grid Strategy:**
    *   Compute CLVs on a coarse temporal grid (expensive intersection step).
    *   Use vector-valued tangent integration to propagate CLVs on a fine grid between coarse points (preserves covariance).

*   **Neutral Vector ($\lambda = 0$):**
    *   For autonomous systems, the neutral CLV is proportional to $\dot{\mathbf{x}} = \mathbf{F}(\mathbf{x})$. This can serve as a check.

### 5.3. General Advice

*   **Numerical Precision:** Use double precision. CLV computation involves many matrix operations that accumulate errors.
*   **Validation:** Check covariance by verifying that $\mathbf{M}(t, t+\Delta t) \mathbf{v}_j(t)$ is parallel to $\mathbf{v}_j(t + \Delta t)$.
*   **Angles between CLVs:** CLVs are generally *not* orthogonal. The angle between $\mathbf{v}_i$ and $\mathbf{v}_j$ (especially adjacent indices) measures hyperbolicity. Near-tangent CLVs indicate weaker hyperbolicity.

---

## 6. References

1.  **Ginelli, F., Poggi, P., Turchi, A., Chaté, H., Livi, R., & Politi, A.** (2007). Characterizing dynamics with covariant Lyapunov vectors. *Physical Review Letters*, 99(13), 130601. [arXiv:0706.0510](https://arxiv.org/abs/0706.0510)

2.  **Wolfe, C. L., & Samelson, R. M.** (2007). An efficient method for recovering Lyapunov vectors from singular vectors. *Tellus A*, 59(3), 355-366. [PDF](https://chaosbk.physics.gatech.edu/library/WoSa07.pdf)
