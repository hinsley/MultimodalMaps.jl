# GPU-Accelerated CLV Parameter Sweep: Implementation Report

## Executive Summary

This document captures all knowledge gained from attempting to implement GPU-accelerated Covariant Lyapunov Vector (CLV) computation for parameter sweeps of the Shimizu-Morioka system. **The Wolfe-Samelson implementation failed validation** and requires debugging or replacement with a working approach.

---

## Objective

Compute the **minimum angle between CLV₃ and the plane spanned by CLV₁ and CLV₂** across a 2D parameter sweep (α × λ) of the Shimizu-Morioka system. Generate a heatmap visualization.

### Requirements
- GPU acceleration via `DiffEqGPU.jl` and `Metal.jl` (Apple Silicon)
- Scalable to fine parameter sweeps (potentially 1000×1000 = 1M trajectories)
- Memory-efficient (bounded, not O(N) in trajectory length)
- Numerically robust (CLVs converge slowly)

---

## The Shimizu-Morioka System

```julia
ẋ = y
ẏ = x - λy - xz
ż = -αz + x²
```

**Jacobian:**
```
J(x,y,z) = [ 0      1     0  ]
           [ 1-z   -λ    -x  ]
           [ 2x     0    -α  ]
```

**Chaotic regime:** α ≈ 0.375, λ ≈ 0.75

---

## Two CLV Algorithms Considered

### 1. Ginelli Algorithm (Working Reference)

**Forward pass:**
1. Integrate state + tangent matrix M
2. Every Δt: QR decompose M → Q, R with positive diagonal
3. Set M ← Q, store Q and R

**Backward pass:**
1. Initialize coefficient matrix C = I
2. Iterate backward: C ← R⁻¹ C, normalize columns
3. Physical CLVs: V = Q × C

**Memory:** O(N) for R matrices — problematic for long trajectories

**Reference implementation:** `lyapunovvectors.jl` (working, tested)

---

### 2. Wolfe-Samelson Algorithm (Failed Implementation)

**Key insight:** CLVs are the intersection of:
- **Backward Singular Vectors (BSVs)** η: span of directions that grew most from distant past
- **Forward Singular Vectors (FSVs)** ξ: complement of directions that will grow most into future

**CLV computation (for n-th CLV):**
1. Form inner product matrix C[i,j] = ⟨ξᵢ, ηⱼ⟩
2. Find null vector y of C[1:n-1, 1:n]
3. CLV_n = Σ yⱼ ηⱼ (linear combination of BSVs)

**Claimed advantage:** No R matrix storage needed — O(1) memory

**Critical detail from paper:** FSVs are obtained by evolving through the **adjoint propagator from the FUTURE back to the PRESENT**. This fundamentally requires backward-in-time integration.

---

## Failed Implementation Details

### What Was Attempted

**Augmented 12D system:**
- State (x, y, z) + tangent matrix M (9 elements)
- GPU integration via `DiffEqGPU.vectorized_asolve`

**Forward pass:**
- Integrate state + BSV tangent
- QR orthonormalize every segment
- Store snapshots for backward pass

**Backward pass (flawed):**
- Integrate FSV tangent using adjoint dynamics: Ṁ = -Jᵀ M
- Compute CLV₃ via nullspace intersection
- Assumed CLV₁ = η₁, CLV₂ = η₂ (BSVs directly)

### Validation Results

Compared against `lyapunovvectors.jl` Ginelli implementation:

| Vector | Expected angle | Actual angle |
|--------|----------------|--------------|
| CLV₃ | < 5° | **80-90°** |
| CLV₁ | < 5° | 21-29° |
| CLV₂ | < 5° | 13-58° |

**Conclusion:** Implementation is fundamentally broken.

---

## Suspected Bugs

### 1. Adjoint Dynamics Sign Convention
Used: `Ṁ = -Jᵀ M`

This may be wrong. Different sources use different conventions. The adjoint of the tangent linear model may be:
- `Ṁ = -Jᵀ M` (time-reversed adjoint)
- `Ṁ = Jᵀ M` (transpose without negation)

Need to verify against literature.

### 2. FSV Evolution Direction
The paper says FSVs come from evolving "from the future back to the present." I attempted to evolve FSVs forward in time thinking the adjoint dynamics would handle this. This may be incorrect.

**Possible fix:** FSVs must be evolved BACKWARD in time (from end to start), not forward.

### 3. CLV₁/CLV₂ = BSV Assumption
I assumed CLV₁ and CLV₂ are just the first two BSVs (η₁, η₂). The full Wolfe-Samelson computes ALL CLVs via intersection, not just CLV₃.

For CLV₁:
- Form 0×1 matrix C (trivially satisfied)
- CLV₁ = η₁ (this is actually correct)

For CLV₂:
- Form 1×2 matrix C[i,j] = ⟨ξ₁, ηⱼ⟩
- Find null vector of C
- CLV₂ = y₁η₁ + y₂η₂

**The assumption CLV₂ = η₂ is WRONG.**

### 4. Frozen Jacobian in Backward Pass
During backward integration of FSVs, I used the saved state snapshot for the entire interval. The Jacobian depends on (x,y,z) which changes during the interval.

**Fix:** Store states at QR resolution (not just output resolution) so backward pass has correct Jacobian at each step.

---

## Memory Scaling Analysis

| Sweep Size | Trajectories | Memory (store all) | Memory (sliding window, 100 steps) |
|------------|-------------|-------------------|-----------------------------------|
| 50×50 | 2,500 | 6 MB | 1.2 MB |
| 200×200 | 40,000 | 91 MB | 19 MB |
| 500×500 | 250,000 | 570 MB | 120 MB |
| 1000×1000 | 1,000,000 | 2.3 GB | 480 MB |

**Conclusion:** Sliding window is essential for large sweeps.

---

## Recommended Approach

Given the complexity of Wolfe-Samelson and the need for robustness:

### Option A: GPU-ified Ginelli with Sliding Window

1. **Forward pass (GPU):**
   - Integrate all trajectories in parallel
   - QR on CPU (or hand-coded GPU QR for 3×3)
   - Store R matrices in sliding window

2. **Backward pass (CPU, multithreaded):**
   - Process trajectories in parallel
   - C ← R⁻¹ C within window
   - Compute CLVs at output points

**Advantage:** Algorithm is proven, matches reference

### Option B: Debug Wolfe-Samelson

1. Verify adjoint dynamics sign convention against literature
2. Implement proper backward FSV evolution
3. Compute ALL CLVs via intersection, not just CLV₃
4. Store states at full QR resolution, not coarse output
5. Add extensive validation at each step

---

## Existing Code

### Working
- `lyapunovvectors.jl`: Ginelli CLV implementation (validated, CPU-only)
- `scans/sm_sweep.jl`: GPU parameter sweep template (no CLVs)

### Broken
- `scans/sm_clv_angle_sweep.jl`: Failed Wolfe-Samelson attempt
- `tests/clv_validation.jl`: Validation test (shows failure)

---

## Key Technical Constraints

### DiffEqGPU.jl
- Works with `StaticArrays` and `Float32`
- Use `vectorized_asolve` for GPU execution
- `save_everystep=false` returns final state directly (not array)

### Metal.jl (Apple Silicon)
- Use `MtlArray` for GPU arrays
- `Metal.@sync` for synchronization
- Device memory is shared with system (unified memory)

### QR Decomposition
- No GPU-native QR in Metal.jl for general matrices
- Hand-coded 3×3 Modified Gram-Schmidt works
- Need positive diagonal convention for CLV uniqueness

---

## Lessons Learned

1. **Validate early:** Should have compared against reference before building full sweep
2. **CLVs converge slowly:** Need long transients (both forward and backward)
3. **Sign conventions matter:** Adjoint dynamics have subtle sign issues
4. **Memory vs accuracy tradeoff:** Coarse output intervals cause frozen Jacobian issues
5. **Wolfe-Samelson is complex:** Despite O(1) memory claim, implementation is tricky

---

## Files in Repository

```
MultimodalMaps.jl/
├── WoSa07.pdf                  # Wolfe-Samelson 2007 paper (algorithm reference)
├── lyapunovvectors.jl          # Working Ginelli CLV (CPU)
├── GPU_sweep.md                # This documentation
├── scans/
│   ├── sm_sweep.jl             # GPU sweep template  
│   └── sm_clv_angle_sweep.jl   # BROKEN Wolfe-Samelson attempt
├── tests/
│   └── clv_validation.jl       # Validation test
└── Project.toml                # Dependencies
```

---

## Dependencies

```toml
DiffEqGPU = "071ae1c0-96b5-11e9-1965-c90190d839ea"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
LinearAlgebra  # stdlib
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DynamicalSystems = "61744808-ddfa-5f27-97ff-6e42cc95d634"
ProgressMeter = "92933f4c-e287-5a05-a399-4b506db050ca"
```

---

## Recommended Next Steps

1. **Verify adjoint convention** by checking Wolfe-Samelson 2007 paper equations directly
2. **Implement full CLV intersection** for all vectors, not just CLV₃
3. **Test on simple system** (e.g., 2D with known CLVs) before Shimizu-Morioka
4. **Consider Ginelli fallback** if Wolfe-Samelson proves too complex

---

## Contact/Context

- Target hardware: Apple M-series Mac with Metal GPU
- Parameter sweep: α ∈ [0.15, 0.65], λ ∈ [0.5, 0.95]
- Output: Heatmap of minimum angle between CLV₃ and plane(CLV₁, CLV₂)
- Performance target: Handle 1M+ trajectories efficiently
