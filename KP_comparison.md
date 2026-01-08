# Kuptsov–Parlitz CLV: `Lyapunov.py` vs `lyapunovvectors.jl`

This note compares the Kuptsov–Parlitz CLV algorithm in `Lyapunov.py` with
the current Julia implementation in `lyapunovvectors.jl` (`clv_wolfe_samelson`).

## High‑level correspondence (algorithmic structure)

Both implementations follow the same staged procedure:

1. **Forward transient** to reach the attractor.
2. **Converge backward Lyapunov vectors (Φ⁻)** via forward QR.
3. **Store Φ⁻ and R** for `n_B` steps.
4. **Store additional R** for `n_C` steps to converge the coefficient matrix A.
5. **Backward reconstruction** of CLVs using stored Φ⁻ and R.

## Parameter mapping

| Python (`Lyapunov.py`) | Julia (`lyapunovvectors.jl`) | Meaning |
| --- | --- | --- |
| `n_forward` | `Ttr` | Initial transient (base only; deviations reset afterward) |
| `n_A` | `Ttr_bsv` | Steps to converge Φ⁻ |
| `n_B` | `N` | Number of stored CLV snapshots |
| `n_C` | `Ttr_fsv` | Steps to converge A (A⁻) |
| `dt` | `Δt` | QR step size |

## Core step equivalence (matches)

**Forward convergence + storage**
- Python: `W = next_LTM(W); W, R = qr(W); Phi_list.append(W); R_list1.append(R)`
- Julia: `step!(tands, Δt); Q=QR(current_deviations); PhiMns[i]=Q; R_list1[i]=R`

**A‑convergence**
- Python: `C = diag(1/||A[:,j]||); B = A @ C; A = solve(R, B)` over `R_list2` reversed.
- Julia: per column normalize `A` into `B`, then `A = R \ B` using `UpperTriangular(R)` over `R_list2` reversed.

**CLV reconstruction**
- Python: iterate backward, update `A` using `R_list1` then `CLV = Q @ A`, normalize columns.
- Julia: same logic, `V_history[i] = PhiMns[i] * A`, normalize columns.

These steps are mathematically the same.

## Differences / potential mismatches

There are no substantive differences remaining in the Kuptsov–Parlitz logic or
indexing. The Julia version mirrors the Python implementation’s:

- base-only transient,
- raw QR (no sign forcing),
- random upper-triangular `A` initialization,
- `N+1` storage (initial Φ⁻ snapshot included),
- backward reconstruction and normalization pattern,
- optional trajectory return and optional “check” copy.

## Bottom line

The Julia implementation now matches the **Kuptsov–Parlitz logic and indexing**
of the Python version.
