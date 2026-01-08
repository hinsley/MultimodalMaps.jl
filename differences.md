# Kuptsov-Parlitz CLV Parity Check (Julia vs Python)

This note verifies that the Julia `clv_wolfe_samelson` implementation matches
the Python `CLV` routine in `Lyapunov.py`.

## Status

- Phase order and data flow match 1:1 (transient, Phi- convergence, store Phi/R,
  store R, A convergence, CLV back-substitution).
- QR uses raw `qr` without sign correction, matching `np.linalg.qr`.
- `A` initialization uses `triu(rand(...))`, and the column-normalize + solve
  step matches the Python `C = diag(1/norm)`, `B = A*C`, `A = solve(R,B)`.
- Storage layout uses `N+1` snapshots and the first CLV is unnormalized, then
  subsequent CLVs are normalized (same as Python).
- Return signature matches Python: `V` only, `(V, history)` if `traj=true`,
  and `check=true` appends a system copy.
- No Lyapunov exponent computation is performed in this routine.

## Notes

- The function name remains `clv_wolfe_samelson` for API compatibility, but the
  implementation is Kuptsov-Parlitz throughout.
