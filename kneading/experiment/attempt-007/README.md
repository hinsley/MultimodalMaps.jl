# Attempt 007 (Rossler Branch-Weighted Entropies)

Rossler entropy analysis using the same workflow and plotting style as `attempt-006`, with branch intervals built from the return-map critical point:

- left branch: `[f(f(c_map)), c_map]`
- right branch: `[c_map, f(c_map)]`

These branch definitions are used for both branch-weighted kneading times and the `h_map / mean(tau)` suspension estimate.

## Run

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-007/EntropiesRossler.py
```

## Outputs

- `entropies_rossler_branch_weighted.png`
- `htop_convergence_branch_vs_pointwise.png`
