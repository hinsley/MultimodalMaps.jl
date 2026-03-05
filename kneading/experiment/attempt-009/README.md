# Attempt 009

This is a reduced, local smoke test of
`PlantChaos/kneading/continuous_critical_itineraries.jl`.

It uses the exact Plant model from that code path, not the Rossler examples in
this repo. The runtime is reduced by:

- cutting the `ΔCa` sweep down to `12` points
- shrinking the return-map probe resolution to `20`
- truncating the SSCS collection at `20` symbols
- omitting the expensive LLE sweep and plotting/output machinery

Vendored sources in `vendor/`:

- `Plant.jl`: exact model source copied from the local sibling repo
  `../SiN_h_current/research/lyapunov_gh_delta_sweep/attempt-005/Plant.jl`
- `equilibria_subset.jl`: exact copied subset from `PlantChaos/tools/equilibria.jl`
- `symbolics_subset.jl`: exact copied subset from `PlantChaos/tools/symbolics.jl`

Still using the MultimodalMaps source in this repo:

- `kneading/power_series.jl`
- `kneading/smallest_root.jl`

Run from the repo root:

```bash
julia --project=. kneading/experiment/attempt-009/main.jl
```

The script writes a reduced summary to
`kneading/experiment/attempt-009/smoke_results.tsv`.
