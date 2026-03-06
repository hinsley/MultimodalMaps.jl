# Attempt 010

This is a low-resolution 2D follow-up to
`kneading/experiment/attempt-009`.

It reuses the exact Plant-model SSCS pipeline from `attempt-009` and scans over:

- `ΔCa ∈ [-33, -20]`
- `Δx ∈ [-1.5, -0.5]`

The default run is intentionally coarse:

- `4` points in `Δx`
- `5` points in `ΔCa`

Each parameter point computes:

1. `T0`
2. `Γ_SD^-`
3. the SSCS of each trajectory
4. a deterministic exact integer encoding of the pair `(T_scs, Γ_SD^-_scs)`

The heatmap uses randomized categorical colors, so nearby encoding values do not
produce nearby colors.

Run from the repo root:

```bash
julia --project=. kneading/experiment/attempt-010/main.jl
```

Outputs:

- `lowres_encoding_results.tsv`
- `lowres_encoding_legend.tsv`
- `lowres_encoding_heatmap.png`

If you want a denser preliminary run later, increase the grid with:

```bash
ATTEMPT010_NX=6 ATTEMPT010_NY=7 julia --project=. kneading/experiment/attempt-010/main.jl
```

Do not use that denser run as the full production scan without checking the low-resolution
results first.
