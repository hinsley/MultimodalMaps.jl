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

The heatmap uses Makie's categorical `:glasbey_bw_n256` colormap without text
labels over the cells.

Run from the repo root:

```bash
JULIA_NUM_THREADS=10 julia --project=. kneading/experiment/attempt-010/main.jl
```

Outputs:

- `lowres_encoding_results.tsv`
- `lowres_encoding_legend.tsv`
- `lowres_encoding_heatmap.png`
- `lowres_benchmark_summary.txt`

The script benchmarks the full coarse grid in serial and threaded modes before
writing the outputs, and it also records per-stage timings for:

- `Γ_SD^-`
- `T0`
- `T_scs`
- `Γ_SD^-_scs`

If you want a denser preliminary run later, increase the grid with:

```bash
ATTEMPT010_NX=6 ATTEMPT010_NY=7 JULIA_NUM_THREADS=10 julia --project=. kneading/experiment/attempt-010/main.jl
```

Do not use that denser run as the full production scan without checking the low-resolution
results first.
