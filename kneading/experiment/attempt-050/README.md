# attempt-050

`attempt-050` is the `attempt-049` cloud sweep/plot pipeline carried forward
with one intended numerical change:

```text
SSCS integration horizon: tmax = 1.0e5
```

The previous attempt-049 run used the inherited SSCS horizon
`SSCS_ODE_TSPAN = (0.0, 5.0e4)`. This attempt overrides `compute_sscs` locally
in `main.jl` so that the signed spike count sequences are computed over
`(0.0, 1.0e5)` without modifying older attempt folders.

Core defaults:

- grid: `1000 x 1000`
- region: `Delta Ca in [-45, -20]`, `Delta x in [-1.5, -0.5]`
- max SSCS length: `12`
- T0 remap resolution: `40`
- output tag: `grid1000_seq12_tmax1e5_prefixes_remap40_newmodel`
- filtered final-figure tag: `grid1000_seq12_tmax1e5_prefixcompatible`

The normal runner is:

```bash
kneading/experiment/attempt-050/run_grid1000_seq12_tmax1e5_prefixes_remap40_newmodel.sh
```

That runner performs the scan, writes the merged TSV and prefix plots, then
generates the filtered prefix-compatible contour figure using
`plot_filtered_full_contours.jl`.

For GCE/GCS operation, see `GCE_GCS_RUNBOOK.md`.
