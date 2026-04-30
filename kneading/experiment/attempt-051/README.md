# attempt-051

`attempt-051` carries forward the `attempt-050` cloud sweep and filtered
contour plotting pipeline, but widens the parameter window and runs three
`g_h` cases.

Locked run settings:

- grid: `1000 x 1000`
- region: `Delta Ca in [-60, 20]`, `Delta x in [-2, 2]`
- model overrides: `tau_y = 2e4`, `g_h in {0, 1e-3, 1e-2}`
- sequence length: `12`
- SSCS integration window: `tspan = (0, 1e5)`
- T0 remap resolution: `40`
- checkpointing: one directory of `column_XXXX.tsv` files per `g_h` case

Primary entrypoint:

```bash
kneading/experiment/attempt-051/run_grid1000_seq12_tmax1e5_gh_sweep.sh
```

Cloud artifact prefix used for the live run:

```bash
gs://carter-kneading-attempt048/attempt-051
```

Each `g_h` case writes a raw merged TSV, full/prefix replay plots, legends,
and the filtered final plot. The deterministic case labels are:

- `gh0p000`
- `gh0p001`
- `gh0p01`

The filtered final plot for each case is named:

```text
grid1000_seq12_tmax1e5_<gh-label>_prefixcompatible_tzero2to12_contours.png
```
