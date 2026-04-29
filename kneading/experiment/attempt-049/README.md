# attempt-049

`attempt-049` extends the successful `attempt-048` cloud workflow to a
`1000 x 1000`, 12-symbol prefix contour sweep on the current shared SiN model
in `models/SiN.jl`.

The important difference from the historical image
`attempt-016/grid500_seq7_prefixes_remap40_contours.png` is model provenance:

- the historical PNG was generated before `models/SiN.jl` existed and used
  `attempt-009/vendor/Plant.jl`
- this attempt includes `attempt-011/main.jl`, which includes
  `attempt-009/main.jl`, which now includes `models/SiN.jl`
- both models have `g_h = 0`, but the current model has the updated shared
  parameter layout and `EL = -40`

Compared with `attempt-048`, this attempt changes:

- grid resolution from `500 x 500` to `1000 x 1000`
- max SSCS length and prefix plots from `7` to `12`
- default contour linewidth from `0.8` to `0.35`
- default plot canvas to `1600 x 1200` with `px_per_unit = 2.0`
- larger axis label, title, and tick-label fonts

No data have been generated in this folder yet.

Run later with:

```bash
kneading/experiment/attempt-049/run_grid1000_seq12_prefixes_remap40_newmodel.sh
```

The default output tag is `grid1000_seq12_prefixes_remap40_newmodel`, so this
does not overwrite `attempt-016` artifacts.

For Google Compute Engine / optional Google Cloud Storage usage, see
`GCE_GCS_RUNBOOK.md`.
