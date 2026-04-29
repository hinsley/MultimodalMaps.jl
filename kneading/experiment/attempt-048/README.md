# attempt-048

`attempt-048` migrates the `attempt-016` `500 x 500`, 7-symbol prefix contour
workflow to the current shared SiN model in `models/SiN.jl`.

The important difference from the historical image
`attempt-016/grid500_seq7_prefixes_remap40_contours.png` is model provenance:

- the historical PNG was generated before `models/SiN.jl` existed and used
  `attempt-009/vendor/Plant.jl`
- this attempt includes `attempt-011/main.jl`, which includes
  `attempt-009/main.jl`, which now includes `models/SiN.jl`
- both models have `g_h = 0`, but the current model has the updated shared
  parameter layout and `EL = -40`

No data have been generated in this folder by this setup change.

Run later with:

```bash
kneading/experiment/attempt-048/run_grid500_seq7_prefixes_remap40_newmodel.sh
```

The default output tag is `grid500_seq7_prefixes_remap40_newmodel`, so this
does not overwrite `attempt-016` artifacts.

For Google Compute Engine / optional Google Cloud Storage usage, see
`GCE_GCS_RUNBOOK.md`.
