# attempt-056

`attempt-056` tests the SSCS-event tangent observable requested after
`attempt-055`.

The T0 continuation path is intentionally inherited from `attempt-050`:

- fixed `Delta Ca` columns
- decreasing `Delta x` within each column
- previous successful `(T0_V, T0_Ca)` used as the continuation seed
- fallback to the full remap initializer
- red T0 validation requires the computed `T_scs` to start with `+/-1`

The new diagnostic differs from `attempt-055` in the event and observable:

- active `g_h = 0` state only: `(x, n, h, Ca, V)`
- passive `y` is absent from both trajectory and tangent dynamics
- events are the SSCS symbol detections, not Ca-minima
- the recorded scalar is the tangent `V` component at each SSCS symbol event
- contours are zero contours of that tangent `V` component

Local 200 x 200 test runner:

```bash
kneading/experiment/attempt-056/run_local_grid200_sscs_vtangent.sh
```

Default settings:

- grid `200 x 200`
- `MAX_ITER=8`
- `tmax=1e5`
- output tag `grid200_sscs_vtangent_dotzero_tmax1e5_iter8_ystub`
- parameter window `Delta Ca in [-45, -20]`, `Delta x in [-1.5, -0.5]`

The runner writes per-column checkpoint TSVs, a merged TSV, one contour PNG, and
one summary file.
