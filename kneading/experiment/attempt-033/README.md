# Attempt 033

This attempt returns to the original Shimizu-Morioka contour quantity from
`attempt-024`:

- integrate the orbit together with a tangent vector,
- orthonormalize that tangent against the flow after every step,
- at each `|x|`-maximum event, record the signed `x`-component of the
  orthonormalized tangent,
- contour those zero sets in parameter space.

The change is the **initial condition** used for the kneading-style scan.

Instead of starting every parameter point from the unstable manifold of the
origin and then reading off events directly, `attempt-033` first constructs a
critical-point initial condition for the 1D `|x|`-maxima return map on the
positive-`x` section `y = 0`.

## Critical-point seeding

For a parameter point `(alpha, lambda)` the scan uses two layers:

1. **Periodic full reseed**
   - integrate one long orbit from the unstable manifold of the origin,
   - collect many `|x|`-maximum events,
   - sort the positive-`x` branch by current `x`,
   - locate the first smooth discrete critical point of
     `x_{n+1}^2` as a function of the current `x_n`,
   - refine that critical point with **golden-section search in `x` only**,
     holding `y = 0` and the sampled `z` value fixed.

2. **Local continuation between reseeds**
   - hold the previous `z` value fixed,
   - scan a small local `x` window around the previous critical point,
   - bracket the nearest local extremum of the same type,
   - refine it again with golden-section search in `x` only.

If the local continuation fails, the code falls back to a full reseed at that
parameter point. If even the full reseed fails, the last seed is carried.

The reseed cadence is controlled by `ATTEMPT033_RESEED_PERIOD`.

## Plot output

The final contour plot overlays only `|x|`-maximum iterates `2:8`.

Default output files:

- `grid500_branch16_critgoldx_floworth_absx_plot8_shimizu_morioka_cpu_results.tsv`
- `grid500_branch16_critgoldx_floworth_absx_plot8_shimizu_morioka_cpu_iterate_colors.tsv`
- `grid500_branch16_critgoldx_floworth_absx_plot8_shimizu_morioka_cpu_contours.png`

Run from repo root:

```bash
./kneading/experiment/attempt-033/run_grid500_branch16_critgoldx_floworth_absx_plot8_upload.sh
```

The runner executes the full `500 x 500` sweep and uploads the final PNG to
TGLFS.
