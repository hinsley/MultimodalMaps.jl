# Attempt 034

This attempt returns to the original Shimizu-Morioka contour quantity from
`attempt-024`:

- integrate the orbit together with a tangent vector,
- orthonormalize that tangent against the flow after every step,
- at each `|x|`-maximum event, record the signed `x`-component of the
  orthonormalized tangent,
- contour those zero sets in parameter space.

The change is the **initial condition** used for the kneading-style scan.

Instead of starting every parameter point from the unstable manifold of the
origin and then reading off events directly, `attempt-034` first constructs a
critical-point initial condition for the 1D `|x|`-maxima return map on the
positive-`x` section `y = 0`.

## Critical-point seeding

For a parameter point `(alpha, lambda)` the scan uses two layers:

1. **Periodic full reseed**
   - start from one unstable-manifold side of the origin, then vary the
     initial `z` value downward from `z = 0` along that near-origin chart,
   - for each such near-origin seed, integrate only far enough to collect the
     first two accepted `|x|`-maximum section hits,
   - use that `z`-family to choose a plausible section layer `z = z_fixed`,
   - at that fixed `z`, run a broader `x`-only scan to locate the first smooth
     discrete critical point of the one-step `|x|`-max return map,
   - refine that critical point with **golden-section search in `x` only**,
     holding `y = 0` and `z = z_fixed` fixed.

2. **Local continuation between reseeds**
   - hold the previous `z` value fixed,
   - scan a small local `x` window around the previous critical point,
   - bracket the nearest local extremum of the same type,
   - refine it again with golden-section search in `x` only.

If the local continuation fails, the code falls back to a full reseed at that
parameter point. If even the full reseed fails, the last seed is carried.

Columns are solved at fixed `alpha`, with `lambda` traversed from
`ATTEMPT033_LAMBDA_MAX` down to `ATTEMPT033_LAMBDA_MIN`, so each column carries
its critical seed downward in `lambda`.

The reseed cadence is controlled by `ATTEMPT033_RESEED_PERIOD`.

## Plot output

The final contour plot overlays only `|x|`-maximum iterates `2:8`.

Default output files:

- `grid500_branch16_seedz_goldx_floworth_absx_plot8_shimizu_morioka_cpu_results.tsv`
- `grid500_branch16_seedz_goldx_floworth_absx_plot8_shimizu_morioka_cpu_iterate_colors.tsv`
- `grid500_branch16_seedz_goldx_floworth_absx_plot8_shimizu_morioka_cpu_contours.png`

Run from repo root:

```bash
./kneading/experiment/attempt-034/run_grid500_branch16_seedz_goldx_floworth_absx_plot8_upload.sh
```

The runner executes the full `500 x 500` sweep and uploads the final PNG to
TGLFS.
