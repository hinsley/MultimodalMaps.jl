# Attempt 040

This attempt reruns the `attempt-038` method unchanged at the numerical level,
but adds explicit diagnostics to investigate the diagonal seam visible in the
`attempt-038` contour plot.

The seam hypothesis from `attempt-038` is:

- it is probably **not** caused by a dense wall of explicit full reseeds,
- it is more likely caused by the local x-only secant corrector snapping onto
  the coarse local scan grid or otherwise returning a best scanned fallback
  point,
- the telltale sign was repeated `x` jumps of roughly `0.015`, which is
  exactly the spacing of the 17-point local scan over width `0.24`.

So `attempt-040` keeps the same continuation and contour quantity, but records
how every local correction was actually obtained.

## Same numerical method as attempt-038

The contour quantity is still:

- integrate the orbit together with a tangent vector,
- orthonormalize that tangent against the flow after every step,
- at each `|x|`-maximum, record the signed `x`-component of the orthonormalized
  tangent,
- overlay the zero contours for iterates `2:8`.

The critical-point initial condition is still computed as follows:

1. **Periodic/full reseed**
   - integrate a long orbit from the unstable manifold of the origin,
   - build the positive-`x` sampled branch of the one-step return map,
   - detect the first smooth critical point of that sampled branch,
   - refine that sampled point by x-only golden-section search at fixed `z_*`,
   - then locally refine by the angle-based x-only secant corrector.

2. **Local continuation**
   - keep `z_*` fixed from the carried seed,
   - vary only `x` on the section point `(x, 0, z_*)`,
   - build the event-corrected `(x, z)` return-map Jacobian with
     `SciMLSensitivity.ODEForwardSensitivityProblem`,
   - compute the minimal eigendirection angle `theta(x)`,
   - approximate `theta'(x)` by a centered two-point stencil,
   - solve `theta'(x) = 0` by a safeguarded secant step.

There is still **no `z` correction** in local continuation.

## New diagnostics

For every parameter point, the code now records:

- whether the point came from a full reseed, fallback full reseed, local
  continuation, or carry,
- whether the local x-only corrector started from a true sign-change bracket or
  only from the nearest neighbor of the best scanned `|theta'(x)|`,
- the input `x`,
- the best scanned `x`,
- the initial scan bracket endpoints,
- the local scan spacing,
- the number of secant iterations actually taken,
- whether the initial local scan bracket was a true sign-change bracket.

The local refine mode is distinguished explicitly. Examples:

- `scan_grad_tol`: the scanned point already had sufficiently small
  `|theta'(x)|`, so the secant solver effectively accepted the scanned point.
- `secant_step_tol`: the secant solver actually moved and converged by small
  step size.
- `secant_eval_fail_best`: a secant trial failed to evaluate, so the best point
  seen so far was returned.
- `secant_max_iters_best`: the iteration limit was hit and the best point seen
  so far was returned.

This is the central change relative to `attempt-038`: the results file no
longer hides scan-grid fallback behind the single label `angle_secant`.

## Diagnostic outputs

In addition to the usual contour plot, this attempt writes:

- `..._diagnostics.tsv`
- `..._diagnostic_overlay.png`

The overlay plot keeps the normal contour lines and marks:

- red `×`: full reseeds or fallback full reseeds,
- orange squares: scan-fallback / unbracketed local corrector points,
- cyan diamonds: large `x` jumps relative to the previous `lambda` in the same
  `alpha` column.

The idea is to see whether the seam lines up with:

- explicit reseed locations,
- scan-grid fallback locations,
- or scan-spacing-sized continuation jumps.

## Outputs

Default output tag:

- `grid500_branch16_attractorseed_anglesecantdiag_floworth_absx_plot8_shimizu_morioka_cpu`

Default files:

- `..._results.tsv`
- `..._iterate_colors.tsv`
- `..._contours.png`
- `..._diagnostics.tsv`
- `..._diagnostic_overlay.png`

Run from repo root:

```bash
./kneading/experiment/attempt-040/run_grid500_branch16_seedz_goldx_floworth_absx_plot8_upload.sh
```

The runner uploads both PNGs to TGLFS at the end.
