# Attempt 041

This attempt is a direct rerun of `attempt-040` with one operational change:
within each fixed-`alpha` column, the scan now starts at the **lowest**
`lambda` value and marches upward to the highest `lambda`.

Everything else is intentionally kept the same:

- the contour quantity is still the flow-orthonormalized tangent-vector
  `x`-component at successive `|x|`-maxima,
- the critical-point initial condition is still continued by the x-only
  angle-secant corrector,
- the local continuation still holds the section `z_*` fixed and updates only
  `x`,
- the seam diagnostics from `attempt-040` are preserved.

## Why this attempt exists

`attempt-040` suggested that some of the visible seam structure was tied to how
the local continuation propagated its carried seed through a column. Since that
propagation is directional, reversing the `lambda` traversal is a clean test:

- if the seam is strongly traversal-dependent, it should move or change shape,
- if it is mostly tied to the local scan-grid fallback itself, the same
  problematic region should still light up in the diagnostic overlay.

So `attempt-041` is meant to isolate the effect of continuation direction while
keeping the same numerical method and the same diagnostics.

## Numerical method

The numerical method matches `attempt-040`:

1. **Periodic/full reseed**
   - integrate a long orbit from the unstable manifold of the origin,
   - build the positive-`x` sampled branch of the one-step return map,
   - detect the first smooth critical point of that sampled branch,
   - refine it by x-only golden-section search at fixed `z_*`,
   - then locally refine by the x-only angle-secant corrector.

2. **Local continuation**
   - keep `z_*` fixed from the carried seed,
   - vary only `x` in the section point `(x, 0, z_*)`,
   - build the event-corrected `(x, z)` return-map Jacobian with
     `SciMLSensitivity.ODEForwardSensitivityProblem`,
   - compute the minimal eigendirection angle `theta(x)`,
   - approximate `theta'(x)` by a centered two-point stencil,
   - solve `theta'(x) = 0` by a safeguarded secant step.

The only traversal change is:

- `attempt-040`: `lambda` descending within each column,
- `attempt-041`: `lambda` ascending within each column.

The diagnostic jump analysis is also ordered along the actual traversal
direction, so the “jump relative to previous lambda” is now measured from lower
to higher `lambda`.

## Outputs

Default output tag:

- `grid500_branch16_attractorseed_anglesecantdiag_ascendinglambda_floworth_absx_plot8_shimizu_morioka_cpu`

Default files:

- `..._results.tsv`
- `..._iterate_colors.tsv`
- `..._contours.png`
- `..._diagnostics.tsv`
- `..._diagnostic_overlay.png`

Run from repo root:

```bash
./kneading/experiment/attempt-041/run_grid500_branch16_seedz_goldx_floworth_absx_plot8_upload.sh
```

The runner uploads both PNGs to TGLFS at the end.
