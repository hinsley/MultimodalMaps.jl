# Attempt 038

This attempt keeps the contour quantity from `attempt-037` unchanged:

- integrate the orbit together with a tangent vector,
- orthonormalize that tangent against the flow after every step,
- at each `|x|`-maximum event, record the signed `x`-component of the
  orthonormalized tangent,
- contour those zero sets in parameter space.

The change is again only the **initial condition** used for the kneading-style
scan.

## Critical-point continuation

For each parameter point, the initial condition is constrained to the section
point

`(x, 0, z_*)` with `z_*` fixed.

There is **no `z` correction step**. Continuation is scalar in `x` only.

The continuation uses two layers:

1. **Column-start reseed from the attractor return map**
   - integrate a long orbit from the unstable manifold of the origin,
   - sample the positive-`x` branch of the one-step map `x_n -> x_{n+1}^2`,
   - detect the first smooth critical point of that sampled branch,
   - refine that sampled critical point with the old x-only golden-section
     search at fixed `z_n`,
   - use the resulting `(x_n, 0, z_n)` as the initial guess for the new
     angle-based corrector below.

2. **Local x-only secant corrector on `theta'(x)`**
   - hold the carried `z_*` fixed,
   - build the Jacobian of the next-return map on the `(x, z)` section by a
     `SciMLSensitivity.ODEForwardSensitivityProblem`,
   - include the event-time correction at the next `|x|`-maximum so the return
     derivatives are tangent to the section,
   - form the `2 x 2` return-map Jacobian in `(x, z)` coordinates,
   - compute the minimal eigendirection angle
     `theta(x) = atan(sqrt((a-d)^2 + 4bc), abs(b-c))`,
   - approximate `theta'(x)` by a centered two-point stencil in `x`,
   - solve `theta'(x) = 0` by a safeguarded secant step, updating `x` only.

If direct local secant continuation fails from the carried point, the code
scans a small local `x` window at fixed `z_*`, looks for either a nearby sign
change of `theta'(x)` or, failing that, the smallest sampled `|theta'(x)|`,
and uses that local information to seed the secant pair. If the local secant
still fails, the best scanned fallback point is used directly.

If the local corrector fails at a point, the code falls back to a fresh
attractor-map reseed there. If that also fails, the previous seed is carried.

Columns are solved at fixed `alpha`, with `lambda` traversed from
`ATTEMPT033_LAMBDA_MAX` down to `ATTEMPT033_LAMBDA_MIN`.

## Numerical concession

`theta(x)` itself is built from a forward-sensitivity return-map Jacobian.
However, to avoid implementing second- and third-order event-time sensitivity
equations for exact higher derivatives of `theta`, this attempt uses a centered
two-point finite-difference stencil in `x` only for `theta'(x)`.

So:

- the return-map Jacobian is sensitivity-based,
- the secant corrector only sees a finite-difference approximation of the
  scalar derivative `theta'(x)`,
- there is no exact `theta''(x)` in this attempt, so the stored
  `critical_theta_dxx` field is `NaN`.

That is the only intended approximation relative to a fully exact higher-order
event-sensitivity method.

## Outputs

The final contour plot overlays iterates `2:8`.

Default output files:

- `grid500_branch16_attractorseed_anglesecant_floworth_absx_plot8_shimizu_morioka_cpu_results.tsv`
- `grid500_branch16_attractorseed_anglesecant_floworth_absx_plot8_shimizu_morioka_cpu_iterate_colors.tsv`
- `grid500_branch16_attractorseed_anglesecant_floworth_absx_plot8_shimizu_morioka_cpu_contours.png`

Run from repo root:

```bash
./kneading/experiment/attempt-038/run_grid500_branch16_seedz_goldx_floworth_absx_plot8_upload.sh
```

The runner executes the full `500 x 500` sweep and uploads the final PNG to
TGLFS.
