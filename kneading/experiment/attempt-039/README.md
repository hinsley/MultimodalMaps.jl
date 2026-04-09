# Attempt 039

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

2. **Local x-only exact higher-order Newton corrector**
   - hold the carried `z_*` fixed,
   - integrate one augmented variational ODE containing the base state together
     with first-, second-, and selected third-order derivatives of the next
     return with respect to the section coordinates `(x, z)`,
   - correct those raw derivatives by the exact event-time shift at the next
     `|x|`-maximum so the returned derivatives are tangent to the section,
   - form the `2 x 2` return-map Jacobian in `(x, z)` coordinates and its
     first and second derivatives with respect to the section `x` coordinate,
   - compute the minimal eigendirection angle
     `theta(x) = atan(sqrt((a-d)^2 + 4bc), abs(b-c))`,
   - evaluate both `theta'(x)` and `theta''(x)` exactly from those corrected
     return derivatives,
   - solve `theta'(x) = 0` by damped Newton, updating `x` only.

If direct local Newton fails from the carried point, the code scans a small
local `x` window at fixed `z_*`, looks first for a nearby sampled point with
positive curvature and small `|theta'(x)|`, and otherwise falls back to the
best sampled local minimum of `theta(x)`. Newton is retried from that scanned
point. If even that still fails, the scanned fallback point is used directly.

If the local corrector fails at a point, the code falls back to a fresh
attractor-map reseed there. If that also fails, the previous seed is carried.

Columns are solved at fixed `alpha`, with `lambda` traversed from
`ATTEMPT033_LAMBDA_MAX` down to `ATTEMPT033_LAMBDA_MIN`.

## Numerical concession

This attempt removes the finite-difference concession from `attempt-038`.
The angle objective and its first two `x` derivatives come from one exact
augmented event solve, including the event-time correction through the needed
second- and third-order terms.

So:

- the stored `critical_theta`, `critical_theta_dx`, and
  `critical_theta_dxx` fields are all produced by the exact higher-order
  event-corrected return solve,
- there is no finite-difference stencil in the local corrector.

The remaining approximation is only the usual numerical ODE integration and
rootfinding error.

## Outputs

The final contour plot overlays iterates `2:8`.

Default output files:

- `grid500_branch16_attractorseed_angleexact_floworth_absx_plot8_shimizu_morioka_cpu_results.tsv`
- `grid500_branch16_attractorseed_angleexact_floworth_absx_plot8_shimizu_morioka_cpu_iterate_colors.tsv`
- `grid500_branch16_attractorseed_angleexact_floworth_absx_plot8_shimizu_morioka_cpu_contours.png`

Run from repo root:

```bash
./kneading/experiment/attempt-039/run_grid500_branch16_seedz_goldx_floworth_absx_plot8_upload.sh
```

The runner executes the full `500 x 500` sweep and uploads the final PNG to
TGLFS.
