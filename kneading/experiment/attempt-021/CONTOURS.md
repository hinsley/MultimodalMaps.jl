# Attempt 021 Contour Definition

This attempt scans the Shimizu-Morioka system

```math
\dot x = y,\qquad
\dot y = x - \lambda y - x z,\qquad
\dot z = -\alpha (z - x^2)
```

over the parameter rectangle

- `alpha in [0.0, 0.7]`
- `lambda in [0.2, 1.6]`

using one orbit per parameter pair: the right unstable branch of the saddle equilibrium at the origin.

## One Orbit Per Grid Point

For each `(alpha, lambda)` pair, the scan integrates a single orbit started from a small displacement along the unstable eigenvector of the origin in the `x-y` block:

- unstable eigenvalue:
  `mu = 0.5 * (-lambda + sqrt(lambda^2 + 4))`
- unstable direction:
  `[1, mu, 0]`
- initial condition:
  `eps0 * normalize([1, mu, 0])`

with `eps0 = ATTEMPT021_EPS0`.

Implementation:

- [`unstable_side_initial_condition`](./main.jl)
- [`scan_orbit`](./main.jl)

## Return-Map Samples Without Building the Return Map

The scan does not explicitly construct a 1D return map. Instead, it extracts the orbit's successive local maxima of `z`, which are the data that would feed such a map.

Because

```math
\dot z = -\alpha (z - x^2),
```

the sign of `z - x^2` determines whether `z` is increasing or decreasing:

- `z - x^2 < 0` means `dot z > 0`
- `z - x^2 > 0` means `dot z < 0`

So a local maximum of `z` occurs when the orbit crosses the nullcline `z = x^2` in the direction

- `z - x^2 : negative -> positive`

The code tracks this scalar:

```julia
zprime_proxy(u) = u[3] - u[1] * u[1]
```

and records up to `ATTEMPT021_MAX_ZMAX` such crossings.

Implementation:

- [`zprime_proxy`](./main.jl)
- crossing test inside [`scan_orbit`](./main.jl)

## Quadratic Estimation of Each `z` Maximum

When a `negative -> positive` crossing of `z - x^2` is detected, the code estimates the corresponding `z` maximum value in two stages:

1. a fallback linear estimate from the two samples bracketing the sign change
2. a higher-quality quadratic estimate from three consecutive `(t, z)` samples

The quadratic refinement fits a parabola through

- previous-previous sample
- previous sample
- current sample

and uses the vertex time if that vertex lies inside the local time window.

Implementation:

- [`quadratic_vertex_time`](./main.jl)
- [`quadratic_interp`](./main.jl)
- refinement block inside [`scan_orbit`](./main.jl)

This matters because the red contours are determined from the sequence of these `z` maxima, so noisy peak values would directly move those contours.

## Blue Contours: Near-Saddle Slowdowns

Blue contours encode where the orbit makes a sufficiently slow and sufficiently close pass near the saddle between two successive `z` maxima.

For each return segment, the code tracks:

- the minimum flow speed encountered in that segment
- the minimum Euclidean radius `norm([x, y, z])` encountered in that segment

The flow speed is

```math
\|\dot u\| = \sqrt{\dot x^2 + \dot y^2 + \dot z^2}.
```

If, during a segment, both of the following hold

- `segment_min_speed <= ATTEMPT021_NEAR_SADDLE_SPEED`
- `segment_min_radius <= ATTEMPT021_NEAR_SADDLE_RADIUS`

then that return segment is flagged as a near-saddle segment.

This is stored in `blue_mask` as a bitmask:

- bit `1` means the first return segment satisfies the near-saddle condition
- bit `2` means the second return segment satisfies it
- etc.

So the blue contour family is not a scalar contour of speed. It is the parameter-space boundary between different near-saddle bitmask categories.

Implementation:

- `flow_speed` in [`main.jl`](./main.jl)
- segment minimum logic in [`scan_orbit`](./main.jl)

## Red Contours: Criticality of the Reduced `z`-Maximum Dynamics

Red contours approximate where the reduced 1D dynamics of successive `z` maxima develops a critical point.

Let the recorded `z` maxima be

```text
z1, z2, z3, ...
```

For every interior triple

```text
a = z_{n-1}, b = z_n, c = z_{n+1},
```

the code checks whether the monotonicity changes at `b`, i.e. whether the discrete slopes

- `b - a`
- `c - b`

change sign. In code:

```julia
rise = b - a
fall = c - b
rise * fall <= 0
```

with a small amplitude cutoff `ATTEMPT021_RED_EPS` to avoid tagging numerical flatness.

If this happens, then iterate `n` is marked as critical in `red_mask`.

Interpretation:

- if the sequence of maxima changes from increasing to decreasing, `z_n` behaves like a local maximum of the return-map iterate sequence
- if it changes from decreasing to increasing, `z_n` behaves like a local minimum

So the red contours approximate parameter values where the reduced map crosses a smooth critical point.

Again, the plotted red curves are boundaries in parameter space between different `red_mask` categories.

Implementation:

- red-mask logic in [`scan_orbit`](./main.jl)

## What Is Actually Plotted

The scan writes one result row per parameter point, including:

- `maxima_count`
- `blue_mask`
- `red_mask`
- recorded `zmax_values`
- recorded `return_times`
- `status`

Then [`contours.jl`](./contours.jl) compresses the distinct masks into categorical IDs:

- one ID for each distinct blue mask
- one ID for each distinct red mask

These IDs are stored on the `alpha x lambda` grid and converted into line segments with a categorical marching-squares routine.

That means:

- blue curves are boundaries between different near-saddle bitmask categories
- red curves are boundaries between different criticality bitmask categories

They are not contours of a single continuous scalar field.

Implementation:

- [`build_category_grids`](./contours.jl)
- [`categorical_marching_squares`](./contours.jl)
- [`save_contour_plot`](./contours.jl)

## Status Semantics

Each point is labeled with one of:

- `ok`: the orbit reached the requested number of `z` maxima
- `short`: the orbit remained finite but did not produce the full requested number of maxima before `T_END`
- `blowup`: the state exceeded `ATTEMPT021_MAX_STATE`
- `nonfinite`: the integrator produced a non-finite state

For plotting, both `ok` and `short` are treated as usable. The others are excluded from category assignment.

## Resulting Interpretation of the Final Figure

The final figure in attempt 021 is best read this way:

- blue lines show where the parameter dependence of near-saddle excursions changes
- red lines show where the parameter dependence of the `z`-maximum itinerary changes via an approximate critical point

So the picture is a qualitative bifurcation-style contour diagram derived from one canonical orbit branch, not a dense atlas of full return maps.
