# attempt-027: High-Resolution Recompute With Retired Squares

## Summary

`attempt-027` combines:

- the full `|x|`-max recomputation pipeline from `attempt-025`
- the retired-square rule from `attempt-026`
- the black background / white accepted / red pruned overlay style from the
  latest `attempt-026` plot
- a larger sweep and image configuration

The intended final plot overlays nominal iterates `2:8`. It uses:

- white contours for accepted squares
- red contours for squares whose skip test fired
- no per-iterate color separation
- no later white contour in any square that has ever already emitted red

## Differences From Previous Attempts

### Relative to attempt-025

`attempt-025` already had:

- full trajectory and event recomputation
- saved cumulative refined hit times in the TSV output
- delta-time conversion during plotting
- local per-square corner skip counters

But `attempt-025` still allowed a square to emit later white contours after a
skip increment had already occurred in that same square.

### Relative to attempt-026

`attempt-026` introduced the retired-square rule, but it was plotting-only and
reused the finished `attempt-025` sweep.

`attempt-027` keeps the same retired-square rule, but recomputes the full sweep
from scratch under the new output tag and then generates the final plot
automatically.

## Sweep Configuration

The default run uses:

- `N_alpha = 10000`
- `N_lambda = 10000`
- `alpha ∈ [0.0, 0.7]`
- `lambda ∈ [0.2, 1.6]`
- stored `|x|`-max iterates: `16`
- plotted nominal iterates: `8`
- plotted overlay range: `2:8`
- line width: `0.35`
- figure size: `10000 x 10000`
- `px_per_unit = 4.0`

## Event Generation

The trajectory and tangent integration stage is intentionally the same as
`attempt-025`.

At each parameter point:

1. Seed the unstable manifold from the saddle equilibrium at the origin.
2. Initialize the tangent vector as `[0, 0, 1]`.
3. Orthogonalize the tangent against the flow and normalize it.
4. Integrate the orbit+tangent system.
5. Detect `|x|`-max events using the sign change of `x*y`.
6. Refine event times using the quadratic vertex estimate on `x^2`.
7. Interpolate the event state and tangent to the refined hit time.
8. Re-orthogonalize and renormalize the tangent at the hit.

For every detected `|x|`-max event, write to the TSV:

- the signed tangent `x`-component used for contouring
- the refined cumulative hit time `t_hit`
- the full event state `(x, y, z)`

So yes: the event times are both refined and saved in the output dataset.

## Plotting Logic

The plotter first converts stored cumulative hit times `T_k` to interval times:

- `Δt_1 = T_1`
- `Δt_k = T_k - T_{k-1}` for `k >= 2`

The skip test then uses those interval times exactly as in `attempt-025` and
`attempt-026`.

For each square and nominal iterate:

1. Evaluate the square with the current local skip counters.
2. If missing-data or constant-sign, emit nothing.
3. If mixed-sign, run the one-miss test.
4. If the skip test fires:
   - emit the current contour in red
   - increment the skip counters on the shorter-time sign side
   - retire the square permanently
   - do not emit any white contour for that square at this iterate
5. If the skip test does not fire:
   - emit the contour in white

Once a square is retired, all later nominal iterates skip it entirely.

## Output Artifacts

The run produces:

- per-column TSV files
- a merged TSV
- iterate statistics TSV
- the final retired-square overlay PNG
- an upload JSON/stderr pair from the TGLFS upload step

## Automation

The run script:

- launches the full recompute
- waits for all columns to finish
- builds the final plot automatically
- uploads the PNG to TGLFS with no encryption password

That flow is implemented by:

- [run_grid10000_branch16_absxskip16_plot8_retired_overlay_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/run_grid10000_branch16_absxskip16_plot8_retired_overlay_upload.sh)
