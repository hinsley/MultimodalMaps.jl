# Attempt-024 Contour Status

## Completed computation

The full `attempt-024` state-augmented sweep completed on March 30, 2026.

The finished column data used for all replots in this attempt is:

- `grid1200_branch8_floworth_state_slip_shimizu_morioka_cpu_columns/`

Those column files contain, for each parameter point:

- the recorded dot-product values for `z_max` events
- the recorded dot-product values for `|x|`-max events
- the full `(x, y, z)` state at each recorded event

No new orbit integrations were needed for the tangency-based replot described below.

## Historical plots from the finished sweep

The first pair of plots produced from the completed full-state sweep used a **state-gap / branch-matching slip mask**, not a section-tangency mask:

- `grid1200_branch8_floworth_state_slip_shimizu_morioka_cpu_zmax_contours.png`
- `grid1200_branch8_floworth_state_slip_shimizu_morioka_cpu_absxmax_contours.png`

Those files remain in the directory as historical artifacts, but they are not the preferred tangency-masked outputs.

## Current preferred plots

The current preferred replot from the finished sweep uses a **tangency-threshold straddle rule** to disqualify marched squares:

- `grid1200_branch8_floworth_tangency_straddle_shimizu_morioka_cpu_zmax_contours.png`
- `grid1200_branch8_floworth_tangency_straddle_shimizu_morioka_cpu_absxmax_contours.png`

These were generated directly from the completed state-augmented columns, with no new sweep.

The earlier tangency-masked pair

- `grid1200_branch8_floworth_tangency_section_shimizu_morioka_cpu_zmax_contours.png`
- `grid1200_branch8_floworth_tangency_section_shimizu_morioka_cpu_absxmax_contours.png`

used the looser "any corner below threshold" masking rule and remain in the directory as intermediate artifacts.

## Tangency-based disqualification rule

For marching-squares contour extraction, a cell is masked only if its four corner event states **straddle** the tangency threshold for the corresponding event section:

- at least one corner has tangency score `<= eps`
- at least one other corner has tangency score `> eps`

This mask is applied only to contour extraction. It does not modify the stored sweep values.

## Event-specific transversality scores

### `z_max` event contours

The event section is the `z`-nullcline:

`phi_z(x, y, z) = z - x^2 = 0`

The transversality score used in the replot is:

`T_z = abs(grad(phi_z) dot f(x, y, z; alpha, lambda))`

with

- `grad(phi_z) = (-2x, 0, 1)`
- `f(x, y, z; alpha, lambda) = (y, x - lambda*y - x*z, -alpha*(z - x^2))`

So the implemented score is:

`T_z = abs(-2*x*y - alpha*(z - x^2))`

A marched square is masked for the `z_max` contour layer if its corner scores straddle:

`eps_z = 0.35`

### `|x|`-max event contours

The event section is treated as the `y = 0` surface, corresponding to `x`-extrema:

`phi_x(x, y, z) = y = 0`

The transversality score used in the replot is:

`T_x = abs(grad(phi_x) dot f(x, y, z; alpha, lambda))`

with

- `grad(phi_x) = (0, 1, 0)`

So the implemented score is:

`T_x = abs(x - lambda*y - x*z)`

A marched square is masked for the `|x|`-max contour layer if its corner scores straddle:

`eps_x = 0.10`

## Marching-squares interpolation

Contour vertices are still placed by **linear interpolation along cell edges** using the already computed dot-product values.

This replot does **not** use edge midpoints as contour vertices. The sub-grid contour placement comes from linear interpolation of the sampled scalar field.

## Files written by the current tangency replot

The current tangency-based replot wrote:

- `grid1200_branch8_floworth_tangency_straddle_shimizu_morioka_cpu_zmax_contours.png`
- `grid1200_branch8_floworth_tangency_straddle_shimizu_morioka_cpu_absxmax_contours.png`
- `grid1200_branch8_floworth_tangency_straddle_shimizu_morioka_cpu_iterate_colors.tsv`

Merged TSVs were intentionally not rewritten for this replot.
