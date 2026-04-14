# attempt-044

`attempt-044` reuses the saved `attempt-027` Shimizu-Morioka sweep columns and
produces a static contour image rather than an interactive explorer.

## Rule change

This experiment works only over nominal iterates `2:8`, matching the earlier
overlay plots.

For each marched square independently:

1. Find the earliest nominal iterate in `2:8` whose current contour scalar
   changes sign across the square.
2. At that earliest iterate, force exactly one skip on the
   shorter-return-time sign class, using the same shorter-side test as the old
   red-slip logic.
3. Freeze that one skip pattern for the square forever.
4. Re-evaluate only later nominal iterates with that fixed shifted indexing.

Coloring is then determined by the outcome:

- if at least one later contour survives under the forced skip, those later
  contours are drawn in black and the original earliest contour is omitted
- if no later contour survives, the original earliest contour is drawn in red

This is intentionally different from the earlier skip logic:

- there is no `err_skip < err_noskip` decision
- there are no repeated skip increments
- the shorter side is identified once and only once, at the first contouring
  iterate for that square

## Inputs and outputs

- input columns:
  [attempt-027 saved sweep](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/grid2000_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_retired_shimizu_morioka_cpu_columns)
- main script:
  [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-044/contours.jl)
- runner:
  [run_grid2000_branch16_absxskip16_plot8_forcedfirstskip_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-044/run_grid2000_branch16_absxskip16_plot8_forcedfirstskip_upload.sh)
