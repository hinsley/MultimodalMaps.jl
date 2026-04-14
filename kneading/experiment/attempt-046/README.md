# attempt-046

`attempt-046` builds a self-contained HTML explorer for the saved
`attempt-027` Shimizu-Morioka `|x|`-maximum sweep. It uses cumulative sign
parity instead of the raw per-iterate tangent sign, and it classifies each
earliest mixed contour in `2:8` as black, red, blue, or green from the two
representative cumulative sign sequences on either side of the contour.

## Goal

Provide an interactive artifact where you can:

- see the overlaid nominal-iterate `2:8` contour plot directly in the browser
- toggle individual nominal iterates on and off with checkboxes
- hide or show red contours globally with a single button
- hide or show blue contours globally with a single button
- inspect exact sampled grid points by hover and click
- recover the exact `(alpha, lambda)` value of the nearest sampled point
- see the per-point sign sequence and return times for iterates `2:8` in both
  the hover table and the selected-point table
- see which iterates place the selected sampled point on the shorter side of a
  forced first skip
- highlight the four marched squares surrounding the selected sampled point
- see cumulative signs, where the sign at iterate `k` is `(-1)^N` and `N`
  counts negative raw tangent symbols seen through iterate `k`

## Rendering model

- if a square first contours at nominal iterate `k in 2:8`, that first contour
  is the only contour drawn for that square
- the contour scalar magnitudes stay equal to the saved `|sign(x) * v_x|`
  magnitudes from `attempt-027`, but each iterate sign is replaced by the
  cumulative parity sign `(-1)^N`
- the two representative sides are still chosen from the first mixed square
  using the shorter-return-time convention from the original skip logic
- blue means grazing: deleting one cumulative sign in `2:8` on either side
  makes the remaining `2:16` sequences match, with suffix inversion applied
  after deleting a `-`
- red means coordinate singularity: exactly one consecutive same-sign pair
  flips sign and the rest of the `2:16` sequences match
- black means a real contour: the two cumulative sign sequences differ in
  exactly one place over `2:16`
- green means the square is mixed in `2:8` but does not satisfy any of the
  black/red/blue tests
- the explorer uses the same saved `attempt-027` tangent magnitudes and return
  times, but swaps in cumulative signs before any contouring or table display
- because the saved dataset stops at iterate `16`, a one-symbol grazing test
  compares the longest aligned suffix available after the deletion and leaves
  one unmatched terminal symbol on the undeleted side
- no trajectories are reintegrated; everything is reconstructed from the saved
  sweep columns

## Data packaging

The generated HTML is self-contained:

- black contour segments are embedded as packed `Float32` endpoint arrays
- red contour segments are embedded the same way
- blue contour segments are embedded the same way
- green contour segments are embedded the same way
- sampled-point sign sequences for iterates `2:8` are embedded as packed
  `UInt16` words, with 2 bits per iterate
- sampled-point skip flags for iterates `2:8` are embedded as packed `UInt8`
  bitmasks
- per-point per-iterate return times for iterates `2:8` are embedded as packed
  `UInt16` words after quantization, then gzip-compressed inside the HTML

The full per-point phase-space state payload from the original `3.7 GB`
results TSV is intentionally not embedded. Only the compact fields needed for
interactive inspection are included so the browser artifact stays usable.

The browser decodes those arrays locally and renders the explorer on an HTML
canvas with zoom, pan, hover, and click interaction.

## Entrypoints

- [build_explorer.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-046/build_explorer.jl)
- [run_build_explorer_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-046/run_build_explorer_upload.sh)
