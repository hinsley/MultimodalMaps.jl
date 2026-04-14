# attempt-046

`attempt-046` builds a self-contained HTML explorer for the saved
`attempt-027` Shimizu-Morioka `|x|`-maximum sweep. It uses the derived
monotone sign sequence instead of the raw per-iterate tangent sign, and it
classifies each mixed contour in `2:8` as black, red, blue, or green from the
two representative monotone sign sequences on either side of the contour.

## Goal

Provide an interactive artifact where you can:

- see the overlaid nominal-iterate `2:8` contour plot directly in the browser
- toggle individual nominal iterates on and off with checkboxes
- hide or show black contours globally with a single button
- hide or show green contours globally with a single button
- hide or show red contours globally with a single button
- hide or show blue contours globally with a single button
- inspect exact sampled grid points by hover and click
- recover the exact `(alpha, lambda)` value of the nearest sampled point
- see the per-point sign sequence and return times for iterates `2:8` in both
  the hover table and the selected-point table
- see both the raw dot-product sign sequence and the monotone sign sequence in
  those tables
- see that the retained skip column stays empty in this no-old-skip variant
- highlight the four marched squares surrounding the selected sampled point
- see monotone signs, where the sign at iterate `k` is `+` if the raw
  dot-product sign stayed the same from iterate `k-1` to `k`, and `-` if it
  flipped

## Rendering model

- old-style skip compression is disabled
- every mixed square is evaluated independently at every nominal iterate in
  `2:8`
- the contour scalar magnitudes stay equal to the saved `|sign(x) * v_x|`
  magnitudes from `attempt-027`
- the contoured monotone sign at iterate `k` is `+` when the raw dot-product
  sign stays the same from iterate `k-1` to `k`, and `-` when it flips
- iterate `2` uses raw iterate `1` as the reference for that monotone sign
- the two representative sides for each mixed square are still chosen using the
  shorter-return-time convention from the original marching-square logic, but
  no skips are applied to the iterate index
- blue means grazing: deleting one monotone sign in `2:8` on either side
  makes the remaining `2:10` sequences match, with suffix inversion applied
  after deleting a `-`
- red means coordinate singularity: exactly one consecutive same-sign pair
  flips sign and the rest of the `2:10` sequences match
- black means a real contour: the two monotone sign sequences differ in
  exactly one place over `2:10`
- green means the square is mixed in `2:8` but does not satisfy any of the
  black/red/blue tests
- the explorer uses the same saved `attempt-027` tangent magnitudes and return
  times, but swaps in those monotone signs before any contouring
- the classification tests are intentionally truncated to iterates `2:10`
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
