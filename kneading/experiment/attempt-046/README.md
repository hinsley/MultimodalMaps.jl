# attempt-046

`attempt-046` builds a self-contained HTML explorer for the saved
`attempt-027` Shimizu-Morioka `|x|`-maximum sweep. It uses the derived
monotone sign sequence instead of the raw per-iterate tangent sign, and it
classifies each mixed contour in `2:8` as black, red, blue, purple, or green
from the two representative monotone sign sequences on either side of the
contour.

## Goal

Provide an interactive artifact where you can:

- see the overlaid nominal-iterate `2:8` contour plot directly in the browser
- toggle individual nominal iterates on and off with checkboxes
- hide or show black contours globally with a single button
- hide or show green contours globally with a single button
- hide or show red contours globally with a single button
- hide or show blue contours globally with a single button
- hide or show purple contours globally with a single button
- switch the grazing detector between symbolic deletion rules and the old
  return-time skip rule
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
- every contour is classified from the monotone-sign suffix starting at its
  own nominal iterate `k`
- the contour scalar magnitudes stay equal to the saved `|sign(x) * v_x|`
  magnitudes from `attempt-027`
- the contoured monotone sign at iterate `k` is `+` when the raw dot-product
  sign stays the same from iterate `k-1` to `k`, and `-` when it flips
- iterate `2` uses raw iterate `1` as the reference for that monotone sign
- the two representative sides for each mixed square are still chosen using the
  shorter-return-time convention from the original marching-square logic, but
  no skips are applied to the iterate index
- in symbolic grazing mode, blue means grazing by `+` deletion: deleting one
  `+` monotone sign in `k:8` on either side makes the remaining suffixes
  through `2:12` match
- in symbolic grazing mode, purple means grazing by `-` deletion: deleting one
  `-` monotone sign in `k:8` on either side and then inverting the suffix
  makes the remaining suffixes through `2:12` match
- in return-time grazing mode, blue means the old return-time skip condition
  would fire at that square and iterate; purple is unused in that mode
- red means coordinate singularity: the red test only checks the local window
  `k:k+2`
- after a red contour is drawn at iterate `k`, only same-edge black follow-up
  segments at iterate `k+1` are suppressed, so unrelated next-iterate contours
  in the same square are still allowed through
- black means a real contour: the black test only checks the local window
  `k:k+1`
- green means the square is mixed in `2:8` but does not satisfy any of the
  black/red/blue/purple tests
- the explorer uses the same saved `attempt-027` tangent magnitudes and return
  times, but swaps in those monotone signs before any contouring
- the classification tests are intentionally truncated to iterates `2:12`
- no trajectories are reintegrated; everything is reconstructed from the saved
  sweep columns

## Data packaging

The generated HTML is self-contained:

- black contour segments are embedded as packed `Float32` endpoint arrays
- red contour segments are embedded the same way
- blue contour segments are embedded the same way
- purple contour segments are embedded the same way
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
