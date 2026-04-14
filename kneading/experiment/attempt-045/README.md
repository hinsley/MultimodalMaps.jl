# attempt-045

`attempt-045` builds a self-contained HTML explorer for the saved
`attempt-027` Shimizu-Morioka `|x|`-maximum sweep, but with the
forced-first-skip contour rule from `attempt-044`.

## Goal

Provide an interactive artifact where you can:

- see the overlaid nominal-iterate `2:8` contour plot directly in the browser
- toggle individual nominal iterates on and off with checkboxes
- hide or show red contours globally with a single button
- inspect exact sampled grid points by hover and click
- recover the exact `(alpha, lambda)` value of the nearest sampled point
- see the per-point sign sequence and return times for iterates `2:8` in both
  the hover table and the selected-point table
- see which iterates place the selected sampled point on the shorter side of a
  forced first skip
- highlight the four marched squares surrounding the selected sampled point

## Rendering model

- if a square first contours at nominal iterate `k in 2:8`, that first contour
  forces exactly one skip on the shorter-return-time sign class
- later iterates in `2:8` are used to decide whether the square has any
  surviving later contour
- later surviving contours drawn in `2:8` are black
- if no later contour survives at all, the original earliest contour is drawn
  in red
- the explorer uses the same saved `attempt-027` contour scalar
  `sign(x) * v_x`
- no trajectories are reintegrated; everything is reconstructed from the saved
  sweep columns

## Data packaging

The generated HTML is self-contained:

- black contour segments are embedded as packed `Float32` endpoint arrays
- red contour segments are embedded the same way
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

- [build_explorer.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-045/build_explorer.jl)
- [run_build_explorer_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-045/run_build_explorer_upload.sh)
