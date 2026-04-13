# attempt-043

`attempt-043` builds a self-contained HTML explorer for the saved
`attempt-027` Shimizu-Morioka `|x|`-maximum sweep.

## Goal

Provide an interactive artifact where you can:

- see the overlaid nominal-iterate `2:8` contour plot directly in the browser
- toggle individual nominal iterates on and off with checkboxes
- inspect exact sampled grid points by hover and click
- recover the exact `(alpha, lambda)` value of the nearest sampled point
- see the per-point sign sequence of the saved dot-product scalar for iterates
  `2:8` in a selected-point table
- see which iterates incremented the selected sampled point via skip logic
- highlight the four marched squares surrounding the selected sampled point

## Rendering model

- accepted retired-square contour segments are drawn in black
- skip-trigger contour segments are drawn in red
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

The full per-point iterate state/time payload from the original `3.7 GB`
results TSV is intentionally not embedded, to keep the explorer artifact
lightweight enough to use interactively.

The browser decodes those arrays locally and renders the explorer on an HTML
canvas with zoom, pan, hover, and click interaction.

## Entrypoints

- [build_explorer.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-043/build_explorer.jl)
- [run_build_explorer_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-043/run_build_explorer_upload.sh)
