# attempt-042

`attempt-042` reuses the saved `2000 x 2000` Shimizu-Morioka sweep from
[attempt-027](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/README.md)
and only reruns the plotting stage.

## What changed

- nominal iterates `1:8` are still processed in order so the same skip-index
  adjustments are accumulated before the visible iterate
- nothing is drawn for nominal iterates `1:7`
- only nominal iterate `8` is rendered
- contour detection uses the original `sign(x) * v_x` scalar from `attempt-027`
- if a square had any skip increment on an earlier processed iterate, then an
  iterate-8 contour in that square is drawn in red instead of white
- current iterate-8 skip detections are still drawn in red using the same
  marching-square interpolation rule as before

## Data source

The plotting script reads the existing columns directory from `attempt-027`
through `ATTEMPT025_SWEEP_DIR`. No trajectories are reintegrated here.

## Entrypoints

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-042/main.jl)
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-042/contours.jl)
- [run_grid2000_branch16_absxskip16_plot8_nominal8_overlay_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-042/run_grid2000_branch16_absxskip16_plot8_nominal8_overlay_upload.sh)
