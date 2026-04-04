# attempt-026

`attempt-026` is a plotting-only variant built on top of the completed
`attempt-025` `|x|`-max sweep.

It keeps the same saved-data inputs and the same delta-time skip test as
`attempt-025`, but changes the square lifecycle rule:

- in `attempt-025`, if a square triggered a skip increment, later nominal
  iterates could still draw accepted contours in that same square
- in `attempt-026`, once a square triggers a skip increment and emits its red
  excluded contour, that square is retired permanently and never emits a later
  white accepted contour again

Primary documentation:

- [attempt026_algorithm_update.md](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-026/attempt026_algorithm_update.md)

Executable entrypoints:

- [overlay_excluded_png.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-026/overlay_excluded_png.jl)
- [run_grid1200_branch16_absxskip16_plot8_retired_overlay_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-026/run_grid1200_branch16_absxskip16_plot8_retired_overlay_upload.sh)
