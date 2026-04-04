# attempt-027

`attempt-027` is the high-resolution recompute of the retired-square `|x|`-max
plotting algorithm from `attempt-026`.

It differs from earlier attempts in four main ways:

- it recomputes the full `|x|`-max sweep from scratch rather than reusing older
  saved sweep data
- it stores up to 16 `|x|`-max events per parameter point, including the refined
  cumulative hit times in the TSV output
- it renders the retired-square black/white/red overlay plot automatically at
  the end of the sweep
- it uses the larger `10000 x 10000` parameter grid together with a `10000 x
  10000` figure and `px_per_unit = 4`

Primary documentation:

- [attempt027_algorithm.md](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/attempt027_algorithm.md)

Executable entrypoints:

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/main.jl)
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/contours.jl)
- [run_grid10000_branch16_absxskip16_plot8_retired_overlay_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-027/run_grid10000_branch16_absxskip16_plot8_retired_overlay_upload.sh)
