# attempt-025

`attempt-025` is reserved for the next plotting algorithm.

This attempt is intended to recompute the needed `|x|`-max dataset from
scratch, using the same event-generation mechanics that were correct in
`attempt-024`, but with the missing `|x|`-max return times stored explicitly so
the new square-local skip-adjusted contour algorithm can use them.

The primary implementation spec is:

- [absx_skip_contour_algorithm.md](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-025/absx_skip_contour_algorithm.md)

The executable entrypoint is:

- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-025/contours.jl)

The Markdown spec remains the source of truth for what `attempt-025` should
compute and how it should plot.
