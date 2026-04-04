# attempt-025

`attempt-025` contains the skip-adjusted `|x|`-max contour pipeline.

It recomputes the `|x|`-max dataset from scratch using the same event-generation
mechanics as `attempt-024`, but stores the `|x|`-max hit times explicitly.
During plotting, those stored cumulative hit times are converted into interval
times before the square-local skip test is applied.

Primary documentation:

- [absx_skip_contour_algorithm.md](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-025/absx_skip_contour_algorithm.md)

Current executable entrypoints:

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-025/main.jl)
  Recompute stage: orbit scan, `|x|`-max event detection, tangent processing,
  and TSV writing.
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-025/contours.jl)
  PNG contour renderer from saved sweep data, including skip-adjusted plotting
  and increment-overlay debug artifacts.
- [nominal_iterate_gif.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-025/nominal_iterate_gif.jl)
  Per-nominal-iterate GIF renderer from saved sweep data, including optional
  excluded-contour overlays.

The Markdown spec is intended to match the current implementation, including the
post-hoc delta-time conversion used by the plot stage.
