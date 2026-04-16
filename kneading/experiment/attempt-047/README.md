# attempt-047

`attempt-047` is the `3000 x 3000` recompute-and-explorer rerun of the
`attempt-046` monotone-sign HTML workflow.

Relative to `attempt-046`, it changes three things:

- it recomputes the full Shimizu-Morioka sweep locally in `attempt-047`
  instead of reusing the saved `attempt-027` columns
- it uses a `3000 x 3000` parameter grid
- the HTML explorer keeps only symbolic grazing detection; the return-time
  grazing toggle is removed

## Pipeline

The intended full run is:

1. recompute all column TSV files for the `3000 x 3000` sweep
2. verify that every column file is complete
3. rebuild the monotone-sign explorer from those saved columns
4. upload the final HTML artifact to TGLFS

The explorer still:

- contours nominal iterates `2:8`
- stores up to `16` `|x|`-maximum events per parameter point
- uses the monotone sign derived from the raw `sign(x) * v_x` event scalar
- classifies contours into black, red, blue, purple, or green
- uses only symbolic deletion rules for grazing detection

## Entrypoints

- [main.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-047/main.jl)
- [contours.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-047/contours.jl)
- [run_columns_only.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-047/run_columns_only.jl)
- [build_explorer.jl](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-047/build_explorer.jl)
- [attempt047_algorithm.md](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-047/attempt047_algorithm.md)
- [run_full_recompute_explorer_upload.sh](/home/guest_coder/github/repos/hinsley/MultimodalMaps.jl/kneading/experiment/attempt-047/run_full_recompute_explorer_upload.sh)

## Practical notes

- `run_columns_only.jl` exists so the expensive flow integrations do not also
  pay for an unnecessary PNG contour pass
- `build_explorer.jl` is self-contained and reads only the saved local column
  files from `attempt-047`
- the full launcher disables merged TSV and old iterate-stats writes during the
  recompute stage to keep the overnight run narrower and lower-risk
