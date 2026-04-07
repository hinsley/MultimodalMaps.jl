# Attempt 028 Alpha Continuation of `|x|`-Map Local Minimum

- Generated: `2026-04-07 09:33:03`
- Fixed `lambda = 0.700000`
- Fixed `B = 0.0`
- Alpha grid matches attempt-027: `range(0.0, 0.7, length=2000)`
- Exact seed alpha: `0.400000`
- Seed target minimum guess near `s = 4.4944`
- Corrector uses `x` as the Newton variable and keeps `z` fixed during each Newton solve
- Corrector cadence: every `5` alpha values
- On cadence-skipped alpha values, the full initial condition is held unchanged from the last corrected point
- Break thresholds: `|Δs| > 0.050000` or `||Δstate|| > 0.050000`

## Seed

- Seed discrete-minimum branch index: `421`
- Seed refined `s* = 4.458572390102`
- Seed initial condition `(x,y,z) = (2.111533184703, 0.000000000000, 1.728468286846)`

## Continuation Outcome

- Decreasing-alpha converged points: `95`
- Increasing-alpha converged points: `171`
- Predictor-only fallback points: `1086`
- Break rows detected: `1`
- First break: direction `decreasing`, alpha `0.226913456728`, status `jump_break`
- Largest predictor correction: `3.095073e-01` at alpha `0.226913456728` (decreasing)
- Largest adjacent state jump: `8.207117e-02` at alpha `0.226913456728` (decreasing)

Outputs:

- Results TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv`
- Trace TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_trace.tsv`
- Summary: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_summary.md`
