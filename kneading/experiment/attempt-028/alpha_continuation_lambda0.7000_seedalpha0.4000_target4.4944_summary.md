# Attempt 028 Alpha Continuation of `|x|`-Map Local Minimum

- Generated: `2026-04-07 01:03:31`
- Fixed `lambda = 0.700000`
- Fixed `B = 0.0`
- Alpha grid matches attempt-027: `range(0.0, 0.7, length=2000)`
- Exact seed alpha: `0.400000`
- Seed target minimum guess near `s = 4.4944`
- Break thresholds: `|Δs| > 0.050000` or `||Δstate|| > 0.050000`

## Seed

- Seed discrete-minimum branch index: `421`
- Seed refined `s* = 4.492776434911`
- Seed initial condition `(x,y,z) = (2.119617049118, 0.000000000000, 1.729688686440)`

## Continuation Outcome

- Decreasing-alpha converged points: `84`
- Increasing-alpha converged points: `28`
- Predictor-only fallback points: `1`
- Break rows detected: `2`
- First break: direction `decreasing`, alpha `0.370135067534`, status `corrector_failed: State blew up during event solve | predictor_failed: State blew up during event solve`
- Largest predictor correction: `7.505185e-02` at alpha `0.409704852426` (increasing)
- Largest adjacent state jump: `1.355393e+00` at alpha `0.409704852426` (increasing)

Outputs:

- Results TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv`
- Trace TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_trace.tsv`
- Summary: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_summary.md`
