# Attempt 028 Alpha Continuation of `|x|`-Map Local Minimum

- Generated: `2026-04-07 00:52:39`
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

- Decreasing-alpha converged points: `50`
- Increasing-alpha converged points: `27`
- Break rows detected: `2`
- First break: direction `decreasing`, alpha `0.382391195598`, status `Damped Newton failed to find an acceptable step from s=4.487281813993368`
- Largest predictor correction: `7.324460e-03` at alpha `0.400000000000` (seed)
- Largest adjacent state jump: `1.466802e-03` at alpha `0.385542771386` (decreasing)

Outputs:

- Results TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv`
- Trace TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_trace.tsv`
- Summary: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_summary.md`
