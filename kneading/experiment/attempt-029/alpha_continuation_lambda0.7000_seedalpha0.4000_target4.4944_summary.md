# Attempt 029 Alpha Continuation of `|x|`-Map Local Minimum

- Generated: `2026-04-07 14:40:56`
- Fixed `lambda = 0.700000`
- Fixed `B = 0.0`
- Alpha grid matches attempt-027: `range(0.0, 0.7, length=2000)`
- Exact seed alpha: `0.400000`
- Seed target minimum guess near `s = 4.4944`
- Every alpha step applies a section-local `(δx,δz)` predictor from return-shift cancellation using forward sensitivities and event-time correction
- The heavier corrector still uses `x` as the Newton variable and keeps `z` fixed during each Newton solve
- Corrector cadence: every `5` alpha values
- On cadence-skipped alpha values, the locally corrected section point is kept without Newton refinement
- Break thresholds: `|Δs| > 0.050000` or `||Δstate|| > 0.050000`

## Seed

- Seed discrete-minimum branch index: `421`
- Seed refined `s* = 4.458572390102`
- Seed initial condition `(x,y,z) = (2.111533184703, 0.000000000000, 1.728468286846)`

## Continuation Outcome

- Decreasing-alpha converged points: `9`
- Increasing-alpha converged points: `171`
- Predictor-only points: `1820`
- Would-stop rows detected: `27`
- First would-stop row: direction `decreasing`, alpha `0.382741370685`, status `would_stop_jump_break_then_predictor_only`
- Largest predictor correction: `2.687959e-02` at alpha `0.400000000000` (seed)
- Largest adjacent state jump: `2.906832e-02` at alpha `0.384492246123` (decreasing)

Outputs:

- Results TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv`
- Trace TSV: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_trace.tsv`
- Summary: `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_summary.md`
