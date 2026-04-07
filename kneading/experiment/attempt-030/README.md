# Attempt 030

This attempt returns to the simpler carry-forward alpha continuation for the hooked local minimum of the `|x|`-maxima return map at fixed:

- `lambda = 0.7`
- `B = 0`

The seed is still the refined minimum near `alpha = 0.4`, `s ≈ 4.4944`, but the continuation logic differs from `attempt-029`:

1. Between scheduled correction batches, the initial condition is held exactly fixed at the last successfully corrected section point `(x, 0, z)`.
2. Every `ATTEMPT028_CONT_CORRECT_EVERY` alpha values, the code backfills the whole pending alpha block since the previous flush.
3. Each pending alpha in that block gets its own x-only damped Newton correction attempt at fixed carried `z`.
4. On the decreasing-alpha side only, the first would-stop jump after the last smooth corrected block is force-accepted once as a diagnostic handoff.
5. After that single forced acceptance, the old policy resumes: any later failed or would-stop correction is recorded as predictor-only and the batch continues from the last accepted corrected point.

This makes the diagnostics more local than the old cadence-only scheme:

- skipped predictor points are revisited later,
- jumps can be localized to specific alpha values inside a 5-step block,
- the continuation never stops early just because a particular correction point fails.

Primary outputs from `continue_alpha_minima.jl`:

- `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv`
- `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_trace.tsv`
- `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_summary.md`

Exploratory note for the single forced-jump experiment:

- `forced_jump_observation.md`
