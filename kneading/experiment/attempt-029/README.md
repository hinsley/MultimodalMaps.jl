# Attempt 029

This attempt moves the hook-minimum continuation work out of `attempt-028` and adds a new every-step local section predictor for the alpha crawl at fixed `lambda = 0.7`, `B = 0`.

The continuation now uses two layers:

1. At **every** alpha step, a section-local `(δx, δz)` predictor from return-shift cancellation:
   - compute the next-return map linearization on the `y = 0` section,
   - include the return-time shift in the sensitivities,
   - solve the local least-squares cancellation problem for the parameter-induced return shift,
   - update the section point before any heavier correction.
2. Every `ATTEMPT028_CONT_CORRECT_EVERY` alpha values, run the existing damped Newton corrector in `x` at fixed predicted `z`.

Important policy differences from the late `attempt-028` continuation:

- the alpha crawl does **not** stop when a step looks bad;
- if the local predictor or the Newton corrector fails, the continuation records the would-stop condition and keeps going with predictor-only state carrying;
- if a corrected point would make a large discontinuous jump, the continuation records that row as a would-stop jump and still keeps going from the predictor-only state.

Primary outputs from `continue_alpha_minima.jl`:

- `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_results.tsv`
- `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_trace.tsv`
- `alpha_continuation_lambda0.7000_seedalpha0.4000_target4.4944_summary.md`

The results TSV includes, for each alpha sample, the applied local predictor step `(local_dx, local_dz)`, its damping, singular values, residual norm, whether Newton was used, and whether that alpha sample would have been considered a stop under the old jump rule.
