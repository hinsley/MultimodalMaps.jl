# Attempt 028

This attempt is a fixed-parameter hook-critical-point refinement test for the Shimizu-Morioka `|x|`-maxima return map at:

- `alpha = 0.4`
- `lambda = 0.7`
- `B = 0`

The goal is not a sweep. It is to verify that a local critical point on the hooked part of the `|x|`-maxima return map can be:

1. estimated from a long unstable-manifold sample,
2. refined with damped Newton,
3. differentiated correctly with return-time shift included, and
4. cross-checked against `SciMLSensitivity.jl` first-order forward sensitivities.

This prototype uses the positive-`x` `|x|`-maxima section, then filters to the `next_x < 0` hook subbranch before building the Newton solve. The section curve used by Newton is an exact natural cubic spline through that sampled subbranch. The current default target is the first discrete local minimum of `F(s) = x_{n+1}^2` along that filtered branch, with the extremum type controlled by `ATTEMPT028_TARGET_EXTREMUM`.

Files written by `main.jl`:

- `alpha0p4_lambda0p7_B0_absxmax_events.tsv`
- `alpha0p4_lambda0p7_B0_positive_absx_nextneg_branch.tsv`
- `alpha0p4_lambda0p7_B0_hook_newton_trace.tsv`
- `alpha0p4_lambda0p7_B0_hook_newton_summary.md`
