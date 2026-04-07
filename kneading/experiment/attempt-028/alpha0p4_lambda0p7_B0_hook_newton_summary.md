# Attempt 028 Hook Newton Test

Fixed parameters:

- `alpha = 0.400000`
- `lambda = 0.700000`
- `B = 0.0`

Method summary:

- Collected a long `|x|`-maxima orbit from one unstable-manifold branch of the origin.
- Kept positive-branch maxima and filtered to the `next_x < 0` hook subbranch before building the sampled map `s_n = x_n^2`, `F(s_n) = x_{n+1}^2`.
- Chose an initial hook guess from the sampled branch near the first discrete local minimum of `F` along the filtered subbranch.
- Parameterized the local `y = 0` section curve as `(x, y, z) = (sqrt(s), 0, z(s))` via an exact natural cubic spline through the sampled subbranch.
- Used event-defined derivatives with return-time shift for the next `|x|`-maximum.
- Cross-checked the first derivative against `SciMLSensitivity.ODEForwardSensitivityProblem`.

Sample sizes:

- Collected `|x|`-maxima: `20000`
- Filtered `next_x < 0` subbranch points after transient cut: `433`
- Filtered subbranch `s` range: `[2.127337953, 4.692711886]`
- Filtered subbranch `F(s)` range: `[1.174267034, 4.790498337]`

Newton result:

- Target extremum type: `minimum`
- Initial guess `s0 = 2.138971949060`
- Final `s* = 2.139161071638`
- Final `F(s*) = 4.432963662379`
- Final `F'(s*) = -8.515922818867e-13`
- Final `F''(s*) = 4.754946725189e+03`
- Final next-event time `T(s*) = 12.182926751265`
- Final SciMLSensitivity first-derivative mismatch `|F'_manual - F'_SciML| = 5.934568294780e-15`

Quadratic-convergence diagnostic:

- Ratios `|F'_{k+1}| / |F'_k|^2`: `8.402863e-03, 8.429344e-03, 5.967947e+00`

Outputs:

- Sampled maxima TSV: `alpha0p4_lambda0p7_B0_absxmax_events.tsv`
- Positive branch TSV: `alpha0p4_lambda0p7_B0_positive_absx_nextneg_branch.tsv`
- Newton trace TSV: `alpha0p4_lambda0p7_B0_hook_newton_trace.tsv`
