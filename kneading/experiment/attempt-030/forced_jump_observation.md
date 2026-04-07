# Forced Jump Observation

This note records the observed behavior from the exploratory `attempt-030`
rerun with the following policy:

- carry-forward predictor between correction batches
- x-only Newton corrector every 5 alpha values
- on the decreasing-alpha side only, force-accept the first would-stop jump
- after that one forced acceptance, return to the normal reject-on-bad-correction
  rule

The run was intentionally stopped after the qualitative behavior became clear,
so this note is not a full-sweep TSV replacement.

## Key observation

The single forced acceptance did re-lock the continuation onto a new smooth
decreasing-alpha branch.

- Last accepted point before the old stuck region:
  - `alpha = 0.228314157079`
  - `x = 1.910158281996`
  - `s = 3.648704662277`
- First would-stop jump that was force-accepted:
  - `alpha = 0.221660830415`
  - predictor `x = 1.910158281996`
  - corrected `x = 1.832361943546`
  - corrected `s = 3.357550292154`
  - `|Δs| = 2.912e-01`
  - `||Δstate|| = 7.780e-02`

## What happened afterward

After that single forced jump, the decreasing-alpha continuation resumed with
small accepted corrections rather than repeated large-jump rejections.

Examples from the log:

- `alpha = 0.181390695348`
  - `x = 1.715713278717`
  - `s = 2.943672054765`
  - `|Δs| = 4.227e-03`
  - `||Δstate|| = 1.231e-03`
- `alpha = 0.176488244122`
  - `x = 1.697976646675`
  - `s = 2.883124692652`
  - `|Δs| = 4.411e-03`
  - `||Δstate|| = 1.298e-03`

There was also a later larger but still accepted correction:

- `alpha = 0.149874937469`
  - `x = 1.579491298055`
  - `s = 2.494792760632`
  - `|Δs| = 3.147e-02`
  - `||Δstate|| = 9.932e-03`

Below that region, the run eventually fell back into repeated
`predictor_only_after_corrector_failure` rows while holding the last corrected
state fixed.

At the time the exploratory rerun was stopped, the last corrected carried state
in that later region was:

- `x = 1.579491298055`
- `s = 2.494792760632`

and predictor-only failures were already appearing by about
`alpha = 0.123261630815`.
