# attempt-051 lab notes

## 2026-04-30 setup

Goal: reproduce the final `attempt-050` filtered SSCS contour workflow on a
wider parameter window and for three values of `g_h`.

Changes from `attempt-050`:

- New folder copied from the committed `attempt-050` baseline, not from local
  untracked result files.
- Removed copied `gcs_results` artifacts so `attempt-051` starts clean.
- Parameter range changed to `Delta Ca in [-60, 20]` and `Delta x in [-2, 2]`.
- Resolution remains `1000 x 1000`.
- `MAX_SEQ_LENGTH = 12` and SSCS integration `tmax = 1e5` are retained.
- `tau_y` is explicitly forced to `2e4`.
- `g_h` is explicitly swept over `0`, `1e-3`, and `1e-2`.
- The hard `T_scs[1] == +/-1` validation was removed from the scan path. The
  plotted red contours already ignore T symbol 1, and this avoids discarding
  otherwise valid Gamma data in the widened parameter window.
- The filtered final replay keeps blue symbol 1 and only ignores/filter-removes
  red symbol 1 behavior.

Validation before cloud launch:

- Check shell syntax with `bash -n`.
- Include `main.jl` locally and assert:
  - `length(DELTA_XS_010) == 1000`
  - `first(DELTA_XS_010) == -2.0`
  - `last(DELTA_XS_010) == 2.0`
  - `length(DELTA_CAS_010) == 1000`
  - `first(DELTA_CAS_010) == -60.0`
  - `last(DELTA_CAS_010) == 20.0`
  - `build_params(...)[4]` matches `ATTEMPT051_G_H`
  - `build_params(...)[15] == 2e4`
  - `ATTEMPT51_SSCS_ODE_TSPAN == (0, 1e5)`

Launch plan:

- Push the code first, then start the VM.
- Pull latest code on the VM.
- Verify a write to `gs://carter-kneading-attempt048/attempt-051`.
- Start the runner in `tmux` session `attempt051`.
- Monitor via column checkpoint counts and the last five log lines per `g_h`
  label.
- The runner chooses `julia +release` when available, because the local default
  `julia` channel was older than the manifest used by the recent cloud runs.
