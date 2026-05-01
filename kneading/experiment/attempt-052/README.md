# attempt-052

Lyapunov-dimension sweep for the current SiN model using the `attempt-050`
parameter window and the three `g_h` cases from `attempt-051`.

Defaults:

- `Delta Ca in [-45, -20]`
- `Delta x in [-1.5, -0.5]`
- `tau_y = 2e4`
- `g_h in {0, 1e-3, 1e-2}`
- production grid: `1000 x 1000`
- local validation grid: `10 x 10`, `g_h = 1e-3`
- Lyapunov spectrum uses `ChaosTools.jl` / `DynamicalSystems.jl` tangent
  dynamics and QR reorthonormalization.

The plotted scalar is the Kaplan-Yorke Lyapunov dimension computed from the
full six-exponent spectrum. A point integrates for up to `1e5` measurement time
after a transient. The default early-stop criterion starts at `3e4` measurement
time and accepts convergence if the spectrum changes by less than
`max(5e-4, 0.02 * spectrum_scale)` between check windows.

Color convention:

- dimension in `[0, 1)`: black ramp
- dimension in `[1, 2)`: blue ramp
- dimension in `[2, 3)`: orange ramp
- dimension in `[3, 4)`: green ramp
- dimension in `[4, 5]`: purple ramp
- failed points: light gray

Local validation:

```bash
kneading/experiment/attempt-052/run_local_10x10_gh0p001.sh
```

Production runner:

```bash
ATTEMPT052_GCS_URI=gs://carter-kneading-attempt048/attempt-052 \
  kneading/experiment/attempt-052/run_grid1000_lyapdim_gh_sweep.sh
```

Each `g_h` case writes:

- `<tag>_columns/column_XXXX.tsv`
- `<tag>_results.tsv`
- `<tag>_dimension.png`
- `<tag>_summary.txt`
- `<tag>.log`

The production runner synchronizes column checkpoints and final artifacts to
GCS when `ATTEMPT052_GCS_URI` is set.
