# Attempt 053: Local `g_h = 0` Lyapunov Dimension Scan

This attempt runs the attempt-052 Lyapunov-dimension implementation locally, but
only for `g_h = 0`.

Configuration:
- Grid: `200 x 200`
- `Delta Ca`: `[-45, -20]`
- `Delta x`: `[-1.5, -0.5]`
- `tau_y = 2e4`
- Lyapunov maximum time: `1e5`
- Early-convergence minimum time: `3e4`
- Output directory: this attempt folder

The numerical implementation is reused from `../attempt-052/main.jl`; this
attempt owns only the local run wrapper and the generated artifacts.
