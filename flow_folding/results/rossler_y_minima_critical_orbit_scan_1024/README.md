# Rössler y-minima critical-orbit scan, 1024x1024

Corrected critical-orbit kneading scan over the same Malykh/Shilnikov-style region used in the lower-resolution run:

- `c in [2.0, 7.0]`, 1024 samples.
- `a in [0.30, 0.55]`, 1024 samples.
- `b = 0.3`.
- y-minima critical event index `4`.
- critical orbit used directly as the initial condition.
- no discarded transient y-minima: `orbit_transient_events = 0`, `initial_event_included = true`.
- tangent-based 8-symbol kneading words with `max_time = 2000.0`, `dt = 0.05`.
- column-parallel continuation on `JULIA_NUM_THREADS = 10`.

## Verification

The raw TSV verifier passed before the TSV was removed locally:

- rows: `1048576`.
- status counts: `ok = 775347`, `orbit_max_time = 273229`.
- critical status counts: `ok = 1048576`.
- finite critical initial conditions: `1048576`.
- ok fraction: `0.739429`.
- max adjacent critical-y jump: `0.0010512984`.

The Julia scan phase took `6827.17` seconds wall time.

The local raw TSV is intentionally absent and ignored by `.gitignore`. It is stored in `tglfs`:

- name: `coarse_scan.tsv`.
- UFID: `af14044569ff1161546983808cfd0c7c5bb0cde92cc682ecbd939efd99c7e22b`.
- size: `382081526` bytes.
- inspect artifact: `coarse_scan_tglfs_inspect.json`.

## Artifacts

The heatmap artifacts are in `heatmaps/`:

- `coarse_scan_8bit_word_heatmap.png`.
- `coarse_scan_8bit_word_heatmap_probe.html`.
- `coarse_scan_7bit_monotone_heatmap.png`.
- `coarse_scan_7bit_monotone_heatmap_probe.html`.

Both PNGs are `6400x4400`. Both standalone HTML probes include live click/drag probing for `c`, `a`, the 8-symbol word, the 7-bit monotone signs, and the selected point's critical initial condition `x`, `y`, `z`.
