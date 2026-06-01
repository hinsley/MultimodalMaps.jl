# Rössler y-Minima Critical-Orbit Scan, 256x256

This run uses the corrected critical-orbit protocol:

- Parameters: `c in [2, 7]`, `a in [0.30, 0.55]`, `b = 0.3`.
- Grid: `256 x 256`.
- Critical-point seed: saddle-focus seeded continuation with `critical_event_index = 4`.
- Orbit initial condition: the continued y-minimum critical point itself.
- Transient y-minima discarded from the critical orbit: `0`.
- Symbol count: first `8` y-minimum tangent symbols, with the initial critical event included as symbol 1.
- Max orbit time: `2000`.

Verification summary from `run.log`:

- TSV rows: `65536`.
- Critical solves: `65536 ok`.
- Complete 8-symbol words: `48420`.
- Incomplete orbits within the time/state limit: `17116`.
- Max adjacent critical-y jump: `0.00421347633`.
- Max absolute critical residual in the TSV: below `1e-6`.

Artifacts:

- `coarse_scan.tsv`: full scan data, including symbol prefixes and critical-point diagnostics.
- `heatmaps/coarse_scan_8bit_word_heatmap.png`: 8-bit word heatmap for complete rows.
- `heatmaps/coarse_scan_8bit_word_heatmap_probe.html`: standalone interactive word probe.
- `heatmaps/coarse_scan_7bit_monotone_heatmap.png`: 7-bit monotone-sign heatmap for complete rows.
- `heatmaps/coarse_scan_7bit_monotone_heatmap_probe.html`: standalone interactive monotone-sign probe.
- `run_256_pipeline.sh`: exact scan, verify, render pipeline.
