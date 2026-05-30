# flow_folding

`flow_folding/` detects and continues critical points in one-dimensional
reductions of Poincare return maps for ODE flows, and records tangent-based
kneading words from extremum sections.

The target workflow is: take an ODE system, choose a state variable, collect
successive minima or maxima of that variable on the attractor, use those event
values as a one-dimensional return coordinate, and identify fold/critical
points of the induced return map.

The Rössler examples in this folder use `y`-minima and tangent signs at those
minima. They intentionally do not use the z-maximum threshold convention from
the Malykh-Shilnikov paper, because the kneading scans here encode the tangent
orientation sequence. The production scan uses a 1024x1024 grid, records 8
accepted `y`-minima events per completed grid point, and reports `max_time` for
points that do not finish within the configured integration horizon.

## Event Convention

For an ODE

```text
du/dt = f(u, p, t)
```

choose a state variable `x = u[i]`. The extremum event function is

```text
h(u, p, t) = f(u, p, t)[i]
```

An accepted maximum of `x` satisfies:

```text
h = 0
d/dt h = Dh(u, p, t) * f(u, p, t) < 0
```

An accepted minimum of `x` satisfies:

```text
h = 0
d/dt h = Dh(u, p, t) * f(u, p, t) > 0
```

Do not add coordinate-sign filters by default. The maxima/minima constraint is
the section definition unless a specific diagnostic explicitly asks for a
restricted section.

## Tangent Kneading Convention

For tangent kneading words, integrate the variational equation alongside the
state, project the tangent transverse to the flow direction, re-normalize it,
and record the sign of the chosen tangent component at each accepted extremum.

For the Rössler scan:

- event variable: `y`
- accepted event: `y`-minimum
- observable tangent component: `dy`
- symbol `1`: positive tangent `y` component
- symbol `0`: negative tangent `y` component

## Criticality Test

After collecting accepted extrema, define the scalar return-map coordinates
from the selected state variable:

```text
X_M     = x(q_M)
X_{M+1} = x(q_{M+1})
```

where `q_M` and `q_{M+1}` are successive accepted extrema. The critical-point
condition should be solved as the seed-family residual:

```text
g = (partial_rho X_{M+1}) / (partial_rho X_M) = 0
```

with a safely nonzero denominator. Derivatives at event times should use the
standard event-time correction before reading the selected state component.
Finite differences are acceptable only for quick prototype checks.

## Current Files

- `FlowFolding.jl` contains the core event, tangent, return-map, seed-ray, and
  continuation functions.
- `examples/rossler_common.jl` defines the Malykh-Shilnikov Rössler variant and
  the `y`-minima problem adapter.
- `examples/rossler_y_minima_tangent_scan.jl` runs a coarse tangent-kneading
  scan over `2 <= c <= 7`, `0.30 <= a <= 0.55`, `b=0.3`; it also generates
  PNG contour artifacts by default.
- `examples/rossler_y_minima_tangent_pngs.py` regenerates symbol, prefix, and
  full-word contour PNGs from an existing scan TSV or TSV.GZ without plotting
  dependencies.
- `examples/verify_rossler_y_minima_tangent_outputs.py` verifies the final
  TSV, heatmap PNGs, legends, and standalone probe HTML files for production
  Rössler scan outputs.
- `examples/monitor_rossler_y_minima_tangent_pipeline.py` reports progress and
  optional live TSV validation for chunked production scan runs.
- `examples/rossler_y_minima_tangent_contours.jl` is the legacy SVG contour
  exporter.
- `examples/rossler_seeded_continuation.jl` shows seeded critical-point
  location and continuation along `c`.
- `docs/index.html` is a static browser-readable guide and scan viewer.
- `results/rossler_y_minima_tangent_scan/coarse_scan.tsv` is the committed
  128x128 Rössler scan used by the browser heatmap docs.
- `results/rossler_y_minima_tangent_scan/coarse_scan_runtime.tsv` logs scan,
  contour, write, and total generation timings for the committed artifacts.
- `results/rossler_y_minima_tangent_scan/contours_png_128/` contains PNGs
  rendered from the last 128x128 scan.
- `results/rossler_y_minima_tangent_scan_1024/coarse_scan.tsv.gz` is the
  compressed 1024x1024 Rössler scan.
- `results/rossler_y_minima_tangent_scan_1024/contours/` contains the generated
  Marching-Squares PNG contours, scan summary, and word legend. The PNGs omit
  max-time-limited gray point markers.
- `results/rossler_y_minima_tangent_scan_4096/run_4096_pipeline.sh` runs the
  chunked 4096x4096 y-minima tangent scan, merges the full-symbol TSV, and
  renders the 8-bit word and 7-bit monotone-sign heatmap PNG/HTML outputs.

## Usage

```bash
julia --project=. flow_folding/examples/rossler_y_minima_tangent_scan.jl
```

The scan defaults to a 1024x1024 grid, 8 tangent symbols after 20 transient
`y`-minima, `MM_FLOW_FOLDING_MAX_TIME=450`, and PNG contour export. Large runs
skip browser JS data by default to avoid writing a multi-hundred-megabyte docs
bundle. A compact browser-data run:

```bash
MM_FLOW_FOLDING_NC=128 \
MM_FLOW_FOLDING_NA=128 \
MM_FLOW_FOLDING_WORD_LENGTH=8 \
MM_FLOW_FOLDING_WRITE_DOCS_DATA=true \
julia --project=. flow_folding/examples/rossler_y_minima_tangent_scan.jl
```

To regenerate contours from the current TSV without re-running the ODE scan:

```bash
python3 flow_folding/examples/rossler_y_minima_tangent_pngs.py \
  flow_folding/results/rossler_y_minima_tangent_scan_1024/coarse_scan.tsv.gz \
  --output-dir flow_folding/results/rossler_y_minima_tangent_scan_1024/contours \
  --stem coarse_scan \
  --clean
```

To run the chunked 4096x4096 heatmap pipeline:

```bash
./flow_folding/results/rossler_y_minima_tangent_scan_4096/run_4096_pipeline.sh
```

After it finishes, verify the full-symbol TSV and both standalone heatmap
viewers with:

```bash
python3 flow_folding/examples/verify_rossler_y_minima_tangent_outputs.py
```

While the chunked pipeline is running, monitor progress and validate present
chunk TSVs with:

```bash
python3 flow_folding/examples/monitor_rossler_y_minima_tangent_pipeline.py --validate
```

Open the local docs from the repository root with:

```bash
python3 -m http.server 8765
```

and browse to:

```text
http://localhost:8765/flow_folding/docs/
```

## Planned Buildout

1. Add higher-order interpolation for fixed-step event/tangent samples.
2. Add plotted diagnostics for sampled return maps and seeded critical branches.
3. Add stronger diagnostics for failed/short attractor scans.
4. Add production-scale parallel scan runners once the local convention is
   stable.
