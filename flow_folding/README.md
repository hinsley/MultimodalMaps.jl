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
orientation sequence. The committed scan uses a 128x128 grid, records 8
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
  contour artifacts by default.
- `examples/rossler_y_minima_tangent_contours.jl` regenerates symbol, prefix,
  and full-word contour SVGs from an existing scan TSV.
- `examples/rossler_seeded_continuation.jl` shows seeded critical-point
  location and continuation along `c`.
- `docs/index.html` is a static browser-readable guide and scan viewer.
- `results/rossler_y_minima_tangent_scan/coarse_scan.tsv` is the committed
  128x128 Rössler scan used by the browser docs.
- `results/rossler_y_minima_tangent_scan/coarse_scan_runtime.tsv` logs scan,
  contour, write, and total generation timings for the committed artifacts.
- `results/rossler_y_minima_tangent_scan/contours/` contains the generated
  Marching-Squares SVG contours, scan summary, and word legend. The SVGs omit
  max-time-limited gray point markers.

## Usage

```bash
julia --project=. flow_folding/examples/rossler_y_minima_tangent_scan.jl
```

The scan defaults to a 128x128 grid, 8 tangent symbols after 20 transient
`y`-minima, and `MM_FLOW_FOLDING_MAX_TIME=450`. A denser run with the same
8-symbol convention:

```bash
MM_FLOW_FOLDING_NC=192 \
MM_FLOW_FOLDING_NA=192 \
MM_FLOW_FOLDING_WORD_LENGTH=8 \
MM_FLOW_FOLDING_TRANSIENT_EVENTS=80 \
MM_FLOW_FOLDING_MAX_TIME=1200 \
julia --project=. flow_folding/examples/rossler_y_minima_tangent_scan.jl
```

To regenerate contours from the current TSV without re-running the ODE scan:

```bash
julia --project=. flow_folding/examples/rossler_y_minima_tangent_contours.jl
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
