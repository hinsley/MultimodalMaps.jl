# flow_folding

`flow_folding/` is a scaffold for detecting critical points in one-dimensional
reductions of Poincare return maps for ODE flows.

The target workflow is: take an ODE system, choose a state variable, collect
successive minima or maxima of that variable on the attractor, use those event
values as a one-dimensional return coordinate, and identify fold/critical
points of the induced return map.

This is intentionally not a full implementation yet.

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

## Intended Criticality Test

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

- `FlowFolding.jl` contains a minimal problem type, extremum event helpers, and
  a deliberately unimplemented `detect_critical_points` entrypoint.

## Planned Buildout

1. Add adapters for in-place and out-of-place SciML vector fields.
2. Add robust extrema collection on long attractor trajectories.
3. Add saddle-focus seeded local ray construction for systems where that
   geometry is available.
4. Add event-corrected sensitivity derivatives.
5. Add root finding and continuation for the criticality residual.
6. Add plotting and validation against sampled extremum return maps.
