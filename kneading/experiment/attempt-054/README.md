# Attempt 054: Tangent Ca-Minimum Zero Contours

This attempt replaces SSCS coding with a tangent-direction diagnostic.

For each parameter point, the scan reuses the same two initial trajectories used
by the recent kneading scans:

- `T0`, continued column-wise in decreasing `Delta x` with the attempt-050 T0
  initialization/corrector path.
- `Gamma_SD^-`, started from the displaced unstable-manifold point computed from
  the upper real saddle.

Each trajectory is augmented with one tangent vector. The tangent vector is
evolved by the variational equation using a ForwardDiff Jacobian-vector product
and is projected/renormalized against the flow after accepted solver steps and
at detected calcium minima. At each local minimum of the actual calcium
coordinate, the code records the tangent vector's calcium component,
equivalently its dot product with the `partial Ca` basis vector. Contours are
drawn as zero-level contours of that scalar field for each fixed calcium-map
iterate.

Default test settings:

- Region: `Delta Ca in [-45, -20]`, `Delta x in [-1.5, -0.5]`.
- Resolution: `200 x 200`.
- Integration horizon: `tmax = 1e5`.
- Iterates: first `8` calcium minima.
- Red contours: `T0` trajectory.
- Blue contours: `Gamma_SD^-` trajectory, with the initial tangent taken from
  the upper saddle's weakest real stable eigendirection after excluding the
  passive decoupled `y` direction that occurs at `g_h = 0`.

Run a smoke test:

```bash
bash kneading/experiment/attempt-054/run_local_10x10.sh
```

Run the requested local test:

```bash
bash kneading/experiment/attempt-054/run_local_200x200.sh
```

The scripts prefer `julia +release` because the current manifest was generated
with Julia `1.12.3`. They also run with `--startup-file=no` so local startup
packages do not affect the scan. Set `JULIA_CMD=julia` or another command if
needed.
