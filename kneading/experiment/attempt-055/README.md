# attempt-055

`attempt-055` reruns the corrected unit-tangent Ca-dot zero-contour diagnostic
from `attempt-054` at `500 x 500` resolution on the original
`Delta Ca in [-45, -20]`, `Delta x in [-1.5, -0.5]` window.

The event definition is the corrected low-branch calcium-minimum event:

- active `g_h = 0` state only: `(x, n, h, Ca, V)`
- passive `y` is absent from both trajectory and tangent dynamics
- Ca-minimum candidate requires `dCa/dt = 0`
- accepted event requires `V <= 0`
- accepted event also requires `x <= x_SF`, where `x_SF` is computed from the
  saddle-focus equilibrium at that parameter point

The cloud runner is:

```bash
kneading/experiment/attempt-055/run_gce_grid500_tangent_zero.sh
```

Default cloud settings:

- `JULIA_NUM_THREADS=48`
- grid `500 x 500`
- `MAX_ITER=8`
- `tmax=1e5`
- output tag `grid500_tangent_ca_dotzero_tmax1e5_iter8_ystub_sf_xfiltered`

The runner writes resumable per-column checkpoint TSVs, then writes one merged
TSV, one contour PNG, and one summary file. If `ATTEMPT055_GCS_URI` is set, it
syncs checkpoints on interruption and uploads final artifacts after completion.
