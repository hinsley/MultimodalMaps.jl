# Branch Weights in Attempt-007 (Rossler)

## What the branch weights are

For the Rossler return map `f` with map critical point `c_map`, we use:

- Left branch: `[f(f(c_map)), c_map]`
- Right branch: `[c_map, f(c_map)]`

From return-map samples `(x_n, x_{n+1}, tau_n)`:

- `tau_left = mean(tau_n over x_n in left branch)`
- `tau_right = mean(tau_n over x_n in right branch)`

Then for entropy parameter `s`:

- `g_L(s) = exp(-s * tau_left)`
- `g_R(s) = exp(-s * tau_right)`

Along the critical itinerary, we assign each iterate to left/right branch and accumulate branch-constant time:

- `T_n = sum_{k=1..n} tau_branch(k)`
- `D_branch(s) = 1 + sum eps_n * exp(-s * T_n)`

The flow topological entropy estimate is the positive root of `D_branch(s)=0`.

## How it is implemented here

In `EntropiesRossler.py` and `RosslerConvergence.py`:

- `classify_left(...)` performs interval classification plus deterministic fallback for rare out-of-range values
- `tau_left`, `tau_right` are computed once from `tau_n`
- `tau_branch` is assigned along kneading iterates
- `T_branch = cumsum(tau_branch)`
- Root is solved from the weighted determinant series

In convergence mode, the root branch is tracked continuously (nearest previous root) to avoid occasional spurious near-zero roots of truncated determinants.

## Measured runtime cost

Benchmark on this machine (3 runs, same settings as attempt-007 entropy script):

- ODE/event extraction (`solve_ivp` with events over `[0,8000]`): mean `9.929759 s`
- Map aggregation (`unique`+bin means): mean `0.005959 s`
- Branch-weight calculation single pass (`tau_left`, `tau_right`): mean `0.000060 s`
- Branch-weight micro cost (`classify + means` only): `0.016365 ms` per call

Estimated branch-weight share of extraction+aggregation+branch stage:

- `0.0006%`

So branch-weight computation is negligible; the cost is dominated by ODE integration/event extraction.

## Values from benchmark run

- `tau_left = 6.167510`
- `tau_right = 5.614223`
