# Branch Weights in Attempt-006 (Lorenz)

## What the branch weights are

For the 1D return map `f` with critical point `c`, we use two branch intervals:

- Left branch: `[f(f(c)), c]`
- Right branch: `[c, f(c)]`

From the extracted return-map data `(x_n, x_{n+1}, tau_n)`, we compute:

- `tau_left = mean(tau_n over x_n in left branch)`
- `tau_right = mean(tau_n over x_n in right branch)`

These are branch-constant roof times. For entropy parameter `s`, branch weights are:

- `g_L(s) = exp(-s * tau_left)`
- `g_R(s) = exp(-s * tau_right)`

The weighted kneading time accumulation along the critical itinerary uses branch constants:

- `T_n = sum_{k=1..n} tau_branch(k)` where `tau_branch(k)` is `tau_left` or `tau_right`
- `D_flow(s) = 1 + sum eps_n * exp(-s * T_n)`

Then `h_top` is the positive root of `D_flow(s)=0`.

## How it is implemented here

In `EntropiesLorenz.py`:

- Branch interval masking is done by `branch_masks(...)`
- `tau_left`, `tau_right` are computed once from `tau_n`
- `tau_branch` is assigned along the kneading itinerary
- `T_branch = cumsum(tau_branch)`
- Flow topological entropy uses `D_flow_weighted(s)`

## Measured runtime cost

Benchmark on this machine (3 runs, same settings as attempt-006):

- ODE/event extraction (`solve_ivp` with events over `[0,1000]`): mean `2.771034 s`
- Map aggregation (`unique`+bin means): mean `0.006884 s`
- Branch-weight calculation single pass (`tau_left`, `tau_right`): mean `0.000066 s`
- Branch-weight micro cost (`mask + means` only): `0.018011 ms` per call

Estimated branch-weight share of extraction+aggregation+branch stage:

- `0.0024%`

So branch-weight calculation is effectively free compared to trajectory integration.

## Values from benchmark run

- `tau_left = 0.749171`
- `tau_right = 0.750385`
