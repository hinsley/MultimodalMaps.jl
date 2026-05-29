# Pointwise vs Branch-Weighted Roof-Time in Lorenz Kneading

For the weighted kneading determinant, the branch weights should be attached to symbols (branches), not to individual sampled points.

## Old computation (pointwise roof-time weighting)
The previous determinant used interpolated pointwise return times along the critical itinerary:

- `tau_k = tau(x_k)`
- `T_n^pw = sum_{k=1}^n tau_k`
- `D_pw(s) = 1 + sum_{n>=1} eps_n * exp(-s * T_n^pw)`

Here, `tau(x)` varies continuously inside each branch.

## Why this is inconsistent with branch-weighted kneading
In a branch-weighted setup, each branch has one weight function (`g_L(s)`, `g_R(s)`), so the contribution of a word depends only on its symbol sequence. The pointwise version violates this by letting two orbits with the same symbolic prefix receive different cumulative weights because they visit different `x` within the same branch.

That mixes symbolic dynamics with intra-branch geometry and no longer matches the intended branch-level weighted system.

## Why it inflates `h_top` (~1.13)
Empirically, the pointwise construction yields an effectively smaller roof time along the kneading sums than the branch-averaged model, so terms in `D_pw(s)` decay too slowly with `s`. The zero of the determinant is then pushed to larger `s`, producing an inflated entropy estimate (`h_top ~ 1.13`).

## Corrected computation (branch-constant weighting)
Use branch constants estimated from data:

- `tau_L = E[tau | x < c]`
- `tau_R = E[tau | x >= c]`
- `T_n^br = N_L(n) * tau_L + N_R(n) * tau_R`
- `D_br(s) = 1 + sum_{n>=1} eps_n * exp(-s * T_n^br)`

This is consistent with branch-wise weighting because weights depend only on the L/R symbol counts in each prefix.

## Current corrected values
- Corrected branch-weighted `h_top`: `~0.924`
- LLE benchmark: `~0.906`

So the corrected kneading entropy is close to the LLE scale (gap `~0.018`, about `2%`).
