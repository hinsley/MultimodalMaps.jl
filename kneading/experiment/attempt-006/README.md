# Attempt 005 (Branch-Weighted Fix)

This folder contains a corrected Lorenz kneading pipeline that uses **branch-constant roof weights** in the weighted kneading determinant.

For the unimodal return map with critical point `c`:

- Left branch weight: `g_L(s) = exp(-s * tau_L)`
- Right branch weight: `g_R(s) = exp(-s * tau_R)`

where `tau_L = E[tau | x_n < c]` and `tau_R = E[tau | x_n >= c]` are estimated from return-map data.

The corrected determinant is evaluated along the critical itinerary as:

`D_branch(s) = 1 + sum_n eps_n * exp(-s * T_n_branch)`

with `T_n_branch = N_L(n)*tau_L + N_R(n)*tau_R`.

This script also computes the old pointwise-time determinant for direct comparison.

## Run

Use the existing attempt-005 Python environment:

```bash
kneading/experiment/attempt-005/venv/bin/python kneading/experiments/attempt-005/EntropiesLorenz.py
```

## Outputs

- `entropies_lorenz_branch_weighted.png`
- `htop_convergence_branch_vs_pointwise.png`
