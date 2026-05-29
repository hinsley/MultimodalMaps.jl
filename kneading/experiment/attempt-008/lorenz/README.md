# Attempt-008 Lorenz: Pressure-Operator h_top

This computes flow topological entropy from the suspension pressure equation
`P_f(-s*tau)=0` using a transfer-operator discretization (nonconstant roof time).

## Run (Single Estimate)

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-008/lorenz/compute_htop_pressure.py
```

## Run (Resolution Study + Convergence Plots)

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-008/lorenz/resolution_study.py
```

## Run (Entropy Comparison Plot + Pressure-Root Line)

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-008/lorenz/EntropiesLorenz_with_pressure.py
```

## Outputs (Single Estimate)

- `pressure_report_lorenz.json`
- `pressure_root_scan_lorenz.png`

## Outputs (Resolution Study)

- `resolution_study_lorenz.json`
- `pressure_htop_convergence_lorenz.png`
- `pressure_htop_error_convergence_lorenz.png`

## Outputs (Entropy Comparison + Pressure)

- `entropies_lorenz_branch_weighted.png`
