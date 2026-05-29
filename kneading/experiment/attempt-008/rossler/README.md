# Attempt-008 Rossler: Pressure-Operator h_top

This computes flow topological entropy from the suspension pressure equation
`P_f(-s*tau)=0` using a transfer-operator discretization (nonconstant roof time).

## Run (Single Estimate)

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-008/rossler/compute_htop_pressure.py
```

## Run (Resolution Study + Convergence Plots)

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-008/rossler/resolution_study.py
```

## Run (Entropy Comparison Plot + Pressure-Root Line)

```bash
/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-005/venv/bin/python \
  /Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-008/rossler/EntropiesRossler_with_pressure.py
```

## Outputs (Single Estimate)

- `pressure_report_rossler.json`
- `pressure_root_scan_rossler.png`

## Outputs (Resolution Study)

- `resolution_study_rossler.json`
- `pressure_htop_convergence_rossler.png`
- `pressure_htop_error_convergence_rossler.png`

## Outputs (Entropy Comparison + Pressure)

- `entropies_rossler_with_pressure.png`
