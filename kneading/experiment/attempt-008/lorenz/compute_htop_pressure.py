import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from flow_pressure_operator import (
    aggregate_return_map,
    build_unimodal_pressure_operator,
    pressure_scan,
    solve_pressure_root,
)


sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0


def lorenz(t, state):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def max_x_event(t, state):
    return state[1] - state[0]


max_x_event.direction = 0


def main():
    t0 = time.perf_counter()
    sol = solve_ivp(
        lorenz,
        (0.0, 1000.0),
        [1.0, 1.0, 1.0],
        method="RK45",
        events=[max_x_event],
        rtol=1e-8,
        atol=1e-8,
    )
    t1 = time.perf_counter()

    y_ev = sol.y_events[0]
    is_max = y_ev[:, 2] > (rho - 1.0)
    abs_x_max = np.abs(y_ev[is_max, 0])
    t_max = sol.t_events[0][is_max]

    x_n = abs_x_max[:-1]
    x_np1 = abs_x_max[1:]
    tau_n = np.diff(t_max)

    x_nodes, f_nodes, tau_nodes = aggregate_return_map(x_n, x_np1, tau_n, round_decimals=4)
    t2 = time.perf_counter()

    op_info = build_unimodal_pressure_operator(x_nodes, f_nodes, tau_nodes, n_grid=800)
    op = op_info["operator"]
    t3 = time.perf_counter()

    root_info = solve_pressure_root(op, s_lo=0.0, s_hi_guess=1.0, tol=1e-9)
    h_top = float(root_info["h_top"])
    t4 = time.perf_counter()

    s_scan_max = max(1.4, 1.6 * h_top)
    s_scan, lam_scan = pressure_scan(op, s_min=0.0, s_max=s_scan_max, n=90)
    t5 = time.perf_counter()

    out_plot = HERE / "pressure_root_scan_lorenz.png"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(s_scan, lam_scan - 1.0, color="navy", linewidth=2, label=r"$\lambda_{max}(\mathcal{L}_s)-1$")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(h_top, color="crimson", linestyle="--", linewidth=1.7, label=f"h_top = {h_top:.6f}")
    ax.set_xlabel("s")
    ax.set_ylabel(r"$\lambda_{max}(\mathcal{L}_s)-1$")
    ax.set_title("Lorenz Pressure Root From Nonconstant Roof Function")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200)

    report = {
        "system": "lorenz",
        "params": {"sigma": sigma, "rho": rho, "beta": beta},
        "n_return_samples": int(len(x_n)),
        "n_unique_nodes": int(len(x_nodes)),
        "c_crit": float(op_info["c_crit"]),
        "coverage_fraction": float(op_info["coverage_fraction"]),
        "left_y_range": list(op_info["left_y_range"]),
        "right_y_range": list(op_info["right_y_range"]),
        "h_top_pressure": h_top,
        "lambda_at_root": float(root_info["lambda_at_root"]),
        "root_bracket": {
            "s_lo": float(root_info["s_lo"]),
            "s_hi": float(root_info["s_hi"]),
            "lambda_lo": float(root_info["lambda_lo"]),
            "lambda_hi": float(root_info["lambda_hi"]),
        },
        "timings_seconds": {
            "ode_and_events": t1 - t0,
            "map_aggregation": t2 - t1,
            "build_operator": t3 - t2,
            "solve_root": t4 - t3,
            "scan_plot_values": t5 - t4,
            "total": t5 - t0,
        },
        "outputs": {
            "plot": str(out_plot),
            "report_json": str(HERE / "pressure_report_lorenz.json"),
        },
    }

    out_json = HERE / "pressure_report_lorenz.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("Lorenz pressure-root computation complete")
    print(f"h_top (pressure root) = {h_top:.9f}")
    print(f"coverage_fraction = {report['coverage_fraction']:.6f}")
    print(f"Saved: {out_plot}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
