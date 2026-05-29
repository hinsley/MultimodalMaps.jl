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
    solve_pressure_root,
)


a, b, c_param = 0.2, 0.2, 5.7

ROUND_DECIMALS = [3, 4, 5, 6]
N_GRID = [200, 300, 400, 600, 800, 1000, 1400, 1800]
BURN = 150


def rossler(t, state):
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c_param)]


def max_x_event(t, state):
    return -state[1] - state[2]


max_x_event.direction = -1


def extract_return_data():
    t0 = time.perf_counter()
    sol = solve_ivp(
        rossler,
        (0.0, 8000.0),
        [1.0, 1.0, 1.0],
        method="RK45",
        events=[max_x_event],
        rtol=1e-10,
        atol=1e-10,
    )
    t1 = time.perf_counter()

    x_events_all = sol.y_events[0][:, 0]
    t_events_all = sol.t_events[0]

    x_events = x_events_all[BURN:]
    t_events = t_events_all[BURN:]

    x_n = x_events[:-1]
    x_np1 = x_events[1:]
    tau_n = np.diff(t_events)

    return x_n, x_np1, tau_n, t1 - t0


def main():
    x_n, x_np1, tau_n, t_extract = extract_return_data()

    records = []

    for rd in ROUND_DECIMALS:
        x_nodes, f_nodes, tau_nodes = aggregate_return_map(x_n, x_np1, tau_n, round_decimals=rd)

        for ng in N_GRID:
            t0 = time.perf_counter()
            try:
                op_info = build_unimodal_pressure_operator(x_nodes, f_nodes, tau_nodes, n_grid=ng)
                root_info = solve_pressure_root(op_info["operator"], s_lo=0.0, s_hi_guess=0.2, tol=1e-10)
                h_top = float(root_info["h_top"])
                lam_root = float(root_info["lambda_at_root"])
                cov = float(op_info["coverage_fraction"])
                ok = True
                err = ""
            except Exception as ex:
                h_top = float("nan")
                lam_root = float("nan")
                cov = float("nan")
                ok = False
                err = str(ex)
            dt = time.perf_counter() - t0

            records.append(
                {
                    "round_decimals": rd,
                    "n_grid": ng,
                    "h_top": h_top,
                    "lambda_at_root": lam_root,
                    "coverage_fraction": cov,
                    "ok": ok,
                    "error": err,
                    "solve_seconds": dt,
                    "n_unique_nodes": int(len(x_nodes)),
                }
            )
            status = "ok" if ok else "fail"
            print(f"rd={rd}, n_grid={ng}: {status}, h_top={h_top}")

    finite = [r for r in records if np.isfinite(r["h_top"])]
    if not finite:
        raise RuntimeError("No successful resolution-study points")

    finite_sorted = sorted(finite, key=lambda r: (r["round_decimals"], r["n_grid"]))
    ref = finite_sorted[-1]["h_top"]

    for r in records:
        r["abs_error_vs_ref"] = float(abs(r["h_top"] - ref)) if np.isfinite(r["h_top"]) else float("nan")

    out_conv = HERE / "pressure_htop_convergence_rossler.png"
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for rd in ROUND_DECIMALS:
        rows = [r for r in records if r["round_decimals"] == rd and np.isfinite(r["h_top"])]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["n_grid"])
        ax.plot(
            [r["n_grid"] for r in rows],
            [r["h_top"] for r in rows],
            marker="o",
            linewidth=1.8,
            label=f"round_decimals={rd}",
        )

    ax.axhline(ref, color="black", linestyle="--", linewidth=1.2, label=f"ref={ref:.9f}")
    ax.set_xlabel("Operator grid size (n_grid)")
    ax.set_ylabel("h_top estimate")
    ax.set_title("Rossler Pressure-Root Convergence vs Resolution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_conv, dpi=200)

    out_err = HERE / "pressure_htop_error_convergence_rossler.png"
    fig2, ax2 = plt.subplots(figsize=(9.5, 6.0))
    for rd in ROUND_DECIMALS:
        rows = [r for r in records if r["round_decimals"] == rd and np.isfinite(r["abs_error_vs_ref"]) and r["abs_error_vs_ref"] > 0]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["n_grid"])
        ax2.semilogy(
            [r["n_grid"] for r in rows],
            [r["abs_error_vs_ref"] for r in rows],
            marker="o",
            linewidth=1.8,
            label=f"round_decimals={rd}",
        )

    ax2.set_xlabel("Operator grid size (n_grid)")
    ax2.set_ylabel("|h_top - h_ref|")
    ax2.set_title("Rossler Pressure-Root Error Convergence")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_err, dpi=200)

    report = {
        "system": "rossler",
        "params": {"a": a, "b": b, "c": c_param},
        "burn": BURN,
        "n_return_samples": int(len(x_n)),
        "extract_seconds": float(t_extract),
        "round_decimals": ROUND_DECIMALS,
        "n_grid": N_GRID,
        "reference_h_top": float(ref),
        "records": records,
        "outputs": {
            "convergence_plot": str(out_conv),
            "error_plot": str(out_err),
            "report_json": str(HERE / "resolution_study_rossler.json"),
        },
    }

    out_json = HERE / "resolution_study_rossler.json"
    out_json.write_text(json.dumps(report, indent=2))

    print("Rossler resolution study complete")
    print(f"reference h_top = {ref:.9f}")
    print(f"Saved: {out_conv}")
    print(f"Saved: {out_err}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
