from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import root_scalar


@dataclass
class BranchInverse:
    y_nodes: np.ndarray
    x_nodes: np.ndarray
    tau_nodes: np.ndarray


@dataclass
class PressureOperator:
    grid: np.ndarray
    # Each tuple is (mask_on_grid, preimage_on_grid, tau_on_preimage)
    branches: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]

    def apply(self, v: np.ndarray, s: float) -> np.ndarray:
        out = np.zeros_like(v)
        for mask, preimage, tau_pre in self.branches:
            if not np.any(mask):
                continue
            v_pre = np.interp(preimage[mask], self.grid, v)
            out[mask] += np.exp(-s * tau_pre[mask]) * v_pre
        return out

    def leading_eigenvalue(self, s: float, tol: float = 1e-12, max_iter: int = 4000) -> Tuple[float, int]:
        # Positive operator on a positive cone: L1-normalized power iteration.
        v = np.ones_like(self.grid, dtype=np.float64)
        v /= np.sum(v)

        lam = 1.0
        for it in range(max_iter):
            w = self.apply(v, s)
            lam_new = float(np.sum(w))
            if not np.isfinite(lam_new) or lam_new <= 0.0:
                raise RuntimeError(f"Invalid leading-eigenvalue iterate at s={s}")

            w /= lam_new
            if np.max(np.abs(w - v)) < tol:
                return lam_new, it + 1

            v = w
            lam = lam_new

        return lam, max_iter


def aggregate_return_map(
    x_n: np.ndarray,
    x_np1: np.ndarray,
    tau_n: np.ndarray,
    round_decimals: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_nodes, idx = np.unique(np.round(x_n, round_decimals), return_inverse=True)
    f_nodes = np.zeros_like(x_nodes, dtype=np.float64)
    tau_nodes = np.zeros_like(x_nodes, dtype=np.float64)

    for i in range(len(x_nodes)):
        mask = idx == i
        f_nodes[i] = np.mean(x_np1[mask])
        tau_nodes[i] = np.mean(tau_n[mask])

    return x_nodes, f_nodes, tau_nodes


def _build_inverse_branch(x_branch: np.ndarray, f_branch: np.ndarray, tau_branch: np.ndarray) -> BranchInverse:
    # Build y -> x and y -> tau representations by sorting on y=f(x).
    order = np.argsort(f_branch)
    y_sorted = f_branch[order]
    x_sorted = x_branch[order]
    tau_sorted = tau_branch[order]

    y_unique, idx = np.unique(np.round(y_sorted, 6), return_inverse=True)
    x_unique = np.zeros_like(y_unique, dtype=np.float64)
    tau_unique = np.zeros_like(y_unique, dtype=np.float64)

    for i in range(len(y_unique)):
        mask = idx == i
        x_unique[i] = np.mean(x_sorted[mask])
        tau_unique[i] = np.mean(tau_sorted[mask])

    # Remove flat y duplicates that survived rounding.
    keep = np.r_[True, np.diff(y_unique) > 1e-10]
    y_unique = y_unique[keep]
    x_unique = x_unique[keep]
    tau_unique = tau_unique[keep]

    if y_unique.size < 2:
        raise RuntimeError("Branch inverse is degenerate")

    return BranchInverse(y_nodes=y_unique, x_nodes=x_unique, tau_nodes=tau_unique)


def build_unimodal_pressure_operator(
    x_nodes: np.ndarray,
    f_nodes: np.ndarray,
    tau_nodes: np.ndarray,
    n_grid: int = 800,
) -> Dict[str, object]:
    c_idx = int(np.argmax(f_nodes))
    c_crit = float(x_nodes[c_idx])

    left_mask = x_nodes <= c_crit
    right_mask = x_nodes >= c_crit
    if np.count_nonzero(left_mask) < 3 or np.count_nonzero(right_mask) < 3:
        raise RuntimeError("Insufficient branch data to build inverse branches")

    left_inv = _build_inverse_branch(x_nodes[left_mask], f_nodes[left_mask], tau_nodes[left_mask])
    right_inv = _build_inverse_branch(x_nodes[right_mask], f_nodes[right_mask], tau_nodes[right_mask])

    x_min = float(np.min(x_nodes))
    x_max = float(np.max(x_nodes))
    grid = np.linspace(x_min, x_max, n_grid)

    branches = []
    coverage = np.zeros_like(grid, dtype=bool)

    for br in (left_inv, right_inv):
        y_min = float(np.min(br.y_nodes))
        y_max = float(np.max(br.y_nodes))
        mask = (grid >= y_min) & (grid <= y_max)

        preimage = np.full_like(grid, np.nan, dtype=np.float64)
        tau_pre = np.full_like(grid, np.nan, dtype=np.float64)
        preimage[mask] = np.interp(grid[mask], br.y_nodes, br.x_nodes)
        tau_pre[mask] = np.interp(grid[mask], br.y_nodes, br.tau_nodes)

        branches.append((mask, preimage, tau_pre))
        coverage |= mask

    operator = PressureOperator(grid=grid, branches=branches)

    return {
        "operator": operator,
        "c_crit": c_crit,
        "coverage_fraction": float(np.mean(coverage)),
        "x_min": x_min,
        "x_max": x_max,
        "left_y_range": (float(left_inv.y_nodes[0]), float(left_inv.y_nodes[-1])),
        "right_y_range": (float(right_inv.y_nodes[0]), float(right_inv.y_nodes[-1])),
    }


def solve_pressure_root(
    operator: PressureOperator,
    s_lo: float,
    s_hi_guess: float,
    tol: float = 1e-8,
    max_expand: int = 32,
) -> Dict[str, object]:
    cache: Dict[float, Tuple[float, int]] = {}

    def lam(s: float) -> float:
        key = float(np.round(s, 14))
        if key not in cache:
            cache[key] = operator.leading_eigenvalue(key)
        return cache[key][0]

    l_lo = lam(s_lo)
    s_hi = s_hi_guess
    l_hi = lam(s_hi)

    expand_count = 0
    while l_hi > 1.0 and expand_count < max_expand:
        s_hi *= 2.0
        l_hi = lam(s_hi)
        expand_count += 1

    if l_lo <= 1.0:
        raise RuntimeError(f"Invalid lower bracket: lambda({s_lo})={l_lo} <= 1")
    if l_hi > 1.0:
        raise RuntimeError(
            f"Could not find upper bracket with lambda(s)<1; last s={s_hi}, lambda={l_hi}"
        )

    def f_root(s: float) -> float:
        return lam(s) - 1.0

    root = float(root_scalar(f_root, bracket=[s_lo, s_hi], method="brentq", xtol=tol).root)
    lam_root = lam(root)

    return {
        "h_top": root,
        "lambda_at_root": lam_root,
        "s_lo": s_lo,
        "s_hi": s_hi,
        "lambda_lo": l_lo,
        "lambda_hi": l_hi,
        "cache": cache,
    }


def pressure_scan(
    operator: PressureOperator,
    s_min: float,
    s_max: float,
    n: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    ss = np.linspace(s_min, s_max, n)
    lam_vals = np.zeros_like(ss)
    for i, s in enumerate(ss):
        lam_vals[i], _ = operator.leading_eigenvalue(float(s))
    return ss, lam_vals
