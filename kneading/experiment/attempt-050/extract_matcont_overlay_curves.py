#!/usr/bin/env python3
"""Extract the MATCONT curves used for the attempt-050 contour overlay.

The source .fig is a MATLAB v5 figure.  The red homSF curve starts in the
pure-red "2 homoclinics" line series, but the lower visible continuation is
stored in adjacent red "primary homoclinic" line series.  Those red pieces are
spliced at their nearest point in the parameter plane.  The unlabeled orange
curve is stored as a single line series with two finite chunks separated by
NaNs; the orange chunks are reconnected at the nearest pair of component
endpoints in the parameter plane.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DEFAULT_FIG_PATH = Path("/Users/carterhinsley/Downloads/jack_red_hil_overlap1.fig")
DEFAULT_OUTPUT_PATH = ROOT / "gcs_results" / "jack_red_hil_overlap1_matcont_overlay_curves.tsv"
X_LIMITS = (-45.0, -20.0)
Y_LIMITS = (-1.5, -0.5)


@dataclass
class LineSeries:
    handle: int
    display_name: str
    color: tuple[float, float, float]
    x: np.ndarray
    y: np.ndarray


def iter_children(obj) -> Iterable[object]:
    children = getattr(obj, "children", None)
    if children is None:
        return
    if isinstance(children, np.ndarray):
        for child in children.flat:
            yield child
    else:
        yield children


def iter_graphics(obj) -> Iterable[object]:
    yield obj
    for child in iter_children(obj):
        yield from iter_graphics(child)


def color_tuple(value) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != 3:
        raise ValueError(f"Expected RGB color triplet, got {arr!r}")
    return tuple(float(x) for x in arr)


def get_line_series(fig_path: Path) -> list[LineSeries]:
    mat = loadmat(fig_path, squeeze_me=True, struct_as_record=False)
    root = mat["hgS_070000"]
    lines: list[LineSeries] = []
    for obj in iter_graphics(root):
        if getattr(obj, "type", None) != "graph2d.lineseries":
            continue
        props = obj.properties
        x = np.asarray(getattr(props, "XData", []), dtype=float).ravel()
        y = np.asarray(getattr(props, "YData", []), dtype=float).ravel()
        if x.size == 0 or y.size == 0:
            continue
        n = min(x.size, y.size)
        display_name = getattr(props, "DisplayName", "")
        if not isinstance(display_name, str):
            display_name = ""
        lines.append(
            LineSeries(
                handle=int(round(float(obj.handle))),
                display_name=display_name,
                color=color_tuple(props.Color),
                x=x[:n],
                y=y[:n],
            )
        )
    return lines


def close_color(color: tuple[float, float, float], target: tuple[float, float, float], tol: float = 1.0e-6) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(color, target))


def orangeish(color: tuple[float, float, float]) -> bool:
    r, g, b = color
    return r > 0.75 and 0.35 < g < 0.65 and b < 0.2


def visible_point_count(line: LineSeries) -> int:
    finite = np.isfinite(line.x) & np.isfinite(line.y)
    visible = (
        finite
        & (line.x >= X_LIMITS[0])
        & (line.x <= X_LIMITS[1])
        & (line.y >= Y_LIMITS[0])
        & (line.y <= Y_LIMITS[1])
    )
    return int(np.count_nonzero(visible))


def split_finite_chunks(x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    current: list[tuple[float, float]] = []
    for xi, yi in zip(x, y):
        if math.isfinite(float(xi)) and math.isfinite(float(yi)):
            current.append((float(xi), float(yi)))
        elif current:
            chunks.append(np.asarray(current, dtype=float))
            current = []
    if current:
        chunks.append(np.asarray(current, dtype=float))
    return [chunk for chunk in chunks if len(chunk) > 0]


def endpoint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def connect_components(components: list[np.ndarray]) -> np.ndarray:
    if not components:
        raise ValueError("No components to connect")
    path = components[0].copy()
    remaining = [component.copy() for component in components[1:]]
    while remaining:
        best: tuple[float, int, str, str] | None = None
        for idx, component in enumerate(remaining):
            options = [
                (endpoint_distance(path[-1], component[0]), idx, "end", "start"),
                (endpoint_distance(path[-1], component[-1]), idx, "end", "end"),
                (endpoint_distance(path[0], component[-1]), idx, "start", "end"),
                (endpoint_distance(path[0], component[0]), idx, "start", "start"),
            ]
            local_best = min(options, key=lambda item: item[0])
            if best is None or local_best[0] < best[0]:
                best = local_best
        assert best is not None
        _, idx, path_endpoint, component_endpoint = best
        component = remaining.pop(idx)
        if path_endpoint == "end" and component_endpoint == "start":
            path = np.vstack([path, component])
        elif path_endpoint == "end" and component_endpoint == "end":
            path = np.vstack([path, component[::-1]])
        elif path_endpoint == "start" and component_endpoint == "end":
            path = np.vstack([component, path])
        else:
            path = np.vstack([component[::-1], path])
    return path


def pick_overlay_curves(lines: list[LineSeries]) -> dict[str, tuple[list[int], np.ndarray]]:
    homsf_lines = [
        line
        for line in lines
        if line.display_name == "2 homoclinics" and close_color(line.color, (1.0, 0.0, 0.0))
    ]
    if not homsf_lines:
        raise RuntimeError("Could not find the red homSF '2 homoclinics' line series")

    orange_candidates = [
        line
        for line in lines
        if line.display_name == ""
        and orangeish(line.color)
        and len(split_finite_chunks(line.x, line.y)) >= 2
        and visible_point_count(line) > 0
    ]
    if not orange_candidates:
        raise RuntimeError("Could not find the unlabeled two-piece orange line series")
    orange_line = max(orange_candidates, key=visible_point_count)

    homsf_by_handle = {line.handle: line for line in homsf_lines}
    required_homsf_handles = (62, 63, 64)
    if all(handle in homsf_by_handle for handle in required_homsf_handles):
        homsf_components = [
            split_finite_chunks(homsf_by_handle[62].x, homsf_by_handle[62].y)[0][::-1],
            split_finite_chunks(homsf_by_handle[63].x, homsf_by_handle[63].y)[0][::-1],
            split_finite_chunks(homsf_by_handle[64].x, homsf_by_handle[64].y)[0],
        ]
        homsf_handles = list(required_homsf_handles)
    else:
        homsf_components = []
        homsf_handles = []
        for line in sorted(homsf_lines, key=lambda item: item.handle):
            homsf_handles.append(line.handle)
            homsf_components.extend(split_finite_chunks(line.x, line.y))

    # The lower red segment in the MATLAB figure is not labeled "homSF";
    # it is the red-toned primary-homoclinic continuation through the ShH point.
    primary_by_handle = {
        line.handle: line
        for line in lines
        if line.display_name == "primary homoclinic" and visible_point_count(line) > 0
    }
    if 66 in primary_by_handle and 67 in primary_by_handle and homsf_components:
        homsf_tail_start = homsf_components[-1][-1]
        tail67 = split_finite_chunks(primary_by_handle[67].x, primary_by_handle[67].y)[0]
        tail_start_idx = int(np.argmin(np.linalg.norm(tail67 - homsf_tail_start, axis=1)))
        tail66 = split_finite_chunks(primary_by_handle[66].x, primary_by_handle[66].y)[0]
        homsf_components.extend([tail67[tail_start_idx:], tail66])
        homsf_handles.extend([67, 66])

    orange_components = split_finite_chunks(orange_line.x, orange_line.y)
    return {
        "homSF": (homsf_handles, np.vstack(homsf_components)),
        "orange_unlabeled": ([orange_line.handle], connect_components(orange_components)),
    }


def write_curves(path: Path, curves: dict[str, tuple[list[int], np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("curve\tpart\tsource_handles\tx\ty\n")
        for curve_name, (handles, points) in curves.items():
            handle_text = ",".join(str(handle) for handle in handles)
            for x, y in points:
                f.write(f"{curve_name}\t1\t{handle_text}\t{x:.17g}\t{y:.17g}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", type=Path, default=DEFAULT_FIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    lines = get_line_series(args.fig)
    curves = pick_overlay_curves(lines)
    write_curves(args.output, curves)
    for curve_name, (handles, points) in curves.items():
        print(
            f"{curve_name}: handles={','.join(str(handle) for handle in handles)} "
            f"points={len(points)} x=[{points[:, 0].min():.6g},{points[:, 0].max():.6g}] "
            f"y=[{points[:, 1].min():.6g},{points[:, 1].max():.6g}]"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
