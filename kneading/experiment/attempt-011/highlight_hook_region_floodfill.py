import csv
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/Users/carterhinsley/Documents/Dev/MultimodalMaps.jl/kneading/experiment/attempt-011")
RESULTS_PATH = ROOT / "grid500_seq7_prefixes_results.tsv"
INPUT_PLOT_NAME = "grid500_seq7_prefixes_contours.png"
OUTPUT_PLOT_NAMES = (
    "grid500_seq7_prefixes_contours_hookfill_flood.png",
    "grid500_seq7_prefixes_prefix07_contours_hookfill_flood.png",
)
SUMMARY_NAME = "grid500_seq7_prefixes_contours_hookfill_flood_summary.txt"

FULL_WIDTH = 2200
FULL_HEIGHT = 1700
PLOT_LEFT = 140
PLOT_TOP = 32
PLOT_RIGHT = 2167
PLOT_BOTTOM = 1581
PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT + 1
PLOT_HEIGHT = PLOT_BOTTOM - PLOT_TOP + 1

SEED_DELTA_CA = -40.0
SEED_DELTA_X = -1.5
SEED_PIXEL_NUDGE_Y = -3
RASTER_LINE_WIDTH = 5

FILL_RGBA = (244, 214, 75, 220)

CASE_SEGMENTS = {
    1: ((3, 0),),
    2: ((0, 1),),
    3: ((3, 1),),
    4: ((1, 2),),
    5: ((3, 0), (1, 2)),
    6: ((0, 2),),
    7: ((3, 2),),
    8: ((2, 3),),
    9: ((0, 2),),
    10: ((2, 3), (0, 1)),
    11: ((1, 2),),
    12: ((1, 3),),
    13: ((0, 1),),
    14: ((3, 0),),
}


def read_results():
    with RESULTS_PATH.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def build_category_grids(rows):
    delta_cas = sorted({float(row["delta_ca"]) for row in rows})
    delta_xs = sorted({float(row["delta_x"]) for row in rows})
    ca_to_idx = {value: idx for idx, value in enumerate(delta_cas)}
    dx_to_idx = {value: idx for idx, value in enumerate(delta_xs)}

    T_lookup = {}
    gamma_lookup = {}
    next_T = 1
    next_gamma = 1

    T_grid = np.zeros((len(delta_cas), len(delta_xs)), dtype=np.int32)
    gamma_grid = np.zeros((len(delta_cas), len(delta_xs)), dtype=np.int32)

    for row in rows:
        i = ca_to_idx[float(row["delta_ca"])]
        j = dx_to_idx[float(row["delta_x"])]
        if row["status"] != "ok":
            continue

        T_key = row["T_scs"]
        gamma_key = row["gamma_scs"]
        if T_key not in T_lookup:
            T_lookup[T_key] = next_T
            next_T += 1
        if gamma_key not in gamma_lookup:
            gamma_lookup[gamma_key] = next_gamma
            next_gamma += 1

        T_grid[i, j] = T_lookup[T_key]
        gamma_grid[i, j] = gamma_lookup[gamma_key]

    return delta_cas, delta_xs, T_grid, gamma_grid


def edge_point(edge, x0, x1, y0, y1):
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    if edge == 0:
        return x0, ym
    if edge == 1:
        return xm, y1
    if edge == 2:
        return x1, ym
    return xm, y0


def categorical_segments(grid, x_values, y_values):
    segments = []
    x_count, y_count = grid.shape
    for x_idx in range(x_count - 1):
        x0 = x_values[x_idx]
        x1 = x_values[x_idx + 1]
        for y_idx in range(y_count - 1):
            y0 = y_values[y_idx]
            y1 = y_values[y_idx + 1]

            bottom_left = int(grid[x_idx, y_idx])
            bottom_right = int(grid[x_idx + 1, y_idx])
            top_right = int(grid[x_idx + 1, y_idx + 1])
            top_left = int(grid[x_idx, y_idx + 1])

            if bottom_left == bottom_right == top_right == top_left:
                continue

            categories = (
                bottom_left,
                bottom_right if bottom_right != bottom_left else 0,
                top_right if top_right not in (bottom_left, bottom_right) else 0,
                top_left if top_left not in (bottom_left, bottom_right, top_right) else 0,
            )

            local_segments = set()
            for category in categories:
                if category <= 0:
                    continue
                mask_case = (
                    (1 if bottom_left == category else 0)
                    + (2 if bottom_right == category else 0)
                    + (4 if top_right == category else 0)
                    + (8 if top_left == category else 0)
                )
                for edge_a, edge_b in CASE_SEGMENTS.get(mask_case, ()):
                    point_a = edge_point(edge_a, x0, x1, y0, y1)
                    point_b = edge_point(edge_b, x0, x1, y0, y1)
                    local_segments.add(tuple(sorted((point_a, point_b))))
            segments.extend(local_segments)
    return segments


def data_to_plot_pixel(delta_ca, delta_x, min_ca, max_ca, min_dx, max_dx):
    x = (delta_ca - min_ca) / (max_ca - min_ca) * (PLOT_WIDTH - 1)
    y = (max_dx - delta_x) / (max_dx - min_dx) * (PLOT_HEIGHT - 1)
    return x, y


def rasterize_segments(T_segments, gamma_segments, delta_cas, delta_xs):
    contour_mask = Image.new("L", (PLOT_WIDTH, PLOT_HEIGHT), 255)
    draw = ImageDraw.Draw(contour_mask)

    min_ca = delta_cas[0]
    max_ca = delta_cas[-1]
    min_dx = delta_xs[0]
    max_dx = delta_xs[-1]

    for segments in (T_segments, gamma_segments):
        for (point_a, point_b) in segments:
            draw.line(
                [
                    data_to_plot_pixel(point_a[0], point_a[1], min_ca, max_ca, min_dx, max_dx),
                    data_to_plot_pixel(point_b[0], point_b[1], min_ca, max_ca, min_dx, max_dx),
                ],
                fill=0,
                width=RASTER_LINE_WIDTH,
            )

    seed_x, seed_y = data_to_plot_pixel(SEED_DELTA_CA, SEED_DELTA_X, min_ca, max_ca, min_dx, max_dx)
    seed = (round(seed_x), max(0, min(PLOT_HEIGHT - 1, round(seed_y) + SEED_PIXEL_NUDGE_Y)))
    return contour_mask, seed


def flood_fill_mask(contour_mask, seed):
    pixels = contour_mask.load()
    if pixels[seed] != 255:
        raise RuntimeError(f"Seed pixel {seed} is not in free space.")

    seen = set([seed])
    queue = deque([seed])

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < PLOT_WIDTH and 0 <= ny < PLOT_HEIGHT and pixels[nx, ny] == 255 and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append((nx, ny))

    mask = Image.new("RGBA", (PLOT_WIDTH, PLOT_HEIGHT), (0, 0, 0, 0))
    mask_pixels = mask.load()
    for x, y in seen:
        mask_pixels[x, y] = FILL_RGBA
    return mask, seen


def summarize_region(rows, filled_pixels, delta_cas, delta_xs, seed):
    min_ca = delta_cas[0]
    max_ca = delta_cas[-1]
    min_dx = delta_xs[0]
    max_dx = delta_xs[-1]

    inside_rows = []
    mask_lookup = set(filled_pixels)
    for row in rows:
        pixel = tuple(round(value) for value in data_to_plot_pixel(float(row["delta_ca"]), float(row["delta_x"]), min_ca, max_ca, min_dx, max_dx))
        if pixel in mask_lookup:
            inside_rows.append(row)

    ok_rows = [row for row in inside_rows if row["status"] == "ok"]
    pair_counts = Counter((row["T_scs"], row["gamma_scs"]) for row in ok_rows)
    T_counts = Counter(row["T_scs"] for row in ok_rows)
    gamma_counts = Counter(row["gamma_scs"] for row in ok_rows)
    delta_ca_values = [float(row["delta_ca"]) for row in inside_rows]
    delta_x_values = [float(row["delta_x"]) for row in inside_rows]

    lines = [
        f"seed_delta_ca\t{SEED_DELTA_CA}",
        f"seed_delta_x\t{SEED_DELTA_X}",
        f"seed_pixel\t{seed}",
        f"raster_line_width\t{RASTER_LINE_WIDTH}",
        f"filled_pixel_count\t{len(filled_pixels)}",
        f"inside_total_rows\t{len(inside_rows)}",
        f"inside_ok_rows\t{len(ok_rows)}",
        f"inside_failed_rows\t{len(inside_rows) - len(ok_rows)}",
        f"inside_unique_pairs\t{len(pair_counts)}",
        f"inside_unique_T\t{len(T_counts)}",
        f"inside_unique_gamma\t{len(gamma_counts)}",
        f"inside_delta_ca_min\t{min(delta_ca_values)}",
        f"inside_delta_ca_max\t{max(delta_ca_values)}",
        f"inside_delta_x_min\t{min(delta_x_values)}",
        f"inside_delta_x_max\t{max(delta_x_values)}",
        "top_pairs",
    ]
    for (T_scs, gamma_scs), count in pair_counts.most_common(10):
        lines.append(f"{count}\t{T_scs}\t{gamma_scs}")

    (ROOT / SUMMARY_NAME).write_text("\n".join(lines) + "\n")


def main():
    rows = read_results()
    delta_cas, delta_xs, T_grid, gamma_grid = build_category_grids(rows)
    T_segments = categorical_segments(T_grid, delta_cas, delta_xs)
    gamma_segments = categorical_segments(gamma_grid, delta_cas, delta_xs)
    contour_mask, seed = rasterize_segments(T_segments, gamma_segments, delta_cas, delta_xs)
    fill_mask, filled_pixels = flood_fill_mask(contour_mask, seed)

    base = Image.open(ROOT / INPUT_PLOT_NAME).convert("RGBA")
    overlay = Image.new("RGBA", (FULL_WIDTH, FULL_HEIGHT), (0, 0, 0, 0))
    overlay.alpha_composite(fill_mask, dest=(PLOT_LEFT, PLOT_TOP))
    highlighted = Image.alpha_composite(base, overlay)

    for output_name in OUTPUT_PLOT_NAMES:
        highlighted.save(ROOT / output_name)

    summarize_region(rows, filled_pixels, delta_cas, delta_xs, seed)


if __name__ == "__main__":
    main()
