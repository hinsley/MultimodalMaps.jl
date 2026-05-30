#!/usr/bin/env python3
"""Verify Rossler y-minima tangent scan TSV, PNG, and HTML outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import math
import os
import re
import struct
import sys
from pathlib import Path


TSV_HEADER = [
    "a",
    "c",
    "b",
    "status",
    "events",
    "word",
    "code",
    "period",
    "gamma",
    "max_time",
    "first_time",
    "last_time",
    "min_y",
    "max_y",
]


def fail(message: str) -> None:
    raise SystemExit(f"verification failed: {message}")


def open_scan(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="")
    return path.open("r", newline="")


def require_file(path: Path) -> Path:
    if not path.is_file():
        fail(f"missing required file {path}")
    return path


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            fail(f"{path} is not a PNG file")
        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)
        data = handle.read(length)
    if chunk_type != b"IHDR" or len(data) < 8:
        fail(f"{path} does not start with a valid IHDR chunk")
    return struct.unpack(">II", data[:8])


def verify_png(path: Path, expected_width: int, expected_height: int) -> None:
    width, height = png_dimensions(path)
    if (width, height) != (expected_width, expected_height):
        fail(
            f"{path} has dimensions {width}x{height}, "
            f"expected {expected_width}x{expected_height}"
        )
    print(f"png ok: {path} {width}x{height}")


def count_tsv_rows(path: Path) -> int:
    with path.open("r", newline="") as handle:
        return max(0, sum(1 for _line in handle) - 1)


def verify_legend_rows(path: Path, expected_rows: int) -> None:
    rows = count_tsv_rows(path)
    if rows != expected_rows:
        fail(f"{path} has {rows} data rows, expected {expected_rows}")
    print(f"legend ok: {path} rows={rows}")


def verify_word_legend(path: Path) -> None:
    rows = count_tsv_rows(path)
    if rows <= 0:
        fail(f"{path} has no data rows")
    print(f"word legend ok: {path} rows={rows}")


def verify_tsv(
    path: Path,
    expected_n_c: int,
    expected_n_a: int,
    word_length: int,
) -> None:
    expected_rows = expected_n_c * expected_n_a
    statuses: dict[str, int] = {}
    c_values: set[str] = set()
    a_values: set[str] = set()
    ok_rows = 0
    total_rows = 0
    bad_examples: list[str] = []

    with open_scan(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != TSV_HEADER:
            fail(f"{path} header is {reader.fieldnames!r}, expected {TSV_HEADER!r}")
        for row in reader:
            total_rows += 1
            status = row["status"]
            statuses[status] = statuses.get(status, 0) + 1
            c_values.add(row["c"])
            a_values.add(row["a"])
            if status == "ok":
                ok_rows += 1
                word = row["word"]
                if len(word) != word_length or any(char not in "01" for char in word):
                    bad_examples.append(f"row {total_rows}: bad word {word!r}")
                try:
                    code = int(row["code"])
                except ValueError:
                    bad_examples.append(f"row {total_rows}: bad code {row['code']!r}")
                else:
                    if code != int(word, 2):
                        bad_examples.append(
                            f"row {total_rows}: code {code} does not match word {word}"
                        )
            elif status != "max_time":
                bad_examples.append(f"row {total_rows}: unexpected status {status!r}")
            if len(bad_examples) >= 10:
                fail("; ".join(bad_examples))

    if total_rows != expected_rows:
        fail(f"{path} has {total_rows} rows, expected {expected_rows}")
    if len(c_values) != expected_n_c:
        fail(f"{path} has {len(c_values)} c values, expected {expected_n_c}")
    if len(a_values) != expected_n_a:
        fail(f"{path} has {len(a_values)} a values, expected {expected_n_a}")
    if ok_rows == 0:
        fail(f"{path} has no ok rows")

    print(
        "tsv ok: "
        f"{path} rows={total_rows} ok={ok_rows} "
        f"max_time={statuses.get('max_time', 0)}"
    )


def extract_octet_script(html: str, element_id: str, path: Path) -> bytes:
    pattern = (
        r'<script id="' + re.escape(element_id) + r'" '
        r'type="application/octet-stream">\s*(.*?)\s*</script>'
    )
    match = re.search(pattern, html, re.DOTALL)
    if match is None:
        fail(f"{path} is missing octet-stream script {element_id}")
    encoded = re.sub(r"\s+", "", match.group(1))
    try:
        return base64.b64decode(encoded)
    except Exception as exc:  # noqa: BLE001
        fail(f"{path} has invalid base64 in {element_id}: {exc}")


def verify_probe_html(
    path: Path,
    expected_cells: int,
    expected_color_mode: str,
) -> None:
    html = path.read_text()
    if "data:image/png;base64," not in html:
        fail(f"{path} does not embed its PNG image")
    color_mode_match = re.search(r"const\s+COLOR_MODE\s*=\s*'([^']+)'", html)
    if color_mode_match is not None:
        actual_color_mode = color_mode_match.group(1)
        if actual_color_mode != expected_color_mode:
            fail(f"{path} declares COLOR_MODE {actual_color_mode!r}")
    elif expected_color_mode == "word" and "8-bit word heatmap" not in html:
        fail(f"{path} does not identify itself as an 8-bit word heatmap")
    elif expected_color_mode == "monotone" and "7-bit monotone heatmap" not in html:
        fail(f"{path} does not identify itself as a 7-bit monotone heatmap")
    if "function monotoneBits" not in html or "function symbolText" not in html:
        fail(f"{path} is missing symbol/monotone probe logic")

    code_bytes = extract_octet_script(html, "codeBytes", path)
    valid_bits = extract_octet_script(html, "validBits", path)
    expected_valid_len = math.ceil(expected_cells / 8)
    if len(code_bytes) != expected_cells:
        fail(f"{path} codeBytes length {len(code_bytes)}, expected {expected_cells}")
    if len(valid_bits) != expected_valid_len:
        fail(f"{path} validBits length {len(valid_bits)}, expected {expected_valid_len}")
    print(
        f"html ok: {path} color_mode={expected_color_mode} "
        f"code_bytes={len(code_bytes)} valid_bits={len(valid_bits)}"
    )


def find_scan_tsv(result_dir: Path, stem: str) -> Path:
    gz_path = result_dir / f"{stem}.tsv.gz"
    plain_path = result_dir / f"{stem}.tsv"
    if gz_path.is_file():
        return gz_path
    if plain_path.is_file():
        return plain_path
    fail(f"missing {gz_path} or {plain_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="flow_folding/results/rossler_y_minima_tangent_scan_4096",
        help="scan result directory",
    )
    parser.add_argument("--stem", default="coarse_scan", help="artifact filename stem")
    parser.add_argument("--n-c", type=int, default=4096, help="expected c grid count")
    parser.add_argument("--n-a", type=int, default=4096, help="expected a grid count")
    parser.add_argument("--word-length", type=int, default=8, help="expected word length")
    parser.add_argument("--png-width", type=int, default=25600, help="expected PNG width")
    parser.add_argument("--png-height", type=int, default=17600, help="expected PNG height")
    parser.add_argument(
        "--skip-tsv",
        action="store_true",
        help="skip full TSV scan validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    contour_dir = result_dir / "contours"
    expected_cells = args.n_c * args.n_a
    tsv_path = find_scan_tsv(result_dir, args.stem)

    if not args.skip_tsv:
        verify_tsv(tsv_path, args.n_c, args.n_a, args.word_length)

    word_png = require_file(contour_dir / f"{args.stem}_8bit_word_heatmap.png")
    word_html = require_file(contour_dir / f"{args.stem}_8bit_word_heatmap_probe.html")
    word_legend = require_file(contour_dir / f"{args.stem}_8bit_word_heatmap_legend.tsv")
    word_counts = require_file(contour_dir / f"{args.stem}_word_legend.tsv")
    mono_png = require_file(contour_dir / f"{args.stem}_7bit_monotone_heatmap.png")
    mono_html = require_file(contour_dir / f"{args.stem}_7bit_monotone_heatmap_probe.html")
    mono_legend = require_file(contour_dir / f"{args.stem}_7bit_monotone_heatmap_legend.tsv")

    verify_png(word_png, args.png_width, args.png_height)
    verify_png(mono_png, args.png_width, args.png_height)
    verify_legend_rows(word_legend, 1 << args.word_length)
    verify_legend_rows(mono_legend, 1 << (args.word_length - 1))
    verify_word_legend(word_counts)
    verify_probe_html(word_html, expected_cells, "word")
    verify_probe_html(mono_html, expected_cells, "monotone")
    print("all requested Rossler y-minima tangent outputs verified")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
