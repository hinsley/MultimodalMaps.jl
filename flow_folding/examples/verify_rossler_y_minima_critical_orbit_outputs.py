#!/usr/bin/env python3
import argparse
import csv
import math
import os
import struct
from collections import Counter, defaultdict


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def word_code(word: str) -> int:
    code = 0
    for char in word:
        code = 2 * code + (1 if char == "1" else 0)
    return code


def png_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path} is not a PNG file")
    return struct.unpack(">II", header[16:24])


def finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify corrected Rössler y-minima critical-orbit scan artifacts."
    )
    parser.add_argument("tsv")
    parser.add_argument("--n-c", type=int, default=256)
    parser.add_argument("--n-a", type=int, default=256)
    parser.add_argument("--word-length", type=int, default=8)
    parser.add_argument("--b", type=float, default=0.3)
    parser.add_argument("--c-min", type=float, default=2.0)
    parser.add_argument("--c-max", type=float, default=7.0)
    parser.add_argument("--a-min", type=float, default=0.30)
    parser.add_argument("--a-max", type=float, default=0.55)
    parser.add_argument("--max-residual", type=float, default=1e-5)
    parser.add_argument("--max-event-value", type=float, default=1e-7)
    parser.add_argument("--max-critical-y-jump", type=float, default=0.5)
    parser.add_argument("--min-ok-fraction", type=float, default=0.5)
    parser.add_argument("--png", action="append", default=[], help="PNG path to verify dimensions for")
    parser.add_argument("--png-width", type=int, default=6400)
    parser.add_argument("--png-height", type=int, default=4400)
    parser.add_argument("--html", action="append", default=[], help="standalone probe HTML to require")
    args = parser.parse_args()

    failures: list[str] = []
    rows: list[dict[str, str]] = []
    with open(args.tsv, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required_columns = {
        "a",
        "c",
        "b",
        "status",
        "events",
        "word",
        "code",
        "max_time",
        "first_time",
        "last_time",
        "critical_status",
        "critical_event_index",
        "critical_rho",
        "critical_residual",
        "critical_x",
        "critical_y",
        "critical_z",
        "critical_event_value",
        "critical_second_derivative",
        "critical_tangent_x",
        "critical_tangent_y",
        "critical_tangent_z",
        "orbit_transient_events",
        "first_event_is_return",
        "initial_event_included",
    }
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        failures.append(f"missing TSV columns: {', '.join(missing)}")

    expected_rows = args.n_c * args.n_a
    if len(rows) != expected_rows:
        failures.append(f"row count {len(rows)} != expected {expected_rows}")

    c_values = sorted({float(row["c"]) for row in rows}) if rows else []
    a_values = sorted({float(row["a"]) for row in rows}) if rows else []
    if len(c_values) != args.n_c:
        failures.append(f"unique c count {len(c_values)} != {args.n_c}")
    if len(a_values) != args.n_a:
        failures.append(f"unique a count {len(a_values)} != {args.n_a}")
    if c_values and (abs(c_values[0] - args.c_min) > 1e-10 or abs(c_values[-1] - args.c_max) > 1e-10):
        failures.append(f"c range {c_values[0]}..{c_values[-1]} != {args.c_min}..{args.c_max}")
    if a_values and (abs(a_values[0] - args.a_min) > 1e-10 or abs(a_values[-1] - args.a_max) > 1e-10):
        failures.append(f"a range {a_values[0]}..{a_values[-1]} != {args.a_min}..{args.a_max}")

    status_counts: Counter[str] = Counter()
    critical_counts: Counter[str] = Counter()
    ok_count = 0
    critical_y_by_c: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)

    for line_number, row in enumerate(rows, start=2):
        prefix = f"line {line_number}"
        status_counts[row["status"]] += 1
        critical_counts[row["critical_status"]] += 1
        try:
            if abs(float(row["b"]) - args.b) > 1e-12:
                failures.append(f"{prefix}: b={row['b']} != {args.b}")
            events = int(row["events"])
            word = row["word"]
            if int(row["orbit_transient_events"]) != 0:
                failures.append(f"{prefix}: orbit_transient_events is not zero")
            if row["critical_status"] == "ok":
                if parse_bool(row["first_event_is_return"]):
                    failures.append(f"{prefix}: first_event_is_return should be false")
                if not parse_bool(row["initial_event_included"]):
                    failures.append(f"{prefix}: initial_event_included should be true")
                if events > 0 and abs(float(row["first_time"])) > 1e-12:
                    failures.append(f"{prefix}: first_time is not the initial critical event")
                residual = abs(finite_float(row, "critical_residual"))
                if residual > args.max_residual:
                    failures.append(f"{prefix}: critical residual {residual:g} exceeds {args.max_residual:g}")
                event_value = abs(finite_float(row, "critical_event_value"))
                if event_value > args.max_event_value:
                    failures.append(f"{prefix}: critical event value {event_value:g} exceeds {args.max_event_value:g}")
                second_derivative = finite_float(row, "critical_second_derivative")
                if second_derivative <= 0:
                    failures.append(f"{prefix}: critical point is not a y-minimum")
                critical_y = finite_float(row, "critical_y")
                critical_y_by_c[row["c"]].append((float(row["a"]), critical_y))
            if row["status"] == "ok":
                ok_count += 1
                if events != args.word_length:
                    failures.append(f"{prefix}: ok row has {events} events")
                if len(word) != args.word_length:
                    failures.append(f"{prefix}: ok row word length {len(word)}")
                if any(char not in "01" for char in word):
                    failures.append(f"{prefix}: word contains non-binary symbols")
                if int(row["code"]) != word_code(word):
                    failures.append(f"{prefix}: code does not match word")
            else:
                if int(row["code"]) != -1:
                    failures.append(f"{prefix}: incomplete row code should be -1")
                if len(word) != events:
                    failures.append(f"{prefix}: incomplete word length does not match events")
        except Exception as exc:
            failures.append(f"{prefix}: {exc}")

    ok_fraction = ok_count / len(rows) if rows else 0.0
    if ok_fraction < args.min_ok_fraction:
        failures.append(f"ok fraction {ok_fraction:.3f} below {args.min_ok_fraction:.3f}")

    jumps: list[float] = []
    for values in critical_y_by_c.values():
        values.sort()
        jumps.extend(abs(y1 - y0) for (_, y0), (_, y1) in zip(values, values[1:]))
    max_jump = max(jumps) if jumps else 0.0
    if max_jump > args.max_critical_y_jump:
        failures.append(f"max adjacent critical_y jump {max_jump:g} exceeds {args.max_critical_y_jump:g}")

    for path in args.png:
        if not os.path.exists(path):
            failures.append(f"missing PNG: {path}")
            continue
        try:
            size = png_size(path)
        except Exception as exc:
            failures.append(str(exc))
            continue
        if size != (args.png_width, args.png_height):
            failures.append(f"{path} size {size[0]}x{size[1]} != {args.png_width}x{args.png_height}")

    for path in args.html:
        if not os.path.exists(path):
            failures.append(f"missing HTML: {path}")
            continue
        text = open(path, encoding="utf-8").read()
        for needle in ("codeBytes", "validBits", "Parameter Probe", "pointermove"):
            if needle not in text:
                failures.append(f"{path} does not contain {needle}")

    print(f"rows={len(rows)} expected={expected_rows}")
    print(f"status={dict(status_counts)}")
    print(f"critical_status={dict(critical_counts)}")
    print(f"ok_fraction={ok_fraction:.6f}")
    print(f"max_adjacent_critical_y_jump={max_jump:.12g}")
    if failures:
        print("verification_failed")
        for failure in failures[:50]:
            print(f"FAIL: {failure}")
        if len(failures) > 50:
            print(f"FAIL: ... {len(failures) - 50} more")
        raise SystemExit(1)
    print("verification_ok")


if __name__ == "__main__":
    main()
