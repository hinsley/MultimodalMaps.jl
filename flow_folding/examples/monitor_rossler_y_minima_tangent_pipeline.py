#!/usr/bin/env python3
"""Monitor a chunked Rossler y-minima tangent scan pipeline."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_FIELDS = 14


@dataclass(frozen=True)
class ChunkStatus:
    chunk: int
    path: Path
    rows: int
    expected_rows: int
    header_bad: int
    bad_field_rows: int
    bad_ok_word_rows: int
    last_log: str

    @property
    def complete(self) -> bool:
        return self.rows == self.expected_rows

    @property
    def percent(self) -> float:
        if self.expected_rows <= 0:
            return 0.0
        return 100.0 * self.rows / self.expected_rows

    @property
    def valid(self) -> bool:
        return (
            self.header_bad == 0
            and self.bad_field_rows == 0
            and self.bad_ok_word_rows == 0
        )


def run_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return ""


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _line in handle)


def latest_log_line(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if line.strip():
                last = line.strip()
    return last


def chunk_bounds(chunk: int, n_a: int, chunks: int) -> tuple[int, int]:
    start_idx = chunk * n_a // chunks + 1
    end_idx = (chunk + 1) * n_a // chunks
    return start_idx, end_idx


def expected_rows_for_chunk(chunk: int, n_c: int, n_a: int, chunks: int) -> int:
    start_idx, end_idx = chunk_bounds(chunk, n_a, chunks)
    return (end_idx - start_idx + 1) * n_c


def validate_tsv(path: Path, word_length: int) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0

    header_bad = 0
    bad_field_rows = 0
    bad_ok_word_rows = 0
    with path.open("r", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if len(header) != EXPECTED_FIELDS:
            header_bad = 1
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != EXPECTED_FIELDS:
                bad_field_rows += 1
                continue
            status = row[3]
            word = row[5]
            if status == "ok" and len(word) != word_length:
                bad_ok_word_rows += 1
    return header_bad, bad_field_rows, bad_ok_word_rows


def chunk_status(
    result_dir: Path,
    chunk: int,
    n_c: int,
    n_a: int,
    chunks: int,
    word_length: int,
    validate: bool,
) -> ChunkStatus:
    chunk_path = result_dir / "chunks" / f"chunk_{chunk:03d}.tsv"
    rows = max(0, line_count(chunk_path) - 1)
    expected_rows = expected_rows_for_chunk(chunk, n_c, n_a, chunks)
    header_bad = bad_field_rows = bad_ok_word_rows = 0
    if validate and chunk_path.exists():
        header_bad, bad_field_rows, bad_ok_word_rows = validate_tsv(chunk_path, word_length)
    last_log = latest_log_line(result_dir / "chunks" / f"chunk_{chunk:03d}.log")
    return ChunkStatus(
        chunk=chunk,
        path=chunk_path,
        rows=rows,
        expected_rows=expected_rows,
        header_bad=header_bad,
        bad_field_rows=bad_field_rows,
        bad_ok_word_rows=bad_ok_word_rows,
        last_log=last_log,
    )


def parse_rate_and_eta(line: str) -> tuple[str, str]:
    rate = "-"
    eta = "-"
    rate_match = re.search(r"rate=([0-9.]+)/s", line)
    eta_match = re.search(r"eta=([0-9.]+)s", line)
    if rate_match is not None:
        rate = rate_match.group(1)
    if eta_match is not None:
        eta = eta_match.group(1)
    return rate, eta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="flow_folding/results/rossler_y_minima_tangent_scan_4096",
        help="pipeline result directory",
    )
    parser.add_argument("--n-c", type=int, default=4096, help="expected c grid count")
    parser.add_argument("--n-a", type=int, default=4096, help="expected a grid count")
    parser.add_argument("--chunks", type=int, default=32, help="expected chunk count")
    parser.add_argument("--word-length", type=int, default=8, help="expected ok word length")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="scan present chunk TSV rows for field count and ok-word length",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir)
    chunk_dir = result_dir / "chunks"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmux_running = subprocess.call(
        ["tmux", "has-session", "-t", "rossler4096"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0
    julia_workers = run_output(["pgrep", "-fl", "julia.*rossler_y_minima_tangent_scan.jl"])
    worker_count = len(julia_workers.splitlines()) if julia_workers else 0

    statuses = [
        chunk_status(
            result_dir,
            chunk,
            args.n_c,
            args.n_a,
            args.chunks,
            args.word_length,
            args.validate,
        )
        for chunk in range(args.chunks)
        if (chunk_dir / f"chunk_{chunk:03d}.tsv").exists()
    ]
    complete = sum(1 for status in statuses if status.complete)
    total_rows = sum(status.rows for status in statuses)
    expected_total_rows = args.n_c * args.n_a
    invalid = [status for status in statuses if not status.valid]

    print(f"time_utc={now}")
    print(f"result_dir={result_dir}")
    print(f"tmux_running={'yes' if tmux_running else 'no'}")
    print(f"julia_workers={worker_count}")
    print(f"failed_marker={'yes' if (chunk_dir / '.failed').exists() else 'no'}")
    print(f"chunk_tsv_count={len(statuses)}")
    print(f"complete_chunk_count={complete}")
    print(f"scanned_rows={total_rows}/{expected_total_rows}")
    if expected_total_rows > 0:
        print(f"scanned_percent={100.0 * total_rows / expected_total_rows:.4f}")
    print(f"root_tsv_gz={'yes' if (result_dir / 'coarse_scan.tsv.gz').exists() else 'no'}")
    print(f"contour_files={len(list((result_dir / 'contours').glob('*'))) if (result_dir / 'contours').exists() else 0}")
    print()
    print("chunk\trows\texpected\tpercent\tcomplete\trate_per_s\teta_s\tvalid")
    for status in statuses:
        rate, eta = parse_rate_and_eta(status.last_log)
        print(
            f"{status.chunk:03d}\t{status.rows}\t{status.expected_rows}\t"
            f"{status.percent:.2f}\t{'yes' if status.complete else 'no'}\t"
            f"{rate}\t{eta}\t{'yes' if status.valid else 'no'}"
        )
        if args.validate and not status.valid:
            print(
                f"  validation: header_bad={status.header_bad} "
                f"bad_field_rows={status.bad_field_rows} "
                f"bad_ok_word_rows={status.bad_ok_word_rows}"
            )
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
