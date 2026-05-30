#!/usr/bin/env python3
"""Render Rössler y-minima tangent scan contours as PNGs.

This script intentionally uses only the Python standard library. It keeps the
flow-folding contour export path independent of Julia plotting stacks and their
binary backends.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import csv
import gzip
import math
import os
import struct
import textwrap
import time
import zlib
from array import array
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence


RGB = tuple[int, int, int]
Segment = tuple[float, float, float, float]

BASE_PNG_WIDTH = 1600
BASE_PNG_HEIGHT = 1100

PALETTE = [
    "#b23a2e",
    "#2f66b3",
    "#238b62",
    "#8f4cb3",
    "#c37a1f",
    "#258ea6",
    "#cc4c8a",
    "#59662b",
]

FONT = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "00000", "11110", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


@dataclass(frozen=True)
class ScanData:
    path: str
    c_values: list[float]
    a_values: list[float]
    ok: bytearray
    words: list[str]
    codes: array
    n_symbols: int
    total_points: int
    ok_points: int
    max_time_limited_points: int
    max_time: float
    word_counts: Counter[str]

    @property
    def n_c(self) -> int:
        return len(self.c_values)

    @property
    def n_a(self) -> int:
        return len(self.a_values)

    @property
    def symbol_value_source(self) -> str:
        return "symbol_signs"


@dataclass(frozen=True)
class PlotSpec:
    label: str
    color: RGB
    stroke_width: float
    alpha: float
    segments: Callable[[], Iterator[Segment]]


class Canvas:
    def __init__(self, width: int, height: int, background: RGB = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def blend_pixel(self, x: int, y: int, color: RGB, alpha: float = 1.0) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        offset = 3 * (y * self.width + x)
        if alpha >= 1.0:
            self.pixels[offset : offset + 3] = bytes(color)
            return
        inv = 1.0 - alpha
        self.pixels[offset] = int(self.pixels[offset] * inv + color[0] * alpha)
        self.pixels[offset + 1] = int(self.pixels[offset + 1] * inv + color[1] * alpha)
        self.pixels[offset + 2] = int(self.pixels[offset + 2] * inv + color[2] * alpha)

    def fill_rect(self, x: int, y: int, width: int, height: int, color: RGB, alpha: float = 1.0) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        if x1 <= x0 or y1 <= y0:
            return
        if alpha >= 1.0:
            row = bytes(color) * (x1 - x0)
            for py in range(y0, y1):
                offset = 3 * (py * self.width + x0)
                self.pixels[offset : offset + len(row)] = row
            return
        for py in range(y0, y1):
            for px in range(x0, x1):
                self.blend_pixel(px, py, color, alpha)

    def draw_brush(self, x: int, y: int, radius: int, color: RGB, alpha: float = 1.0) -> None:
        radius = max(0, radius)
        if radius <= 0:
            self.blend_pixel(x, y, color, alpha)
            return
        r2 = radius * radius
        for yy in range(y - radius, y + radius + 1):
            dy = yy - y
            for xx in range(x - radius, x + radius + 1):
                dx = xx - x
                if dx * dx + dy * dy <= r2:
                    self.blend_pixel(xx, yy, color, alpha)

    def draw_thin_line(self, x0: float, y0: float, x1: float, y1: float, color: RGB, width: float, alpha: float) -> None:
        dx = x1 - x0
        dy = y1 - y0
        steps = max(1, int(math.ceil(max(abs(dx), abs(dy)))))
        coverage = max(0.0, min(1.0, width)) * max(0.0, min(1.0, alpha))
        if coverage <= 0.0:
            return
        for step in range(steps + 1):
            theta = step / steps
            self.blend_pixel(int(round(x0 + theta * dx)), int(round(y0 + theta * dy)), color, coverage)

    def draw_line(self, x0: float, y0: float, x1: float, y1: float, color: RGB, width: float = 1.0, alpha: float = 1.0) -> None:
        if width < 1.0:
            self.draw_thin_line(x0, y0, x1, y1, color, width, alpha)
            return
        x0i = int(round(x0))
        y0i = int(round(y0))
        x1i = int(round(x1))
        y1i = int(round(y1))
        dx = abs(x1i - x0i)
        sx = 1 if x0i < x1i else -1
        dy = -abs(y1i - y0i)
        sy = 1 if y0i < y1i else -1
        err = dx + dy
        radius = max(0, int(round(width)) // 2)
        while True:
            self.draw_brush(x0i, y0i, radius, color, alpha)
            if x0i == x1i and y0i == y1i:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0i += sx
            if e2 <= dx:
                err += dx
                y0i += sy

    def draw_text(self, x: int, y: int, text: str, color: RGB, scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT.get(char, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cursor + gx * scale, y + gy * scale, scale, scale, color)
            cursor += 6 * scale

    def text_width(self, text: str, scale: int = 2) -> int:
        return max(0, len(text) * 6 * scale - scale)


def hex_rgb(value: str) -> RGB:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def style_scale(width: int, height: int) -> float:
    return min(width / BASE_PNG_WIDTH, height / BASE_PNG_HEIGHT)


def hsv_rgb(hue: float, saturation: float, value: float) -> RGB:
    hue = hue % 1.0
    chroma = value * saturation
    x = chroma * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
    match_value = value - chroma
    sector = int(hue * 6.0)
    if sector == 0:
        red, green, blue = chroma, x, 0.0
    elif sector == 1:
        red, green, blue = x, chroma, 0.0
    elif sector == 2:
        red, green, blue = 0.0, chroma, x
    elif sector == 3:
        red, green, blue = 0.0, x, chroma
    elif sector == 4:
        red, green, blue = x, 0.0, chroma
    else:
        red, green, blue = chroma, 0.0, x
    return (
        int(round((red + match_value) * 255.0)),
        int(round((green + match_value) * 255.0)),
        int(round((blue + match_value) * 255.0)),
    )


def word_heatmap_color(code: int) -> RGB:
    hue = ((code * 137) % 256) / 256.0
    saturation = 0.55 + 0.28 * ((code & 0x03) / 3.0)
    value = 0.76 + 0.18 * (((code >> 2) & 0x03) / 3.0)
    return hsv_rgb(hue, saturation, value)


def monotone_palette_code(code: int) -> int:
    return 0x80 | code


def monotone_heatmap_color(code: int) -> RGB:
    return word_heatmap_color(monotone_palette_code(code))


def open_scan(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="")


def safe_float(value: str) -> float:
    return float(value) if value else math.nan


def scan_axes(path: str) -> tuple[list[float], list[float], int, int, int, float]:
    c_values: set[float] = set()
    a_values: set[float] = set()
    total = 0
    ok = 0
    n_symbols = 0
    max_time = math.nan
    with open_scan(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total += 1
            c_values.add(float(row["c"]))
            a_values.add(float(row["a"]))
            if row["status"] == "ok":
                ok += 1
                n_symbols = max(n_symbols, len(row["word"]))
            if "max_time" in row:
                max_time = safe_float(row["max_time"])
    return sorted(c_values), sorted(a_values), total, ok, n_symbols, max_time


def load_scan(path: str) -> ScanData:
    c_values, a_values, total, ok_points, n_symbols, max_time = scan_axes(path)
    if total == 0:
        raise SystemExit(f"no scan rows found in {path}")
    if n_symbols == 0:
        raise SystemExit(f"no completed kneading words found in {path}")

    n_c = len(c_values)
    n_a = len(a_values)
    c_lookup = {value: idx for idx, value in enumerate(c_values)}
    a_lookup = {value: idx for idx, value in enumerate(a_values)}
    ok = bytearray(n_c * n_a)
    words = [""] * (n_c * n_a)
    codes = array("i", [-1]) * (n_c * n_a)
    word_counts: Counter[str] = Counter()

    with open_scan(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            c_idx = c_lookup[float(row["c"])]
            a_idx = a_lookup[float(row["a"])]
            idx = a_idx * n_c + c_idx
            if row["status"] != "ok":
                continue
            word = row["word"]
            ok[idx] = 1
            words[idx] = word
            codes[idx] = int(row["code"]) if row.get("code") else word_code(word)
            word_counts[word] += 1

    return ScanData(
        path=path,
        c_values=c_values,
        a_values=a_values,
        ok=ok,
        words=words,
        codes=codes,
        n_symbols=n_symbols,
        total_points=total,
        ok_points=ok_points,
        max_time_limited_points=total - ok_points,
        max_time=max_time,
        word_counts=word_counts,
    )


def word_code(word: str) -> int:
    code = 0
    for char in word:
        code = 2 * code + (1 if char == "1" else 0)
    return code


def monotone_bits(word: str) -> str:
    return "".join("1" if word[idx] == word[idx + 1] else "0" for idx in range(len(word) - 1))


def monotone_code(word: str) -> int:
    return word_code(monotone_bits(word))


def sign_text(bits: str) -> str:
    return " ".join("+" if char == "1" else "-" for char in bits)


def require_eight_symbol_words(data: ScanData) -> None:
    if data.n_symbols != 8:
        raise SystemExit(f"7-bit monotone heatmaps require 8-symbol words; found {data.n_symbols}")


def least_period(word: str) -> int:
    n = len(word)
    if n == 0:
        return 0
    for period in range(1, n + 1):
        if n % period:
            continue
        if all(word[idx] == word[idx % period] for idx in range(n)):
            return period
    return n


def binary_sequence_value(word: str) -> float:
    n = len(word)
    value = 0.0
    for idx, char in enumerate(word):
        if char == "1":
            value += 1.0 / (2.0 ** (n - idx))
    return value


def code_word(code: int, n_symbols: int) -> str:
    return format(code, f"0{n_symbols}b")


def cell_index(data: ScanData, a_idx: int, c_idx: int) -> int:
    return a_idx * data.n_c + c_idx


def symbol_value(data: ScanData, a_idx: int, c_idx: int, symbol_index: int) -> float | None:
    idx = cell_index(data, a_idx, c_idx)
    if not data.ok[idx]:
        return None
    word = data.words[idx]
    if len(word) < symbol_index:
        return None
    return 1.0 if word[symbol_index - 1] == "1" else -1.0


def edge_zero_point(
    edge_id: int,
    values: tuple[float, float, float, float],
    points: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    level: float = 0.0,
) -> tuple[float, float] | None:
    if edge_id == 1:
        z1, z2 = values[0], values[1]
        (x1, y1), (x2, y2) = points[0], points[1]
    elif edge_id == 2:
        z1, z2 = values[1], values[2]
        (x1, y1), (x2, y2) = points[1], points[2]
    elif edge_id == 3:
        z1, z2 = values[2], values[3]
        (x1, y1), (x2, y2) = points[2], points[3]
    else:
        z1, z2 = values[3], values[0]
        (x1, y1), (x2, y2) = points[3], points[0]

    d1 = z1 - level
    d2 = z2 - level
    if not (math.isfinite(d1) and math.isfinite(d2)) or (d1 == 0.0 and d2 == 0.0):
        return None
    if d1 == 0.0:
        theta = 0.0
    elif d2 == 0.0:
        theta = 1.0
    elif (d1 < 0.0) == (d2 < 0.0):
        return None
    else:
        theta = d1 / (d1 - d2)
    return (1.0 - theta) * x1 + theta * x2, (1.0 - theta) * y1 + theta * y2


def iter_symbol_segments(data: ScanData, symbol_index: int) -> Iterator[Segment]:
    for a_idx in range(data.n_a - 1):
        y_t = data.a_values[a_idx]
        y_b = data.a_values[a_idx + 1]
        for c_idx in range(data.n_c - 1):
            z_tl = symbol_value(data, a_idx, c_idx, symbol_index)
            z_tr = symbol_value(data, a_idx, c_idx + 1, symbol_index)
            z_br = symbol_value(data, a_idx + 1, c_idx + 1, symbol_index)
            z_bl = symbol_value(data, a_idx + 1, c_idx, symbol_index)
            if z_tl is None or z_tr is None or z_br is None or z_bl is None:
                continue
            case_idx = (
                (8 if z_tl >= 0.0 else 0)
                + (4 if z_tr >= 0.0 else 0)
                + (2 if z_br >= 0.0 else 0)
                + (1 if z_bl >= 0.0 else 0)
            )
            if case_idx == 0 or case_idx == 15:
                continue
            x_l = data.c_values[c_idx]
            x_r = data.c_values[c_idx + 1]
            points = ((x_l, y_t), (x_r, y_t), (x_r, y_b), (x_l, y_b))
            values = (z_tl, z_tr, z_br, z_bl)
            edge_points = {edge: edge_zero_point(edge, values, points) for edge in range(1, 5)}
            if case_idx in (5, 10):
                center_value = 0.25 * sum(values)
                if case_idx == 5:
                    pairing = ((1, 2), (3, 4)) if center_value >= 0.0 else ((1, 4), (2, 3))
                else:
                    pairing = ((1, 4), (2, 3)) if center_value >= 0.0 else ((1, 2), (3, 4))
            else:
                pairings = {
                    1: ((4, 3),),
                    2: ((3, 2),),
                    3: ((4, 2),),
                    4: ((1, 2),),
                    6: ((1, 3),),
                    7: ((1, 4),),
                    8: ((1, 4),),
                    9: ((1, 3),),
                    11: ((1, 2),),
                    12: ((4, 2),),
                    13: ((3, 2),),
                    14: ((4, 3),),
                }
                pairing = pairings.get(case_idx, ())
            for edge_a, edge_b in pairing:
                p1 = edge_points.get(edge_a)
                p2 = edge_points.get(edge_b)
                if p1 is None or p2 is None:
                    continue
                yield p1[0], p1[1], p2[0], p2[1]


def iter_category_segments(data: ScanData, values: Sequence[int]) -> Iterator[Segment]:
    for a_idx in range(data.n_a - 1):
        y_t = data.a_values[a_idx]
        y_b = data.a_values[a_idx + 1]
        y_m = 0.5 * (y_t + y_b)
        for c_idx in range(data.n_c - 1):
            x_l = data.c_values[c_idx]
            x_r = data.c_values[c_idx + 1]
            x_m = 0.5 * (x_l + x_r)
            z_tl = values[cell_index(data, a_idx, c_idx)]
            z_tr = values[cell_index(data, a_idx, c_idx + 1)]
            z_br = values[cell_index(data, a_idx + 1, c_idx + 1)]
            z_bl = values[cell_index(data, a_idx + 1, c_idx)]

            if z_tl != z_tr and z_tr == z_br and z_br == z_bl:
                yield x_m, y_t, x_l, y_m
            elif z_tr != z_tl and z_tl == z_br and z_br == z_bl:
                yield x_r, y_m, x_m, y_t
            elif z_br != z_tr and z_tr == z_tl and z_tl == z_bl:
                yield x_m, y_b, x_r, y_m
            elif z_bl != z_br and z_br == z_tr and z_tr == z_tl:
                yield x_l, y_m, x_m, y_b
            elif z_tl == z_tr and z_bl == z_br and z_tl != z_bl:
                yield x_l, y_m, x_r, y_m
            elif z_tl == z_bl and z_tr == z_br and z_tl != z_tr:
                yield x_m, y_t, x_m, y_b


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def write_png(path: str, canvas: Canvas) -> None:
    ihdr = struct.pack(">IIBBBBB", canvas.width, canvas.height, 8, 2, 0, 0, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(png_chunk(b"IHDR", ihdr))
        compressor = zlib.compressobj(level=6)
        pending = bytearray()
        stride = canvas.width * 3
        threshold = 1 << 20
        for y in range(canvas.height):
            start = y * stride
            compressed = compressor.compress(b"\x00" + canvas.pixels[start : start + stride])
            if compressed:
                pending.extend(compressed)
                if len(pending) >= threshold:
                    handle.write(png_chunk(b"IDAT", bytes(pending)))
                    pending.clear()
        pending.extend(compressor.flush())
        if pending:
            handle.write(png_chunk(b"IDAT", bytes(pending)))
        handle.write(png_chunk(b"IEND", b""))


def clean_output_dir(output_dir: str, stem: str) -> None:
    if not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if not name.startswith(stem + "_"):
            continue
        if name.endswith((".png", ".svg")):
            os.remove(os.path.join(output_dir, name))


def render_plot(path: str, title: str, data: ScanData, specs: Sequence[PlotSpec], width: int, height: int) -> list[int]:
    canvas = Canvas(width, height)
    scale = style_scale(width, height)

    def sp(value: float) -> int:
        return int(round(value * scale))

    left = sp(96)
    right = sp(300)
    top = sp(76)
    bottom = sp(92)
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width <= 0 or plot_height <= 0:
        raise SystemExit(f"PNG size {width}x{height} is too small for scaled plot layout")
    c_min, c_max = data.c_values[0], data.c_values[-1]
    a_min, a_max = data.a_values[0], data.a_values[-1]

    def xpix(c_value: float) -> float:
        return left + (c_value - c_min) / (c_max - c_min) * plot_width

    def ypix(a_value: float) -> float:
        return top + (a_max - a_value) / (a_max - a_min) * plot_height

    grid_color = hex_rgb("#e1e5e0")
    axis_color = hex_rgb("#17201c")
    text_color = hex_rgb("#59635d")
    title_color = hex_rgb("#17201c")

    title_text_scale = max(1, int(round(3 * scale)))
    tick_text_scale = max(1, int(round(2 * scale)))
    axis_text_scale = max(1, int(round(3 * scale)))
    grid_width = max(1.0, 1.0 * scale)
    axis_width = max(1.0, 2.0 * scale)

    title_width = canvas.text_width(title, scale=title_text_scale)
    canvas.draw_text(max(sp(8), (width - title_width) // 2), sp(26), title, title_color, scale=title_text_scale)

    for idx in range(6):
        c_tick = c_min + (c_max - c_min) * idx / 5
        x = xpix(c_tick)
        canvas.draw_line(x, top, x, top + plot_height, grid_color, width=grid_width)
        label = f"{c_tick:.2f}"
        canvas.draw_text(
            int(x - canvas.text_width(label, scale=tick_text_scale) / 2),
            top + plot_height + sp(20),
            label,
            text_color,
            scale=tick_text_scale,
        )
    for idx in range(6):
        a_tick = a_min + (a_max - a_min) * idx / 5
        y = ypix(a_tick)
        canvas.draw_line(left, y, left + plot_width, y, grid_color, width=grid_width)
        label = f"{a_tick:.3f}"
        canvas.draw_text(
            max(sp(4), left - canvas.text_width(label, scale=tick_text_scale) - sp(14)),
            int(y - sp(7)),
            label,
            text_color,
            scale=tick_text_scale,
        )

    canvas.draw_line(left, top, left + plot_width, top, axis_color, width=axis_width)
    canvas.draw_line(left + plot_width, top, left + plot_width, top + plot_height, axis_color, width=axis_width)
    canvas.draw_line(left + plot_width, top + plot_height, left, top + plot_height, axis_color, width=axis_width)
    canvas.draw_line(left, top + plot_height, left, top, axis_color, width=axis_width)

    counts: list[int] = []
    for spec in specs:
        count = 0
        for x0, y0, x1, y1 in spec.segments():
            canvas.draw_line(
                xpix(x0),
                ypix(y0),
                xpix(x1),
                ypix(y1),
                spec.color,
                width=spec.stroke_width * scale,
                alpha=spec.alpha,
            )
            count += 1
        counts.append(count)

    legend_y = top + sp(16)
    legend_x = left + plot_width + sp(28)
    for spec in specs:
        canvas.draw_line(
            legend_x,
            legend_y,
            legend_x + sp(34),
            legend_y,
            spec.color,
            width=max(1.0, spec.stroke_width * scale),
            alpha=spec.alpha,
        )
        canvas.draw_text(legend_x + sp(48), legend_y - sp(8), spec.label, axis_color, scale=tick_text_scale)
        legend_y += sp(28)

    canvas.draw_text(int(left + plot_width / 2 - sp(5)), height - sp(30), "C", axis_color, scale=axis_text_scale)
    canvas.draw_text(sp(34), int(top + plot_height / 2 - sp(10)), "A", axis_color, scale=axis_text_scale)
    write_png(path, canvas)
    return counts


def parameter_edges(values: Sequence[float]) -> list[float]:
    if len(values) == 1:
        return [values[0] - 0.5, values[0] + 0.5]
    edges = [values[0]]
    edges.extend(0.5 * (values[idx] + values[idx + 1]) for idx in range(len(values) - 1))
    edges.append(values[-1])
    return edges


def render_heatmap_cells(
    canvas: Canvas,
    data: ScanData,
    xpix: Callable[[float], float],
    ypix: Callable[[float], float],
    color_for_index: Callable[[int], bytes],
) -> None:
    c_edges = parameter_edges(data.c_values)
    a_edges = parameter_edges(data.a_values)
    x_bounds = [max(0, min(canvas.width, int(round(xpix(value))))) for value in c_edges]
    y_bounds = [max(0, min(canvas.height, int(round(ypix(value))))) for value in a_edges]
    start_x = x_bounds[0]
    end_x = x_bounds[-1]
    row_length = max(0, end_x - start_x) * 3
    if row_length == 0:
        return

    background = bytes((255, 255, 255))
    for a_idx in range(data.n_a):
        row = bytearray()
        for c_idx in range(data.n_c):
            width = max(0, x_bounds[c_idx + 1] - x_bounds[c_idx])
            if width == 0:
                continue
            idx = cell_index(data, a_idx, c_idx)
            row.extend(color_for_index(idx) * width)
        if len(row) < row_length:
            row.extend(background * ((row_length - len(row)) // 3))
        elif len(row) > row_length:
            del row[row_length:]

        y0 = min(y_bounds[a_idx], y_bounds[a_idx + 1])
        y1 = max(y_bounds[a_idx], y_bounds[a_idx + 1])
        if y1 <= y0:
            y1 = min(canvas.height, y0 + 1)
        for y in range(y0, y1):
            offset = 3 * (y * canvas.width + start_x)
            canvas.pixels[offset : offset + row_length] = row


def render_word_heatmap(path: str, data: ScanData, width: int, height: int) -> None:
    canvas = Canvas(width, height)
    scale = style_scale(width, height)

    def sp(value: float) -> int:
        return int(round(value * scale))

    left = sp(96)
    right = sp(300)
    top = sp(76)
    bottom = sp(92)
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width <= 0 or plot_height <= 0:
        raise SystemExit(f"PNG size {width}x{height} is too small for scaled plot layout")

    c_min, c_max = data.c_values[0], data.c_values[-1]
    a_min, a_max = data.a_values[0], data.a_values[-1]

    def xpix(c_value: float) -> float:
        return left + (c_value - c_min) / (c_max - c_min) * plot_width

    def ypix(a_value: float) -> float:
        return top + (a_max - a_value) / (a_max - a_min) * plot_height

    colors = [bytes(word_heatmap_color(code)) for code in range(1 << data.n_symbols)]
    background = bytes((255, 255, 255))

    def color_for_index(idx: int) -> bytes:
        code = data.codes[idx]
        return colors[code] if code >= 0 else background

    render_heatmap_cells(canvas, data, xpix, ypix, color_for_index)

    axis_color = hex_rgb("#17201c")
    text_color = hex_rgb("#59635d")
    title_color = hex_rgb("#17201c")

    title_text_scale = max(1, int(round(3 * scale)))
    tick_text_scale = max(1, int(round(2 * scale)))
    axis_text_scale = max(1, int(round(3 * scale)))
    axis_width = max(1.0, 2.0 * scale)
    tick_length = sp(12)

    title = f"ROSSLER Y-MIN {data.n_symbols}-BIT WORD HEATMAP"
    title_width = canvas.text_width(title, scale=title_text_scale)
    canvas.draw_text(max(sp(8), (width - title_width) // 2), sp(26), title, title_color, scale=title_text_scale)

    for idx in range(6):
        c_tick = c_min + (c_max - c_min) * idx / 5
        x = xpix(c_tick)
        label = f"{c_tick:.2f}"
        canvas.draw_text(
            int(x - canvas.text_width(label, scale=tick_text_scale) / 2),
            top + plot_height + sp(20),
            label,
            text_color,
            scale=tick_text_scale,
        )
    for idx in range(6):
        a_tick = a_min + (a_max - a_min) * idx / 5
        y = ypix(a_tick)
        label = f"{a_tick:.3f}"
        canvas.draw_text(
            max(sp(4), left - canvas.text_width(label, scale=tick_text_scale) - sp(14)),
            int(y - sp(7)),
            label,
            text_color,
            scale=tick_text_scale,
        )

    canvas.draw_line(left, top, left + plot_width, top, axis_color, width=axis_width)
    canvas.draw_line(left + plot_width, top, left + plot_width, top + plot_height, axis_color, width=axis_width)
    canvas.draw_line(left + plot_width, top + plot_height, left, top + plot_height, axis_color, width=axis_width)
    canvas.draw_line(left, top + plot_height, left, top, axis_color, width=axis_width)
    for idx in range(6):
        c_tick = c_min + (c_max - c_min) * idx / 5
        x = xpix(c_tick)
        canvas.draw_line(x, top + plot_height, x, top + plot_height + tick_length, axis_color, width=axis_width)
    for idx in range(6):
        a_tick = a_min + (a_max - a_min) * idx / 5
        y = ypix(a_tick)
        canvas.draw_line(left - tick_length, y, left, y, axis_color, width=axis_width)

    legend_x = left + plot_width + sp(28)
    canvas.draw_text(legend_x, top + sp(8), "8-BIT WORD", axis_color, scale=tick_text_scale)
    canvas.draw_text(legend_x, top + sp(42), "COLORS", axis_color, scale=tick_text_scale)
    canvas.draw_text(legend_x, top + sp(76), "SEE TSV", axis_color, scale=tick_text_scale)

    swatch_size = max(1, sp(8))
    swatch_gap = max(0, sp(2))
    swatches_per_row = 16
    swatch_start_y = top + sp(122)
    for code, color in enumerate(colors):
        row = code // swatches_per_row
        col = code % swatches_per_row
        canvas.fill_rect(
            legend_x + col * (swatch_size + swatch_gap),
            swatch_start_y + row * (swatch_size + swatch_gap),
            swatch_size,
            swatch_size,
            color,
        )

    canvas.draw_text(int(left + plot_width / 2 - sp(5)), height - sp(30), "C", axis_color, scale=axis_text_scale)
    canvas.draw_text(sp(34), int(top + plot_height / 2 - sp(10)), "A", axis_color, scale=axis_text_scale)
    write_png(path, canvas)


def render_monotone_heatmap(path: str, data: ScanData, width: int, height: int) -> None:
    canvas = Canvas(width, height)
    scale = style_scale(width, height)

    def sp(value: float) -> int:
        return int(round(value * scale))

    left = sp(96)
    right = sp(300)
    top = sp(76)
    bottom = sp(92)
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width <= 0 or plot_height <= 0:
        raise SystemExit(f"PNG size {width}x{height} is too small for scaled plot layout")

    c_min, c_max = data.c_values[0], data.c_values[-1]
    a_min, a_max = data.a_values[0], data.a_values[-1]

    def xpix(c_value: float) -> float:
        return left + (c_value - c_min) / (c_max - c_min) * plot_width

    def ypix(a_value: float) -> float:
        return top + (a_max - a_value) / (a_max - a_min) * plot_height

    sign_count = data.n_symbols - 1
    colors = [bytes(monotone_heatmap_color(code)) for code in range(1 << sign_count)]
    background = bytes((255, 255, 255))

    def color_for_index(idx: int) -> bytes:
        return colors[monotone_code(data.words[idx])] if data.ok[idx] else background

    render_heatmap_cells(canvas, data, xpix, ypix, color_for_index)

    axis_color = hex_rgb("#17201c")
    text_color = hex_rgb("#59635d")
    title_color = hex_rgb("#17201c")

    title_text_scale = max(1, int(round(3 * scale)))
    tick_text_scale = max(1, int(round(2 * scale)))
    axis_text_scale = max(1, int(round(3 * scale)))
    axis_width = max(1.0, 2.0 * scale)
    tick_length = sp(12)

    title = f"ROSSLER Y-MIN {sign_count}-BIT MONOTONE HEATMAP"
    title_width = canvas.text_width(title, scale=title_text_scale)
    canvas.draw_text(max(sp(8), (width - title_width) // 2), sp(26), title, title_color, scale=title_text_scale)

    for idx in range(6):
        c_tick = c_min + (c_max - c_min) * idx / 5
        x = xpix(c_tick)
        label = f"{c_tick:.2f}"
        canvas.draw_text(
            int(x - canvas.text_width(label, scale=tick_text_scale) / 2),
            top + plot_height + sp(20),
            label,
            text_color,
            scale=tick_text_scale,
        )
    for idx in range(6):
        a_tick = a_min + (a_max - a_min) * idx / 5
        y = ypix(a_tick)
        label = f"{a_tick:.3f}"
        canvas.draw_text(
            max(sp(4), left - canvas.text_width(label, scale=tick_text_scale) - sp(14)),
            int(y - sp(7)),
            label,
            text_color,
            scale=tick_text_scale,
        )

    canvas.draw_line(left, top, left + plot_width, top, axis_color, width=axis_width)
    canvas.draw_line(left + plot_width, top, left + plot_width, top + plot_height, axis_color, width=axis_width)
    canvas.draw_line(left + plot_width, top + plot_height, left, top + plot_height, axis_color, width=axis_width)
    canvas.draw_line(left, top + plot_height, left, top, axis_color, width=axis_width)
    for idx in range(6):
        c_tick = c_min + (c_max - c_min) * idx / 5
        x = xpix(c_tick)
        canvas.draw_line(x, top + plot_height, x, top + plot_height + tick_length, axis_color, width=axis_width)
    for idx in range(6):
        a_tick = a_min + (a_max - a_min) * idx / 5
        y = ypix(a_tick)
        canvas.draw_line(left - tick_length, y, left, y, axis_color, width=axis_width)

    legend_x = left + plot_width + sp(28)
    canvas.draw_text(legend_x, top + sp(8), "7-BIT", axis_color, scale=tick_text_scale)
    canvas.draw_text(legend_x, top + sp(42), "MONOTONE", axis_color, scale=tick_text_scale)
    canvas.draw_text(legend_x, top + sp(76), "SEE TSV", axis_color, scale=tick_text_scale)

    swatch_size = max(1, sp(8))
    swatch_gap = max(0, sp(2))
    swatches_per_row = 16
    swatch_start_y = top + sp(122)
    for code, color in enumerate(colors):
        row = code // swatches_per_row
        col = code % swatches_per_row
        canvas.fill_rect(
            legend_x + col * (swatch_size + swatch_gap),
            swatch_start_y + row * (swatch_size + swatch_gap),
            swatch_size,
            swatch_size,
            color,
        )

    canvas.draw_text(int(left + plot_width / 2 - sp(5)), height - sp(30), "C", axis_color, scale=axis_text_scale)
    canvas.draw_text(sp(34), int(top + plot_height / 2 - sp(10)), "A", axis_color, scale=axis_text_scale)
    write_png(path, canvas)


def write_heatmap_legend(path: str, data: ScanData) -> None:
    counts = Counter(code for code in data.codes if code >= 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "code", "red", "green", "blue", "hex", "count"])
        for code in range(1 << data.n_symbols):
            red, green, blue = word_heatmap_color(code)
            writer.writerow([
                code_word(code, data.n_symbols),
                code,
                red,
                green,
                blue,
                f"#{red:02x}{green:02x}{blue:02x}",
                counts.get(code, 0),
            ])


def write_monotone_heatmap_legend(path: str, data: ScanData) -> None:
    sign_count = data.n_symbols - 1
    counts = Counter(monotone_code(word) for word in data.words if word)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "monotone_signs",
            "monotone_bits",
            "monotone_code",
            "palette_code",
            "palette_word",
            "red",
            "green",
            "blue",
            "hex",
            "count",
        ])
        for code in range(1 << sign_count):
            bits = code_word(code, sign_count)
            palette_code = monotone_palette_code(code)
            red, green, blue = monotone_heatmap_color(code)
            writer.writerow([
                sign_text(bits),
                bits,
                code,
                palette_code,
                code_word(palette_code, data.n_symbols),
                red,
                green,
                blue,
                f"#{red:02x}{green:02x}{blue:02x}",
                counts.get(code, 0),
            ])


def wrap_base64(raw: bytes, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(base64.b64encode(raw).decode("ascii"), width))


def byte_arrays_for_probe(data: ScanData) -> tuple[bytes, bytes]:
    code_bytes = bytearray(data.n_c * data.n_a)
    valid_bits = bytearray(math.ceil(len(code_bytes) / 8))
    for idx, code in enumerate(data.codes):
        if code >= 0:
            code_bytes[idx] = code
            valid_bits[idx >> 3] |= 1 << (idx & 7)
    return bytes(code_bytes), bytes(valid_bits)


def write_monotone_probe_html(
    path: str,
    data: ScanData,
    image_path: str,
    width: int,
    height: int,
    color_mode: str = "monotone",
) -> None:
    if color_mode not in {"word", "monotone"}:
        raise SystemExit(f"unsupported probe color mode {color_mode!r}")

    scale = style_scale(width, height)

    def sp(value: float) -> int:
        return int(round(value * scale))

    left = sp(96)
    top = sp(76)
    right = sp(300)
    bottom = sp(92)
    plot_width = width - left - right
    plot_height = height - top - bottom
    view_width = left + plot_width + sp(26)
    sign_count = data.n_symbols - 1
    code_bytes, valid_bits = byte_arrays_for_probe(data)

    with open(image_path, "rb") as handle:
        image_base64 = wrap_base64(handle.read())

    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
  :root {
    color-scheme: light;
    --ink: #17201c;
    --muted: #59635d;
    --line: #cfd6d1;
    --paper: #ffffff;
    --panel: #f6f8f5;
    --accent: #d92845;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    color: var(--ink);
    background: var(--paper);
  }
  main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    min-height: 100vh;
  }
  .stage {
    min-width: 0;
    padding: 16px;
    overflow: auto;
  }
  .image-wrap {
    position: relative;
    width: min(100%, ${view_width}px);
    aspect-ratio: ${view_width} / ${height};
    margin: 0 auto;
    border: 1px solid var(--line);
    background: white;
    overflow: hidden;
  }
  #heatmap {
    display: block;
    width: calc(100% * ${width} / ${view_width});
    height: auto;
    cursor: crosshair;
    touch-action: none;
    user-select: none;
    -webkit-user-drag: none;
  }
  #marker {
    position: absolute;
    width: 25px;
    height: 25px;
    transform: translate(-50%, -50%);
    pointer-events: none;
    display: none;
  }
  #marker::before,
  #marker::after {
    content: "";
    position: absolute;
    background: var(--marker-color, var(--accent));
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.95), 0 0 0 2px rgba(23, 32, 28, 0.35);
  }
  #marker::before {
    left: 12px;
    top: 0;
    width: 1px;
    height: 25px;
  }
  #marker::after {
    left: 0;
    top: 12px;
    width: 25px;
    height: 1px;
  }
  aside {
    border-left: 1px solid var(--line);
    background: var(--panel);
    padding: 18px 16px;
  }
  h1 {
    margin: 0 0 16px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0;
  }
  dl {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 10px 12px;
    margin: 0;
    align-items: baseline;
  }
  dt { color: var(--muted); font-size: 12px; }
  dd { margin: 0; font-size: 13px; text-align: right; }
  .status {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }
  .swatch {
    display: inline-block;
    width: 13px;
    height: 13px;
    border: 1px solid var(--line);
    vertical-align: -2px;
    margin-right: 6px;
    background: white;
  }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    aside { border-left: 0; border-top: 1px solid var(--line); }
    dl { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    dt, dd { text-align: left; }
  }
</style>
</head>
<body>
<main>
  <section class="stage" aria-label="Heatmap image">
    <div class="image-wrap">
      <img id="heatmap" alt="$image_alt" src="data:image/png;base64,$image_base64">
      <div id="marker"></div>
    </div>
  </section>
  <aside>
    <h1>Parameter Probe</h1>
    <dl>
      <dt>c</dt><dd id="cValue">-</dd>
      <dt>a</dt><dd id="aValue">-</dd>
      <dt>pixel x</dt><dd id="xValue">-</dd>
      <dt>pixel y</dt><dd id="yValue">-</dd>
      <dt>grid c</dt><dd id="cIndex">-</dd>
      <dt>grid a</dt><dd id="aIndex">-</dd>
      <dt>code byte</dt><dd id="codeValue">-</dd>
      <dt>word bits</dt><dd id="wordBits">-</dd>
      <dt>symbols</dt><dd id="symbols">-</dd>
      <dt>monotone sign</dt><dd id="monotoneSigns">-</dd>
      <dt>monotone code</dt><dd id="monotoneCode">-</dd>
      <dt>color</dt><dd id="colorValue"><span class="swatch" id="colorSwatch"></span>-</dd>
    </dl>
    <div class="status" id="status">No point selected.</div>
  </aside>
</main>
<script id="codeBytes" type="application/octet-stream">
$code_bytes
</script>
<script id="validBits" type="application/octet-stream">
$valid_bits
</script>
<script>
(() => {
  const IMAGE = { width: ${width}, height: ${height} };
  const VIEW = { width: ${view_width}, height: ${height} };
  const PLOT = { left: ${plot_left}, top: ${plot_top}, width: ${plot_width}, height: ${plot_height} };
  PLOT.right = PLOT.left + PLOT.width;
  PLOT.bottom = PLOT.top + PLOT.height;
  const C = { min: ${c_min}, max: ${c_max}, count: ${c_count} };
  const A = { min: ${a_min}, max: ${a_max}, count: ${a_count} };
  const CELL_COUNT = C.count * A.count;
  const MONOTONE_BITS = ${sign_count};
  const COLOR_MODE = '$color_mode';

  const image = document.getElementById('heatmap');
  const marker = document.getElementById('marker');
  const fields = {
    c: document.getElementById('cValue'),
    a: document.getElementById('aValue'),
    x: document.getElementById('xValue'),
    y: document.getElementById('yValue'),
    ci: document.getElementById('cIndex'),
    ai: document.getElementById('aIndex'),
    code: document.getElementById('codeValue'),
    bits: document.getElementById('wordBits'),
    symbols: document.getElementById('symbols'),
    monotoneSigns: document.getElementById('monotoneSigns'),
    monotoneCode: document.getElementById('monotoneCode'),
    color: document.getElementById('colorValue'),
    swatch: document.getElementById('colorSwatch'),
    status: document.getElementById('status'),
  };

  function decodeByteChars(id, expectedLength) {
    const encoded = document.getElementById(id).textContent.replace(/\s+/g, '');
    const byteChars = atob(encoded);
    if (byteChars.length !== expectedLength) {
      throw new Error(`${id} length ${byteChars.length} did not match expected ${expectedLength}`);
    }
    const bytes = new Uint8Array(byteChars.length);
    for (let idx = 0; idx < byteChars.length; idx += 1) {
      bytes[idx] = byteChars.charCodeAt(idx) & 0xff;
    }
    return bytes;
  }

  const codeBytes = decodeByteChars('codeBytes', CELL_COUNT);
  const validBits = decodeByteChars('validBits', Math.ceil(CELL_COUNT / 8));
  const sampleCanvas = document.createElement('canvas');
  sampleCanvas.width = 1;
  sampleCanvas.height = 1;
  const sampleContext = sampleCanvas.getContext('2d', { willReadFrequently: true });
  let dragging = false;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function setText(element, value) {
    element.textContent = value;
  }

  function isValid(index) {
    return (validBits[index >> 3] & (1 << (index & 7))) !== 0;
  }

  function wordBits(code) {
    return code.toString(2).padStart(8, '0');
  }

  function symbolText(bits) {
    return bits.replace(/1/g, '+').replace(/0/g, '-').split('').join(' ');
  }

  function monotoneBits(bits) {
    let signs = '';
    for (let idx = 0; idx < bits.length - 1; idx += 1) {
      signs += bits[idx] === bits[idx + 1] ? '1' : '0';
    }
    return signs;
  }

  function monotoneSignText(bits) {
    return bits.replace(/1/g, '+').replace(/0/g, '-').split('').join(' ');
  }

  function monotoneCode(bits) {
    let code = 0;
    for (let idx = 0; idx < bits.length; idx += 1) {
      code = 2 * code + (bits[idx] === '1' ? 1 : 0);
    }
    return code;
  }

  function hsvRgb(hue, saturation, value) {
    hue = ((hue % 1) + 1) % 1;
    const chroma = value * saturation;
    const x = chroma * (1 - Math.abs((hue * 6) % 2 - 1));
    const matchValue = value - chroma;
    const sector = Math.floor(hue * 6);
    let red = 0;
    let green = 0;
    let blue = 0;
    if (sector === 0) { [red, green, blue] = [chroma, x, 0]; }
    else if (sector === 1) { [red, green, blue] = [x, chroma, 0]; }
    else if (sector === 2) { [red, green, blue] = [0, chroma, x]; }
    else if (sector === 3) { [red, green, blue] = [0, x, chroma]; }
    else if (sector === 4) { [red, green, blue] = [x, 0, chroma]; }
    else { [red, green, blue] = [chroma, 0, x]; }
    return [red, green, blue].map(channel => Math.round((channel + matchValue) * 255));
  }

  function wordHeatmapColor(code) {
    const hue = ((code * 137) % 256) / 256.0;
    const saturation = 0.55 + 0.28 * ((code & 0x03) / 3.0);
    const value = 0.76 + 0.18 * (((code >> 2) & 0x03) / 3.0);
    return hsvRgb(hue, saturation, value);
  }

  function monotonePaletteCode(code) {
    return 0x80 | code;
  }

  function monotoneHeatmapColor(code) {
    return wordHeatmapColor(monotonePaletteCode(code));
  }

  function rgbHex(rgb) {
    return '#' + rgb.map(value => value.toString(16).padStart(2, '0')).join('');
  }

  function sampleImageColor(pixelX, pixelY) {
    if (!sampleContext) {
      return '#d92845';
    }
    const sx = clamp(Math.floor(pixelX), 0, IMAGE.width - 1);
    const sy = clamp(Math.floor(pixelY), 0, IMAGE.height - 1);
    try {
      sampleContext.clearRect(0, 0, 1, 1);
      sampleContext.drawImage(image, sx, sy, 1, 1, 0, 0, 1, 1);
      const pixel = sampleContext.getImageData(0, 0, 1, 1).data;
      return rgbHex([pixel[0], pixel[1], pixel[2]]);
    } catch {
      return '#d92845';
    }
  }

  function update(event) {
    event.preventDefault();
    const rect = image.getBoundingClientRect();
    const displayX = event.clientX - rect.left;
    const displayY = event.clientY - rect.top;
    const pixelX = displayX * IMAGE.width / rect.width;
    const pixelY = displayY * IMAGE.height / rect.height;

    marker.style.left = `${displayX}px`;
    marker.style.top = `${displayY}px`;
    marker.style.display = 'block';
    marker.style.setProperty('--marker-color', sampleImageColor(pixelX, pixelY));

    setText(fields.x, pixelX.toFixed(1));
    setText(fields.y, pixelY.toFixed(1));

    const inside = pixelX >= PLOT.left && pixelX <= PLOT.right && pixelY >= PLOT.top && pixelY <= PLOT.bottom;
    const plotX = clamp(pixelX, PLOT.left, PLOT.right) - PLOT.left;
    const plotY = clamp(pixelY, PLOT.top, PLOT.bottom) - PLOT.top;
    const c = C.min + plotX / PLOT.width * (C.max - C.min);
    const a = A.max - plotY / PLOT.height * (A.max - A.min);
    const cIndex = clamp(Math.round((c - C.min) / (C.max - C.min) * (C.count - 1)), 0, C.count - 1);
    const aIndex = clamp(Math.round((a - A.min) / (A.max - A.min) * (A.count - 1)), 0, A.count - 1);
    const cellIndex = aIndex * C.count + cIndex;

    setText(fields.c, c.toFixed(10));
    setText(fields.a, a.toFixed(10));
    setText(fields.ci, String(cIndex + 1));
    setText(fields.ai, String(aIndex + 1));

    if (isValid(cellIndex)) {
      const code = codeBytes[cellIndex];
      const bits = wordBits(code);
      const monoBits = monotoneBits(bits);
      const monoCode = monotoneCode(monoBits);
      const hex = rgbHex(COLOR_MODE === 'word' ? wordHeatmapColor(code) : monotoneHeatmapColor(monoCode));
      marker.style.setProperty('--marker-color', hex);
      setText(fields.code, `${code} / 0x${code.toString(16).padStart(2, '0').toUpperCase()}`);
      setText(fields.bits, bits);
      setText(fields.symbols, symbolText(bits));
      setText(fields.monotoneSigns, monotoneSignText(monoBits));
      setText(fields.monotoneCode, `${monoCode} / 0x${monoCode.toString(16).padStart(2, '0').toUpperCase()}`);
      fields.swatch.style.backgroundColor = hex;
      fields.color.lastChild.textContent = hex;
      fields.status.textContent = inside ? 'Inside plot area.' : 'Outside plot area; values are clamped to the nearest plot edge.';
    } else {
      setText(fields.code, '-');
      setText(fields.bits, '-');
      setText(fields.symbols, '-');
      setText(fields.monotoneSigns, '-');
      setText(fields.monotoneCode, '-');
      fields.swatch.style.backgroundColor = 'white';
      fields.color.lastChild.textContent = '-';
      fields.status.textContent = inside ? 'No completed 8-symbol word at the nearest grid point.' : 'Outside plot area; values are clamped to the nearest plot edge.';
    }
  }

  image.addEventListener('pointerdown', event => {
    dragging = true;
    image.setPointerCapture(event.pointerId);
    update(event);
  });

  image.addEventListener('pointermove', event => {
    if (dragging) {
      update(event);
    }
  });

  image.addEventListener('pointerup', event => {
    dragging = false;
    if (image.hasPointerCapture(event.pointerId)) {
      image.releasePointerCapture(event.pointerId);
    }
  });

  image.addEventListener('pointercancel', event => {
    dragging = false;
    if (image.hasPointerCapture(event.pointerId)) {
      image.releasePointerCapture(event.pointerId);
    }
  });
})();
</script>
</body>
</html>
"""
    values = {
        "title": (
            "Rossler Y-Min 8-Bit Word Heatmap Probe"
            if color_mode == "word"
            else "Rossler Y-Min 7-Bit Monotone Heatmap Probe"
        ),
        "image_alt": (
            "Rossler y-minima 8-bit word heatmap"
            if color_mode == "word"
            else "Rossler y-minima 7-bit monotone heatmap"
        ),
        "image_base64": image_base64,
        "code_bytes": wrap_base64(code_bytes),
        "valid_bits": wrap_base64(valid_bits),
        "width": width,
        "height": height,
        "view_width": view_width,
        "plot_left": left,
        "plot_top": top,
        "plot_width": plot_width,
        "plot_height": plot_height,
        "c_min": repr(data.c_values[0]),
        "c_max": repr(data.c_values[-1]),
        "c_count": data.n_c,
        "a_min": repr(data.a_values[0]),
        "a_max": repr(data.a_values[-1]),
        "a_count": data.n_a,
        "sign_count": sign_count,
        "color_mode": color_mode,
    }
    html = template
    for key, value in values.items():
        html = html.replace("${" + key + "}", str(value)).replace("$" + key, str(value))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write(html)


def write_word_legend(path: str, data: ScanData) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["word", "count", "code", "period", "gamma"])
        for word, count in sorted(data.word_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([word, count, word_code(word), least_period(word), f"{binary_sequence_value(word):.12g}"])


def write_summary(
    path: str,
    data: ScanData,
    symbol_counts: Sequence[int],
    results_path: str,
    output_dir: str,
    scan_seconds: float,
    write_tsv_seconds: float,
    contour_generation_seconds: float,
    width: int,
    height: int,
    render_style_scale: float,
    line_width_scale: float,
    alpha: float,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        def write(key: str, value) -> None:
            handle.write(f"{key}\t{value}\n")

        write("results_path", results_path)
        write("output_dir", output_dir)
        write("image_format", "png")
        write("png_width", width)
        write("png_height", height)
        write("style_scale", f"{render_style_scale:.12g}")
        write("line_width_scale", line_width_scale)
        write("contour_alpha", alpha)
        write("symbol_value_source", data.symbol_value_source)
        write("all_symbol_draw_order", ",".join(str(idx) for idx in range(data.n_symbols, 0, -1)))
        write("total_points", data.total_points)
        write("ok_points", data.ok_points)
        write("max_time_limited_points", data.max_time_limited_points)
        write("word_length", data.n_symbols)
        write("max_time", data.max_time)
        write("scan_seconds", scan_seconds)
        write("write_tsv_seconds", write_tsv_seconds)
        write("contour_generation_seconds", contour_generation_seconds)
        for idx, count in enumerate(symbol_counts, start=1):
            write(f"symbol{idx:02d}_segments", count)


def render_all(data: ScanData, args: argparse.Namespace) -> str:
    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.clean:
        clean_output_dir(args.output_dir, args.stem)

    symbol_counts: list[int] = []
    for symbol_index in range(1, data.n_symbols + 1):
        color = hex_rgb(PALETTE[(symbol_index - 1) % len(PALETTE)])
        specs = [
            PlotSpec(
                label=f"SYMBOL {symbol_index:02d}",
                color=color,
                stroke_width=3.0 * args.line_width_scale,
                alpha=args.alpha,
                segments=lambda symbol_index=symbol_index: iter_symbol_segments(data, symbol_index),
            )
        ]
        path = os.path.join(args.output_dir, f"{args.stem}_symbol{symbol_index:02d}_contours.png")
        before = time.time()
        counts = render_plot(path, f"ROSSLER Y-MIN SYMBOL {symbol_index:02d}", data, specs, args.width, args.height)
        symbol_counts.append(counts[0])
        print(f"wrote {path} segments={counts[0]} seconds={time.time() - before:.3f}", flush=True)

    all_symbol_order = list(range(data.n_symbols, 0, -1))
    all_specs = [
        PlotSpec(
            label=f"SYMBOL {symbol_index:02d}",
            color=hex_rgb(PALETTE[(symbol_index - 1) % len(PALETTE)]),
            stroke_width=2.0 * args.line_width_scale,
            alpha=args.alpha,
            segments=lambda symbol_index=symbol_index: iter_symbol_segments(data, symbol_index),
        )
        for symbol_index in all_symbol_order
    ]
    path = os.path.join(args.output_dir, f"{args.stem}_all_symbol_contours.png")
    before = time.time()
    render_plot(path, "ROSSLER Y-MIN SYMBOL CONTOURS", data, all_specs, args.width, args.height)
    print(f"wrote {path} seconds={time.time() - before:.3f}", flush=True)

    path = os.path.join(args.output_dir, f"{args.stem}_word_boundary_contours.png")
    before = time.time()
    render_plot(
        path,
        "ROSSLER Y-MIN WORD BOUNDARIES",
        data,
        [
            PlotSpec(
                label=f"{data.n_symbols}-SYMBOL WORD",
                color=hex_rgb("#111111"),
                stroke_width=2.0 * args.line_width_scale,
                alpha=args.alpha,
                segments=lambda: iter_category_segments(data, data.codes),
            )
        ],
        args.width,
        args.height,
    )
    print(f"wrote {path} seconds={time.time() - before:.3f}", flush=True)

    heatmap_path = os.path.join(args.output_dir, f"{args.stem}_8bit_word_heatmap.png")
    before = time.time()
    render_word_heatmap(heatmap_path, data, args.width, args.height)
    print(f"wrote {heatmap_path} seconds={time.time() - before:.3f}", flush=True)
    write_heatmap_legend(os.path.join(args.output_dir, f"{args.stem}_8bit_word_heatmap_legend.tsv"), data)

    write_word_legend(os.path.join(args.output_dir, f"{args.stem}_word_legend.tsv"), data)
    write_summary(
        os.path.join(args.output_dir, f"{args.stem}_contour_summary.tsv"),
        data,
        symbol_counts,
        results_path=data.path,
        output_dir=args.output_dir,
        scan_seconds=args.scan_seconds,
        write_tsv_seconds=args.write_tsv_seconds,
        contour_generation_seconds=time.time() - started,
        width=args.width,
        height=args.height,
        render_style_scale=style_scale(args.width, args.height),
        line_width_scale=args.line_width_scale,
        alpha=args.alpha,
    )
    return args.output_dir


def render_heatmap_only(data: ScanData, args: argparse.Namespace) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{args.stem}_8bit_word_heatmap.png")
    started = time.time()
    render_word_heatmap(path, data, args.width, args.height)
    write_heatmap_legend(os.path.join(args.output_dir, f"{args.stem}_8bit_word_heatmap_legend.tsv"), data)
    write_word_legend(os.path.join(args.output_dir, f"{args.stem}_word_legend.tsv"), data)
    if args.write_heatmap_probe:
        html_path = os.path.join(args.output_dir, f"{args.stem}_8bit_word_heatmap_probe.html")
        write_monotone_probe_html(html_path, data, path, args.width, args.height, color_mode="word")
        print(f"wrote {html_path}", flush=True)
    print(f"wrote {path} seconds={time.time() - started:.3f}", flush=True)
    return args.output_dir


def render_monotone_heatmap_only(data: ScanData, args: argparse.Namespace) -> str:
    require_eight_symbol_words(data)
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.time()
    path = os.path.join(args.output_dir, f"{args.stem}_7bit_monotone_heatmap.png")
    render_monotone_heatmap(path, data, args.width, args.height)
    write_monotone_heatmap_legend(os.path.join(args.output_dir, f"{args.stem}_7bit_monotone_heatmap_legend.tsv"), data)
    if args.write_monotone_probe:
        html_path = os.path.join(args.output_dir, f"{args.stem}_7bit_monotone_heatmap_probe.html")
        write_monotone_probe_html(html_path, data, path, args.width, args.height)
        print(f"wrote {html_path}", flush=True)
    print(f"wrote {path} seconds={time.time() - started:.3f}", flush=True)
    return args.output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path", help="scan TSV or TSV.GZ to render")
    parser.add_argument("--output-dir", required=True, help="directory for PNGs and summaries")
    parser.add_argument("--stem", default="coarse_scan", help="output filename stem")
    parser.add_argument("--scan-seconds", type=float, default=math.nan)
    parser.add_argument("--write-tsv-seconds", type=float, default=math.nan)
    parser.add_argument("--width", type=int, default=6400)
    parser.add_argument("--height", type=int, default=4400)
    parser.add_argument("--line-width-scale", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--clean", action="store_true", help="remove existing stem-matching PNG/SVG files first")
    parser.add_argument("--only-heatmap", action="store_true", help="write only the full-word heatmap PNG and legend TSV")
    parser.add_argument("--only-monotone-heatmap", action="store_true", help="write only the 7-bit monotone-sign heatmap PNG and legend TSV")
    parser.add_argument("--write-heatmap-probe", action="store_true", help="also write a standalone HTML probe for --only-heatmap")
    parser.add_argument("--write-monotone-probe", action="store_true", help="also write a standalone HTML probe for --only-monotone-heatmap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    data = load_scan(args.results_path)
    print(
        f"loaded {args.results_path}: points={data.total_points} ok={data.ok_points} "
        f"grid={data.n_c}x{data.n_a} word_length={data.n_symbols} "
        f"symbol_values={data.symbol_value_source} seconds={time.time() - started:.3f}",
        flush=True,
    )
    if args.only_heatmap:
        written = render_heatmap_only(data, args)
        print(f"wrote PNG heatmap in {written}", flush=True)
    elif args.only_monotone_heatmap:
        written = render_monotone_heatmap_only(data, args)
        print(f"wrote monotone PNG heatmap in {written}", flush=True)
    else:
        written = render_all(data, args)
        print(f"wrote PNG contours in {written}", flush=True)


if __name__ == "__main__":
    main()
