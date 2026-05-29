#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import html
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "gcs_results"
COLUMNS = RESULTS / "grid1000_seq12_tmax1e5_prefixes_remap40_newmodel_columns"
IMAGE = RESULTS / "grid1000_seq12_tmax1e5_prefixcompatible_tzero2to12_matcont_overlay_signonly_stripes_contours.png"
OUTPUT = RESULTS / "grid1000_seq12_tmax1e5_prefixcompatible_tzero2to12_interactive_sscs.html"

NX = 1000
NY = 1000
X_MIN = -45.0
X_MAX = -20.0
Y_MIN = -1.5
Y_MAX = -0.5

# Measured from the 3200 x 2400 Makie raster. These are image-pixel coordinates
# of the actual data rectangle, excluding axis labels and tick labels.
PANEL = {
    "left": 244.0,
    "right": 3020.0,
    "top": 39.0,
    "bottom": 2194.0,
}


def b64_bytes(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return "\n".join(encoded[i : i + 96] for i in range(0, len(encoded), 96))


def b64_file(path: Path) -> str:
    return b64_bytes(path.read_bytes())


def b64_gzip(data: bytes) -> str:
    return b64_bytes(gzip.compress(data, compresslevel=9))


def dict_indexer() -> tuple[list[str], dict[str, int]]:
    values = [""]
    index = {"": 0}
    return values, index


def get_index(value: str, values: list[str], index: dict[str, int]) -> int:
    found = index.get(value)
    if found is not None:
        return found
    found = len(values)
    values.append(value)
    index[value] = found
    return found


def build_lookup() -> tuple[list[str], list[str], bytes, bytes]:
    t_values, t_index = dict_indexer()
    g_values, g_index = dict_indexer()
    t_indices = bytearray(NX * NY * 4)
    g_indices = bytearray(NX * NY * 4)

    column_paths = sorted(COLUMNS.glob("column_*.tsv"))
    if len(column_paths) != NY:
        raise RuntimeError(f"Expected {NY} column files, found {len(column_paths)} in {COLUMNS}")

    for col, path in enumerate(column_paths):
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            t_col = header.index("T_scs")
            g_col = header.index("gamma_scs")
            for row, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                offset = (row * NY + col) * 4
                struct.pack_into("<I", t_indices, offset, get_index(parts[t_col], t_values, t_index))
                struct.pack_into("<I", g_indices, offset, get_index(parts[g_col], g_values, g_index))

        if col % 100 == 99:
            print(f"packed {col + 1}/{NY} columns")

    return t_values, g_values, bytes(t_indices), bytes(g_indices)


def main() -> None:
    t_values, g_values, t_indices, g_indices = build_lookup()

    metadata = {
        "nx": NX,
        "ny": NY,
        "xMin": X_MIN,
        "xMax": X_MAX,
        "yMin": Y_MIN,
        "yMax": Y_MAX,
        "panel": PANEL,
        "imageWidth": 3200,
        "imageHeight": 2400,
        "tDictionarySize": len(t_values),
        "gammaDictionarySize": len(g_values),
        "sourceImage": IMAGE.name,
        "sourceColumns": COLUMNS.name,
    }

    image_b64 = b64_file(IMAGE)
    t_dict_b64 = b64_gzip("\n".join(t_values).encode("utf-8"))
    g_dict_b64 = b64_gzip("\n".join(g_values).encode("utf-8"))
    t_index_b64 = b64_gzip(t_indices)
    g_index_b64 = b64_gzip(g_indices)

    template = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>SiN Symbolic Explorer</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #171717;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: max(12px, env(safe-area-inset-top)) max(12px, env(safe-area-inset-right)) max(16px, env(safe-area-inset-bottom)) max(12px, env(safe-area-inset-left));
      min-height: 100vh;
    }}
    .page {{
      width: min(100%, 1180px);
      margin: 0 auto;
      display: grid;
      gap: 10px;
    }}
    .plot-card {{
      background: #fff;
      border: 1px solid #d6d6d0;
      border-radius: 8px;
      padding: 8px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }}
    .plot-title {{
      margin: 4px 0 8px;
      text-align: center;
      font-size: clamp(22px, 3.5vw, 38px);
      font-weight: 650;
      line-height: 1.1;
      letter-spacing: 0;
    }}
    .plot-wrap {{
      position: relative;
      width: 100%;
      touch-action: manipulation;
      user-select: none;
      -webkit-user-select: none;
    }}
    #plot {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 4px;
    }}
    #probe {{
      position: absolute;
      width: 13px;
      height: 13px;
      margin-left: -6.5px;
      margin-top: -6.5px;
      border-radius: 999px;
      border: 2px solid #111;
      background: #fff;
      box-shadow: 0 0 0 2px rgba(255,255,255,0.8), 0 2px 10px rgba(0,0,0,0.35);
      pointer-events: none;
      transform: translate3d(0,0,0);
      display: none;
    }}
    .readout {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      background: #171717;
      color: #f7f7f4;
      border-radius: 8px;
      padding: 10px;
      font-size: 14px;
      line-height: 1.25;
    }}
    .cell {{
      min-width: 0;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 6px;
      padding: 8px;
    }}
    .label {{
      display: block;
      color: #b9b9b2;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 3px;
    }}
    .value {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: normal;
      overflow-wrap: anywhere;
    }}
    .status {{
      font-size: 13px;
      color: #4a4a45;
      padding: 0 2px;
    }}
    @media (max-width: 720px) {{
      body {{
        padding-left: max(8px, env(safe-area-inset-left));
        padding-right: max(8px, env(safe-area-inset-right));
      }}
      .plot-card {{
        padding: 5px;
        border-radius: 6px;
      }}
      .readout {{
        grid-template-columns: 1fr 1fr;
        font-size: 13px;
        gap: 6px;
        padding: 8px;
      }}
      .cell {{
        padding: 7px;
      }}
      .wide {{
        grid-column: 1 / -1;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="plot-card">
      <h1 class="plot-title">SiN Symbolic Explorer</h1>
      <div id="plotWrap" class="plot-wrap" aria-label="Interactive SSCS contour plot">
        <img id="plot" alt="SSCS contours with Hopf, HomSF, and Shilnikov-Hopf overlays" draggable="false" src="data:image/png;base64,{image_b64}">
        <div id="probe"></div>
      </div>
    </section>
    <section id="readout" class="readout" aria-live="polite">
      <div class="cell"><span class="label">ΔCa</span><span id="caValue" class="value">—</span></div>
      <div class="cell"><span class="label">Δx</span><span id="dxValue" class="value">—</span></div>
      <div class="cell wide"><span class="label">T SSCS</span><span id="tValue" class="value">—</span></div>
      <div class="cell wide"><span class="label">Gamma SSCS</span><span id="gValue" class="value">—</span></div>
    </section>
    <div id="status" class="status">Click or tap inside the plotted axes to select a point.</div>
  </main>

  <script>
    const META = {json.dumps(metadata, separators=(",", ":"))};
    const T_DICT_GZ_B64 = `{t_dict_b64}`;
    const G_DICT_GZ_B64 = `{g_dict_b64}`;
    const T_INDEX_GZ_B64 = `{t_index_b64}`;
    const G_INDEX_GZ_B64 = `{g_index_b64}`;

    function cleanBase64(s) {{
      return s.replace(/\\s+/g, "");
    }}

    function decodeBase64Bytes(s) {{
      const binary = atob(cleanBase64(s));
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }}

    async function gunzipBase64Bytes(s) {{
      if (!("DecompressionStream" in window)) {{
        throw new Error("This browser does not support DecompressionStream for the standalone compressed lookup tables.");
      }}
      const compressed = decodeBase64Bytes(s);
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
      return new Uint8Array(await new Response(stream).arrayBuffer());
    }}

    async function decodeTextDictionary(s) {{
      return new TextDecoder("utf-8").decode(await gunzipBase64Bytes(s)).split("\\n");
    }}

    async function decodeUint32Map(s) {{
      const bytes = await gunzipBase64Bytes(s);
      return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    }}

    let T_DICT = [];
    let G_DICT = [];
    let T_INDEX = new Uint32Array();
    let G_INDEX = new Uint32Array();
    let lookupReady = false;

    const plot = document.getElementById("plot");
    const wrap = document.getElementById("plotWrap");
    const probe = document.getElementById("probe");
    const statusEl = document.getElementById("status");
    const caValue = document.getElementById("caValue");
    const dxValue = document.getElementById("dxValue");
    const tValue = document.getElementById("tValue");
    const gValue = document.getElementById("gValue");

    function clamp(x, lo, hi) {{
      return Math.min(hi, Math.max(lo, x));
    }}

    function formatSeq(s) {{
      if (!s) return "—";
      return "[" + s.split(",").join(", ") + "]";
    }}

    function lookupAtImagePixel(px, py) {{
      if (!lookupReady) return null;
      const p = META.panel;
      if (px < p.left || px > p.right || py < p.top || py > p.bottom) return null;

      const xFrac = clamp((px - p.left) / (p.right - p.left), 0, 1);
      const yFrac = clamp((py - p.top) / (p.bottom - p.top), 0, 1);
      const ca = META.xMin + xFrac * (META.xMax - META.xMin);
      const dx = META.yMax - yFrac * (META.yMax - META.yMin);
      const col = clamp(Math.round((ca - META.xMin) / (META.xMax - META.xMin) * (META.ny - 1)), 0, META.ny - 1);
      const row = clamp(Math.round((dx - META.yMin) / (META.yMax - META.yMin) * (META.nx - 1)), 0, META.nx - 1);
      const offset = row * META.ny + col;
      const gridCa = META.xMin + col / (META.ny - 1) * (META.xMax - META.xMin);
      const gridDx = META.yMin + row / (META.nx - 1) * (META.yMax - META.yMin);
      return {{
        px, py, col, row,
        ca: gridCa,
        dx: gridDx,
        t: T_DICT[T_INDEX[offset]] || "",
        g: G_DICT[G_INDEX[offset]] || "",
      }};
    }}

    function updateFromPointer(ev) {{
      if (!lookupReady) {{
        statusEl.textContent = "Loading embedded SSCS lookup tables...";
        return;
      }}
      const rect = plot.getBoundingClientRect();
      const relX = (ev.clientX - rect.left) * META.imageWidth / rect.width;
      const relY = (ev.clientY - rect.top) * META.imageHeight / rect.height;
      const hit = lookupAtImagePixel(relX, relY);
      if (!hit) {{
        probe.style.display = "none";
        statusEl.textContent = "Move inside the plotted axes.";
        return;
      }}
      probe.style.display = "block";
      probe.style.left = `${{(hit.px / META.imageWidth) * 100}}%`;
      probe.style.top = `${{(hit.py / META.imageHeight) * 100}}%`;
      caValue.textContent = hit.ca.toFixed(6);
      dxValue.textContent = hit.dx.toFixed(6);
      tValue.textContent = formatSeq(hit.t);
      gValue.textContent = formatSeq(hit.g);
      statusEl.textContent = `grid column ${{hit.col + 1}} / ${{META.ny}}, row ${{hit.row + 1}} / ${{META.nx}}`;
    }}

    wrap.addEventListener("pointerdown", (ev) => {{
      wrap.setPointerCapture(ev.pointerId);
      updateFromPointer(ev);
    }});

    async function initializeLookup() {{
      try {{
        statusEl.textContent = "Loading embedded SSCS lookup tables...";
        [T_DICT, G_DICT, T_INDEX, G_INDEX] = await Promise.all([
          decodeTextDictionary(T_DICT_GZ_B64),
          decodeTextDictionary(G_DICT_GZ_B64),
          decodeUint32Map(T_INDEX_GZ_B64),
          decodeUint32Map(G_INDEX_GZ_B64),
        ]);
        lookupReady = true;
        statusEl.textContent = "Click or tap inside the plotted axes to select a point.";
      }} catch (err) {{
        statusEl.textContent = err.message;
      }}
    }}

    initializeLookup();
  </script>
</body>
</html>
"""

    OUTPUT.write_text(template, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"T dictionary: {len(t_values)} entries; Gamma dictionary: {len(g_values)} entries")
    print(f"HTML size: {OUTPUT.stat().st_size / (1024 * 1024):.1f} MiB")


if __name__ == "__main__":
    main()
