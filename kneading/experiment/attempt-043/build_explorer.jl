using Pkg

const ATTEMPT043_ROOT = @__DIR__
const REPO_ROOT_043 = normpath(joinpath(ATTEMPT043_ROOT, "..", "..", ".."))
Pkg.activate(REPO_ROOT_043)

using Base64
using Printf

module Attempt027
include(normpath(joinpath(@__DIR__, "..", "attempt-027", "contours.jl")))
end

const A27 = Attempt027
const OUTPUT_TAG_043 = get(
    ENV,
    "ATTEMPT043_OUTPUT_TAG",
    "grid2000_branch16_absxskip16_plot8_deltatfix_nominal_iterates2_8_black_red_retired_explorer_shimizu_morioka_cpu",
)
const HTML_PATH_043 = joinpath(ATTEMPT043_ROOT, "$(OUTPUT_TAG_043).html")
const STATS_PATH_043 = joinpath(ATTEMPT043_ROOT, "$(OUTPUT_TAG_043)_iterate_stats.tsv")

@inline sign_code_043(value::Float64) = value > 0.0 ? UInt16(0x2) : value < 0.0 ? UInt16(0x1) : UInt16(0x0)

function pack_sign_word_043(result::A27.SMAbsXResult25)
    word = UInt16(0)
    max_iter = min(result.absxmax_count, 8, length(result.absxmax_dot_values))
    for nominal_iterate in 2:8
        code = nominal_iterate <= max_iter ? sign_code_043(result.absxmax_dot_values[nominal_iterate]) : UInt16(0)
        word |= code << (2 * (nominal_iterate - 2))
    end
    return word
end

function build_sign_words_043()
    n_alpha = length(A27.ALPHAS_025)
    n_lambda = length(A27.LAMBDAS_025)
    sign_words = fill(UInt16(0), n_alpha * n_lambda)

    for col_idx in eachindex(A27.ALPHAS_025)
        path = A27.column_path_025(col_idx)
        A27.row_is_complete_025(path, n_lambda) || error("Missing or incomplete column file: $(path)")
        open(path, "r") do io
            readline(io)
            row_idx = 0
            for line in eachline(io)
                row_idx += 1
                result = A27.parse_result_025(split(line, '\t'))
                linear_idx = (row_idx - 1) * n_alpha + col_idx
                sign_words[linear_idx] = pack_sign_word_043(result)
            end
            row_idx == n_lambda || error("Column $(col_idx) ended at $(row_idx) rows, expected $(n_lambda)")
        end
    end

    return sign_words
end

function collect_retired_overlay_segments_043()
    dot_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = A27.cumulative_to_interval_grids_025(cumulative_time_grids)

    skip_state = A27.initialize_skip_state_025()
    retired_cells = falses(length(A27.LAMBDAS_025) - 1, length(A27.ALPHAS_025) - 1)
    accepted_segments = NTuple{4, Float64}[]
    excluded_segments = NTuple{4, Float64}[]
    iterate_stats = [A27.zero_iterate_stats_025() for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]

    for nominal_iterate in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP
        local_accepted =
            nominal_iterate >= 2 && nominal_iterate <= 8 ?
            accepted_segments :
            nothing
        local_excluded =
            nominal_iterate >= 2 && nominal_iterate <= 8 ?
            NTuple{4, Float64}[] :
            nothing

        stats = A27.process_nominal_iterate_027(
            nominal_iterate,
            dot_grids,
            time_grids,
            skip_state,
            retired_cells,
            local_accepted,
            nothing,
            local_excluded,
        )
        iterate_stats[nominal_iterate] = stats
        local_excluded !== nothing && append!(excluded_segments, local_excluded)
    end

    return accepted_segments, excluded_segments, iterate_stats
end

function flatten_segments_043(segments::Vector{NTuple{4, Float64}})
    flat = Vector{Float32}(undef, 4 * length(segments))
    idx = 1
    @inbounds for (x1, y1, x2, y2) in segments
        flat[idx] = Float32(x1); idx += 1
        flat[idx] = Float32(y1); idx += 1
        flat[idx] = Float32(x2); idx += 1
        flat[idx] = Float32(y2); idx += 1
    end
    return flat
end

function base64_bytes_043(values::AbstractVector{T}) where {T}
    io = IOBuffer()
    write(io, reinterpret(UInt8, values))
    return base64encode(take!(io))
end

function write_iterate_stats_043(path::String, stats)
    open(path, "w") do io
        println(io, "nominal_iterate\tmissing_data_squares\tconstant_sign_squares\tincremented_squares\tcontoured_squares\temitted_segments")
        for iterate in 1:length(stats)
            stat = stats[iterate]
            println(
                io,
                join([
                    string(iterate),
                    string(stat.missing_data),
                    string(stat.constant_sign),
                    string(stat.incremented),
                    string(stat.contoured_squares),
                    string(stat.emitted_segments),
                ], '\t'),
            )
        end
    end
end

function write_html_043(
    path::String,
    black_segments_b64::String,
    red_segments_b64::String,
    sign_words_b64::String,
)
    n_alpha = length(A27.ALPHAS_025)
    n_lambda = length(A27.LAMBDAS_025)
    alpha_min = A27.ATTEMPT025_ALPHA_MIN
    alpha_max = A27.ATTEMPT025_ALPHA_MAX
    lambda_min = A27.ATTEMPT025_LAMBDA_MIN
    lambda_max = A27.ATTEMPT025_LAMBDA_MAX

    open(path, "w") do io
        print(io, """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shimizu-Morioka Contour Explorer</title>
  <style>
    :root {
      color-scheme: light;
      --panel: #f3f4f6;
      --ink: #111111;
      --muted: #5f6368;
      --accent: #0f766e;
      --skip: #b91c1c;
      --border: #d0d7de;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: #ffffff; color: var(--ink); font-family: Menlo, Consolas, monospace; }
    #app { display: flex; height: 100%; width: 100%; }
    #viewerPane { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; }
    #toolbar {
      display: flex; align-items: center; gap: 12px; padding: 10px 14px;
      border-bottom: 1px solid var(--border); background: #fafafa; color: var(--ink);
    }
    #toolbar button {
      border: 1px solid var(--border); background: white; color: var(--ink);
      padding: 6px 10px; border-radius: 6px; cursor: pointer;
    }
    #toolbar .note { color: var(--muted); font-size: 12px; }
    #viewerWrap { position: relative; flex: 1 1 auto; min-height: 0; background: white; }
    canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
    #sidebar {
      width: 360px; max-width: 40vw; border-left: 1px solid var(--border);
      background: var(--panel); padding: 14px 16px; overflow: auto;
    }
    h1, h2 { margin: 0 0 10px 0; font-size: 16px; }
    h2 { margin-top: 18px; font-size: 14px; }
    .box { border: 1px solid var(--border); background: white; border-radius: 8px; padding: 10px 12px; }
    .kv { display: grid; grid-template-columns: 120px 1fr; gap: 6px 10px; font-size: 12px; }
    .label { color: var(--muted); }
    .mono { white-space: pre-wrap; word-break: break-word; }
    .legend-row { display: flex; align-items: center; gap: 8px; font-size: 12px; margin: 6px 0; }
    .swatch { width: 20px; height: 3px; border-radius: 2px; }
    .swatch.black { background: #000000; }
    .swatch.red { background: #c00000; }
    .swatch.cyan { background: #0ea5e9; }
    .small { font-size: 12px; color: var(--muted); }
    .signs { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
    .chip {
      border: 1px solid var(--border); border-radius: 999px; padding: 3px 8px;
      font-size: 12px; background: white;
    }
    .chip.pos { color: #0f766e; }
    .chip.neg { color: #b91c1c; }
    .chip.missing { color: #6b7280; }
  </style>
</head>
<body>
  <div id="app">
    <section id="viewerPane">
      <div id="toolbar">
        <button id="resetView">Reset View</button>
        <button id="clearSelection">Clear Selection</button>
        <span class="note">Wheel: zoom. Drag: pan. Hover snaps to the nearest sampled parameter point. Click pins a point.</span>
      </div>
      <div id="viewerWrap">
        <canvas id="baseCanvas"></canvas>
        <canvas id="overlayCanvas"></canvas>
      </div>
    </section>
    <aside id="sidebar">
      <h1>Attempt-043 Explorer</h1>
      <div class="box small">
        Self-contained HTML explorer built from the saved attempt-027 `2000 x 2000` sweep.
        Contours are rendered for nominal iterates `2:8` with the same retired-square rule:
        accepted contours in black and skip-trigger contours in red.
      </div>
      <h2>Legend</h2>
      <div class="box">
        <div class="legend-row"><span class="swatch black"></span><span>accepted contour segments</span></div>
        <div class="legend-row"><span class="swatch red"></span><span>skip-trigger / retired-square contour segments</span></div>
        <div class="legend-row"><span class="swatch cyan"></span><span>selected sampled grid point</span></div>
      </div>
      <h2>Hover</h2>
      <div class="box">
        <div id="hoverInfo" class="kv"></div>
        <div id="hoverSigns" class="signs"></div>
      </div>
      <h2>Selected</h2>
      <div class="box">
        <div id="selectedInfo" class="kv"></div>
        <div id="selectedSigns" class="signs"></div>
      </div>
      <h2>View</h2>
      <div class="box">
        <div id="viewInfo" class="kv"></div>
      </div>
    </aside>
  </div>
  <script>
    const CONFIG = {
      nAlpha: $(n_alpha),
      nLambda: $(n_lambda),
      alphaMin: $(alpha_min),
      alphaMax: $(alpha_max),
      lambdaMin: $(lambda_min),
      lambdaMax: $(lambda_max)
    };
    const BLACK_SEGMENTS_B64 = '""")
        print(io, black_segments_b64)
        print(io, """';
    const RED_SEGMENTS_B64 = '""")
        print(io, red_segments_b64)
        print(io, """';
    const SIGN_WORDS_B64 = '""")
        print(io, sign_words_b64)
        print(io, """';

    function decodeBase64Bytes(b64) {
      const raw = atob(b64);
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
      return bytes;
    }

    function decodeUint16Array(b64) {
      const bytes = decodeBase64Bytes(b64);
      return new Uint16Array(bytes.buffer);
    }

    function decodeFloat32Array(b64) {
      const bytes = decodeBase64Bytes(b64);
      return new Float32Array(bytes.buffer);
    }

    const blackSegments = decodeFloat32Array(BLACK_SEGMENTS_B64);
    const redSegments = decodeFloat32Array(RED_SEGMENTS_B64);
    const signWords = decodeUint16Array(SIGN_WORDS_B64);

    const baseCanvas = document.getElementById('baseCanvas');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const baseCtx = baseCanvas.getContext('2d');
    const overlayCtx = overlayCanvas.getContext('2d');
    const viewerWrap = document.getElementById('viewerWrap');
    const hoverInfo = document.getElementById('hoverInfo');
    const hoverSigns = document.getElementById('hoverSigns');
    const selectedInfo = document.getElementById('selectedInfo');
    const selectedSigns = document.getElementById('selectedSigns');
    const viewInfo = document.getElementById('viewInfo');
    const resetViewButton = document.getElementById('resetView');
    const clearSelectionButton = document.getElementById('clearSelection');

    const state = {
      view: { a0: CONFIG.alphaMin, a1: CONFIG.alphaMax, l0: CONFIG.lambdaMin, l1: CONFIG.lambdaMax },
      hover: null,
      selected: null,
      dragging: null
    };

    function cssRect() {
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      return { x: 70, y: 20, w: Math.max(50, w - 90), h: Math.max(50, h - 60) };
    }

    function resizeCanvases() {
      const dpr = window.devicePixelRatio || 1;
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      for (const canvas of [baseCanvas, overlayCanvas]) {
        canvas.width = Math.max(1, Math.floor(w * dpr));
        canvas.height = Math.max(1, Math.floor(h * dpr));
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
      }
      baseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawBase();
      drawOverlay();
    }

    function alphaStep() {
      return (CONFIG.alphaMax - CONFIG.alphaMin) / (CONFIG.nAlpha - 1);
    }

    function lambdaStep() {
      return (CONFIG.lambdaMax - CONFIG.lambdaMin) / (CONFIG.nLambda - 1);
    }

    function alphaAt(i) {
      return CONFIG.alphaMin + i * alphaStep();
    }

    function lambdaAt(j) {
      return CONFIG.lambdaMin + j * lambdaStep();
    }

    function pointIndex(i, j) {
      return j * CONFIG.nAlpha + i;
    }

    function plotX(alpha) {
      const r = cssRect();
      return r.x + (alpha - state.view.a0) * r.w / (state.view.a1 - state.view.a0);
    }

    function plotY(lambda) {
      const r = cssRect();
      return r.y + r.h - (lambda - state.view.l0) * r.h / (state.view.l1 - state.view.l0);
    }

    function screenToData(clientX, clientY) {
      const bounds = overlayCanvas.getBoundingClientRect();
      const x = clientX - bounds.left;
      const y = clientY - bounds.top;
      const r = cssRect();
      if (x < r.x || x > r.x + r.w || y < r.y || y > r.y + r.h) return null;
      const alpha = state.view.a0 + (x - r.x) * (state.view.a1 - state.view.a0) / r.w;
      const lambda = state.view.l0 + (r.h - (y - r.y)) * (state.view.l1 - state.view.l0) / r.h;
      return { alpha, lambda, x, y };
    }

    function clampInt(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function sampleNearestPoint(alpha, lambda) {
      const i = clampInt(Math.round((alpha - CONFIG.alphaMin) / alphaStep()), 0, CONFIG.nAlpha - 1);
      const j = clampInt(Math.round((lambda - CONFIG.lambdaMin) / lambdaStep()), 0, CONFIG.nLambda - 1);
      const idx = pointIndex(i, j);
      return { i, j, idx, alpha: alphaAt(i), lambda: lambdaAt(j), signWord: signWords[idx] };
    }

    function decodeSignWord(word) {
      const result = [];
      for (let nominal = 2; nominal <= 8; nominal += 1) {
        const code = (word >> (2 * (nominal - 2))) & 0x3;
        result.push({ nominal, code });
      }
      return result;
    }

    function codeText(code) {
      return code === 2 ? '+' : code === 1 ? '-' : '·';
    }

    function codeClass(code) {
      return code === 2 ? 'pos' : code === 1 ? 'neg' : 'missing';
    }

    function setInfo(target, point) {
      if (!point) {
        target.innerHTML = '<div class="small">No point selected.</div>';
        return;
      }
      target.innerHTML = [
        ['alpha', point.alpha.toFixed(6)],
        ['lambda', point.lambda.toFixed(6)],
        ['grid index', '(' + point.i + ', ' + point.j + ')'],
        ['flat index', String(point.idx)],
        ['sign word', '0x' + point.signWord.toString(16).padStart(4, '0')]
      ].map(function(pair) {
        return '<div class="label">' + pair[0] + '</div><div class="mono">' + pair[1] + '</div>';
      }).join('');
    }

    function setSigns(target, point) {
      if (!point) {
        target.innerHTML = '';
        return;
      }
      const chips = decodeSignWord(point.signWord).map(function(entry) {
        return '<span class="chip ' + codeClass(entry.code) + '">k=' + entry.nominal + ': ' + codeText(entry.code) + '</span>';
      });
      target.innerHTML = chips.join('');
    }

    function niceTickStep(span, targetTicks) {
      const raw = span / Math.max(1, targetTicks);
      const power = Math.pow(10, Math.floor(Math.log10(raw)));
      const scaled = raw / power;
      let nice = 1;
      if (scaled > 5) nice = 10;
      else if (scaled > 2) nice = 5;
      else if (scaled > 1) nice = 2;
      return nice * power;
    }

    function tickValues(min, max, targetTicks) {
      const step = niceTickStep(max - min, targetTicks);
      const first = Math.ceil(min / step) * step;
      const values = [];
      for (let value = first; value <= max + 0.5 * step; value += step) values.push(value);
      return values;
    }

    function drawAxes() {
      const r = cssRect();
      baseCtx.strokeStyle = '#000000';
      baseCtx.lineWidth = 1;
      baseCtx.strokeRect(r.x, r.y, r.w, r.h);
      baseCtx.fillStyle = '#111111';
      baseCtx.font = '12px Menlo, Consolas, monospace';

      const alphaTicks = tickValues(state.view.a0, state.view.a1, 6);
      for (const alpha of alphaTicks) {
        const x = plotX(alpha);
        baseCtx.beginPath();
        baseCtx.moveTo(x, r.y + r.h);
        baseCtx.lineTo(x, r.y + r.h + 5);
        baseCtx.stroke();
        baseCtx.fillText(alpha.toFixed(3), x - 14, r.y + r.h + 18);
      }

      const lambdaTicks = tickValues(state.view.l0, state.view.l1, 7);
      for (const lambda of lambdaTicks) {
        const y = plotY(lambda);
        baseCtx.beginPath();
        baseCtx.moveTo(r.x - 5, y);
        baseCtx.lineTo(r.x, y);
        baseCtx.stroke();
        baseCtx.fillText(lambda.toFixed(3), 8, y + 4);
      }

      baseCtx.fillText('alpha', r.x + r.w / 2 - 16, r.y + r.h + 38);
      baseCtx.save();
      baseCtx.translate(18, r.y + r.h / 2 + 16);
      baseCtx.rotate(-Math.PI / 2);
      baseCtx.fillText('lambda', 0, 0);
      baseCtx.restore();
    }

    function drawSegmentArray(array, color) {
      const r = cssRect();
      const a0 = state.view.a0;
      const a1 = state.view.a1;
      const l0 = state.view.l0;
      const l1 = state.view.l1;
      baseCtx.beginPath();
      for (let idx = 0; idx < array.length; idx += 4) {
        const x1 = array[idx];
        const y1 = array[idx + 1];
        const x2 = array[idx + 2];
        const y2 = array[idx + 3];
        if ((x1 < a0 && x2 < a0) || (x1 > a1 && x2 > a1) || (y1 < l0 && y2 < l0) || (y1 > l1 && y2 > l1)) continue;
        baseCtx.moveTo(plotX(x1), plotY(y1));
        baseCtx.lineTo(plotX(x2), plotY(y2));
      }
      baseCtx.strokeStyle = color;
      baseCtx.lineWidth = 1.1;
      baseCtx.stroke();
    }

    function drawBase() {
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      baseCtx.clearRect(0, 0, w, h);
      baseCtx.fillStyle = '#ffffff';
      baseCtx.fillRect(0, 0, w, h);
      const r = cssRect();
      baseCtx.save();
      baseCtx.beginPath();
      baseCtx.rect(r.x, r.y, r.w, r.h);
      baseCtx.clip();
      drawSegmentArray(blackSegments, '#000000');
      drawSegmentArray(redSegments, '#c00000');
      baseCtx.restore();
      drawAxes();
      updateViewInfo();
    }

    function drawPointMarker(ctx, point, color, radius, dash) {
      if (!point) return;
      const x = plotX(point.alpha);
      const y = plotY(point.lambda);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - radius - 6, y);
      ctx.lineTo(x + radius + 6, y);
      ctx.moveTo(x, y - radius - 6);
      ctx.lineTo(x, y + radius + 6);
      ctx.stroke();
      ctx.restore();
    }

    function drawOverlay() {
      const w = viewerWrap.clientWidth;
      const h = viewerWrap.clientHeight;
      overlayCtx.clearRect(0, 0, w, h);
      drawPointMarker(overlayCtx, state.hover, '#6b7280', 4, [4, 4]);
      drawPointMarker(overlayCtx, state.selected, '#0891b2', 6, null);
      setInfo(hoverInfo, state.hover);
      setSigns(hoverSigns, state.hover);
      setInfo(selectedInfo, state.selected);
      setSigns(selectedSigns, state.selected);
    }

    function updateViewInfo() {
      const rows = [
        ['alpha range', state.view.a0.toFixed(6) + ' .. ' + state.view.a1.toFixed(6)],
        ['lambda range', state.view.l0.toFixed(6) + ' .. ' + state.view.l1.toFixed(6)],
        ['grid', CONFIG.nAlpha + ' x ' + CONFIG.nLambda],
        ['segments', (blackSegments.length / 4).toLocaleString() + ' black, ' + (redSegments.length / 4).toLocaleString() + ' red']
      ];
      viewInfo.innerHTML = rows.map(function(pair) {
        return '<div class="label">' + pair[0] + '</div><div class="mono">' + pair[1] + '</div>';
      }).join('');
    }

    function zoomAbout(clientX, clientY, zoomFactor) {
      const data = screenToData(clientX, clientY);
      if (!data) return;
      const aSpan = state.view.a1 - state.view.a0;
      const lSpan = state.view.l1 - state.view.l0;
      const newASpan = Math.max(alphaStep() * 8, Math.min(CONFIG.alphaMax - CONFIG.alphaMin, aSpan * zoomFactor));
      const newLSpan = Math.max(lambdaStep() * 8, Math.min(CONFIG.lambdaMax - CONFIG.lambdaMin, lSpan * zoomFactor));
      const aRatio = (data.alpha - state.view.a0) / aSpan;
      const lRatio = (data.lambda - state.view.l0) / lSpan;
      state.view.a0 = data.alpha - aRatio * newASpan;
      state.view.a1 = state.view.a0 + newASpan;
      state.view.l0 = data.lambda - lRatio * newLSpan;
      state.view.l1 = state.view.l0 + newLSpan;
      clampView();
      drawBase();
      drawOverlay();
    }

    function clampView() {
      const fullA = CONFIG.alphaMax - CONFIG.alphaMin;
      const fullL = CONFIG.lambdaMax - CONFIG.lambdaMin;
      const spanA = state.view.a1 - state.view.a0;
      const spanL = state.view.l1 - state.view.l0;
      if (state.view.a0 < CONFIG.alphaMin) {
        state.view.a0 = CONFIG.alphaMin;
        state.view.a1 = state.view.a0 + spanA;
      }
      if (state.view.a1 > CONFIG.alphaMax) {
        state.view.a1 = CONFIG.alphaMax;
        state.view.a0 = state.view.a1 - spanA;
      }
      if (state.view.l0 < CONFIG.lambdaMin) {
        state.view.l0 = CONFIG.lambdaMin;
        state.view.l1 = state.view.l0 + spanL;
      }
      if (state.view.l1 > CONFIG.lambdaMax) {
        state.view.l1 = CONFIG.lambdaMax;
        state.view.l0 = state.view.l1 - spanL;
      }
      if (spanA >= fullA) { state.view.a0 = CONFIG.alphaMin; state.view.a1 = CONFIG.alphaMax; }
      if (spanL >= fullL) { state.view.l0 = CONFIG.lambdaMin; state.view.l1 = CONFIG.lambdaMax; }
    }

    overlayCanvas.addEventListener('mousemove', function(event) {
      if (state.dragging) {
        const r = cssRect();
        const dx = event.clientX - state.dragging.clientX;
        const dy = event.clientY - state.dragging.clientY;
        const aShift = -dx * (state.view.a1 - state.view.a0) / r.w;
        const lShift = dy * (state.view.l1 - state.view.l0) / r.h;
        state.view.a0 = state.dragging.view.a0 + aShift;
        state.view.a1 = state.dragging.view.a1 + aShift;
        state.view.l0 = state.dragging.view.l0 + lShift;
        state.view.l1 = state.dragging.view.l1 + lShift;
        clampView();
        drawBase();
        drawOverlay();
        return;
      }
      const data = screenToData(event.clientX, event.clientY);
      state.hover = data ? sampleNearestPoint(data.alpha, data.lambda) : null;
      drawOverlay();
    });

    overlayCanvas.addEventListener('mouseleave', function() {
      if (!state.dragging) {
        state.hover = null;
        drawOverlay();
      }
    });

    overlayCanvas.addEventListener('mousedown', function(event) {
      if (event.button !== 0) return;
      state.dragging = {
        clientX: event.clientX,
        clientY: event.clientY,
        view: { a0: state.view.a0, a1: state.view.a1, l0: state.view.l0, l1: state.view.l1 }
      };
    });

    window.addEventListener('mouseup', function(event) {
      if (!state.dragging) return;
      const moved = Math.hypot(event.clientX - state.dragging.clientX, event.clientY - state.dragging.clientY);
      const dragState = state.dragging;
      state.dragging = null;
      if (moved < 4) {
        const data = screenToData(event.clientX, event.clientY);
        state.selected = data ? sampleNearestPoint(data.alpha, data.lambda) : null;
        drawOverlay();
      }
    });

    overlayCanvas.addEventListener('wheel', function(event) {
      event.preventDefault();
      const zoomFactor = event.deltaY < 0 ? 0.85 : 1.18;
      zoomAbout(event.clientX, event.clientY, zoomFactor);
    }, { passive: false });

    resetViewButton.addEventListener('click', function() {
      state.view = { a0: CONFIG.alphaMin, a1: CONFIG.alphaMax, l0: CONFIG.lambdaMin, l1: CONFIG.lambdaMax };
      drawBase();
      drawOverlay();
    });

    clearSelectionButton.addEventListener('click', function() {
      state.selected = null;
      drawOverlay();
    });

    window.addEventListener('resize', resizeCanvases);
    resizeCanvases();
  </script>
</body>
</html>
""")
    end
end

function main()
    println("Building attempt-043 interactive explorer from saved attempt-027 sweep.")
    println("Source columns: $(A27.SWEEP_DIR_025)")
    flush(stdout)

    sign_words = build_sign_words_043()
    println("Packed sign words for $(length(sign_words)) sampled grid points.")
    flush(stdout)

    black_segments, red_segments, iterate_stats = collect_retired_overlay_segments_043()
    println("Collected $(length(black_segments)) black segments and $(length(red_segments)) red segments.")
    flush(stdout)

    black_blob = base64_bytes_043(flatten_segments_043(black_segments))
    red_blob = base64_bytes_043(flatten_segments_043(red_segments))
    sign_blob = base64_bytes_043(sign_words)

    write_iterate_stats_043(STATS_PATH_043, iterate_stats)
    write_html_043(HTML_PATH_043, black_blob, red_blob, sign_blob)

    println("Saved iterate stats to $(STATS_PATH_043)")
    println("Saved explorer HTML to $(HTML_PATH_043)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
