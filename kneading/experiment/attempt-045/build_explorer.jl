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
    "ATTEMPT045_OUTPUT_TAG",
    get(
        ENV,
        "ATTEMPT043_OUTPUT_TAG",
        "grid2000_branch16_absxskip16_plot8_forcedfirstskip_black_red_explorer_shimizu_morioka_cpu",
    ),
)
const HTML_PATH_043 = joinpath(ATTEMPT043_ROOT, "$(OUTPUT_TAG_043).html")
const STATS_PATH_043 = joinpath(ATTEMPT043_ROOT, "$(OUTPUT_TAG_043)_iterate_stats.tsv")
const MISSING_TIME_WORD_043 = UInt16(0xffff)

@inline sign_code_043(value::Float64) = value > 0.0 ? UInt16(0x2) : value < 0.0 ? UInt16(0x1) : UInt16(0x0)
@inline skip_bit_043(nominal_iterate::Int) = UInt8(1) << (nominal_iterate - 2)
@inline point_linear_index_043(i::Int, j::Int, n_alpha::Int) = (j - 1) * n_alpha + i

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

function mark_point_skip_masks_043!(
    point_skip_masks::Vector{UInt8},
    j::Int,
    i::Int,
    signs::NTuple{4, Int8},
    shorter_sign::Int8,
    nominal_iterate::Int,
    n_alpha::Int,
)
    2 <= nominal_iterate <= 8 || return nothing
    bit = skip_bit_043(nominal_iterate)
    if signs[1] == shorter_sign
        point_skip_masks[point_linear_index_043(i, j, n_alpha)] |= bit
    end
    if signs[2] == shorter_sign
        point_skip_masks[point_linear_index_043(i + 1, j, n_alpha)] |= bit
    end
    if signs[3] == shorter_sign
        point_skip_masks[point_linear_index_043(i + 1, j + 1, n_alpha)] |= bit
    end
    if signs[4] == shorter_sign
        point_skip_masks[point_linear_index_043(i, j + 1, n_alpha)] |= bit
    end
    return nothing
end

@inline function increment_local_skip_045(
    skip::NTuple{4, Int},
    signs::NTuple{4, Int8},
    shorter_sign::Int8,
)
    return (
        skip[1] + (signs[1] == shorter_sign ? 1 : 0),
        skip[2] + (signs[2] == shorter_sign ? 1 : 0),
        skip[3] + (signs[3] == shorter_sign ? 1 : 0),
        skip[4] + (signs[4] == shorter_sign ? 1 : 0),
    )
end

function evaluate_square_local_045(
    j::Int,
    i::Int,
    nominal_iterate::Int,
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
    skip::NTuple{4, Int},
)
    k_tl = nominal_iterate + skip[1]
    k_tr = nominal_iterate + skip[2]
    k_br = nominal_iterate + skip[3]
    k_bl = nominal_iterate + skip[4]
    ks = (k_tl, k_tr, k_br, k_bl)

    any(k -> k < 1 || k > length(dot_grids) || k + 1 > length(time_grids), ks) && return A27.missing_evaluation_025()

    d_tl = dot_grids[k_tl][j, i]
    d_tr = dot_grids[k_tr][j, i + 1]
    d_br = dot_grids[k_br][j + 1, i + 1]
    d_bl = dot_grids[k_bl][j + 1, i]
    all(isfinite, (d_tl, d_tr, d_br, d_bl)) || return A27.missing_evaluation_025()

    t_tl = time_grids[k_tl][j, i]
    t_tr = time_grids[k_tr][j, i + 1]
    t_br = time_grids[k_br][j + 1, i + 1]
    t_bl = time_grids[k_bl][j + 1, i]
    all(isfinite, (t_tl, t_tr, t_br, t_bl)) || return A27.missing_evaluation_025()

    t2_tl = time_grids[k_tl + 1][j, i]
    t2_tr = time_grids[k_tr + 1][j, i + 1]
    t2_br = time_grids[k_br + 1][j + 1, i + 1]
    t2_bl = time_grids[k_bl + 1][j + 1, i]
    all(isfinite, (t2_tl, t2_tr, t2_br, t2_bl)) || return A27.missing_evaluation_025()

    signs = (
        A27.sign_class_025(d_tl),
        A27.sign_class_025(d_tr),
        A27.sign_class_025(d_br),
        A27.sign_class_025(d_bl),
    )
    any(==(Int8(0)), signs) && return A27.missing_evaluation_025()

    status = all(==(signs[1]), signs) ? A27.EVAL_CONSTANT_025 : A27.EVAL_MIXED_025
    return A27.SquareEvaluation25(
        status,
        (d_tl, d_tr, d_br, d_bl),
        (t_tl, t_tr, t_br, t_bl),
        (t2_tl, t2_tr, t2_br, t2_bl),
        signs,
        ks,
    )
end

function build_grids_043()
    dot_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_dot_values)
    cumulative_time_grids = A27.build_iterate_grids_025(result -> result.absxmax_count, result -> result.absxmax_return_times)
    time_grids = A27.cumulative_to_interval_grids_025(cumulative_time_grids)
    return dot_grids, time_grids
end

function collect_forcedfirstskip_overlay_segments_045(
    dot_grids::Vector{Matrix{Float64}},
    time_grids::Vector{Matrix{Float64}},
)
    n_plot = A27.ATTEMPT025_PLOT_ITERATE_CAP
    plot_iterate_end = min(8, n_plot)
    stored_iterate_end = min(length(dot_grids), length(time_grids) - 1)
    n_lambda_cells = length(A27.LAMBDAS_025) - 1
    n_alpha_cells = length(A27.ALPHAS_025) - 1
    n_threads = Threads.maxthreadid()

    black_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    red_tls = [[NTuple{4, Float64}[] for _ in 1:n_plot] for _ in 1:n_threads]
    earliest_source_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    black_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    black_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    red_cell_tls = [zeros(Int, n_plot) for _ in 1:n_threads]
    red_segment_tls = [zeros(Int, n_plot) for _ in 1:n_threads]

    earliest_iterate_cells = zeros(UInt8, n_lambda_cells, n_alpha_cells)
    shorter_sign_cells = zeros(Int8, n_lambda_cells, n_alpha_cells)

    Threads.@threads :dynamic for j in 1:n_lambda_cells
        tid = Threads.threadid()
        black_local = black_tls[tid]
        red_local = red_tls[tid]
        earliest_local = earliest_source_tls[tid]
        black_cell_local = black_cell_tls[tid]
        black_segment_local = black_segment_tls[tid]
        red_cell_local = red_cell_tls[tid]
        red_segment_local = red_segment_tls[tid]

        y_tl = Float64(A27.LAMBDAS_025[j])
        y_bl = Float64(A27.LAMBDAS_025[j + 1])

        for i in 1:n_alpha_cells
            x_tl = Float64(A27.ALPHAS_025[i])
            x_tr = Float64(A27.ALPHAS_025[i + 1])
            skip = (0, 0, 0, 0)
            earliest_nominal = 0
            earliest_evaluation = A27.missing_evaluation_025()
            later_found = false

            for nominal_iterate in 2:stored_iterate_end
                evaluation = evaluate_square_local_045(j, i, nominal_iterate, dot_grids, time_grids, skip)
                evaluation.status == A27.EVAL_MIXED_025 || continue

                if earliest_nominal == 0
                    nominal_iterate <= plot_iterate_end || break
                    earliest_nominal = nominal_iterate
                    earliest_evaluation = evaluation
                    earliest_local[nominal_iterate] += 1
                    shorter_sign, _, _ = A27.choose_representatives_025(evaluation)
                    earliest_iterate_cells[j, i] = UInt8(nominal_iterate)
                    shorter_sign_cells[j, i] = shorter_sign
                    skip = increment_local_skip_045(skip, evaluation.sign, shorter_sign)
                    continue
                end

                later_found = true
                if nominal_iterate <= plot_iterate_end
                    added = A27.append_march_square_zero_segments_025!(
                        black_local[nominal_iterate],
                        evaluation.current_dot,
                        x_tl,
                        y_tl,
                        x_tr,
                        y_tl,
                        x_tr,
                        y_bl,
                        x_tl,
                        y_bl,
                    )
                    if added > 0
                        black_cell_local[nominal_iterate] += 1
                        black_segment_local[nominal_iterate] += added
                    end
                end
            end

            if earliest_nominal != 0 && !later_found
                added = A27.append_march_square_zero_segments_025!(
                    red_local[earliest_nominal],
                    earliest_evaluation.current_dot,
                    x_tl,
                    y_tl,
                    x_tr,
                    y_tl,
                    x_tr,
                    y_bl,
                    x_tl,
                    y_bl,
                )
                if added > 0
                    red_cell_local[earliest_nominal] += 1
                    red_segment_local[earliest_nominal] += added
                end
            end
        end
    end

    black_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    red_segments_by_iter = [NTuple{4, Float64}[] for _ in 1:n_plot]
    earliest_source_cells = zeros(Int, n_plot)
    black_contoured_cells = zeros(Int, n_plot)
    black_segments_count = zeros(Int, n_plot)
    red_contoured_cells = zeros(Int, n_plot)
    red_segments_count = zeros(Int, n_plot)

    for tid in 1:n_threads
        for iterate in 2:plot_iterate_end
            append!(black_segments_by_iter[iterate], black_tls[tid][iterate])
            append!(red_segments_by_iter[iterate], red_tls[tid][iterate])
            earliest_source_cells[iterate] += earliest_source_tls[tid][iterate]
            black_contoured_cells[iterate] += black_cell_tls[tid][iterate]
            black_segments_count[iterate] += black_segment_tls[tid][iterate]
            red_contoured_cells[iterate] += red_cell_tls[tid][iterate]
            red_segments_count[iterate] += red_segment_tls[tid][iterate]
        end
    end

    point_skip_masks = fill(UInt8(0), length(A27.ALPHAS_025) * length(A27.LAMBDAS_025))
    n_alpha_points = length(A27.ALPHAS_025)
    zero_skip = (0, 0, 0, 0)
    for j in 1:n_lambda_cells
        for i in 1:n_alpha_cells
            nominal_iterate = Int(earliest_iterate_cells[j, i])
            nominal_iterate == 0 && continue
            evaluation = evaluate_square_local_045(j, i, nominal_iterate, dot_grids, time_grids, zero_skip)
            evaluation.status == A27.EVAL_MIXED_025 || continue
            mark_point_skip_masks_043!(
                point_skip_masks,
                j,
                i,
                evaluation.sign,
                shorter_sign_cells[j, i],
                nominal_iterate,
                n_alpha_points,
            )
        end
    end

    iterate_stats = (
        earliest_source_cells=earliest_source_cells,
        black_contoured_cells=black_contoured_cells,
        black_segments_count=black_segments_count,
        red_contoured_cells=red_contoured_cells,
        red_segments_count=red_segments_count,
    )

    return black_segments_by_iter, red_segments_by_iter, point_skip_masks, iterate_stats
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

function base64_gzip_bytes_043(values::AbstractVector{T}) where {T}
    raw = reinterpret(UInt8, values)
    tmp_path = tempname()
    write(tmp_path, raw)
    compressed = try
        read(`gzip -c $tmp_path`)
    finally
        isfile(tmp_path) && rm(tmp_path; force=true)
    end
    return base64encode(compressed)
end

function choose_time_scale_043(time_grids::Vector{Matrix{Float64}})
    max_time = 0.0
    for nominal_iterate in 2:min(8, length(time_grids))
        grid = time_grids[nominal_iterate]
        @inbounds for value in grid
            isfinite(value) || continue
            value > max_time && (max_time = value)
        end
    end

    for scale in (1000, 200, 100, 50, 10)
        max_time <= (65534 / scale) && return scale
    end
    return 1
end

function build_time_words_043(time_grids::Vector{Matrix{Float64}})
    n_alpha = length(A27.ALPHAS_025)
    n_lambda = length(A27.LAMBDAS_025)
    time_scale = choose_time_scale_043(time_grids)
    words = fill(MISSING_TIME_WORD_043, n_alpha * n_lambda * 7)

    for nominal_iterate in 2:min(8, length(time_grids))
        grid = time_grids[nominal_iterate]
        offset = nominal_iterate - 2
        for col_idx in 1:n_alpha
            for row_idx in 1:n_lambda
                value = grid[row_idx, col_idx]
                isfinite(value) || continue
                quantized = round(Int, value * time_scale)
                0 <= quantized <= 65534 || continue
                linear_idx = ((row_idx - 1) * n_alpha + (col_idx - 1)) * 7 + offset + 1
                words[linear_idx] = UInt16(quantized)
            end
        end
    end

    return words, time_scale
end

function write_iterate_stats_043(path::String, stats)
    open(path, "w") do io
        println(io, "nominal_iterate\tearliest_source_cells\tblack_contoured_cells\tblack_segments\tred_contoured_cells\tred_segments")
        for iterate in 2:min(8, length(stats.earliest_source_cells))
            println(
                io,
                join([
                    string(iterate),
                    string(stats.earliest_source_cells[iterate]),
                    string(stats.black_contoured_cells[iterate]),
                    string(stats.black_segments_count[iterate]),
                    string(stats.red_contoured_cells[iterate]),
                    string(stats.red_segments_count[iterate]),
                ], '\t'),
            )
        end
    end
end

function write_html_043(
    path::String,
    black_segments_b64_by_iter::Vector{String},
    red_segments_b64_by_iter::Vector{String},
    sign_words_b64::String,
    skip_words_b64::String,
    time_words_gz_b64::String,
    time_scale::Int,
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
  <title>Shimizu-Morioka Forced-First-Skip Explorer</title>
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
      width: 344px; max-width: 38vw; border-left: 1px solid var(--border);
      background: var(--panel); padding: 8px 9px; overflow: auto;
    }
    h1, h2 { margin: 0 0 6px 0; font-size: 14px; }
    h2 { margin-top: 9px; font-size: 12px; }
    .box { border: 1px solid var(--border); background: white; border-radius: 8px; padding: 6px 8px; }
    .kv { display: grid; grid-template-columns: 78px 1fr; gap: 2px 6px; font-size: 10px; }
    .label { color: var(--muted); }
    .mono { white-space: pre-wrap; word-break: break-word; }
    .legend-row { display: flex; align-items: center; gap: 7px; font-size: 10px; margin: 3px 0; }
    .swatch { width: 20px; height: 3px; border-radius: 2px; }
    .swatch.black { background: #000000; }
    .swatch.red { background: #c00000; }
    .swatch.cyan { background: #0ea5e9; }
    .small { font-size: 10px; color: var(--muted); }
    .chip {
      border: 1px solid var(--border); border-radius: 999px; padding: 3px 8px;
      font-size: 12px; background: white;
    }
    .chip.pos { color: #111111; }
    .chip.neg { color: #111111; }
    .chip.missing { color: #6b7280; }
    .iter-controls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 4px; margin-top: 6px; }
    .iter-controls label { display: flex; align-items: center; gap: 4px; font-size: 10px; }
    .iter-buttons { display: flex; gap: 5px; margin-bottom: 6px; }
    .iter-table { width: 100%; border-collapse: collapse; font-size: 10px; table-layout: fixed; }
    .iter-table th, .iter-table td { border-bottom: 1px solid var(--border); padding: 3px 4px; text-align: left; }
    .iter-table th { color: var(--muted); font-weight: 600; background: #fbfbfb; position: sticky; top: 0; }
    .iter-row.skip { color: var(--skip); font-weight: 600; }
    .iter-row.normal { color: #111111; }
    .highlight-note { margin-top: 4px; font-size: 10px; color: var(--muted); }
    .compact-meta { margin-bottom: 4px; }
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
      <h1>Attempt-045 Explorer</h1>
      <div class="box small">
        Self-contained HTML explorer built from the saved attempt-027 `2000 x 2000` sweep.
        This uses the attempt-044 rule: the first contouring iterate in `2:8` forces one
        shorter-return-time skip, later iterates up through the stored 16th iterate decide
        whether that square stays red or gets a later black contour, and only iterates `2:8`
        are actually drawn.
      </div>
      <h2>Legend</h2>
      <div class="box">
        <div class="legend-row"><span class="swatch black"></span><span>later surviving contours after the forced first skip</span></div>
        <div class="legend-row"><span class="swatch red"></span><span>earliest contour only, when no later contour survives</span></div>
        <div class="legend-row"><span class="swatch cyan"></span><span>selected sampled grid point</span></div>
        <div class="legend-row"><span class="swatch" style="background:#bcbcbc;"></span><span>four marched squares around the selected point</span></div>
      </div>
      <h2>Contours</h2>
      <div class="box">
        <div class="iter-buttons">
          <button id="showAllIterates">Show All</button>
          <button id="hideAllIterates">Hide All</button>
          <button id="toggleRedContours">Hide Red</button>
        </div>
        <div id="iterateControls" class="iter-controls"></div>
      </div>
      <h2>Hover</h2>
      <div class="box">
        <div id="hoverInfo" class="kv compact-meta"></div>
        <table class="iter-table">
          <thead>
            <tr><th>Iter</th><th>Sign</th><th>Time</th><th>Skip</th></tr>
          </thead>
          <tbody id="hoverTableBody"></tbody>
        </table>
      </div>
      <h2>Selected</h2>
      <div class="box">
        <div id="selectedInfo" class="kv compact-meta"></div>
        <div class="highlight-note">Rows are red where the selected sampled point lies on the shorter-return-time side of the forced first skip for some surrounding square at that nominal iterate.</div>
        <table class="iter-table">
          <thead>
            <tr><th>Iter</th><th>Sign</th><th>Time</th><th>Skip</th></tr>
          </thead>
          <tbody id="selectedTableBody"></tbody>
        </table>
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
    const BLACK_SEGMENTS_B64_BY_ITER = {
""")
        for idx in 2:length(black_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, black_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
    const RED_SEGMENTS_B64_BY_ITER = {
""")
        for idx in 2:length(red_segments_b64_by_iter)
            print(io, "      ")
            print(io, idx)
            print(io, ": '")
            print(io, red_segments_b64_by_iter[idx])
            print(io, "',\n")
        end
        print(io, """    };
    const SIGN_WORDS_B64 = '""")
        print(io, sign_words_b64)
        print(io, """';
    const SKIP_WORDS_B64 = '""")
        print(io, skip_words_b64)
        print(io, """';
    const TIME_WORDS_GZ_B64 = '""")
        print(io, time_words_gz_b64)
        print(io, """';
    const TIME_SCALE = $(time_scale);

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

    async function decodeGzipUint16Array(b64) {
      if (!b64) return new Uint16Array(0);
      if (typeof DecompressionStream === 'undefined') {
        console.warn('DecompressionStream is unavailable; per-iterate times will not be shown.');
        return new Uint16Array(0);
      }
      const bytes = decodeBase64Bytes(b64);
      const stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
      const decompressed = await new Response(stream).arrayBuffer();
      return new Uint16Array(decompressed);
    }

    const blackSegmentsByIter = {};
    const redSegmentsByIter = {};
    for (let nominal = 2; nominal <= 8; nominal += 1) {
      blackSegmentsByIter[nominal] = BLACK_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(BLACK_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
      redSegmentsByIter[nominal] = RED_SEGMENTS_B64_BY_ITER[nominal] ? decodeFloat32Array(RED_SEGMENTS_B64_BY_ITER[nominal]) : new Float32Array(0);
    }
    const signWords = decodeUint16Array(SIGN_WORDS_B64);
    const skipWords = decodeBase64Bytes(SKIP_WORDS_B64);
    let timeWords = null;

    const baseCanvas = document.getElementById('baseCanvas');
    const overlayCanvas = document.getElementById('overlayCanvas');
    const baseCtx = baseCanvas.getContext('2d');
    const overlayCtx = overlayCanvas.getContext('2d');
    const viewerWrap = document.getElementById('viewerWrap');
    const hoverInfo = document.getElementById('hoverInfo');
    const hoverTableBody = document.getElementById('hoverTableBody');
    const selectedInfo = document.getElementById('selectedInfo');
    const selectedTableBody = document.getElementById('selectedTableBody');
    const viewInfo = document.getElementById('viewInfo');
    const iterateControls = document.getElementById('iterateControls');
    const resetViewButton = document.getElementById('resetView');
    const clearSelectionButton = document.getElementById('clearSelection');
    const showAllIteratesButton = document.getElementById('showAllIterates');
    const hideAllIteratesButton = document.getElementById('hideAllIterates');
    const toggleRedContoursButton = document.getElementById('toggleRedContours');

    const state = {
      view: { a0: CONFIG.alphaMin, a1: CONFIG.alphaMax, l0: CONFIG.lambdaMin, l1: CONFIG.lambdaMax },
      hover: null,
      selected: null,
      dragging: null,
      visibleIterates: new Set([2, 3, 4, 5, 6, 7, 8]),
      showRedContours: true
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
      return {
        i,
        j,
        idx,
        alpha: alphaAt(i),
        lambda: lambdaAt(j),
        signWord: signWords[idx],
        skipWord: skipWords[idx]
      };
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

    function setInfo(target, point, emptyText) {
      if (!point) {
        target.innerHTML = '<div class="small">' + emptyText + '</div>';
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

    function decodeSkipWord(word) {
      const result = [];
      for (let nominal = 2; nominal <= 8; nominal += 1) {
        result.push({ nominal, skip: !!(word & (1 << (nominal - 2))) });
      }
      return result;
    }

    function decodeTimeForPoint(pointIdx, nominalIterate) {
      if (!timeWords || timeWords.length === 0) return null;
      const word = timeWords[pointIdx * 7 + (nominalIterate - 2)];
      if (word === undefined || word === 0xffff) return null;
      return word / TIME_SCALE;
    }

    function formatTimeValue(value) {
      if (value === null) return '·';
      if (TIME_SCALE >= 100) return value.toFixed(2);
      if (TIME_SCALE >= 10) return value.toFixed(1);
      return value.toFixed(0);
    }

    function setPointTable(target, point, emptyText) {
      if (!point) {
        target.innerHTML = '<tr><td colspan="4" class="small">' + emptyText + '</td></tr>';
        return;
      }
      const signs = decodeSignWord(point.signWord);
      const skips = decodeSkipWord(point.skipWord);
      const rows = [];
      for (let idx = 0; idx < signs.length; idx += 1) {
        const signEntry = signs[idx];
        const skipEntry = skips[idx];
        const timeValue = formatTimeValue(decodeTimeForPoint(point.idx, signEntry.nominal));
        rows.push(
          '<tr class="iter-row ' + (skipEntry.skip ? 'skip' : 'normal') + '">' +
            '<td>' + signEntry.nominal + '</td>' +
            '<td>' + codeText(signEntry.code) + '</td>' +
            '<td>' + timeValue + '</td>' +
            '<td>' + (skipEntry.skip ? 'yes' : 'no') + '</td>' +
          '</tr>'
        );
      }
      target.innerHTML = rows.join('');
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

    function drawSelectedNeighborCells() {
      if (!state.selected) return;
      const cellCoords = [
        [state.selected.i - 1, state.selected.j - 1],
        [state.selected.i, state.selected.j - 1],
        [state.selected.i - 1, state.selected.j],
        [state.selected.i, state.selected.j]
      ];
      overlayCtx.save();
      overlayCtx.fillStyle = 'rgba(170, 170, 170, 0.18)';
      overlayCtx.strokeStyle = 'rgba(90, 90, 90, 0.95)';
      overlayCtx.lineWidth = 1.2;
      for (const [ci, cj] of cellCoords) {
        if (ci < 0 || cj < 0 || ci >= CONFIG.nAlpha - 1 || cj >= CONFIG.nLambda - 1) continue;
        const x0 = plotX(alphaAt(ci));
        const x1 = plotX(alphaAt(ci + 1));
        const yTop = plotY(lambdaAt(cj + 1));
        const yBot = plotY(lambdaAt(cj));
        overlayCtx.fillRect(x0, yTop, x1 - x0, yBot - yTop);
        overlayCtx.strokeRect(x0, yTop, x1 - x0, yBot - yTop);
      }
      overlayCtx.restore();
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
      for (let nominal = 2; nominal <= 8; nominal += 1) {
        if (!state.visibleIterates.has(nominal)) continue;
        drawSegmentArray(blackSegmentsByIter[nominal], '#000000');
        if (state.showRedContours) drawSegmentArray(redSegmentsByIter[nominal], '#c00000');
      }
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
      drawSelectedNeighborCells();
      drawPointMarker(overlayCtx, state.hover, '#6b7280', 4, [4, 4]);
      drawPointMarker(overlayCtx, state.selected, '#0891b2', 6, null);
      setInfo(hoverInfo, state.hover, 'No point under cursor.');
      setPointTable(hoverTableBody, state.hover, 'No point under cursor.');
      setInfo(selectedInfo, state.selected, 'No point selected.');
      setPointTable(selectedTableBody, state.selected, 'No point selected.');
    }

    function updateViewInfo() {
      let visibleBlack = 0;
      let visibleRed = 0;
      let totalRed = 0;
      for (let nominal = 2; nominal <= 8; nominal += 1) {
        if (!state.visibleIterates.has(nominal)) continue;
        visibleBlack += blackSegmentsByIter[nominal].length / 4;
        totalRed += redSegmentsByIter[nominal].length / 4;
      }
      visibleRed = state.showRedContours ? totalRed : 0;
      const rows = [
        ['alpha range', state.view.a0.toFixed(6) + ' .. ' + state.view.a1.toFixed(6)],
        ['lambda range', state.view.l0.toFixed(6) + ' .. ' + state.view.l1.toFixed(6)],
        ['grid', CONFIG.nAlpha + ' x ' + CONFIG.nLambda],
        ['visible iterates', Array.from(state.visibleIterates).sort(function(a, b) { return a - b; }).join(', ') || '(none)'],
        ['red contours', state.showRedContours ? 'shown' : 'hidden'],
        ['segments', visibleBlack.toLocaleString() + ' black, ' + visibleRed.toLocaleString() + ' red']
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

    function renderIterateControls() {
      const html = [];
      for (let nominal = 2; nominal <= 8; nominal += 1) {
        const checked = state.visibleIterates.has(nominal) ? 'checked' : '';
        html.push(
          '<label><input type="checkbox" class="iterateToggle" data-iterate="' + nominal + '" ' + checked + '>k=' + nominal + '</label>'
        );
      }
      iterateControls.innerHTML = html.join('');
      for (const input of iterateControls.querySelectorAll('.iterateToggle')) {
        input.addEventListener('change', function(event) {
          const nominal = Number(event.target.getAttribute('data-iterate'));
          if (event.target.checked) state.visibleIterates.add(nominal);
          else state.visibleIterates.delete(nominal);
          drawBase();
          drawOverlay();
        });
      }
    }

    showAllIteratesButton.addEventListener('click', function() {
      state.visibleIterates = new Set([2, 3, 4, 5, 6, 7, 8]);
      renderIterateControls();
      drawBase();
      drawOverlay();
    });

    hideAllIteratesButton.addEventListener('click', function() {
      state.visibleIterates = new Set();
      renderIterateControls();
      drawBase();
      drawOverlay();
    });

    function updateRedToggleButton() {
      toggleRedContoursButton.textContent = state.showRedContours ? 'Hide Red' : 'Show Red';
    }

    toggleRedContoursButton.addEventListener('click', function() {
      state.showRedContours = !state.showRedContours;
      updateRedToggleButton();
      drawBase();
      drawOverlay();
    });

    window.addEventListener('resize', resizeCanvases);
    renderIterateControls();
    updateRedToggleButton();
    decodeGzipUint16Array(TIME_WORDS_GZ_B64)
      .then(function(words) {
        timeWords = words;
        drawOverlay();
      })
      .catch(function(error) {
        console.error('Failed to decode time payload:', error);
        timeWords = new Uint16Array(0);
        drawOverlay();
      });
    resizeCanvases();
  </script>
</body>
</html>
""")
    end
end

function main()
    println("Building attempt-045 interactive explorer from saved attempt-027 sweep.")
    println("Source columns: $(A27.SWEEP_DIR_025)")
    flush(stdout)

    sign_words = build_sign_words_043()
    println("Packed sign words for $(length(sign_words)) sampled grid points.")
    flush(stdout)

    dot_grids, time_grids = build_grids_043()
    time_words, time_scale = build_time_words_043(time_grids)
    println("Packed $(length(time_words)) quantized return-time words at scale $(time_scale).")
    flush(stdout)

    black_segments_by_iter, red_segments_by_iter, point_skip_masks, iterate_stats =
        collect_forcedfirstskip_overlay_segments_045(dot_grids, time_grids)
    total_black = sum(length, black_segments_by_iter)
    total_red = sum(length, red_segments_by_iter)
    println("Collected $(total_black) black segments and $(total_red) red segments.")
    flush(stdout)

    black_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    red_blobs = [base64_bytes_043(Float32[]) for _ in 1:A27.ATTEMPT025_PLOT_ITERATE_CAP]
    for nominal_iterate in 2:min(8, A27.ATTEMPT025_PLOT_ITERATE_CAP)
        black_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(black_segments_by_iter[nominal_iterate]))
        red_blobs[nominal_iterate] = base64_bytes_043(flatten_segments_043(red_segments_by_iter[nominal_iterate]))
    end
    sign_blob = base64_bytes_043(sign_words)
    skip_blob = base64_bytes_043(point_skip_masks)
    time_blob = base64_gzip_bytes_043(time_words)

    write_iterate_stats_043(STATS_PATH_043, iterate_stats)
    write_html_043(HTML_PATH_043, black_blobs, red_blobs, sign_blob, skip_blob, time_blob, time_scale)

    println("Saved iterate stats to $(STATS_PATH_043)")
    println("Saved explorer HTML to $(HTML_PATH_043)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
