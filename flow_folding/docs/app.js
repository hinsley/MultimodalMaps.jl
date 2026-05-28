(function () {
  const data = window.FLOW_FOLDING_ROSSLER_SCAN;
  const canvas = document.querySelector("#scan-canvas");
  const detail = document.querySelector("#point-detail");
  const summary = document.querySelector("#scan-summary");
  const legend = document.querySelector("#word-legend");

  if (!data || !canvas) {
    return;
  }

  const rows = data.rows || [];
  const config = data.config || {};
  const cValues = uniqueSorted(rows.map((row) => row.c));
  const aValues = uniqueSorted(rows.map((row) => row.a));
  const byKey = new Map(rows.map((row) => [`${row.a}|${row.c}`, row]));
  const ctx = canvas.getContext("2d");

  function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((a, b) => a - b);
  }

  function codeColor(row) {
    if (!row || row.status !== "ok") {
      return "#d0d5d1";
    }
    const span = Math.max(1, 2 ** (config.word_length || 8) - 1);
    const hue = 25 + 260 * (row.code / span);
    const light = row.period > 0 && row.period <= 2 ? 45 : 58;
    return `hsl(${hue} 72% ${light}%)`;
  }

  function setCanvasSize() {
    const rect = canvas.getBoundingClientRect();
    const scale = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(rect.width * scale);
    canvas.height = Math.round(rect.height * scale);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
  }

  function draw() {
    setCanvasSize();
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const padLeft = 52;
    const padRight = 14;
    const padTop = 14;
    const padBottom = 42;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;
    const cellW = plotW / cValues.length;
    const cellH = plotH / aValues.length;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    for (let ai = 0; ai < aValues.length; ai += 1) {
      for (let ci = 0; ci < cValues.length; ci += 1) {
        const a = aValues[ai];
        const c = cValues[ci];
        const row = byKey.get(`${a}|${c}`);
        const x = padLeft + ci * cellW;
        const y = padTop + (aValues.length - 1 - ai) * cellH;
        ctx.fillStyle = codeColor(row);
        ctx.fillRect(x, y, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
      }
    }

    ctx.strokeStyle = "#17201c";
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    ctx.fillStyle = "#17201c";
    ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("c", padLeft + plotW / 2, height - 10);
    ctx.save();
    ctx.translate(15, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("a", 0, 0);
    ctx.restore();

    drawTick(ctx, config.c_min, padLeft, height - padBottom + 16, "center");
    drawTick(ctx, config.c_max, padLeft + plotW, height - padBottom + 16, "center");
    drawTick(ctx, config.a_min, padLeft - 8, padTop + plotH + 4, "right");
    drawTick(ctx, config.a_max, padLeft - 8, padTop + 4, "right");
  }

  function drawTick(ctx, value, x, y, align) {
    ctx.fillStyle = "#647067";
    ctx.textAlign = align;
    ctx.fillText(Number(value).toFixed(2), x, y);
  }

  function eventToRow(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const padLeft = 52;
    const padRight = 14;
    const padTop = 14;
    const padBottom = 42;
    const plotW = canvas.clientWidth - padLeft - padRight;
    const plotH = canvas.clientHeight - padTop - padBottom;
    if (x < padLeft || x > padLeft + plotW || y < padTop || y > padTop + plotH) {
      return null;
    }
    const ci = Math.min(cValues.length - 1, Math.max(0, Math.floor(((x - padLeft) / plotW) * cValues.length)));
    const aiFromTop = Math.min(aValues.length - 1, Math.max(0, Math.floor(((y - padTop) / plotH) * aValues.length)));
    const ai = aValues.length - 1 - aiFromTop;
    return byKey.get(`${aValues[ai]}|${cValues[ci]}`);
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) {
      return "n/a";
    }
    return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(4);
  }

  function renderDetail(row) {
    if (!detail) {
      return;
    }
    if (!row) {
      detail.innerHTML = "<strong>Hover a cell</strong><span>Point details will appear here.</span>";
      return;
    }
    detail.innerHTML = [
      `<strong>a=${row.a.toFixed(4)}, c=${row.c.toFixed(4)}</strong>`,
      `<span>status: ${row.status}, events: ${row.events}</span>`,
      `<span>word: <code>${row.word || "none"}</code></span>`,
      `<span>code: ${row.code}, period: ${row.period}, gamma: ${formatNumber(row.gamma)}</span>`,
      `<span>last event: ${formatNumber(row.last_time)} / max time: ${formatNumber(row.max_time || config.max_time)}</span>`,
      `<span>y-min range: ${formatNumber(row.min_y)} to ${formatNumber(row.max_y)}</span>`,
    ].join("");
  }

  function renderSummary() {
    if (!summary || !legend) {
      return;
    }
    const ok = rows.filter((row) => row.status === "ok").length;
    const maxTime = Number.isFinite(config.max_time) ? config.max_time : Math.max(...rows.map((row) => row.max_time || 0));
    const words = new Map();
    for (const row of rows) {
      if (row.status === "ok") {
        words.set(row.word, (words.get(row.word) || 0) + 1);
      }
    }
    const topWords = Array.from(words.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    summary.innerHTML = [
      `<div class="metric"><strong>${rows.length}</strong><span>grid points</span></div>`,
      `<div class="metric"><strong>${ok}</strong><span>completed words</span></div>`,
      `<div class="metric"><strong>${config.n_c} x ${config.n_a}</strong><span>coarse region grid</span></div>`,
      `<div class="metric"><strong>${config.word_length}</strong><span>tangent symbols after ${config.transient_events} transient y-minima</span></div>`,
      `<div class="metric"><strong>${formatNumber(maxTime)}</strong><span>max integration time per grid point</span></div>`,
    ].join("");
    legend.innerHTML = topWords
      .map(([word, count]) => `<tr><td><code>${word}</code></td><td>${count}</td></tr>`)
      .join("");
  }

  window.addEventListener("resize", draw);
  canvas.addEventListener("mousemove", (event) => renderDetail(eventToRow(event)));
  canvas.addEventListener("mouseleave", () => renderDetail(null));

  renderSummary();
  renderDetail(null);
  draw();
})();
