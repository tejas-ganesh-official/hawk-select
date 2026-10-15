// Replay of a logged binary-HHO run: hawk x feature bit grid, convergence, escaping energy.
import { STRATEGY_COLORS, STRATEGY_LABELS, C, fitCanvas } from "./theme.js";

const BASE_STEP_MS = 420; // per iteration at 1x
const FLASH_MS = 380;

export async function initReplay(root, url = "data/replay.json") {
  let run;
  try {
    run = await (await fetch(url)).json();
  } catch (err) {
    root.querySelector("[data-r=status]").textContent = "Could not load replay data.";
    console.error(err);
    return;
  }

  const H = run.history;
  const T = H.length - 1;
  const N = run.n_hawks;
  const names = run.feature_names;
  const D = names.length;
  const bits = H.map((h) => h.masks.map((m) => Uint8Array.from(m, (c) => c === "1")));
  const best = H.map((h) => Uint8Array.from(h.best_mask, (c) => c === "1"));

  const $ = (k) => root.querySelector(`[data-r=${k}]`);
  const gridCv = $("grid"), convCv = $("conv"), energyCv = $("energy");
  const slider = $("slider"), playBtn = $("play"), speedSel = $("speed");
  const tip = $("tip");
  slider.max = T;
  $("status").remove();
  root.classList.add("ready");

  // static header text
  $("title").textContent = `${run.algorithm} · ${run.n_hawks} hawks × ${run.iterations} iterations · seed ${run.seed}`;

  // strategy legend with live counts
  const legendCounts = STRATEGY_LABELS.map((label, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot" style="--c:${STRATEGY_COLORS[i]}"></span><span>${label}</span><b>0</b>`;
    $("legend").appendChild(li);
    return li.querySelector("b");
  });

  // feature chips
  const chips = names.map((n) => {
    const el = document.createElement("span");
    el.className = "chip";
    el.textContent = n;
    $("chips").appendChild(el);
    return el;
  });

  renderMetrics($("metrics"), run);

  let t = 0;
  let playing = false;
  let changedAt = 0; // timestamp of last iteration change, for flip flashes
  let speed = 1;
  let acc = 0, last = 0;
  let hover = null;
  let layout = null;

  // ---------- bit grid ----------
  function gridLayout(w) {
    const gutter = 46;
    const cell = Math.max(6, Math.floor((w - gutter - 4) / D));
    const top = 28; // column-frequency bars
    const rabbitGap = 10;
    return { gutter, cell, top, rabbitGap, gridTop: top + cell + rabbitGap, labelTop: top + cell + rabbitGap + N * cell + 6 };
  }

  function drawGrid(now) {
    const { ctx, w, h } = fitCanvas(gridCv);
    const L = (layout = gridLayout(w));
    const { gutter, cell, top, gridTop } = L;
    const cur = bits[t], prev = bits[Math.max(0, t - 1)];
    const flash = t > 0 ? Math.max(0, 1 - (now - changedAt) / FLASH_MS) : 0;
    const rabbit = best[t];
    ctx.clearRect(0, 0, w, h);

    // population frequency per feature
    for (let d = 0; d < D; d++) {
      let on = 0;
      for (let i = 0; i < N; i++) on += cur[i][d];
      const f = on / N;
      const x = gutter + d * cell;
      ctx.fillStyle = "rgba(255,255,255,.05)";
      ctx.fillRect(x + 1, 0, cell - 2, top - 6);
      ctx.fillStyle = rabbit[d] ? C.accent : "rgba(47,214,196,.55)";
      const bh = (top - 6) * f;
      ctx.fillRect(x + 1, top - 6 - bh, cell - 2, bh);
    }

    // rabbit row
    ctx.fillStyle = C.muted;
    ctx.font = "600 10px 'JetBrains Mono', monospace";
    ctx.textBaseline = "middle";
    ctx.fillText("BEST", 4, top + cell / 2);
    for (let d = 0; d < D; d++) {
      const x = gutter + d * cell;
      ctx.fillStyle = rabbit[d] ? C.accent : "rgba(255,255,255,.04)";
      if (rabbit[d]) { ctx.shadowColor = C.accent; ctx.shadowBlur = 8; }
      roundRect(ctx, x + 1, top + 1, cell - 2, cell - 2, 2);
      ctx.shadowBlur = 0;
    }

    // hawks
    const strat = H[t].strategy;
    for (let i = 0; i < N; i++) {
      const y = gridTop + i * cell;
      const s = strat[i];
      ctx.fillStyle = s == null ? C.muted : STRATEGY_COLORS[s];
      ctx.beginPath();
      ctx.arc(10, y + cell / 2, 3.5, 0, Math.PI * 2);
      ctx.fill();
      if (cell >= 11 || i % 2 === 0) { // thin out row labels when cells get small
        ctx.fillStyle = C.muted;
        ctx.font = `${labelFont(cell)}px 'JetBrains Mono', monospace`;
        ctx.fillText(`H${String(i + 1).padStart(2, "0")}`, 18, y + cell / 2);
      }
      for (let d = 0; d < D; d++) {
        const x = gutter + d * cell;
        const on = cur[i][d];
        ctx.fillStyle = on ? "rgba(47,214,196,.85)" : "rgba(255,255,255,.035)";
        roundRect(ctx, x + 1, y + 1, cell - 2, cell - 2, 2);
        if (flash > 0 && on !== prev[i][d]) {
          ctx.strokeStyle = on ? `rgba(60,240,138,${flash})` : `rgba(255,77,109,${flash})`;
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1.5, y + 1.5, cell - 3, cell - 3);
        }
      }
    }

    // hover crosshair
    if (hover) {
      ctx.strokeStyle = "rgba(255,255,255,.5)";
      ctx.lineWidth = 1;
      if (hover.d != null) ctx.strokeRect(gutter + hover.d * cell + 0.5, top + 0.5, cell - 1, gridTop - top + N * cell - 1);
      if (hover.i != null) ctx.strokeRect(gutter + 0.5, gridTop + hover.i * cell + 0.5, D * cell - 1, cell - 1);
    }

    // rotated feature labels
    ctx.save();
    ctx.font = `${labelFont(cell)}px 'JetBrains Mono', monospace`;
    ctx.textBaseline = "middle";
    for (let d = 0; d < D; d++) {
      ctx.save();
      ctx.translate(gutter + d * cell + cell / 2, L.labelTop);
      ctx.rotate(Math.PI / 2);
      ctx.fillStyle = rabbit[d] ? C.accent : hover?.d === d ? C.text : C.muted;
      ctx.fillText(names[d], 0, 0);
      ctx.restore();
    }
    ctx.restore();
  }

  function sizeGrid() {
    const w = gridCv.parentElement.clientWidth;
    const L = gridLayout(w);
    // longest name is 27 chars; monospace glyphs are ~0.6em wide
    gridCv.style.height = `${L.labelTop + Math.ceil(27 * 0.62 * labelFont(L.cell)) + 8}px`;
  }

  // ---------- convergence ----------
  const bestF = H.map((h) => h.best_fitness);
  const meanF = H.map((h) => h.mean_fitness);
  const nSel = H.map((h) => h.best_n_selected);
  const sortedMean = [...meanF].sort((a, b) => a - b);
  const yMax = Math.max(bestF[0], sortedMean[Math.floor(sortedMean.length * 0.9)]) * 1.08;
  const yMin = Math.min(...bestF) * 0.94;

  function drawConv() {
    const { ctx, w, h } = fitCanvas(convCv);
    const p = { l: 52, r: 52, t: 14, b: 26 };
    const X = (i) => p.l + (i / T) * (w - p.l - p.r);
    const Y = (v) => p.t + (1 - (Math.min(v, yMax) - yMin) / (yMax - yMin)) * (h - p.t - p.b);
    const Yn = (n) => p.t + (1 - n / D) * (h - p.t - p.b);
    ctx.clearRect(0, 0, w, h);
    axes(ctx, w, h, p, yMin, yMax, (v) => v.toFixed(4), T);

    // right axis: features selected
    ctx.fillStyle = C.teal; ctx.font = "10px 'JetBrains Mono', monospace"; ctx.textAlign = "left";
    [0, 20, 41].forEach((n) => ctx.fillText(n, w - p.r + 6, Yn(n)));
    ctx.textAlign = "start";

    const line = (arr, y, color, upto, dash = [], width = 2) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i <= upto; i++) i ? ctx.lineTo(X(i), y(arr[i])) : ctx.moveTo(X(i), y(arr[i]));
      ctx.stroke(); ctx.setLineDash([]);
    };
    // ghost of full run, then the played part
    ctx.globalAlpha = 0.18; line(bestF, Y, C.accent, T); line(nSel, Yn, C.teal, T, [], 1.5); ctx.globalAlpha = 1;
    line(meanF, Y, "rgba(138,152,176,.7)", t, [4, 4], 1.2);
    line(nSel, Yn, C.teal, t, [], 1.5);
    line(bestF, Y, C.accent, t, [], 2.2);

    ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(t), p.t); ctx.lineTo(X(t), h - p.b); ctx.stroke();
    ctx.fillStyle = C.accent;
    ctx.beginPath(); ctx.arc(X(t), Y(bestF[t]), 4, 0, Math.PI * 2); ctx.fill();
  }

  // ---------- escaping energy ----------
  function drawEnergy() {
    const { ctx, w, h } = fitCanvas(energyCv);
    const p = { l: 52, r: 52, t: 10, b: 22 };
    const X = (i) => p.l + (i / T) * (w - p.l - p.r);
    const Y = (e) => p.t + (1 - (e + 2) / 4) * (h - p.t - p.b);
    ctx.clearRect(0, 0, w, h);

    // |E| >= 1 exploration band
    ctx.fillStyle = "rgba(107,124,255,.07)";
    ctx.fillRect(p.l, Y(2), w - p.l - p.r, Y(1) - Y(2));
    ctx.fillRect(p.l, Y(-1), w - p.l - p.r, Y(-2) - Y(-1));
    ctx.fillStyle = "rgba(255,179,71,.05)";
    ctx.fillRect(p.l, Y(0.5), w - p.l - p.r, Y(-0.5) - Y(0.5));

    ctx.font = "10px 'JetBrains Mono', monospace"; ctx.fillStyle = C.muted; ctx.textAlign = "right";
    [-2, -1, -0.5, 0, 0.5, 1, 2].forEach((e) => {
      ctx.fillText(e, p.l - 8, Y(e));
      ctx.strokeStyle = e === 0 ? "rgba(255,255,255,.18)" : C.grid;
      ctx.beginPath(); ctx.moveTo(p.l, Y(e)); ctx.lineTo(w - p.r, Y(e)); ctx.stroke();
    });
    ctx.textAlign = "left"; ctx.fillStyle = "rgba(107,124,255,.8)";
    ctx.fillText("explore", w - p.r + 4, Y(1.5));
    ctx.fillStyle = "rgba(255,179,71,.8)";
    ctx.fillText("hard", w - p.r + 4, Y(0));
    ctx.textAlign = "start";

    // envelope ±2(1 - t/T)
    ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.setLineDash([3, 4]);
    for (const sgn of [1, -1]) {
      ctx.beginPath(); ctx.moveTo(X(0), Y(sgn * 2)); ctx.lineTo(X(T), Y(0)); ctx.stroke();
    }
    ctx.setLineDash([]);

    for (let k = 1; k <= t; k++) {
      const alpha = k === t ? 1 : 0.45;
      H[k].energy.forEach((e, i) => {
        ctx.globalAlpha = alpha;
        ctx.fillStyle = STRATEGY_COLORS[H[k].strategy[i]];
        ctx.fillRect(X(k) - 1.2, Y(e) - 1.2, k === t ? 4 : 2.4, k === t ? 4 : 2.4);
      });
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "rgba(255,255,255,.35)";
    ctx.beginPath(); ctx.moveTo(X(t), p.t); ctx.lineTo(X(t), h - p.b); ctx.stroke();
  }

  // ---------- side stats ----------
  function updateStats() {
    $("iter").textContent = `${t} / ${T}`;
    $("best").textContent = bestF[t].toFixed(5);
    $("nsel").textContent = `${nSel[t]} / ${D}`;
    $("emax").textContent = (2 * (1 - Math.max(0, t - 1) / T)).toFixed(2);
    const counts = new Array(6).fill(0);
    H[t].strategy.forEach((s) => s != null && counts[s]++);
    legendCounts.forEach((b, i) => (b.textContent = counts[i]));
    const prevBest = best[Math.max(0, t - 1)];
    chips.forEach((el, d) => {
      const on = best[t][d], was = prevBest[d];
      el.classList.toggle("on", !!on);
      el.classList.toggle("added", t > 0 && on && !was);
      el.classList.toggle("removed", t > 0 && !on && was);
    });
    slider.value = t;
    slider.style.setProperty("--p", `${(t / T) * 100}%`);
  }

  function setT(next, animate = true) {
    next = Math.max(0, Math.min(T, next));
    if (next === t) return;
    t = next;
    changedAt = animate ? performance.now() : 0;
    updateStats();
    drawConv();
    drawEnergy();
  }

  function setPlaying(p) {
    playing = p;
    playBtn.textContent = p ? "Pause" : t >= T ? "Replay" : "Play";
    playBtn.setAttribute("aria-pressed", p);
  }

  playBtn.addEventListener("click", () => {
    if (!playing && t >= T) setT(0, false);
    setPlaying(!playing);
  });
  $("prev").addEventListener("click", () => { setPlaying(false); setT(t - 1); });
  $("next").addEventListener("click", () => { setPlaying(false); setT(t + 1); });
  slider.addEventListener("input", () => { setPlaying(false); setT(+slider.value, false); });
  speedSel.addEventListener("change", () => (speed = +speedSel.value));
  root.addEventListener("keydown", (e) => {
    if (e.target === speedSel) return;
    if (e.key === " ") { e.preventDefault(); playBtn.click(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); setPlaying(false); setT(t + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); setPlaying(false); setT(t - 1); }
  });

  gridCv.addEventListener("mousemove", (e) => {
    if (!layout) return;
    const r = gridCv.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const { gutter, cell, top, gridTop } = layout;
    const d = Math.floor((x - gutter) / cell);
    const inCols = d >= 0 && d < D;
    const i = Math.floor((y - gridTop) / cell);
    const inRows = i >= 0 && i < N;
    const onRabbit = y >= top && y < top + cell;
    if (!inCols || !(inRows || onRabbit || y < top)) { hover = null; tip.hidden = true; return; }
    hover = { d, i: inRows ? i : null };
    let on = 0;
    for (let k = 0; k < N; k++) on += bits[t][k][d];
    const who = inRows
      ? `Hawk ${i + 1} · ${bits[t][i][d] ? "uses" : "drops"} it · ${H[t].strategy[i] == null ? "initial" : STRATEGY_LABELS[H[t].strategy[i]]}`
      : `Best-so-far ${best[t][d] ? "uses" : "drops"} it`;
    tip.innerHTML = `<b>${names[d]}</b><span>${who}</span><span>${on}/${N} hawks use it</span>`;
    tip.hidden = false;
    const tx = Math.min(x + 14, r.width - tip.offsetWidth - 4);
    tip.style.transform = `translate(${tx}px, ${y + 14}px)`;
  });
  gridCv.addEventListener("mouseleave", () => { hover = null; tip.hidden = true; });

  function frame(now) {
    if (playing) {
      acc += now - (last || now);
      if (acc >= BASE_STEP_MS / speed) {
        acc = 0;
        setT(t + 1);
        if (t >= T) setPlaying(false);
      }
    }
    last = now;
    drawGrid(now);
    requestAnimationFrame(frame);
  }

  const relayout = () => { sizeGrid(); drawConv(); drawEnergy(); };
  new ResizeObserver(relayout).observe(root);
  relayout();
  updateStats();

  // autoplay once when scrolled into view (respecting reduced motion)
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce) {
    const io = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) { setPlaying(true); io.disconnect(); }
    }, { threshold: 0.35 });
    io.observe(gridCv);
  }
  requestAnimationFrame(frame);
}

function renderMetrics(el, run) {
  const t = run.test, b = run.baseline_test;
  if (!t) return;
  const rows = [
    ["Test accuracy", "accuracy", true, pct],
    ["Macro-F1", "macro_f1", true, pct],
    ["Detection rate", "detection_rate", true, pct],
    ["False alarm rate", "false_alarm_rate", false, pct],
    ["Features", "n_features", false, (v) => v],
  ];
  el.innerHTML = rows.map(([label, k, higherBetter, fmt]) => {
    const v = t[k], base = b?.[k];
    let delta = "";
    if (base != null) {
      const diff = v - base;
      const good = higherBetter ? diff > 0 : diff < 0;
      const txt = k === "n_features" ? `${diff > 0 ? "+" : ""}${diff}` : `${diff > 0 ? "+" : ""}${(diff * 100).toFixed(1)} pp`;
      delta = `<span class="delta ${diff === 0 ? "" : good ? "good" : "bad"}">${txt}</span>`;
    }
    return `<div class="metric"><span class="label">${label}</span><strong>${fmt(v)}</strong>${delta}
      ${base != null ? `<span class="base">all features: ${fmt(base)}</span>` : ""}</div>`;
  }).join("");
}

// rotated labels sit one per column, so the font can't be taller than a cell
const labelFont = (cell) => Math.max(7, Math.min(10, cell - 1));

const pct = (v) =>`${(v * 100).toFixed(1)}%`;

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
  ctx.fill();
}

function axes(ctx, w, h, p, lo, hi, fmt, T) {
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo);
    const y = p.t + (1 - k / 4) * (h - p.t - p.b);
    ctx.strokeStyle = C.grid;
    ctx.beginPath(); ctx.moveTo(p.l, y); ctx.lineTo(w - p.r, y); ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.fillText(fmt(v), p.l - 6, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let k = 0; k <= 4; k++) {
    ctx.fillText(`t${Math.round((k * T) / 4)}`,p.l + (k / 4) * (w - p.l - p.r), h - p.b + 8);
  }
  ctx.textAlign = "start";
  ctx.textBaseline = "middle";
}
