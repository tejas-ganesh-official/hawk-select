// Section 04: 30-seed comparison of HHO against GA, PSO, filters and all features.
import { C, ALGO_COLORS, GROUP_COLORS, fitCanvas } from "./theme.js";
import { MONO, linear, ticks, drawAxes, tooltip, localXY, onResize, pct } from "./charts.js";

const GROUPS = [["basic", 9], ["content", 13], ["traffic", 9], ["host", 10]];
const GROUP_OF = GROUPS.flatMap(([g, n]) => Array(n).fill(g));

export async function initResults(root) {
  const data = await (await fetch("data/results.json")).json();
  const algos = data.algos;
  const byKey = Object.fromEntries(algos.map((a) => [a.key, a]));
  const visible = new Set(algos.map((a) => a.key));
  const $ = (k) => root.querySelector(`[data-res=${k}]`);
  const redraws = [];
  const redrawAll = () => redraws.forEach((f) => f());

  // ----- algorithm toggles -----
  $("toggles").innerHTML = algos.map((a) =>
    `<button class="toggle on" data-key="${a.key}" style="--c:${ALGO_COLORS[a.key]}" aria-pressed="true">
       <i></i>${a.label}</button>`).join("");
  $("toggles").addEventListener("click", (e) => {
    const b = e.target.closest(".toggle");
    if (!b) return;
    const k = b.dataset.key;
    visible.has(k) ? visible.delete(k) : visible.add(k);
    b.classList.toggle("on", visible.has(k));
    b.setAttribute("aria-pressed", visible.has(k));
    redrawAll();
  });

  summaryTable($("table"), algos);
  redraws.push(convergence($("conv"), algos, visible));
  redraws.push(pareto($("pareto"), algos, visible));
  redraws.push(radar($("radar"), algos, visible));
  redraws.push(selection($("freq"), algos, visible, data.feature_names));
  confusion($("cm"), algos, data.class_names, byKey);
}

// ---------- summary table ----------
function summaryTable(host, algos) {
  const cols = [
    ["fitness", "Fitness ↓", (v) => v.toFixed(4), false],
    ["n_selected", "Features ↓", (v) => v.toFixed(1), false],
    ["accuracy", "Test acc ↑", (v) => pct(v), true],
    ["macro_f1", "Macro-F1 ↑", (v) => pct(v), true],
    ["detection_rate", "Detection ↑", (v) => pct(v), true],
    ["false_alarm_rate", "False alarms ↓", (v) => pct(v, 2), false],
    ["fitness_evals", "Evaluations", (v) => Math.round(v).toLocaleString(), null],
    ["runtime_s", "Time / run", (v) => (v < 60 ? `${v.toFixed(1)} s` : `${(v / 60).toFixed(1)} min`), null],
  ];
  const best = Object.fromEntries(cols.map(([k, , , hi]) => {
    if (hi === null) return [k, null];
    const vals = algos.filter((a) => a.key !== "all").map((a) => a.mean[k]);
    return [k, hi ? Math.max(...vals) : Math.min(...vals)];
  }));
  host.innerHTML = `<table>
    <thead><tr><th>Method</th>${cols.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead>
    <tbody>${algos.map((a) => `<tr>
      <th><i class="sw-dot" style="--c:${ALGO_COLORS[a.key]}"></i>${a.label}</th>
      ${cols.map(([k, , fmt]) => {
        const isBest = best[k] !== null && a.key !== "all" && a.mean[k] === best[k];
        const sd = a.n_runs > 1 && a.std[k] > 0 && k !== "fitness_evals" && k !== "runtime_s"
          ? `<small>±${k === "n_selected" ? a.std[k].toFixed(1) : k === "fitness" ? a.std[k].toFixed(4) : (a.std[k] * 100).toFixed(1)}</small>` : "";
        return `<td class="${isBest ? "best" : ""}">${fmt(a.mean[k])}${sd}</td>`;
      }).join("")}
    </tr>`).join("")}</tbody></table>
    <p class="note">Mean over ${algos[0].n_runs} seeds (± one standard deviation). Best non-baseline value per column highlighted. Test = KNN (k=5) on KDDTest+.</p>`;
}

// ---------- convergence with ±std bands ----------
function convergence(host, algos, visible) {
  const cv = host.querySelector("canvas");
  const tip = tooltip(host);
  const iter = algos.filter((a) => a.convergence);
  const T = iter[0].convergence.mean.length - 1;
  let hoverT = null;

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    const area = { l: 58, r: w - 12, t: 12, b: h - 40 };
    const shown = iter.filter((a) => visible.has(a.key));
    const lo = Math.min(...shown.flatMap((a) => a.convergence.mean.map((m, i) => m - a.convergence.std[i])), 0.02);
    const hi = Math.max(...shown.flatMap((a) => a.convergence.mean.slice(2).map((m, i) => m + a.convergence.std[i + 2])), 0);
    const x = linear([0, T], [area.l, area.r]);
    const y = linear([Math.max(0, lo * 0.95), hi * 1.03 || 0.02], [area.b, area.t]);
    ctx.clearRect(0, 0, w, h);
    drawAxes(ctx, { x, y, area, xTicks: ticks(0, T, 5), yTicks: ticks(y.invert(area.b), y.invert(area.t), 4),
      fmtY: (v) => v.toFixed(3), xLabel: "iteration", yLabel: "best fitness (lower is better)" });

    ctx.save();
    ctx.beginPath(); ctx.rect(area.l, area.t, area.r - area.l, area.b - area.t); ctx.clip();
    for (const a of shown) {
      const { mean, std } = a.convergence, col = ALGO_COLORS[a.key];
      ctx.fillStyle = col; ctx.globalAlpha = 0.12;
      ctx.beginPath();
      mean.forEach((m, i) => (i ? ctx.lineTo(x(i), y(m + std[i])) : ctx.moveTo(x(i), y(m + std[i]))));
      for (let i = T; i >= 0; i--) ctx.lineTo(x(i), y(mean[i] - std[i]));
      ctx.fill();
      ctx.globalAlpha = 1; ctx.strokeStyle = col; ctx.lineWidth = 2.2;
      ctx.beginPath();
      mean.forEach((m, i) => (i ? ctx.lineTo(x(i), y(m)) : ctx.moveTo(x(i), y(m))));
      ctx.stroke();
    }
    ctx.restore();

    if (hoverT != null) {
      ctx.strokeStyle = "rgba(255,255,255,.35)";
      ctx.beginPath(); ctx.moveTo(x(hoverT), area.t); ctx.lineTo(x(hoverT), area.b); ctx.stroke();
      for (const a of shown) {
        ctx.fillStyle = ALGO_COLORS[a.key];
        ctx.beginPath(); ctx.arc(x(hoverT), y(a.convergence.mean[hoverT]), 4, 0, Math.PI * 2); ctx.fill();
      }
    }
    draw.x = x; draw.area = area;
  }

  cv.addEventListener("mousemove", (e) => {
    const [px, py] = localXY(e, cv);
    const { x, area } = draw;
    if (px < area.l || px > area.r) { hoverT = null; tip.hide(); draw(); return; }
    hoverT = Math.round(x.invert(px));
    draw();
    const rows = iter.filter((a) => visible.has(a.key))
      .sort((a, b) => a.convergence.mean[hoverT] - b.convergence.mean[hoverT])
      .map((a) => `<span><i class="sw-dot" style="--c:${ALGO_COLORS[a.key]}"></i>${a.label}: ${a.convergence.mean[hoverT].toFixed(4)} · ${a.convergence.nsel_mean[hoverT].toFixed(1)} feat.</span>`);
    tip.show(`<b>iteration ${hoverT}</b>${rows.join("")}`, px, py);
  });
  cv.addEventListener("mouseleave", () => { hoverT = null; tip.hide(); draw(); });
  onResize(cv, draw);
  return draw;
}

// ---------- Pareto: features vs test accuracy, every run ----------
function pareto(host, algos, visible) {
  const cv = host.querySelector("canvas");
  const tip = tooltip(host);
  const pts = algos.flatMap((a) => a.runs.map((r) => ({ a, r })));
  let near = null;

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    const area = { l: 50, r: w - 14, t: 12, b: h - 40 };
    const accs = pts.map((p) => p.r.accuracy);
    const x = linear([0, 42], [area.l, area.r]);
    const y = linear([Math.min(...accs) - 0.005, Math.max(...accs) + 0.005], [area.b, area.t]);
    ctx.clearRect(0, 0, w, h);
    drawAxes(ctx, { x, y, area, xTicks: [0, 10, 20, 30, 41], yTicks: ticks(y.invert(area.b), y.invert(area.t), 4),
      fmtY: (v) => pct(v, 0), xLabel: "features selected", yLabel: "test accuracy" });

    // Pareto front over visible runs (fewer features, higher accuracy)
    const vis = pts.filter((p) => visible.has(p.a.key)).sort((p, q) => p.r.n_selected - q.r.n_selected || q.r.accuracy - p.r.accuracy);
    const front = [];
    let bestAcc = -1;
    for (const p of vis) if (p.r.accuracy > bestAcc) { front.push(p); bestAcc = p.r.accuracy; }
    if (front.length > 1) {
      ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.setLineDash([4, 4]);
      ctx.beginPath();
      front.forEach((p, i) => {
        const px = x(p.r.n_selected), py = y(p.r.accuracy);
        if (i) ctx.lineTo(px, y(front[i - 1].r.accuracy));
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.stroke(); ctx.setLineDash([]);
    }

    for (const p of vis) {
      // jitter identical points a little so stacks stay visible
      const jx = ((p.r.seed * 7919) % 11 - 5) * 0.6;
      p.px = x(p.r.n_selected) + jx; p.py = y(p.r.accuracy);
      ctx.fillStyle = ALGO_COLORS[p.a.key];
      ctx.globalAlpha = near && near !== p ? 0.45 : 0.85;
      ctx.beginPath(); ctx.arc(p.px, p.py, near === p ? 6 : 3.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // per-algorithm mean
    for (const a of algos.filter((a) => visible.has(a.key))) {
      const px = x(a.mean.n_selected), py = y(a.mean.accuracy);
      ctx.strokeStyle = ALGO_COLORS[a.key]; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(px - 7, py); ctx.lineTo(px + 7, py); ctx.moveTo(px, py - 7); ctx.lineTo(px, py + 7); ctx.stroke();
      ctx.lineWidth = 1;
    }
    draw.vis = vis;
  }

  cv.addEventListener("mousemove", (e) => {
    const [mx, my] = localXY(e, cv);
    let best = null, bd = 144;
    for (const p of draw.vis || []) {
      const d = (p.px - mx) ** 2 + (p.py - my) ** 2;
      if (d < bd) { bd = d; best = p; }
    }
    if (best !== near) { near = best; draw(); }
    if (!best) return tip.hide();
    const r = best.r;
    tip.show(`<b>${best.a.label} · seed ${r.seed}</b><span>${r.n_selected} features · acc ${pct(r.accuracy, 2)}</span>
      <span>detection ${pct(r.detection_rate)} · false alarms ${pct(r.false_alarm_rate, 2)}</span>`, mx, my);
  });
  cv.addEventListener("mouseleave", () => { near = null; tip.hide(); draw(); });
  onResize(cv, draw);
  return draw;
}

// ---------- radar ----------
function radar(host, allAlgos, visible) {
  // "all features" is the reference point, not a selection method; including it would
  // stretch the compactness and speed axes so far that the real methods all collapse.
  const algos = allAlgos.filter((a) => a.key !== "all");
  const cv = host.querySelector("canvas");
  const tip = tooltip(host);
  const maxRt = Math.max(...algos.map((a) => a.mean.runtime_s));
  const axes = [
    ["Accuracy", (a) => a.mean.accuracy, (v) => pct(v)],
    ["Macro-F1", (a) => a.mean.macro_f1, (v) => pct(v)],
    ["Detection", (a) => a.mean.detection_rate, (v) => pct(v)],
    ["Low false alarms", (a) => 1 - a.mean.false_alarm_rate, (v) => pct(1 - v, 2) + " FAR"],
    ["Compactness", (a) => 1 - a.mean.n_selected / 41, (v) => `${Math.round((1 - v) * 41)} feat.`],
    ["Speed", (a) => 1 - Math.log10(1 + a.mean.runtime_s) / Math.log10(1 + maxRt), (_, a) => `${a.mean.runtime_s.toFixed(1)} s/run`],
  ];
  // min-max scale each axis across methods so differences of a few % are visible
  const ranges = axes.map(([, f]) => {
    const vs = algos.map(f);
    return [Math.min(...vs), Math.max(...vs)];
  });
  const norm = (k, v) => 0.15 + 0.85 * ((v - ranges[k][0]) / (ranges[k][1] - ranges[k][0] || 1));
  let hoverAlgo = null;

  function draw() {
    const { ctx, w, h } = fitCanvas(cv);
    const cx = w / 2, cy = h / 2 + 4, R = Math.min(w, h) / 2 - 46;
    const ang = (k) => -Math.PI / 2 + (k / axes.length) * Math.PI * 2;
    const P = (k, v) => [cx + Math.cos(ang(k)) * R * v, cy + Math.sin(ang(k)) * R * v];
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = C.grid;
    for (const ring of [0.25, 0.5, 0.75, 1]) {
      ctx.beginPath();
      axes.forEach((_, k) => { const [px, py] = P(k, ring); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath(); ctx.stroke();
    }
    ctx.font = `11px ${MONO}`; ctx.fillStyle = C.muted; ctx.textBaseline = "middle";
    axes.forEach(([label], k) => {
      const [px, py] = P(k, 1);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
      const [lx, ly] = P(k, 1.16);
      ctx.textAlign = Math.abs(lx - cx) < 4 ? "center" : lx > cx ? "left" : "right";
      ctx.fillText(label, lx, ly);
    });
    ctx.textAlign = "start";
    const shown = algos.filter((a) => visible.has(a.key));
    for (const a of shown) {
      const col = ALGO_COLORS[a.key];
      const dim = hoverAlgo && hoverAlgo !== a;
      ctx.beginPath();
      axes.forEach(([, f], k) => { const [px, py] = P(k, norm(k, f(a))); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.closePath();
      ctx.fillStyle = col; ctx.globalAlpha = dim ? 0.02 : hoverAlgo === a ? 0.25 : 0.07; ctx.fill();
      ctx.globalAlpha = dim ? 0.2 : 1; ctx.strokeStyle = col; ctx.lineWidth = hoverAlgo === a ? 2.6 : 1.6; ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.lineWidth = 1;
    draw.geo = { cx, cy, R, P, shown };
  }

  // hover a legend row to focus one method
  const legend = host.querySelector(".radar-legend");
  legend.innerHTML = algos.map((a) => `<li data-key="${a.key}"><i class="sw-dot" style="--c:${ALGO_COLORS[a.key]}"></i>${a.label}</li>`).join("");
  legend.addEventListener("mouseover", (e) => {
    const li = e.target.closest("li");
    hoverAlgo = li ? algos.find((a) => a.key === li.dataset.key) : null;
    draw();
    if (hoverAlgo) {
      tip.show(`<b>${hoverAlgo.label}</b>${axes.map(([l, f, fmt]) => `<span>${l}: ${fmt(f(hoverAlgo), hoverAlgo)}</span>`).join("")}`, 8, 40);
    }
  });
  legend.addEventListener("mouseleave", () => { hoverAlgo = null; tip.hide(); draw(); });
  onResize(cv, draw);
  return draw;
}

// ---------- selection frequency strip: methods x features ----------
function selection(host, algos, visible, names) {
  const cv = host.querySelector("canvas");
  const tip = tooltip(host);
  const search = algos.filter((a) => a.key !== "all");
  // order features by how often the metaheuristics pick them (mean over HHO/GA/PSO)
  const meta = algos.filter((a) => ["hho_s", "hho_v", "ga", "pso"].includes(a.key));
  const order = names.map((_, d) => d).sort((i, j) =>
    meta.reduce((s, a) => s + a.selection_freq[j] - a.selection_freq[i], 0));
  let hover = null;

  function draw() {
    const rows = search.filter((a) => visible.has(a.key));
    const w0 = cv.parentElement.clientWidth;
    const labelW = 92, cell = Math.max(8, (w0 - labelW - 4) / names.length);
    const labelH = 150;
    cv.style.height = `${rows.length * (cell + 6) + labelH + 10}px`;
    const { ctx, w } = fitCanvas(cv);
    ctx.clearRect(0, 0, w, cv.clientHeight);
    const rowH = cell + 6;
    ctx.font = `11px ${MONO}`; ctx.textBaseline = "middle";
    rows.forEach((a, r) => {
      ctx.fillStyle = ALGO_COLORS[a.key];
      ctx.fillText(a.label, 0, r * rowH + rowH / 2);
      order.forEach((d, c) => {
        const f = a.selection_freq[d];
        const x = labelW + c * cell, y = r * rowH + 3;
        ctx.fillStyle = "rgba(255,255,255,.04)";
        ctx.fillRect(x + 0.5, y, cell - 1, cell);
        ctx.fillStyle = ALGO_COLORS[a.key];
        ctx.globalAlpha = f;
        ctx.fillRect(x + 0.5, y, cell - 1, cell);
        ctx.globalAlpha = 1;
        if (hover && hover.r === r && hover.c === c) {
          ctx.strokeStyle = "#fff"; ctx.strokeRect(x + 0.5, y, cell - 1, cell);
        }
      });
    });
    // feature labels with group colour tick
    const top = rows.length * rowH + 6;
    order.forEach((d, c) => {
      const x = labelW + c * cell + cell / 2;
      ctx.fillStyle = GROUP_COLORS[GROUP_OF[d]];
      ctx.fillRect(x - cell / 2 + 1, top, cell - 2, 3);
      ctx.save();
      ctx.translate(x, top + 8);
      ctx.rotate(Math.PI / 2);
      ctx.font = `${Math.min(10, cell - 1)}px ${MONO}`;
      ctx.fillStyle = hover?.c === c ? C.text : C.muted;
      ctx.fillText(names[d], 0, 0);
      ctx.restore();
    });
    draw.geo = { rows, labelW, cell, rowH };
  }

  cv.addEventListener("mousemove", (e) => {
    const [mx, my] = localXY(e, cv);
    const { rows, labelW, cell, rowH } = draw.geo;
    const c = Math.floor((mx - labelW) / cell), r = Math.floor(my / rowH);
    if (c < 0 || c >= names.length || r < 0 || r >= rows.length) { hover = null; tip.hide(); draw(); return; }
    hover = { r, c };
    draw();
    const d = order[c], a = rows[r];
    tip.show(`<b>${names[d]}</b><span>${GROUP_OF[d]} feature</span><span>${a.label}: picked in ${Math.round(a.selection_freq[d] * a.n_runs)}/${a.n_runs} runs</span>`, mx, my);
  });
  cv.addEventListener("mouseleave", () => { hover = null; tip.hide(); draw(); });
  onResize(host, draw);
  return draw;
}

// ---------- confusion matrices ----------
function confusion(host, algos, classNames, byKey) {
  const sel = host.querySelector("select");
  const grid = host.querySelector(".cm-grid");
  sel.innerHTML = algos.map((a) => `<option value="${a.key}">${a.label}</option>`).join("");
  sel.value = "hho_s";

  function render() {
    const a = byKey[sel.value], base = byKey.all;
    const n = classNames.length;
    let html = `<div class="cm" style="--n:${n}"><span class="cm-corner">true ↓ / predicted →</span>`;
    html += classNames.map((c) => `<span class="cm-head">${c}</span>`).join("");
    a.confusion.forEach((row, i) => {
      html += `<span class="cm-head row">${classNames[i]}</span>`;
      row.forEach((v, j) => {
        const diag = i === j;
        const d = v - base.confusion[i][j];
        const good = diag ? d > 0 : d < 0;
        const delta = a.key === "all" || Math.abs(d) < 0.0005 ? "" :
          `<small class="${good ? "good" : "bad"}">${d > 0 ? "+" : ""}${(d * 100).toFixed(1)} pp</small>`;
        html += `<div class="cm-cell ${diag ? "diag" : "off"}" style="--a:${diag ? v : Math.min(1, v * 3)}">
          <strong>${pct(v)}</strong>${delta}</div>`;
      });
    });
    grid.innerHTML = html + "</div>";
  }
  sel.addEventListener("change", render);
  render();
}
