// Analyzer UI: load a CSV, configure the target and hunt, stream the run, then show insights.
import { parseCSV, profile, prepare, targetProblem, skipReason, LIMITS } from "./dataset.js";
import { renderReport } from "./insights.js";
import { STRATEGY_COLORS, STRATEGY_LABELS, C, fitCanvas } from "../theme.js";
import { tooltip, localXY, onResize, pct } from "../charts.js";

const DEFAULTS = { hawks: 16, iterations: 50, alpha: 0.99, transfer: "s", k: 5, seed: 7 };

export function initApp(root) {
  const $ = (k) => root.querySelector(`[data-el="${k}"]`);
  const act = (k) => root.querySelector(`[data-act="${k}"]`);
  const S = { parsed: null, prof: null, fileName: "", target: null, included: new Set(), prep: null, worker: null, run: null };

  // ---------- navigation ----------
  function go(step) {
    root.querySelectorAll("[data-step]").forEach((el) => (el.hidden = +el.dataset.step !== step));
    root.querySelectorAll("[data-step-tab]").forEach((li) => {
      const n = +li.dataset.stepTab;
      li.classList.toggle("current", n === step);
      li.classList.toggle("done", n < step);
    });
    root.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  // ---------- step 1: load ----------
  const drop = $("drop"), fileInput = $("file");
  drop.addEventListener("click", (e) => { if (e.target !== fileInput) { e.preventDefault(); fileInput.click(); } });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
  fileInput.addEventListener("change", () => fileInput.files[0] && loadFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));

  fetch("data/samples/index.json").then((r) => r.json()).then((list) => {
    $("samples").innerHTML = list.map((s, i) => `
      <button class="sample" data-i="${i}">
        <strong>${s.name}</strong><span>${s.desc}</span><em>Load →</em>
      </button>`).join("");
    $("samples").addEventListener("click", async (e) => {
      const b = e.target.closest(".sample");
      if (!b) return;
      const s = list[+b.dataset.i];
      b.classList.add("loading");
      try {
        const text = await (await fetch(`data/samples/${s.file}`)).text();
        ingest(text, s.file, s.target);
      } catch (err) { showLoadError(err); }
      b.classList.remove("loading");
    });
  });

  function showLoadError(err) {
    const el = $("load-error");
    el.textContent = err.message || String(err);
    el.hidden = false;
  }

  async function loadFile(file) {
    $("load-error").hidden = true;
    if (file.size > LIMITS.maxBytes) return showLoadError(new Error(`That file is ${(file.size / 1048576).toFixed(1)} MB; the limit is 15 MB.`));
    try { ingest(await file.text(), file.name); } catch (err) { showLoadError(err); }
    fileInput.value = "";
  }

  function ingest(text, name, preferredTarget) {
    $("load-error").hidden = true;
    try {
      const parsed = parseCSV(text);
      if (parsed.header.length < 3) throw new Error("Need at least two feature columns plus a target column. Is the file comma, semicolon or tab separated?");
      const prof = profile(parsed);
      S.parsed = parsed; S.prof = prof; S.fileName = name;
      const pref = prof.cols.find((c) => c.name === preferredTarget);
      setTarget(pref ? pref.index : prof.target);
      renderConfig();
      go(2);
    } catch (err) { showLoadError(err); }
  }

  // ---------- step 2: configure ----------
  const settings = Object.fromEntries(["hawks", "iterations", "alpha", "transfer", "k", "seed"].map((n) => [n, root.querySelector(`[name="${n}"]`)]));
  const showSetting = () => {
    root.querySelector('[data-o="hawks"]').textContent = settings.hawks.value;
    root.querySelector('[data-o="iterations"]').textContent = settings.iterations.value;
    const a = +settings.alpha.value;
    root.querySelector('[data-o="alpha"]').textContent = a >= 0.995 ? "accuracy first" : a >= 0.975 ? "balanced" : "lean subsets";
    estimate();
  };
  Object.values(settings).forEach((el) => el.addEventListener("input", showSetting));
  act("defaults").addEventListener("click", () => {
    for (const [k, v] of Object.entries(DEFAULTS)) settings[k].value = v;
    showSetting();
  });
  showSetting();
  act("change-file").addEventListener("click", () => go(1));

  function setTarget(idx) {
    S.target = idx;
    // default feature set: everything except the target and columns with a skipReason
    S.included = new Set(S.prof.cols.filter((c) => c.index !== idx && !skipReason(c)).map((c) => c.index));
  }

  function renderConfig() {
    const { prof, parsed } = S;
    $("file-name").textContent = S.fileName;
    const missingCells = prof.cols.reduce((a, c) => a + c.missing, 0);
    const t = prof.cols[S.target];
    $("kpis").innerHTML = [
      [prof.n.toLocaleString(), prof.n > LIMITS.maxRows ? `rows (sampling ${LIMITS.maxRows.toLocaleString()})` : "rows"],
      [S.included.size, "candidate features"],
      [t.unique, "classes to predict"],
      missingCells ? [pct(missingCells / (prof.n * prof.cols.length)), "missing cells (filled in)"] : ["0", "missing values"],
    ].map(([v, l]) => `<div class="kpi"><strong>${v}</strong><span>${l}</span></div>`).join("");

    $("target").innerHTML = prof.cols.map((c) => `<option value="${c.index}" ${c.index === S.target ? "selected" : ""}>${esc(c.name)} · ${c.unique} values</option>`).join("");
    $("target-hint").textContent = S.target === prof.target
      ? "Guessed from the column names. Double-check it is the outcome you care about."
      : "";
    const problem = targetProblem(t, prof.n);
    $("target-error").hidden = !problem;
    $("target-error").textContent = problem || "";

    const counts = t.unique <= 20 ? classCounts(parsed, S.target) : [];
    const max = Math.max(1, ...counts.map((c) => c[1]));
    const total = counts.reduce((a, c) => a + c[1], 0);
    $("classes").innerHTML = counts.map(([name, n]) => `
      <div class="class-row"><span class="mono">${esc(name)}</span>
        <span class="class-bar"><i style="width:${(n / max) * 100}%"></i></span>
        <span class="mono muted">${pct(n / total)}</span></div>`).join("");

    const flag = (c) => c.index === S.target ? `<span class="badge target">target</span>`
      : skipReason(c) ? `<span class="badge warn">${skipReason(c)}</span>` : "";
    $("columns").innerHTML = `<thead><tr><th>Use</th><th>Column</th><th>Type</th><th>Missing</th><th>Distinct</th><th>Examples</th><th></th></tr></thead>
      <tbody>${prof.cols.map((c) => `<tr class="${c.index === S.target ? "is-target" : S.included.has(c.index) ? "" : "off"}">
        <td><input type="checkbox" data-col="${c.index}" ${S.included.has(c.index) ? "checked" : ""} ${c.index === S.target ? "disabled" : ""} aria-label="Use ${esc(c.name)}"></td>
        <td class="mono">${esc(c.name)}</td>
        <td><span class="badge ${c.type}">${c.type === "numeric" ? "number" : "category"}</span></td>
        <td class="mono">${c.missing ? pct(c.missing / prof.n) : "–"}</td>
        <td class="mono">${c.unique.toLocaleString()}</td>
        <td class="examples">${c.top.slice(0, 3).map((t) => esc(t.value)).join(", ")}</td>
        <td>${flag(c)}</td></tr>`).join("")}</tbody>`;
    $("col-summary").textContent = `${S.included.size} of ${prof.cols.length - 1} columns used as features`;
    act("run").disabled = !!problem || S.included.size < 2;
    estimate();
  }

  $("target").addEventListener("change", (e) => { setTarget(+e.target.value); renderConfig(); });
  $("columns").addEventListener("change", (e) => {
    const j = e.target.dataset.col;
    if (j == null) return;
    e.target.checked ? S.included.add(+j) : S.included.delete(+j);
    renderConfig();
  });

  function estimate() {
    if (!S.prof) return;
    const D = S.included.size, n = Math.min(S.prof.n, LIMITS.maxRows);
    // mirrors prepare(): 5-fold CV up to 1,200 train+val rows, else one capped holdout split
    const pool = n * 0.8;
    const pairs = pool <= 1200 ? 5 * (pool * 0.8) * (pool * 0.2) : Math.min(1500, n * 0.6) * Math.min(600, n * 0.2);
    // rough: ~2.5e8 multiply-adds per second, with early exit trimming roughly half
    const perEval = (pairs * Math.max(1, D / 2) * 0.5) / 2.5e8;
    const evalsN = D + +settings.hawks.value * (+settings.iterations.value + 1) * 1.3;
    const sec = perEval * evalsN;
    $("estimate").textContent = D < 2 ? "Select at least two feature columns." :
      `Searching ${D} features: about ${Math.round(evalsN).toLocaleString()} models, roughly ${sec < 60 ? Math.max(2, Math.round(sec)) + " seconds" : Math.round(sec / 60) + " minutes"} on a typical laptop.`;
  }

  // ---------- step 3: run ----------
  act("run").addEventListener("click", startRun);
  act("retune").addEventListener("click", () => go(2));
  act("new").addEventListener("click", () => go(1));
  act("cancel").addEventListener("click", () => {
    S.worker?.terminate(); S.worker = null;
    if (S.run) S.run.cancelled = true;
    go(2);
  });
  act("skip").addEventListener("click", () => { if (S.run) S.run.skip = true; });

  const flockCv = $("flock");
  const flockTip = tooltip(flockCv.parentElement);
  const curveCv = $("h-curve");
  const legendCounts = STRATEGY_LABELS.map((label, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="dot" style="--c:${STRATEGY_COLORS[i]}"></span><span>${label}</span><b>0</b>`;
    $("h-legend").appendChild(li);
    return li.querySelector("b");
  });

  function startRun() {
    const featureIdx = [...S.included].sort((a, b) => a - b);
    const config = Object.fromEntries(Object.entries(settings).map(([k, el]) => [k, k === "transfer" ? el.value : +el.value]));
    let prep;
    try { prep = prepare(S.parsed, S.prof, S.target, featureIdx, config.seed); }
    catch (err) { $("target-error").textContent = err.message; $("target-error").hidden = false; return; }
    if (prep.classNames.length < 2) { $("target-error").textContent = "The target has a single class after removing missing values."; $("target-error").hidden = false; return; }

    S.prep = prep;
    if (S.run) S.run.cancelled = true;
    S.run = { config, iters: [], last: null, baseline: null, hoverFeature: null, queue: [], lastShown: 0, pumping: false, skip: false };
    $("h-chips").innerHTML = prep.featureNames.map((n) => `<span class="chip">${esc(n)}</span>`).join("");
    $("h-baseline").textContent = "";
    setPhase("baseline");
    $("progress").style.width = "0%";
    go(3);

    S.worker?.terminate();
    const w = new Worker(new URL("./worker.js", import.meta.url));
    S.worker = w;
    w.onmessage = (e) => onWorker(e.data);
    w.onerror = (e) => { alert(`The analysis failed: ${e.message}`); go(2); };
    const strip = ({ X, y, n }) => ({ X, y, n });
    w.postMessage({
      type: "run", config,
      data: { D: prep.D, classNames: prep.classNames, isCat: prep.isCat, train: strip(prep.train), test: strip(prep.test),
        folds: prep.folds.map((f) => ({ train: strip(f.train), val: strip(f.val) })) },
    });
  }

  function setPhase(name) {
    const order = ["baseline", "hho", "final"];
    $("phases").querySelectorAll("span").forEach((s) => {
      const k = order.indexOf(s.dataset.phase), cur = order.indexOf(name);
      s.classList.toggle("current", k === cur);
      s.classList.toggle("done", k < cur);
    });
  }

  // Small datasets finish in well under a second. Worker messages are queued and played back
  // at a minimum pace so the hunt is visible; "Skip" flushes the queue.
  const MIN_HUNT_MS = 4000;
  function onWorker(m) {
    if (m.type === "error") {
      S.worker?.terminate(); S.worker = null;
      alert(`The analysis failed: ${m.message}`);
      go(2);
      return;
    }
    if (m.type === "done") { S.worker?.terminate(); S.worker = null; }
    S.run.queue.push(m);
    if (!S.run.pumping) { S.run.pumping = true; requestAnimationFrame(pump); }
  }

  function pump(now) {
    const R = S.run;
    if (!R || R.cancelled) return;
    const gap = R.skip ? 0 : MIN_HUNT_MS / R.config.iterations;
    while (R.queue.length) {
      const m = R.queue[0];
      if (m.type === "iter" && now - R.lastShown < gap) break;
      R.queue.shift();
      if (m.type === "iter") R.lastShown = now;
      show(m);
      if (m.type === "iter" && !R.skip) break;
    }
    if (R.queue.length || S.worker) requestAnimationFrame(pump);
    else R.pumping = false;
  }

  function show(m) {
    const R = S.run;
    if (m.type === "phase") setPhase(m.name);
    else if (m.type === "baseline") {
      R.baseline = m;
      $("h-baseline").textContent = `all ${S.prep.D} features: ${pct(1 - m.all.err)} validation accuracy · best statistical filter: ${pct(1 - m.filter.err)} with ${m.filter.k}`;
    } else if (m.type === "iter") {
      R.iters.push({ t: m.t, bestFit: m.bestFit, meanFit: m.meanFit });
      R.last = m;
      $("progress").style.width = `${(m.t / m.T) * 100}%`;
      $("h-iter").textContent = `${m.t} / ${m.T}`;
      $("h-acc").textContent = pct(1 - m.bestErr);
      const nsel = m.best.reduce((a, b) => a + b, 0);
      $("h-nsel").textContent = `${nsel} / ${S.prep.D}`;
      $("h-evals").textContent = m.evals.toLocaleString();
      $("h-energy").style.width = `${(1 - m.t / m.T) * 100}%`;
      const counts = new Array(6).fill(0);
      m.strategy.forEach((s) => s >= 0 && counts[s]++);
      legendCounts.forEach((b, i) => (b.textContent = counts[i]));
      $("h-chips").querySelectorAll(".chip").forEach((el, d) => el.classList.toggle("on", !!m.best[d]));
      drawFlock();
      drawCurve();
    } else if (m.type === "done") {
      $("progress").style.width = "100%";
      renderReport($("report"), { prep: S.prep, prof: S.prof, result: m.result, config: R.config, iters: R.iters, fileName: S.fileName, parsed: S.parsed });
      go(4);
    }
  }

  function drawFlock() {
    const m = S.run?.last;
    if (!m) return;
    const D = S.prep.D, N = m.masks.length / D;
    const w0 = flockCv.parentElement.clientWidth;
    const gutter = 28;
    const cell = Math.max(3, Math.min(18, Math.floor((w0 - gutter) / D)));
    const gap = 8;
    flockCv.style.width = `${gutter + cell * D}px`;
    flockCv.style.height = `${cell * (N + 1) + gap}px`;
    const { ctx, w, h } = fitCanvas(flockCv);
    ctx.clearRect(0, 0, w, h);
    const r = Math.min(2, cell / 4);
    const box = (x, y, on, col) => {
      ctx.fillStyle = on ? col : "rgba(255,255,255,.045)";
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x + 0.5, y + 0.5, cell - 1, cell - 1, r) : ctx.rect(x + 0.5, y + 0.5, cell - 1, cell - 1); ctx.fill();
    };
    ctx.fillStyle = C.accent; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.textBaseline = "middle";
    ctx.fillText("best", 0, cell / 2);
    for (let d = 0; d < D; d++) box(gutter + d * cell, 0, m.best[d], C.accent);
    for (let i = 0; i < N; i++) {
      const y = cell + gap + i * cell;
      const s = m.strategy[i];
      ctx.fillStyle = s >= 0 ? STRATEGY_COLORS[s] : C.muted;
      ctx.beginPath(); ctx.arc(10, y + cell / 2, Math.min(4, cell / 3), 0, Math.PI * 2); ctx.fill();
      for (let d = 0; d < D; d++) box(gutter + d * cell, y, m.masks[i * D + d], "rgba(47,214,196,.85)");
    }
    if (S.run.hoverFeature != null) {
      ctx.strokeStyle = "rgba(255,255,255,.6)";
      ctx.strokeRect(gutter + S.run.hoverFeature * cell + 0.5, 0.5, cell - 1, h - 1);
    }
    drawFlock.geo = { gutter, cell, D, N, gap };
  }

  flockCv.addEventListener("mousemove", (e) => {
    const g = drawFlock.geo;
    if (!g) return;
    const [x, y] = localXY(e, flockCv);
    const d = Math.floor((x - g.gutter) / g.cell);
    if (d < 0 || d >= g.D) { S.run.hoverFeature = null; flockTip.hide(); drawFlock(); return; }
    S.run.hoverFeature = d;
    const m = S.run.last;
    let on = 0;
    for (let i = 0; i < g.N; i++) on += m.masks[i * g.D + d];
    flockTip.show(`<b>${esc(S.prep.featureNames[d])}</b><span>${m.best[d] ? "in the best subset" : "not in the best subset"}</span><span>${on}/${g.N} hawks using it</span>`, x, y);
    drawFlock();
  });
  flockCv.addEventListener("mouseleave", () => { if (S.run) S.run.hoverFeature = null; flockTip.hide(); drawFlock(); });
  onResize(flockCv.parentElement, drawFlock);

  function drawCurve() {
    const it = S.run.iters;
    const { ctx, w, h } = fitCanvas(curveCv);
    ctx.clearRect(0, 0, w, h);
    if (it.length < 2) return;
    const T = S.run.config.iterations;
    const lo = Math.min(...it.map((p) => p.bestFit)), hi = it[0].meanFit;
    const X = (t) => 4 + (t / T) * (w - 8);
    const Y = (v) => h - 4 - ((Math.min(v, hi) - lo) / (hi - lo || 1)) * (h - 8);
    ctx.strokeStyle = "rgba(138,152,176,.6)"; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); it.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p.meanFit)) : ctx.moveTo(X(p.t), Y(p.meanFit)))); ctx.stroke();
    ctx.setLineDash([]); ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
    ctx.beginPath(); it.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p.bestFit)) : ctx.moveTo(X(p.t), Y(p.bestFit)))); ctx.stroke();
  }
}

function classCounts({ rows }, j) {
  const m = new Map();
  for (const r of rows) {
    const v = r[j].trim();
    if (v) m.set(v, (m.get(v) || 0) + 1);
  }
  return [...m.entries()].sort((a, b) => b[1] - a[1]);
}

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
