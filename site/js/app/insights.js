// Step 4: turn a finished run into a readable report with findings, rankings and downloads.
import { C, fitCanvas } from "../theme.js";
import { pct } from "../charts.js";
import { skipReason } from "./dataset.js";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pp = (d) => `${d >= 0 ? "+" : "−"}${Math.abs(d * 100).toFixed(1)} pp`;
const code = (s) => `<code>${esc(s)}</code>`;

export function renderReport(host, { prep, prof, result, config, iters, fileName, parsed }) {
  const names = prep.featureNames, D = prep.D;
  const sel = result.mask.map((b, d) => (b ? d : -1)).filter((d) => d >= 0);
  const selSet = new Set(sel);
  const T = result.test;
  const k = sel.length;
  const reduction = 1 - k / D;
  const dAcc = T.hho.accuracy - T.all.accuracy;

  // ---- feature-level facts ----
  const imp = new Map(result.importance.map((r) => [r.feature, r.delta]));
  const ranked = [...sel].sort((a, b) => imp.get(b) - imp.get(a));
  const fRank = result.scores.map((_, i) => i).sort((a, b) => result.scores[b] - result.scores[a]);
  const fPos = new Map(fRank.map((d, r) => [d, r + 1]));
  const medianF = [...result.scores].sort((a, b) => a - b)[D >> 1];

  const dropped = names.map((_, d) => d).filter((d) => !selSet.has(d));
  const redundant = [], lowSignal = [], other = [];
  for (const d of dropped) {
    let bestR = 0, twin = -1;
    for (const s of sel) {
      const r = Math.abs(prep.corr[d][s]);
      if (r > bestR) { bestR = r; twin = s; }
    }
    if (bestR >= 0.85) redundant.push({ d, twin, r: bestR });
    else if (result.scores[d] <= medianF) lowSignal.push({ d });
    else other.push({ d });
  }
  redundant.sort((a, b) => b.r - a.r);

  const excluded = prof.cols.filter((c) => c.index !== prep.targetIdx && !prep.featureIdx.includes(c.index));

  // ---- findings, in plain language ----
  const F = [];
  const testN = T.hho.perClass.reduce((a, c) => a + c.support, 0);
  const majorityTest = Math.max(...T.hho.perClass.map((c) => c.support)) / testN;
  const weak = T.hho.accuracy - majorityTest < 0.05; // barely beats always guessing the commonest class
  const rowsApart = Math.round(Math.abs(dAcc) * testN);
  if (dAcc < 0 && rowsApart <= Math.max(1, testN * 0.005)) {
    F.push([weak ? "info" : "good", `With just <b>${k} of ${D}</b> features the model matches using every feature: ${pct(T.hho.accuracy)} vs ${pct(T.all.accuracy)},
      a difference of ${rowsApart} row${rowsApart > 1 ? "s" : ""} out of ${testN} in the test set.`]);
  } else if (dAcc >= -0.005) {
    F.push([weak ? "info" : "good", `With just <b>${k} of ${D}</b> features the model is ${dAcc > 0.005 ? "<b>more accurate</b>" : "just as accurate"} on unseen data:
      ${pct(T.hho.accuracy)} vs ${pct(T.all.accuracy)} using every feature${Math.abs(dAcc) < 0.0005 ? "" : ` (${pp(dAcc)})`}.`]);
  } else {
    F.push(["warn", `Cutting to ${k} features cost ${pp(dAcc).replace("−", "")} of accuracy on unseen data (${pct(T.hho.accuracy)} vs ${pct(T.all.accuracy)}).
      If every point matters, slide <b>Priority</b> towards higher accuracy and run again.`]);
  }
  if (weak) {
    F.push(["warn", `Careful: always guessing the most common class would already score ${pct(majorityTest)}, so ${pct(T.hho.accuracy)} is only
      ${((T.hho.accuracy - majorityTest) * 100).toFixed(1)} points better. These columns say little about ${code(prep.targetName)}; check that it is the column you meant to predict.`]);
  }
  if (ranked.length && !(k === 1 && imp.get(ranked[0]) <= 0.001)) {
    const top = ranked[0];
    const d = imp.get(top);
    F.push(["info", d > 0.001
      ? k === 1
        ? `${code(names[top])} does all the work: it lifts validation accuracy ${(d * 100).toFixed(1)} points above always guessing the most common class (${pct(result.majorityAcc)}).`
        : `${code(names[top])} carries the most weight: taking it out of the chosen set drops validation accuracy by ${(d * 100).toFixed(1)} points.`
      : `No single chosen feature is critical on its own; the selected columns back each other up.`]);
  }
  if (redundant.length) {
    const ex = redundant.slice(0, 2).map((r) => `${code(names[r.d])} ≈ ${code(names[r.twin])} (r = ${r.r.toFixed(2)})`).join(", ");
    F.push(["info", `${redundant.length} dropped feature${redundant.length > 1 ? "s were" : " was"} near-duplicates of kept ones, so they added no new information: ${ex}.`]);
  }
  const ff = filterFinding(T, k, result.filterK, testN);
  if (weak && ff[0] === "good") ff[0] = "info";
  F.push(ff);

  const K = prep.classNames.length;
  const counts = T.hho.perClass.map((c) => c.support);
  const totalTest = counts.reduce((a, b) => a + b, 0);
  const rare = counts.map((c, i) => [i, c / totalTest]).filter(([, f]) => f < 0.1 && f > 0);
  if (rare.length) {
    const [ri, rf] = rare.sort((a, b) => a[1] - b[1])[0];
    const rec = T.hho.perClass[ri].recall;
    F.push(rec < T.hho.accuracy - 0.1
      ? ["warn", `The class ${code(prep.classNames[ri])} is rare (${pct(rf)} of rows) and the model only recognises ${pct(rec)} of them,
          so the overall accuracy hides a weak spot. Check the per-class numbers below.`]
      : ["info", `The class ${code(prep.classNames[ri])} is rare (${pct(rf)} of rows), but the model still recognises ${pct(rec)} of them.`]);
  } else if (K > 2) {
    const worst = T.hho.perClass.map((c, i) => [i, c.recall]).sort((a, b) => a[1] - b[1])[0];
    F.push(["info", `Hardest class to recognise: ${code(prep.classNames[worst[0]])} (${pct(worst[1])} recall).`]);
  }
  const stable = sel.filter((d) => result.consensus[d] >= 0.9);
  if (stable.length && stable.length < k) {
    const list = stable.slice(0, 3).map((d) => code(names[d])).join(", ");
    F.push(["info", stable.length === 1
      ? `${list} was used by at least 90% of the flock at the end, the most dependable pick; the other chosen features were closer calls.`
      : `${stable.length} chosen features (${list}${stable.length > 3 ? "…" : ""}) were used by at least 90% of the flock at the end, the most dependable picks; the rest were closer calls.`]);
  }
  if (testN < 150) {
    F.push(["info", `The held-out test set is small (${testN} rows), so each row is worth ${(100 / testN).toFixed(1)} accuracy points.
      Treat small differences between the options as noise.`]);
  }
  const notes = [];
  if (prep.cv) notes.push("scored each candidate subset with 5-fold cross-validation, since the dataset is small");
  const why = (c) => skipReason(c) || "unticked";
  if (excluded.length) {
    notes.push(`left out ${excluded.map((c) => `${code(c.name)} (${why(c)})`).slice(0, 4).join(", ")}${excluded.length > 4 ? ` and ${excluded.length - 4} more` : ""}`);
  }
  const missingCols = prep.featureIdx.filter((j) => prof.cols[j].missing).length;
  if (missingCols) notes.push(`filled missing values in ${missingCols} column${missingCols > 1 ? "s" : ""} (median for numbers, most common value for categories)`);
  if (prep.sampled) notes.push(`analysed a stratified sample of ${prep.N.toLocaleString()} rows`);
  const logs = prep.transforms.filter((t) => t.log).length;
  if (logs) notes.push(`log-scaled ${logs} heavily skewed column${logs > 1 ? "s" : ""}`);

  // ---- markup ----
  const fmtTime = (ms) => (ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : `${(ms / 60000).toFixed(1)} min`);
  const maxImp = Math.max(0.001, ...ranked.map((d) => imp.get(d)));

  host.innerHTML = `
    <div class="report-head">
      <p class="eyebrow">Report · ${esc(fileName)} → predicting ${code(prep.targetName)}</p>
      <h2 class="report-title">Keep <span class="accent">${k}</span> of ${D} features.</h2>
      <p class="lede">${reduction > 0 ? `That's ${pct(reduction, 0)} fewer columns to collect, store and feed to a model` : "Every feature earned its place"}, with
        ${pct(T.hho.accuracy)} accuracy on rows the search never saw.</p>
    </div>

    <div class="kpis">
      <div class="kpi"><strong>${k}<small>/${D}</small></strong><span>features kept</span></div>
      <div class="kpi"><strong>${pct(T.hho.accuracy)}</strong><span>test accuracy <em class="${dAcc >= 0 ? "up" : "down"}">${pp(dAcc)} vs all features</em></span></div>
      <div class="kpi"><strong>${pct(T.hho.macroF1)}</strong><span>macro-F1 (balances all classes)</span></div>
      <div class="kpi"><strong>${fmtTime(result.elapsed)}</strong><span>${result.evals.toLocaleString()} models evaluated</span></div>
    </div>

    <div class="panel findings">
      <div class="panel-head"><h3>What we found</h3></div>
      <ul>${F.map(([kind, html]) => `<li class="${kind}"><i aria-hidden="true"></i><p>${html}</p></li>`).join("")}</ul>
      ${notes.length ? `<p class="note">Along the way we ${notes.join("; ")}.</p>` : ""}
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="panel-head"><h3>Features to keep</h3><span class="hint">ranked by how much accuracy drops without them</span></div>
        <table class="feat-table">
          <thead><tr><th>Feature</th><th>Impact if removed</th><th title="Share of the final flock that also used it">Flock agreement</th></tr></thead>
          <tbody>${ranked.map((d) => `<tr>
            <td class="mono">${esc(names[d])}</td>
            <td><span class="imp"><i style="width:${Math.max(2, (Math.max(0, imp.get(d)) / maxImp) * 100)}%"></i></span>
              <span class="mono small">${imp.get(d) > 0 ? "−" + (imp.get(d) * 100).toFixed(1) + " pts" : "≈ 0"}</span></td>
            <td class="mono">${pct(result.consensus[d], 0)}</td></tr>`).join("")}</tbody>
        </table>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Features to drop</h3><span class="hint">${dropped.length} columns</span></div>
        ${dropGroup("Duplicates of a kept feature", redundant.map((r) => `${code(names[r.d])} <span class="muted">≈ ${esc(names[r.twin])} · r ${r.r.toFixed(2)}</span>`))}
        ${dropGroup("Weak signal on their own", lowSignal.map((r) => code(names[r.d])))}
        ${dropGroup("Useful alone, but redundant in combination", other.map((r) => `${code(names[r.d])} <span class="muted">· filter rank #${fPos.get(r.d)}</span>`))}
        ${excluded.length ? dropGroup("Not analysed", excluded.map((c) => `${code(c.name)} <span class=\"muted\">· ${why(c)}</span>`)) : ""}
      </div>
    </div>

    <div class="two-col">
      <div class="panel">
        <div class="panel-head"><h3>How the options compare</h3><span class="hint">same KNN model, held-out 20% of rows</span></div>
        <table class="compare">
          <thead><tr><th></th><th>Features</th><th>Test accuracy</th><th>Macro-F1</th></tr></thead>
          <tbody>
            ${cmpRow("All features", D, T.all)}
            ${cmpRow("Statistical filter", result.filterK, T.filter)}
            ${cmpRow("Harris Hawks", k, T.hho, true)}
          </tbody>
        </table>
        <canvas class="mini-chart conv" data-el="r-curve" aria-label="Convergence of the hunt"></canvas>
        <p class="note">The line shows the flock's best score improving over ${config.iterations} rounds (lower is better).</p>
      </div>
      <div class="panel">
        <div class="panel-head"><h3>Where the model gets it right</h3><span class="hint">rows = actual class, columns = predicted</span></div>
        ${confusion(T.hho, prep.classNames)}
      </div>
    </div>

    <div class="panel downloads">
      <div class="panel-head"><h3>Take it with you</h3></div>
      <div class="dl-row">
        <button class="btn" data-dl="csv">Reduced dataset (.csv)</button>
        <button class="btn" data-dl="txt">Feature list (.txt)</button>
        <button class="btn" data-dl="json">Full report (.json)</button>
      </div>
      <p class="note">The reduced CSV keeps every original row with just the ${k} chosen columns plus ${code(prep.targetName)}.
        Settings: ${config.hawks} hawks · ${config.iterations} rounds · ${config.transfer.toUpperCase()}-shaped bit rule · k = ${config.k} · seed ${config.seed}.</p>
    </div>`;

  drawConvergence(host.querySelector('[data-el="r-curve"]'), iters);

  host.querySelector(".downloads").addEventListener("click", (e) => {
    const kind = e.target.dataset.dl;
    if (!kind) return;
    const base = fileName.replace(/\.[^.]+$/, "");
    const keepCols = [...sel.map((d) => prep.featureIdx[d]), prep.targetIdx];
    if (kind === "csv") {
      const q = (v) => (/[",\n\r]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
      const lines = [keepCols.map((j) => q(parsed.header[j])).join(",")];
      for (const r of parsed.rows) lines.push(keepCols.map((j) => q(r[j])).join(","));
      save(`${base}_hawkselect.csv`, lines.join("\n"), "text/csv");
    } else if (kind === "txt") {
      save(`${base}_features.txt`, ranked.map((d) => names[d]).join("\n") + "\n", "text/plain");
    } else {
      save(`${base}_report.json`, JSON.stringify({
        file: fileName, target: prep.targetName, classes: prep.classNames, settings: config,
        selected: ranked.map((d) => ({ feature: names[d], impact: imp.get(d), agreement: result.consensus[d] })),
        dropped: { duplicates: redundant.map((r) => ({ feature: names[r.d], duplicate_of: names[r.twin], r: r.r })),
          weak: lowSignal.map((r) => names[r.d]), other: other.map((r) => names[r.d]), excluded: excluded.map((c) => c.name) },
        test: T, filter_k: result.filterK, evaluations: result.evals, runtime_ms: result.elapsed,
        convergence: iters.map((p) => p.bestFit),
      }, null, 2), "application/json");
    }
  });
}

function filterFinding(T, k, fk, testN) {
  const h = T.hho.accuracy, f = T.filter.accuracy;
  const rows = Math.round(Math.abs(h - f) * testN);
  const same = rows <= Math.max(1, testN * 0.005);
  const lead = `A quick statistical filter (ranking columns one at a time) picked ${fk} feature${fk > 1 ? "s" : ""} for ${pct(f)} test accuracy`;
  const n = Math.abs(fk - k), feats = `${n} feature${n > 1 ? "s" : ""}`;
  if (same) {
    if (k < fk) return ["good", `${lead}. The hawks matched it with ${feats} fewer.`];
    if (k > fk) return ["info", `${lead}. The hawks scored the same with ${feats} more, so the filter's set is the leaner choice here.`];
    return ["info", `${lead}. The hawks landed on the same size and score.`];
  }
  if (h > f) {
    return ["good", k <= fk
      ? `${lead}. The hawks did better on both counts: ${pct(h)} with ${k < fk ? `${feats} fewer` : "the same number of features"}, because they judge columns in combination rather than one at a time.`
      : `${lead}. The hawks scored higher (${pct(h)}) by adding ${feats}, picking columns that only help in combination.`];
  }
  return [k < fk ? "info" : "warn", k < fk
    ? `${lead}, higher than the hawks' ${pct(h)}, but it needed ${feats} more. Slide <b>Priority</b> towards accuracy if those points matter.`
    : `${lead}, beating the hawks' ${pct(h)}${k > fk ? ` with ${feats} fewer` : ""}. On this data, ranking columns one at a time was enough.`];
}

function dropGroup(title, items) {
  if (!items.length) return "";
  return `<div class="drop-group"><h4>${title} <span class="muted">${items.length}</span></h4><p>${items.join(" ")}</p></div>`;
}

function cmpRow(label, n, m, hl) {
  return `<tr class="${hl ? "hl" : ""}"><th>${label}</th><td class="mono">${n}</td><td class="mono">${pct(m.accuracy)}</td><td class="mono">${pct(m.macroF1)}</td></tr>`;
}

function confusion(m, classes) {
  const K = classes.length;
  const cells = m.confusion.map((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    return row.map((v, j) => {
      const f = v / tot;
      return `<div class="cm-cell ${i === j ? "diag" : "off"}" style="--a:${i === j ? f : Math.min(1, f * 3)}" title="${v} rows"><strong>${pct(f, 0)}</strong><small>${v}</small></div>`;
    });
  });
  const head = classes.map((c) => `<span class="cm-head" title="${esc(c)}">${esc(c)}</span>`).join("");
  const rows = cells.map((r, i) => `<span class="cm-head row" title="${esc(classes[i])}">${esc(classes[i])}</span>${r.join("")}`).join("");
  const recall = m.perClass.map((c, i) => `<span><b>${esc(classes[i])}</b> ${pct(c.recall)} recall · ${pct(c.precision)} precision</span>`).join("");
  return `<div class="cm ${K > 4 ? "compact" : ""}" style="--n:${K}"><span class="cm-corner"></span>${head}${rows}</div>
    <div class="per-class">${recall}</div>`;
}

function drawConvergence(cv, iters) {
  if (!cv || iters.length < 2) return;
  requestAnimationFrame(() => {
    const { ctx, w, h } = fitCanvas(cv);
    const lo = Math.min(...iters.map((p) => p.bestFit)), hi = Math.max(...iters.map((p) => p.bestFit));
    const T = iters[iters.length - 1].t || 1;
    const X = (t) => 6 + (t / T) * (w - 12), Y = (v) => h - 6 - ((v - lo) / (hi - lo || 1)) * (h - 12);
    ctx.fillStyle = "rgba(255,179,71,.08)";
    ctx.beginPath(); ctx.moveTo(X(0), h);
    iters.forEach((p) => ctx.lineTo(X(p.t), Y(p.bestFit)));
    ctx.lineTo(X(T), h); ctx.fill();
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
    ctx.beginPath(); iters.forEach((p, i) => (i ? ctx.lineTo(X(p.t), Y(p.bestFit)) : ctx.moveTo(X(p.t), Y(p.bestFit)))); ctx.stroke();
  });
}

function save(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
