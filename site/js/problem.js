// Section 01: NSL-KDD at a glance: class shift between train/test, redundancy between features.
import { C, GROUP_COLORS, fitCanvas } from "./theme.js";
import { MONO, tooltip, localXY, onResize, pct } from "./charts.js";

const CLASS_LABELS = { normal: "Normal", dos: "DoS", probe: "Probe", r2l: "R2L", u2r: "U2R" };

export async function initProblem(root) {
  const data = await (await fetch("data/dataset.json")).json();
  const $ = (k) => root.querySelector(`[data-p=${k}]`);

  const nAttackTrain = data.attack_types.train.length;
  $("stats").innerHTML = [
    [data.n_train.toLocaleString(), "training connections"],
    [data.n_test.toLocaleString(), "test connections"],
    [data.features.length, "features per connection"],
    [data.novel_test_types.length, `attack types only in test (train has ${nAttackTrain})`],
  ].map(([v, l]) => `<div class="kpi"><strong>${v}</strong><span>${l}</span></div>`).join("");

  $("novel").textContent = data.novel_test_types.join(", ");

  classBars($("classes"), data);
  heatmap($("heat"), data);
  miBars($("mi"), data);
}

function classBars(host, data) {
  const cls = Object.keys(CLASS_LABELS);
  const tot = (s) => cls.reduce((a, c) => a + data.classes[s][c], 0);
  const share = (s, c) => data.classes[s][c] / tot(s);
  host.innerHTML = cls.map((c) => {
    const tr = share("train", c), te = share("test", c);
    const ratio = te / tr;
    const shift = ratio > 1.5 ? `<span class="shift up">×${ratio.toFixed(1)} in test</span>` : "";
    return `<div class="cls-row">
      <span class="cls-name">${CLASS_LABELS[c]}</span>
      <div class="cls-bars">
        <div class="bar train" style="--w:${Math.max(tr * 100, 0.4)}%" title="${data.classes.train[c].toLocaleString()} rows"><span>${pct(tr)}</span></div>
        <div class="bar test" style="--w:${Math.max(te * 100, 0.4)}%" title="${data.classes.test[c].toLocaleString()} rows"><span>${pct(te)}</span></div>
      </div>
      ${shift}
    </div>`;
  }).join("");
}

function heatmap(host, data) {
  const cv = host.querySelector("canvas");
  const tip = tooltip(host);
  const F = data.features, D = F.length;
  let geo = null;

  function color(r) {
    // diverging: negative -> violet, positive -> amber, 0 -> background
    const a = Math.min(1, Math.abs(r));
    return r >= 0 ? `rgba(255,179,71,${a})` : `rgba(164,107,255,${a})`;
  }

  function draw(hi) {
    const { ctx, w } = fitCanvas(cv);
    const pad = 10;
    const cell = (w - pad) / D;
    geo = { pad, cell };
    ctx.clearRect(0, 0, w, w);
    for (let i = 0; i < D; i++) {
      // group strip along both edges
      ctx.fillStyle = GROUP_COLORS[F[i].group];
      ctx.fillRect(0, pad + i * cell, 5, cell);
      ctx.fillRect(pad + i * cell, 0, cell, 5);
      for (let j = 0; j < D; j++) {
        ctx.fillStyle = i === j ? "rgba(255,255,255,.12)" : color(data.corr[i][j]);
        ctx.fillRect(pad + j * cell, pad + i * cell, cell - 0.6, cell - 0.6);
      }
    }
    if (hi) {
      ctx.strokeStyle = "rgba(255,255,255,.7)";
      ctx.strokeRect(pad, pad + hi[0] * cell, D * cell, cell);
      ctx.strokeRect(pad + hi[1] * cell, pad, cell, D * cell);
    }
  }

  cv.addEventListener("mousemove", (e) => {
    const [x, y] = localXY(e, cv);
    const j = Math.floor((x - geo.pad) / geo.cell), i = Math.floor((y - geo.pad) / geo.cell);
    if (i < 0 || j < 0 || i >= D || j >= D) { tip.hide(); draw(); return; }
    draw([i, j]);
    const r = data.corr[i][j];
    const note = F[i].constant || F[j].constant ? "<span>constant column: no variance</span>" : "";
    tip.show(`<b>${F[i].name}</b><b>× ${F[j].name}</b><span>r = ${r.toFixed(2)}</span>${note}`, x, y);
  });
  cv.addEventListener("mouseleave", () => { tip.hide(); draw(); });
  onResize(cv, () => draw());

  let strong = 0;
  for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++) if (Math.abs(data.corr[i][j]) >= 0.9) strong++;
  host.closest(".panel").querySelector("[data-p=strong]").textContent = strong;
}

function miBars(host, data) {
  const max = Math.max(...data.features.map((f) => f.mi));
  const sorted = [...data.features].sort((a, b) => b.mi - a.mi);
  host.innerHTML = sorted.map((f) => `
    <div class="mi-row" title="${f.group} feature · MI ${f.mi.toFixed(3)}">
      <span class="mi-name ${f.constant ? "dead" : ""}">${f.name}</span>
      <span class="mi-bar"><i style="--w:${(f.mi / max) * 100}%;--c:${GROUP_COLORS[f.group]}"></i></span>
      <span class="mi-val">${f.mi.toFixed(2)}</span>
    </div>`).join("");
}
