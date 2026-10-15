import { HHO2D, landscape, BOUNDS, OPTIMUM, STRATEGY_NAMES } from "./hho2d.js";
import { STRATEGY_COLORS } from "./theme.js";

const FRAMES_PER_ITER = 14;
const TRAIL = 10;

export function initHero(root) {
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const hud = {
    iter: root.querySelector("[data-hud=iter]"),
    energy: root.querySelector("[data-hud=energy]"),
    energyBar: root.querySelector("[data-hud=energy-bar]"),
    phase: root.querySelector("[data-hud=phase]"),
    best: root.querySelector("[data-hud=best]"),
    spark: root.querySelector("[data-hud=spark]"),
    legend: root.querySelector("[data-hud=legend]"),
  };
  const btnPause = root.querySelector("[data-act=pause]");
  const btnRestart = root.querySelector("[data-act=restart]");

  const legendRows = STRATEGY_NAMES.map((name, i) => {
    const row = document.createElement("li");
    row.innerHTML = `<span class="dot" style="--c:${STRATEGY_COLORS[i]}"></span><span>${name}</span><b>0</b>`;
    hud.legend.appendChild(row);
    return row.querySelector("b");
  });

  const hho = new HHO2D();
  let prev = hho.X.map((x) => [...x]);
  let trails = hho.X.map((x) => [[...x]]);
  let frame = 0;
  let paused = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let terrain = null;
  let dpr = 1, W = 0, H = 0;

  const toPx = ([x, y]) => [
    ((x - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])) * W,
    (1 - (y - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0])) * H,
  ];

  function buildTerrain() {
    // Render the landscape at low res into an offscreen canvas, contour bands included.
    const res = 260;
    const off = document.createElement("canvas");
    off.width = off.height = res;
    const octx = off.getContext("2d");
    const img = octx.createImageData(res, res);
    const span = BOUNDS[1] - BOUNDS[0];
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const x = BOUNDS[0] + (i / res) * span;
        const y = BOUNDS[1] - (j / res) * span;
        const v = Math.min(1, Math.sqrt(landscape(x, y) / 80)); // 0 = deep valley
        const band = Math.abs(((v * 14) % 1) - 0.5) < 0.06 ? 1 : 0;
        const k = (j * res + i) * 4;
        // valleys glow teal, peaks fade to near-black navy
        img.data[k] = 8 + (1 - v) * 10 + band * 20;
        img.data[k + 1] = 14 + (1 - v) * 70 + band * 30;
        img.data[k + 2] = 26 + (1 - v) * 80 + band * 40;
        img.data[k + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    terrain = off;
  }

  function resize() {
    dpr = Math.min(2, devicePixelRatio || 1);
    const r = canvas.getBoundingClientRect();
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function restart() {
    hho.reset();
    prev = hho.X.map((x) => [...x]);
    trails = hho.X.map((x) => [[...x]]);
    frame = 0;
  }

  function advance() {
    prev = hho.X.map((x) => [...x]);
    hho.step();
    hho.X.forEach((x, i) => {
      trails[i].push([...x]);
      if (trails[i].length > TRAIL) trails[i].shift();
    });
  }

  function drawHawk(x, y, angle, color) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.beginPath(); // stylised raptor: swept wings
    ctx.moveTo(9, 0);
    ctx.lineTo(-6, -8);
    ctx.lineTo(-2, 0);
    ctx.lineTo(-6, 8);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(terrain, 0, 0, W, H);

    // true optimum (faint crosshair)
    const [ox, oy] = toPx(OPTIMUM);
    ctx.strokeStyle = "rgba(255,255,255,.18)";
    ctx.setLineDash([3, 5]);
    ctx.beginPath(); ctx.moveTo(ox - 18, oy); ctx.lineTo(ox + 18, oy); ctx.moveTo(ox, oy - 18); ctx.lineTo(ox, oy + 18); ctx.stroke();
    ctx.setLineDash([]);

    const ease = 1 - Math.pow(1 - t, 3);
    hho.X.forEach((cur, i) => {
      const color = STRATEGY_COLORS[hho.strategy[i]];
      const tr = trails[i];
      ctx.strokeStyle = color;
      for (let k = 1; k < tr.length; k++) {
        const [ax, ay] = toPx(tr[k - 1]), [bx, by] = toPx(tr[k]);
        ctx.globalAlpha = (k / tr.length) * 0.35;
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const p = prev[i];
      const pos = [p[0] + (cur[0] - p[0]) * ease, p[1] + (cur[1] - p[1]) * ease];
      const [px, py] = toPx(pos);
      const [qx, qy] = toPx(cur), [sx, sy] = toPx(p);
      const angle = Math.hypot(qx - sx, qy - sy) > 0.5 ? Math.atan2(qy - sy, qx - sx) : -Math.PI / 2;
      drawHawk(px, py, angle, color);
    });

    // rabbit = best solution so far, pulsing
    const [rx, ry] = toPx(hho.rabbit);
    const pulse = 6 + 3 * Math.sin(performance.now() / 180);
    ctx.fillStyle = "#fff";
    ctx.shadowColor = "#fff"; ctx.shadowBlur = 16;
    ctx.beginPath(); ctx.arc(rx, ry, 4, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(255,255,255,.6)";
    ctx.beginPath(); ctx.arc(rx, ry, pulse + 4, 0, Math.PI * 2); ctx.stroke();
  }

  function drawSpark() {
    const c = hud.spark, g = c.getContext("2d");
    const w = (c.width = c.clientWidth * dpr), h = (c.height = c.clientHeight * dpr);
    g.clearRect(0, 0, w, h);
    const hs = hho.history;
    if (hs.length < 2) return;
    const max = Math.log10(hs[0] + 1e-6) , min = Math.log10(1e-6);
    g.strokeStyle = "#ffb347"; g.lineWidth = 1.5 * dpr;
    g.beginPath();
    hs.forEach((v, i) => {
      const x = (i / (hho.T - 1)) * w;
      const y = h - ((Math.log10(v + 1e-6) - min) / (max - min || 1)) * (h - 4) - 2;
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  }

  function updateHud() {
    hud.iter.textContent = `${hho.t} / ${hho.T}`;
    // Bar shows the |E| envelope 2(1 - t/T); phase reflects what most hawks are actually doing.
    const e = hho.Emax;
    hud.energy.textContent = e.toFixed(2);
    hud.energyBar.style.width = `${(e / 2) * 100}%`;
    const exploring = hho.strategy.filter((s) => s <= 1).length;
    const phase = hho.done ? "done" : hho.t === 0 || exploring * 2 >= hho.n ? "explore" : "exploit";
    hud.phase.textContent = { done: "Rabbit caught", explore: "Exploration", exploit: "Exploitation" }[phase];
    hud.phase.dataset.phase = phase;
    hud.best.textContent = hho.rabbitFit.toExponential(2);
    const counts = new Array(STRATEGY_NAMES.length).fill(0);
    hho.strategy.forEach((s) => counts[s]++);
    legendRows.forEach((b, i) => (b.textContent = counts[i]));
    drawSpark();
  }

  let holdUntil = 0;
  function loop(now) {
    if (!paused && now > holdUntil) {
      frame++;
      if (frame >= FRAMES_PER_ITER) {
        frame = 0;
        if (hho.done) restart();
        else { advance(); updateHud(); }
        if (hho.done) holdUntil = now + 2500; // linger on the catch before restarting
      }
    }
    draw(paused ? 1 : frame / FRAMES_PER_ITER);
    requestAnimationFrame(loop);
  }

  btnPause.addEventListener("click", () => {
    paused = !paused;
    btnPause.textContent = paused ? "Play" : "Pause";
  });
  btnRestart.addEventListener("click", () => { restart(); updateHud(); });
  if (paused) btnPause.textContent = "Play";

  new ResizeObserver(() => { resize(); drawSpark(); }).observe(canvas);
  resize();
  buildTerrain();
  updateHud();
  requestAnimationFrame(loop);
}
