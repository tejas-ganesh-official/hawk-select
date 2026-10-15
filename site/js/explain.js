// Section 02: interactive explainers for escaping energy, the strategy decision, Lévy flights
// and the binary transfer functions.
import { C, STRATEGY_COLORS, fitCanvas } from "./theme.js";
import { MONO, linear, drawAxes, onResize } from "./charts.js";

export function initExplainers(root) {
  energyExplainer(root.querySelector("[data-x=energy]"));
  levyExplainer(root.querySelector("[data-x=levy]"));
  transferExplainer(root.querySelector("[data-x=transfer]"));
}

// ---------- escaping energy + decision tree ----------
const FORMULAS = [
  "X(t+1) = X<sub>rand</sub> − r₁·|X<sub>rand</sub> − 2r₂·X|",
  "X(t+1) = (X<sub>rabbit</sub> − X<sub>mean</sub>) − r₃·(LB + r₄(UB − LB))",
  "X(t+1) = ΔX − E·|J·X<sub>rabbit</sub> − X|",
  "X(t+1) = X<sub>rabbit</sub> − E·|ΔX|",
  "Y = X<sub>rabbit</sub> − E·|J·X<sub>rabbit</sub> − X|,  Z = Y + S·Lévy(D),  keep the better",
  "Y = X<sub>rabbit</sub> − E·|J·X<sub>rabbit</sub> − X<sub>mean</sub>|,  Z = Y + S·Lévy(D),  keep the better",
];
const PLAIN = [
  "The hawk perches next to a randomly chosen family member: wide, undirected scouting.",
  "The hawk perches relative to the rabbit and the flock's average position.",
  "The rabbit still has energy, so hawks encircle it softly and wait for it to tire.",
  "The rabbit is exhausted: hawks close in hard, jumping straight towards it.",
  "The rabbit tries to escape with zig-zag jumps; hawks try progressive Lévy dives, accepting a dive only if it lands closer.",
  "Same dives, but now aimed from the flock's mean position at an exhausted rabbit.",
];

function energyExplainer(host) {
  const cv = host.querySelector("canvas");
  const s = {
    t: host.querySelector("[name=t]"), e0: host.querySelector("[name=e0]"),
    r: host.querySelector("[name=r]"), q: host.querySelector("[name=q]"),
  };
  const out = (k) => host.querySelector(`[data-o=${k}]`);
  const nodes = [...host.querySelectorAll("[data-node]")];
  const T = 100;

  function strategy(E, r, q) {
    const a = Math.abs(E);
    if (a >= 1) return q >= 0.5 ? 0 : 1;
    if (r >= 0.5) return a >= 0.5 ? 2 : 3;
    return a >= 0.5 ? 4 : 5;
  }

  function update() {
    const t = +s.t.value, e0 = +s.e0.value, r = +s.r.value, q = +s.q.value;
    const E = 2 * e0 * (1 - t / T);
    const st = strategy(E, r, q);
    out("t").textContent = t;
    out("e0").textContent = e0.toFixed(2);
    out("r").textContent = r.toFixed(2);
    out("q").textContent = q.toFixed(2);
    out("E").textContent = E.toFixed(3);
    out("absE").textContent = Math.abs(E).toFixed(3);
    out("name").textContent = host.querySelector(`[data-node="s${st}"] b`).textContent;
    out("name").style.color = STRATEGY_COLORS[st];
    out("formula").innerHTML = FORMULAS[st];
    out("plain").textContent = PLAIN[st];

    // highlight the active path through the tree
    const a = Math.abs(E);
    const path = new Set(["root", a >= 1 ? "explore" : "exploit", `s${st}`]);
    if (a < 1) path.add(r >= 0.5 ? "besiege" : "dive");
    nodes.forEach((n) => n.classList.toggle("active", path.has(n.dataset.node)));
    host.style.setProperty("--active", STRATEGY_COLORS[st]);

    draw(t, E, st);
  }

  function draw(t, E, st) {
    const { ctx, w, h } = fitCanvas(cv);
    const area = { l: 40, r: w - 12, t: 12, b: h - 30 };
    const x = linear([0, T], [area.l, area.r]);
    const y = linear([-2, 2], [area.b, area.t]);
    ctx.clearRect(0, 0, w, h);

    // zones
    const band = (lo, hi, col) => { ctx.fillStyle = col; ctx.fillRect(area.l, y(hi), area.r - area.l, y(lo) - y(hi)); };
    band(1, 2, "rgba(107,124,255,.10)"); band(-2, -1, "rgba(107,124,255,.10)");
    band(0.5, 1, "rgba(47,214,196,.07)"); band(-1, -0.5, "rgba(47,214,196,.07)");
    band(-0.5, 0.5, "rgba(255,179,71,.08)");
    drawAxes(ctx, { x, y, xTicks: [0, 25, 50, 75, 100], yTicks: [-2, -1, -0.5, 0, 0.5, 1, 2], area, xLabel: "iteration t" });

    ctx.font = `10px ${MONO}`;
    ctx.fillStyle = "rgba(140,155,255,.9)"; ctx.fillText("exploration |E| ≥ 1", area.l + 6, y(1.75));
    ctx.fillStyle = "rgba(47,214,196,.9)"; ctx.fillText("soft  0.5 ≤ |E| < 1", area.l + 6, y(0.8));
    ctx.fillStyle = "rgba(255,179,71,.9)"; ctx.fillText("hard  |E| < 0.5", area.l + 6, y(0.22));

    // envelope ±2(1 - t/T) and the reachable region at this t
    ctx.strokeStyle = "rgba(255,255,255,.5)";
    ctx.setLineDash([4, 4]);
    for (const sg of [1, -1]) { ctx.beginPath(); ctx.moveTo(x(0), y(2 * sg)); ctx.lineTo(x(T), y(0)); ctx.stroke(); }
    ctx.setLineDash([]);
    const env = 2 * (1 - t / T);
    ctx.strokeStyle = "rgba(255,255,255,.25)";
    ctx.beginPath(); ctx.moveTo(x(t), y(env)); ctx.lineTo(x(t), y(-env)); ctx.stroke();

    // current E
    const col = STRATEGY_COLORS[st];
    ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 14;
    ctx.beginPath(); ctx.arc(x(t), y(E), 6, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }

  Object.values(s).forEach((el) => el.addEventListener("input", update));
  host.querySelector("[data-act=random]").addEventListener("click", () => {
    s.e0.value = (Math.random() * 2 - 1).toFixed(2);
    s.r.value = Math.random().toFixed(2);
    s.q.value = Math.random().toFixed(2);
    update();
  });
  onResize(cv, update);
}

// ---------- Lévy flight vs Gaussian walk ----------
function gauss() {
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const SIGMA = 0.6965745; // Mantegna, beta = 1.5
const levyStep = () => (gauss() * SIGMA) / Math.pow(Math.abs(gauss()), 1 / 1.5);

function levyExplainer(host) {
  const cvs = [...host.querySelectorAll("canvas")];
  const STEPS = 700;
  let paths = [], drawn = 0, raf = 0;

  function generate() {
    const make = (step) => {
      const p = [[0, 0]];
      for (let i = 0; i < STEPS; i++) {
        const [x, y] = p[p.length - 1];
        const a = Math.random() * Math.PI * 2, len = Math.abs(step());
        p.push([x + Math.cos(a) * len, y + Math.sin(a) * len]);
      }
      return p;
    };
    paths = [make(gauss), make(levyStep)];
    drawn = 0;
    cancelAnimationFrame(raf);
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) { drawn = STEPS; render(); } else animate();
  }

  function render() {
    paths.forEach((p, k) => {
      const { ctx, w, h } = fitCanvas(cvs[k]);
      // shared scale per path so the shape, not the size, is compared
      const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]);
      const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
      const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      const s = (Math.min(w, h) - 24) / span;
      const X = (v) => w / 2 + (v - cx) * s, Y = (v) => h / 2 + (v - cy) * s;
      ctx.clearRect(0, 0, w, h);
      const col = k ? C.accent : C.teal;
      for (let i = 1; i <= Math.min(drawn, STEPS); i++) {
        const big = Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]) * s > 25;
        ctx.strokeStyle = col;
        ctx.globalAlpha = big ? 0.95 : 0.55;
        ctx.lineWidth = big ? 1.6 : 1;
        ctx.beginPath(); ctx.moveTo(X(p[i - 1][0]), Y(p[i - 1][1])); ctx.lineTo(X(p[i][0]), Y(p[i][1])); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      const end = p[Math.min(drawn, STEPS)];
      ctx.fillStyle = "#fff";
      ctx.beginPath(); ctx.arc(X(end[0]), Y(end[1]), 3.5, 0, Math.PI * 2); ctx.fill();
    });
  }

  function animate() {
    drawn = Math.min(STEPS, drawn + 6);
    render();
    if (drawn < STEPS) raf = requestAnimationFrame(animate);
  }

  host.querySelector("[data-act=levy]").addEventListener("click", generate);
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { generate(); io.disconnect(); } }, { threshold: 0.3 });
  io.observe(host);
  onResize(host, () => paths.length && render());
}

// ---------- transfer functions ----------
function transferExplainer(host) {
  const cv = host.querySelector("canvas");
  const slider = host.querySelector("[name=x]");
  const out = (k) => host.querySelector(`[data-o=${k}]`);
  const bitsS = host.querySelector("[data-o=bitsS]"), bitsV = host.querySelector("[data-o=bitsV]");
  const D = 41;
  const prev = Array.from({ length: D }, () => Math.random() < 0.5);
  const S = (x) => 1 / (1 + Math.exp(-x));
  const V = (x) => Math.abs(Math.tanh(x));

  const mkBits = (el) => Array.from({ length: D }, () => el.appendChild(document.createElement("i")));
  const cellsS = mkBits(bitsS), cellsV = mkBits(bitsV);

  function sample() {
    const x = +slider.value;
    let onS = 0, flips = 0;
    cellsS.forEach((c) => { const b = Math.random() < S(x); onS += b; c.className = b ? "on" : ""; });
    cellsV.forEach((c, d) => {
      const flip = Math.random() < V(x);
      flips += flip;
      c.className = (flip ? !prev[d] : prev[d]) ? (flip ? "on flip" : "on") : (flip ? "flip" : "");
    });
    out("onS").textContent = onS;
    out("flips").textContent = flips;
  }

  function draw() {
    const x0 = +slider.value;
    out("x").textContent = x0.toFixed(2);
    out("S").textContent = S(x0).toFixed(3);
    out("V").textContent = V(x0).toFixed(3);
    const { ctx, w, h } = fitCanvas(cv);
    const area = { l: 36, r: w - 10, t: 10, b: h - 28 };
    const x = linear([-6, 6], [area.l, area.r]), y = linear([0, 1], [area.b, area.t]);
    ctx.clearRect(0, 0, w, h);
    drawAxes(ctx, { x, y, xTicks: [-6, -4, -2, 0, 2, 4, 6], yTicks: [0, 0.5, 1], area, fmtY: (v) => v.toFixed(1) });
    for (const [f, col] of [[S, C.teal], [V, C.accent]]) {
      ctx.strokeStyle = col; ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let px = area.l; px <= area.r; px++) {
        const v = f(x.invert(px));
        px === area.l ? ctx.moveTo(px, y(v)) : ctx.lineTo(px, y(v));
      }
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255,255,255,.4)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(x0), area.t); ctx.lineTo(x(x0), area.b); ctx.stroke();
    for (const [f, col] of [[S, C.teal], [V, C.accent]]) {
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(x(x0), y(f(x0)), 5, 0, Math.PI * 2); ctx.fill();
    }
  }

  slider.addEventListener("input", () => { draw(); sample(); });
  host.querySelector("[data-act=resample]").addEventListener("click", sample);
  onResize(cv, draw);
  sample();
}
