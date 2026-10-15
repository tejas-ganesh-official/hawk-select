// Small canvas-chart helpers shared by the problem, explainer and results sections.
import { C } from "./theme.js";

export const MONO = "'JetBrains Mono', monospace";

export function linear([d0, d1], [r0, r1]) {
  const s = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  s.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
  return s;
}

export function ticks(lo, hi, n = 5) {
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

// Grid lines + labels for a plot area. y/x are scales; fmt functions format tick labels.
export function drawAxes(ctx, { x, y, xTicks, yTicks, fmtX = String, fmtY = String, area, xLabel, yLabel }) {
  ctx.font = `10px ${MONO}`;
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";
  for (const v of yTicks) {
    ctx.strokeStyle = C.grid;
    ctx.beginPath(); ctx.moveTo(area.l, y(v)); ctx.lineTo(area.r, y(v)); ctx.stroke();
    ctx.fillStyle = C.muted;
    ctx.fillText(fmtY(v), area.l - 6, y(v));
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of xTicks) ctx.fillText(fmtX(v), x(v), area.b + 6);
  if (xLabel) ctx.fillText(xLabel, (area.l + area.r) / 2, area.b + 20);
  if (yLabel) {
    ctx.save();
    ctx.translate(12, (area.t + area.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
  ctx.textAlign = "start";
  ctx.textBaseline = "alphabetic";
}

// A floating tooltip positioned inside `host` (which must be position: relative).
export function tooltip(host) {
  const el = document.createElement("div");
  el.className = "tip";
  el.hidden = true;
  host.appendChild(el);
  return {
    show(html, x, y) {
      el.innerHTML = html;
      el.hidden = false;
      const w = el.offsetWidth, h = el.offsetHeight;
      const tx = x + 14 + w > host.clientWidth ? x - w - 14 : x + 14;
      const ty = Math.min(Math.max(0, y - h / 2), host.clientHeight - h);
      el.style.transform = `translate(${Math.max(0, tx)}px, ${ty}px)`;
    },
    hide() { el.hidden = true; },
  };
}

export function localXY(e, el) {
  const r = el.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

// Redraw whenever the element changes size.
export function onResize(el, fn) {
  new ResizeObserver(() => fn()).observe(el);
}

export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
