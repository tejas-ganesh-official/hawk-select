// Shared palette; strategy order matches STRATEGY in hho2d.js and hho.py.
export const STRATEGY_COLORS = ["#6b7cff", "#a46bff", "#2fd6c4", "#ffb347", "#3cf08a", "#ff4d6d"];
export const STRATEGY_LABELS = [
  "Perch: random hawk", "Perch: family mean", "Soft besiege",
  "Hard besiege", "Soft + rapid dives", "Hard + rapid dives",
];
export const ALGO_COLORS = {
  hho_s: "#ffb347", hho_v: "#2fd6c4", ga: "#a46bff", pso: "#6b7cff",
  mi: "#3cf08a", chi2: "#ff7eb6", all: "#8a98b0",
};
export const GROUP_COLORS = { basic: "#6b7cff", content: "#a46bff", traffic: "#2fd6c4", host: "#ffb347" };

export const C = {
  bg: "#070b14",
  grid: "rgba(120,160,220,0.12)",
  text: "#e6edf7",
  muted: "#8a98b0",
  accent: "#ffb347",
  teal: "#2fd6c4",
  danger: "#ff4d6d",
  good: "#3cf08a",
};

// Size a canvas to its CSS box at device pixel ratio; returns a context drawing in CSS px.
export function fitCanvas(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: width, h: height };
}
