// Web Worker: KNN wrapper fitness, baselines and binary Harris Hawks Optimization.
// A port of hho.py / fitness.py so the browser analyzer behaves like the research code.

let D, K, data, cfg, rand, cache, evals;

self.onmessage = (e) => {
  if (e.data.type !== "run") return;
  try {
    run(e.data.data, e.data.config);
  } catch (err) {
    self.postMessage({ type: "error", message: err.message || String(err) });
  }
};

// ---------- RNG ----------
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss() {
  const u = 1 - rand(), v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const LEVY_SIGMA = 0.6965745; // Mantegna, beta = 1.5
const levy = () => (0.01 * gauss() * LEVY_SIGMA) / Math.pow(Math.abs(gauss()), 1 / 1.5);

// ---------- KNN ----------
// Brute force with partial-distance early exit: stop summing once a row can't make the top k.
// Mixed data uses HEOM: squared difference for numeric (0-1 scaled) features, overlap
// (0 if same category, 1 if not) for categorical ones.
//
// Every training row tied with the k-th nearest distance gets a vote (up to a buffer of 16k).
// On discrete data many rows sit at exactly the same distance; cutting them off at k by row
// order makes the prediction arbitrary, and a pure-noise column then looks "useful" simply
// because it breaks those ties.
function knnPredict(train, test, cols, k) {
  const cat = data.isCat;
  const { X: A, y: Ay, n } = train;
  const { X: B, n: m } = test;
  const c = cols.length;
  const M = 16 * k;
  const preds = new Int32Array(m);
  const bd = new Float64Array(M), bl = new Int32Array(M);
  const votes = new Int32Array(K);
  for (let i = 0; i < m; i++) {
    let cnt = 0;
    const bo = i * D;
    for (let j = 0; j < n; j++) {
      const ao = j * D;
      const worst = cnt >= k ? bd[k - 1] : Infinity;
      let s = 0, q = 0;
      for (; q < c; q++) {
        const f = cols[q];
        if (cat[f]) { if (B[bo + f] !== A[ao + f]) s += 1; }
        else { const diff = B[bo + f] - A[ao + f]; s += diff * diff; }
        if (s > worst) break;
      }
      if (q < c || s > worst) continue;
      if (cnt === M && s >= bd[M - 1]) continue; // tie buffer full: keep the ones we have
      let p = cnt < M ? cnt : M - 1;
      while (p > 0 && bd[p - 1] > s) { bd[p] = bd[p - 1]; bl[p] = bl[p - 1]; p--; }
      bd[p] = s; bl[p] = Ay[j];
      if (cnt < M) cnt++;
      if (cnt > k) { const kth = bd[k - 1]; while (cnt > k && bd[cnt - 1] > kth) cnt--; }
    }
    votes.fill(0);
    let maxV = 0;
    for (let p = 0; p < cnt; p++) maxV = Math.max(maxV, ++votes[bl[p]]);
    let best = bl[0];
    for (let p = 0; p < cnt; p++) if (votes[bl[p]] === maxV) { best = bl[p]; break; } // vote tie: nearest wins
    preds[i] = best;
  }
  return preds;
}

const colsOf = (mask) => {
  const out = [];
  for (let d = 0; d < D; d++) if (mask[d]) out.push(d);
  return Int32Array.from(out);
};

// Error over the fitness folds (5-fold CV on small data, one holdout split otherwise).
function errorRate(mask) {
  const cols = colsOf(mask);
  let wrong = 0, total = 0;
  for (const { train, val } of data.folds) {
    const p = knnPredict(train, val, cols, cfg.k);
    for (let i = 0; i < val.n; i++) if (p[i] !== val.y[i]) wrong++;
    total += val.n;
  }
  return wrong / total;
}

function fitness(mask) {
  const key = mask.join("");
  const hit = cache.get(key);
  if (hit) return hit;
  let res;
  let n = 0;
  for (let d = 0; d < D; d++) n += mask[d];
  if (!n) res = { f: 1, err: 1 };
  else {
    const err = errorRate(mask);
    evals++;
    res = { f: cfg.alpha * err + (1 - cfg.alpha) * (n / D), err };
  }
  cache.set(key, res);
  return res;
}

function metrics(y, p) {
  const cm = Array.from({ length: K }, () => new Array(K).fill(0));
  for (let i = 0; i < y.length; i++) cm[y[i]][p[i]]++;
  let correct = 0, f1sum = 0, present = 0;
  const perClass = [];
  for (let c = 0; c < K; c++) {
    correct += cm[c][c];
    const tp = cm[c][c];
    const fn = cm[c].reduce((a, b) => a + b, 0) - tp;
    const fp = cm.reduce((a, r) => a + r[c], 0) - tp;
    const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    if (tp + fn) { f1sum += f1; present++; }
    perClass.push({ precision, recall, f1, support: tp + fn });
  }
  return { accuracy: correct / y.length, macroF1: f1sum / (present || 1), confusion: cm, perClass };
}

function testMetrics(mask) {
  const p = knnPredict(data.train, data.test, colsOf(mask), cfg.k);
  return metrics(data.test.y, p);
}

// Mutual information between each feature and the class (the univariate filter baseline).
// Numeric features are binned into 10 equal-width bins on their 0-1 scale; categories are used
// as-is. Unlike an F-test this is comparable across numeric and categorical columns.
function miScores() {
  const { X, y, n } = data.folds[0].train;
  const pc = new Float64Array(K);
  for (let i = 0; i < n; i++) pc[y[i]]++;
  const scores = new Float64Array(D);
  for (let d = 0; d < D; d++) {
    const joint = new Map(), px = new Map();
    for (let i = 0; i < n; i++) {
      const v = X[i * D + d];
      const bin = data.isCat[d] ? v : Math.min(9, Math.floor(v * 10));
      const key = bin * K + y[i];
      joint.set(key, (joint.get(key) || 0) + 1);
      px.set(bin, (px.get(bin) || 0) + 1);
    }
    let mi = 0;
    for (const [key, c] of joint) {
      const bin = Math.floor(key / K), cls = key - bin * K;
      mi += (c / n) * Math.log((c * n) / (px.get(bin) * pc[cls]));
    }
    scores[d] = mi;
  }
  return Array.from(scores);
}

// ---------- binary HHO ----------
const LB = -4, UB = 4;
const clamp = (v) => (v < LB ? LB : v > UB ? UB : v);

function transfer(xNew, xOld, bOld) {
  const b = new Uint8Array(D);
  for (let d = 0; d < D; d++) {
    if (cfg.transfer === "s") b[d] = rand() < 1 / (1 + Math.exp(-xNew[d])) ? 1 : 0;
    else b[d] = rand() < Math.abs(Math.tanh(xNew[d] - xOld[d])) ? 1 - bOld[d] : bOld[d];
  }
  return b;
}

function repair(b) {
  let any = 0;
  for (let d = 0; d < D; d++) any |= b[d];
  if (!any) b[Math.floor(rand() * D)] = 1;
  return b;
}

function hho(onIter) {
  const N = cfg.hawks, T = cfg.iterations;
  const X = Array.from({ length: N }, () => Float64Array.from({ length: D }, () => LB + rand() * (UB - LB)));
  const B = Array.from({ length: N }, () => repair(Uint8Array.from({ length: D }, () => (rand() < 0.5 ? 1 : 0))));
  const F = B.map((b) => fitness(b).f);
  let bi = F.indexOf(Math.min(...F));
  let rx = Float64Array.from(X[bi]), rb = Uint8Array.from(B[bi]), rf = F[bi];
  const strategy = new Array(N).fill(-1), energy = new Array(N).fill(2);
  onIter(0, B, strategy, energy, F, rb, rf);

  for (let t = 0; t < T; t++) {
    const mean = new Float64Array(D);
    for (const x of X) for (let d = 0; d < D; d++) mean[d] += x[d] / N;

    for (let i = 0; i < N; i++) {
      const xo = X[i], bo = B[i];
      const E = 2 * (2 * rand() - 1) * (1 - t / T);
      const J = 2 * (1 - rand());
      const a = Math.abs(E);
      let xn, bn, fn = null, s;

      if (a >= 1) {
        xn = new Float64Array(D);
        if (rand() >= 0.5) {
          const xr = X[Math.floor(rand() * N)], r1 = rand(), r2 = rand();
          for (let d = 0; d < D; d++) xn[d] = clamp(xr[d] - r1 * Math.abs(xr[d] - 2 * r2 * xo[d]));
          s = 0;
        } else {
          const r3 = rand(), r4 = rand();
          for (let d = 0; d < D; d++) xn[d] = clamp(rx[d] - mean[d] - r3 * (LB + r4 * (UB - LB)));
          s = 1;
        }
        bn = transfer(xn, xo, bo);
      } else if (rand() >= 0.5) {
        xn = new Float64Array(D);
        if (a >= 0.5) {
          for (let d = 0; d < D; d++) xn[d] = clamp(rx[d] - xo[d] - E * Math.abs(J * rx[d] - xo[d]));
          s = 2;
        } else {
          for (let d = 0; d < D; d++) xn[d] = clamp(rx[d] - E * Math.abs(rx[d] - xo[d]));
          s = 3;
        }
        bn = transfer(xn, xo, bo);
      } else {
        const ref = a >= 0.5 ? xo : mean;
        s = a >= 0.5 ? 4 : 5;
        const Y = new Float64Array(D);
        for (let d = 0; d < D; d++) Y[d] = clamp(rx[d] - E * Math.abs(J * rx[d] - ref[d]));
        const bY = repair(transfer(Y, xo, bo)), fY = fitness(bY).f;
        if (fY < F[i]) { xn = Y; bn = bY; fn = fY; }
        else {
          const Z = new Float64Array(D);
          for (let d = 0; d < D; d++) Z[d] = clamp(Y[d] + rand() * levy());
          const bZ = repair(transfer(Z, xo, bo)), fZ = fitness(bZ).f;
          if (fZ < F[i]) { xn = Z; bn = bZ; fn = fZ; } else { xn = xo; bn = bo; fn = F[i]; }
        }
      }
      repair(bn);
      X[i] = xn; B[i] = bn;
      F[i] = fn ?? fitness(bn).f;
      strategy[i] = s;
      energy[i] = E;
    }
    bi = F.indexOf(Math.min(...F));
    if (F[bi] < rf) { rx = Float64Array.from(X[bi]); rb = Uint8Array.from(B[bi]); rf = F[bi]; }
    onIter(t + 1, B, strategy, energy, F, rb, rf);
  }
  return { mask: rb, fitness: rf, population: B };
}

// ---------- orchestration ----------
function run(d, config) {
  data = d; cfg = config;
  D = d.D; K = d.classNames.length;
  rand = mulberry(cfg.seed);
  cache = new Map(); evals = 0;
  const t0 = performance.now();
  const post = (m, transfer) => self.postMessage(m, transfer || []);

  // baselines
  post({ type: "phase", name: "baseline" });
  const allMask = new Uint8Array(D).fill(1);
  const all = fitness(allMask);
  const scores = miScores();
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a]);
  let filt = null;
  for (let k = 1; k <= D; k++) {
    const m = new Uint8Array(D);
    for (const j of order.slice(0, k)) m[j] = 1;
    const r = fitness(m);
    if (!filt || r.f < filt.f) filt = { ...r, mask: m, k };
  }
  post({ type: "baseline", all: { err: all.err, f: all.f }, filter: { err: filt.err, f: filt.f, k: filt.k }, scores, msPerEval: (performance.now() - t0) / Math.max(1, evals) });

  // HHO
  post({ type: "phase", name: "hho" });
  const res = hho((t, B, strategy, energy, F, rb, rf) => {
    const masks = new Uint8Array(B.length * D);
    B.forEach((b, i) => masks.set(b, i * D));
    let meanF = 0;
    for (const f of F) meanF += f / F.length;
    post({
      type: "iter", t, T: cfg.iterations, masks, strategy: [...strategy], energy: [...energy],
      best: Array.from(rb), bestFit: rf, bestErr: fitness(rb).err, meanFit: meanF,
      evals, elapsed: performance.now() - t0,
    }, [masks.buffer]);
  });

  // final evaluation on the held-out test split
  post({ type: "phase", name: "final" });
  const sel = colsOf(res.mask);
  const baseErr = fitness(res.mask).err;
  // With no features left the best a model can do is always guess the most common class.
  const counts = new Array(K).fill(0);
  for (const { val } of data.folds) for (let i = 0; i < val.n; i++) counts[val.y[i]]++;
  const majorityErr = 1 - Math.max(...counts) / counts.reduce((a, b) => a + b, 0);
  const importance = Array.from(sel, (f) => {
    if (sel.length === 1) return { feature: f, delta: majorityErr - baseErr };
    const m = Uint8Array.from(res.mask); m[f] = 0;
    return { feature: f, delta: errorRate(m) - baseErr };
  });
  const consensus = new Array(D).fill(0);
  for (const b of res.population) for (let i = 0; i < D; i++) consensus[i] += b[i] / res.population.length;

  post({
    type: "done",
    result: {
      mask: Array.from(res.mask), fitness: res.fitness, valErr: baseErr,
      test: { hho: testMetrics(res.mask), all: testMetrics(allMask), filter: testMetrics(filt.mask) },
      filterMask: Array.from(filt.mask), filterK: filt.k, allValErr: all.err, filterValErr: filt.err,
      importance, consensus, scores, evals, majorityAcc: 1 - majorityErr,
      elapsed: performance.now() - t0,
    },
  });
}
