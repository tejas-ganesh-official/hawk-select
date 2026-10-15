// CSV parsing, column profiling and preprocessing for the in-browser analyzer.
// Everything here runs on the main thread; the prepared matrices are handed to the worker.

export const LIMITS = { maxBytes: 15 * 1024 * 1024, maxRows: 20000, maxFeatures: 300, maxClasses: 20 };
const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "?", "-", "missing"]);
const TARGET_NAMES = ["label", "labels", "class", "target", "y", "diagnosis", "outcome", "attack", "survived", "churn", "churned",
  "default", "fraud", "is_fraud", "result", "status", "species", "category", "type"];

// ---------- parsing ----------
function sniffDelimiter(text) {
  const firstLine = text.slice(0, text.indexOf("\n") >>> 0 || 2000);
  let best = ",", n = 0;
  for (const d of [",", ";", "\t", "|"]) {
    const c = firstLine.split(d).length;
    if (c > n) { n = c; best = d; }
  }
  return best;
}

// RFC-4180-ish: quoted fields, escaped quotes, CRLF.
export function parseCSV(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const d = sniffDelimiter(text);
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else q = false;
      } else field += ch;
    } else if (ch === '"') q = true;
    else if (ch === d) { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  if (rows.length < 2) throw new Error("The file needs a header and at least a few rows of data.");

  const isNum = (v) => v.trim() !== "" && Number.isFinite(+v);
  const first = rows[0], second = rows[1];
  const looksLikeHeader = first.some((v, i) => !isNum(v) && isNum(second[i] ?? "")) || first.every((v) => !isNum(v));
  const header = looksLikeHeader ? first.map((h, i) => h.trim() || `column_${i + 1}`) : first.map((_, i) => `column_${i + 1}`);
  const body = looksLikeHeader ? rows.slice(1) : rows;
  const width = header.length;
  const clean = body.filter((r) => r.length === width || (r.length > width * 0.5));
  return { header: dedupe(header), rows: clean.map((r) => (r.length === width ? r : [...r, ...Array(width).fill("")].slice(0, width))), delimiter: d, dropped: body.length - clean.length };
}

function dedupe(names) {
  const seen = new Map();
  return names.map((n) => {
    const k = seen.get(n) || 0;
    seen.set(n, k + 1);
    return k ? `${n}_${k + 1}` : n;
  });
}

// ---------- profiling ----------
export function profile({ header, rows }) {
  const n = rows.length;
  const cols = header.map((name, j) => {
    let missing = 0, numeric = 0;
    const counts = new Map();
    const nums = [];
    for (let i = 0; i < n; i++) {
      const raw = rows[i][j].trim();
      if (MISSING.has(raw.toLowerCase())) { missing++; continue; }
      counts.set(raw, (counts.get(raw) || 0) + 1);
      const v = +raw;
      if (Number.isFinite(v)) { numeric++; nums.push(v); }
    }
    const present = n - missing;
    const type = present && numeric / present >= 0.95 ? "numeric" : "categorical";
    const unique = counts.size;
    const col = { name, index: j, type, missing, unique, constant: unique <= 1 };
    if (type === "numeric") {
      nums.sort((a, b) => a - b);
      col.min = nums[0]; col.max = nums[nums.length - 1];
      col.median = nums[nums.length >> 1];
      col.mean = nums.reduce((a, b) => a + b, 0) / nums.length;
      col.integer = nums.every(Number.isInteger);
    }
    col.top = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5).map(([value, count]) => ({ value, count }));
    // identifiers: every value distinct (and not a real-valued measurement)
    col.idLike = unique >= n * 0.98 && n > 20 && (type === "categorical" || col.integer);
    // free text such as ticket numbers: mostly unique categories carry no shareable pattern
    col.freeText = !col.idLike && type === "categorical" && unique > Math.max(50, present * 0.5);
    col.mostlyMissing = missing > n * 0.6;
    return col;
  });
  return { n, cols, target: guessTarget(cols, n) };
}

// Why a column is left out of the feature set by default (null = use it).
export function skipReason(col) {
  if (col.constant) return "never changes";
  if (col.idLike) return "looks like an ID";
  if (col.freeText) return `mostly unique text, ${col.unique.toLocaleString()} values`;
  if (col.mostlyMissing) return `mostly missing`;
  return null;
}

export function targetProblem(col, n) {
  if (!col) return "Pick the column you want to predict.";
  if (col.unique < 2) return "This column has only one value, so there is nothing to predict.";
  if (col.unique > LIMITS.maxClasses) {
    return col.type === "numeric"
      ? `This looks like a continuous number (${col.unique} distinct values). The analyzer does classification, so pick a column with categories.`
      : `This column has ${col.unique} categories; the limit is ${LIMITS.maxClasses}.`;
  }
  // top holds every value when there are at most 5 classes
  if (col.unique <= col.top.length && Math.min(...col.top.map((t) => t.count)) < 3) return "Some classes have fewer than 3 rows, which is too few to train and test on.";
  return null;
}

function guessTarget(cols, n) {
  const key = (c) => c.name.trim().toLowerCase();
  // earliest entry in TARGET_NAMES wins, so "survived" beats a generic "type" column
  const named = cols.filter((c) => TARGET_NAMES.includes(key(c)) && !targetProblem(c, n))
    .sort((a, b) => TARGET_NAMES.indexOf(key(a)) - TARGET_NAMES.indexOf(key(b)))[0];
  if (named) return named.index;
  for (let j = cols.length - 1; j >= 0; j--) if (!targetProblem(cols[j], n)) return j;
  return cols.length - 1;
}

// ---------- preprocessing ----------
// Seeded RNG so a given dataset + seed always gives the same split.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(arr, rand) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Stratified split of row indices into fractions (e.g. [0.6, 0.2, 0.2]).
function stratifiedSplit(y, fracs, rand) {
  const byClass = new Map();
  y.forEach((c, i) => (byClass.get(c) || byClass.set(c, []).get(c)).push(i));
  const parts = fracs.map(() => []);
  for (const idx of byClass.values()) {
    shuffle(idx, rand);
    let start = 0;
    fracs.forEach((f, k) => {
      const end = k === fracs.length - 1 ? idx.length : start + Math.max(1, Math.round(idx.length * f));
      parts[k].push(...idx.slice(start, Math.min(end, idx.length)));
      start = Math.min(end, idx.length);
    });
  }
  return parts.map((p) => shuffle(p, rand));
}

function stratifiedCap(idx, y, cap, rand) {
  if (idx.length <= cap) return idx;
  const keep = cap / idx.length;
  const [sub] = stratifiedSplit(idx.map((i) => y[i]), [keep, 1 - keep], rand);
  return sub.map((k) => idx[k]);
}

/**
 * Turn parsed rows into numeric matrices.
 * Numeric features: median-imputed, log-transformed if heavy-tailed, min-max scaled on train.
 * Categorical features: integer category codes (mode-imputed), left unscaled. The KNN treats
 * them with an overlap distance (same category = 0, different = 1) instead of pretending the
 * codes are ordered, which would make e.g. ticket #400 "close" to ticket #401.
 */
export function prepare(parsed, prof, targetIdx, featureIdx, seed = 42) {
  const rand = rng(seed);
  const tcol = prof.cols[targetIdx];
  const isMissing = (v) => MISSING.has(v.trim().toLowerCase());

  // drop rows with a missing target
  let rows = parsed.rows.filter((r) => !isMissing(r[targetIdx]));
  const classNames = [...new Set(rows.map((r) => r[targetIdx].trim()))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const classOf = new Map(classNames.map((c, i) => [c, i]));
  let y = rows.map((r) => classOf.get(r[targetIdx].trim()));

  let sampled = false;
  if (rows.length > LIMITS.maxRows) {
    const keep = stratifiedCap(rows.map((_, i) => i), y, LIMITS.maxRows, rand);
    rows = keep.map((i) => rows[i]); y = keep.map((i) => y[i]);
    sampled = true;
  }

  const D = featureIdx.length, N = rows.length;
  const X = new Float32Array(N * D);
  const isCat = Uint8Array.from(featureIdx, (j) => (prof.cols[j].type === "categorical" ? 1 : 0));
  const transforms = featureIdx.map((j, d) => {
    const col = prof.cols[j];
    if (col.type === "categorical") {
      const codes = new Map();
      // frequency order over the full column (profile.top only keeps 5, so recount)
      const freq = new Map();
      for (const r of rows) { const v = r[j].trim(); if (!isMissing(v)) freq.set(v, (freq.get(v) || 0) + 1); }
      [...freq.entries()].sort((a, b) => b[1] - a[1]).forEach(([v], k) => codes.set(v, k));
      const mode = 0;
      for (let i = 0; i < N; i++) {
        const v = rows[i][j].trim();
        X[i * D + d] = isMissing(v) ? mode : codes.get(v);
      }
      return { kind: "categorical", levels: codes.size };
    }
    const med = col.median ?? 0;
    for (let i = 0; i < N; i++) {
      const v = rows[i][j].trim();
      X[i * D + d] = isMissing(v) || !Number.isFinite(+v) ? med : +v;
    }
    const heavy = col.min >= 0 && col.max > 100 * (Math.abs(col.median) + 1);
    if (heavy) for (let i = 0; i < N; i++) X[i * D + d] = Math.log1p(X[i * D + d]);
    return { kind: "numeric", log: heavy };
  });

  const [tr, va, te] = stratifiedSplit(y, [0.6, 0.2, 0.2], rand);

  // min-max scale numeric features using training rows only
  for (let d = 0; d < D; d++) {
    if (isCat[d]) continue;
    let lo = Infinity, hi = -Infinity;
    for (const i of tr) { const v = X[i * D + d]; if (v < lo) lo = v; if (v > hi) hi = v; }
    const span = hi - lo || 1;
    for (let i = 0; i < N; i++) X[i * D + d] = Math.min(1, Math.max(0, (X[i * D + d] - lo) / span));
  }

  const take = (idx) => {
    const out = new Float32Array(idx.length * D);
    idx.forEach((i, r) => out.set(X.subarray(i * D, i * D + D), r * D));
    return { X: out, y: Int32Array.from(idx, (i) => y[i]), n: idx.length };
  };

  // Fitness folds. Small data: 5-fold CV over train+val, because a single small validation
  // split is so noisy the search overfits it. Larger data: one capped, stratified holdout
  // split keeps each evaluation fast.
  const pool = [...tr, ...va];
  let folds;
  if (pool.length <= 1200) {
    const parts = stratifiedSplit(pool.map((i) => y[i]), [0.2, 0.2, 0.2, 0.2, 0.2], rand).map((p) => p.map((k) => pool[k]));
    folds = parts.map((valIdx, f) => ({ train: take(parts.filter((_, g) => g !== f).flat()), val: take(valIdx) }));
  } else {
    folds = [{ train: take(stratifiedCap(tr, y, 1500, rand)), val: take(stratifiedCap(va, y, 600, rand)) }];
  }

  return {
    D, N, classNames, sampled, transforms, isCat,
    featureNames: featureIdx.map((j) => prof.cols[j].name),
    featureIdx, targetIdx, targetName: tcol.name,
    // final models train on train+val and are scored once on the untouched test split
    train: take(pool), test: take(te),
    folds, cv: folds.length > 1,
    // O(n·D²): a capped sample is plenty. Pearson is only meaningful for numeric and two-level columns.
    corr: correlation(take(stratifiedCap(tr, y, 1500, rand)), D, transforms.map((t) => t.kind === "numeric" || t.levels <= 2)),
  };
}

// Pearson correlation between features on the training split (for redundancy insights).
function correlation({ X, n }, D, ordered) {
  const mean = new Float64Array(D), sd = new Float64Array(D);
  for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) mean[d] += X[i * D + d];
  for (let d = 0; d < D; d++) mean[d] /= n;
  for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) sd[d] += (X[i * D + d] - mean[d]) ** 2;
  for (let d = 0; d < D; d++) sd[d] = Math.sqrt(sd[d] / n);
  const C = Array.from({ length: D }, () => new Float32Array(D));
  for (let a = 0; a < D; a++) {
    C[a][a] = 1;
    for (let b = a + 1; b < D; b++) {
      if (!sd[a] || !sd[b] || !ordered[a] || !ordered[b]) continue;
      let s = 0;
      for (let i = 0; i < n; i++) s += (X[i * D + a] - mean[a]) * (X[i * D + b] - mean[b]);
      C[a][b] = C[b][a] = s / (n * sd[a] * sd[b]);
    }
  }
  return C;
}
