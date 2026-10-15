"""Wrapper fitness for feature selection.

fitness(mask) = alpha * error_rate + (1 - alpha) * n_selected / n_features   (lower is better)

The error rate comes from KNN trained on a stratified subsample of the training split
and scored on a stratified subsample of the validation split. 5000/3000 rows rank random
masks as faithfully as 15000/5000 (Spearman ~0.97 against full training data) at ~4x
less cost. The test set is never
touched here; use `test_metrics` once an optimizer has finished.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from data import Dataset, load


def _stratified_subsample(X, y, n, seed):
    if n is None or n >= len(y):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


class Fitness:
    def __init__(self, ds: Dataset, alpha: float = 0.99, k: int = 5,
                 n_train: int | None = 5000, n_val: int | None = 3000, seed: int = 0):
        self.ds = ds
        self.alpha = alpha
        self.k = k
        self.n_features = ds.X_train.shape[1]
        self.X_tr, self.y_tr = _stratified_subsample(ds.X_train, ds.y_train, n_train, seed)
        self.X_va, self.y_va = _stratified_subsample(ds.X_val, ds.y_val, n_val, seed)
        self.cache: dict[bytes, tuple[float, float]] = {}  # mask bytes -> (fitness, error)
        self.evals = 0  # classifier fits actually run
        self.calls = 0

    def error(self, mask: np.ndarray) -> float:
        cols = np.flatnonzero(mask)
        # brute force, single-threaded: threads barely help at this size, and run.py
        # parallelises across seeds with one process per core instead.
        clf = KNeighborsClassifier(n_neighbors=self.k, algorithm="brute", n_jobs=1)
        clf.fit(self.X_tr[:, cols], self.y_tr)
        return 1.0 - accuracy_score(self.y_va, clf.predict(self.X_va[:, cols]))

    def evaluate(self, mask: np.ndarray) -> tuple[float, float]:
        """Return (fitness, validation error) for a 0/1 mask, cached by mask bytes."""
        mask = np.asarray(mask, dtype=bool)
        self.calls += 1
        key = np.packbits(mask).tobytes()
        if key in self.cache:
            return self.cache[key]
        n_sel = int(mask.sum())
        if n_sel == 0:
            result = (1.0, 1.0)  # empty subset is the worst possible solution
        else:
            err = self.error(mask)
            self.evals += 1
            result = (self.alpha * err + (1 - self.alpha) * n_sel / self.n_features, err)
        self.cache[key] = result
        return result

    def __call__(self, mask: np.ndarray) -> float:
        return self.evaluate(mask)[0]

    @property
    def hit_rate(self) -> float:
        return 1 - self.evals / self.calls if self.calls else 0.0


def test_metrics(ds: Dataset, mask: np.ndarray, k: int = 5) -> dict:
    """Final report numbers: KNN on the full training split, scored on KDDTest+."""
    cols = np.flatnonzero(np.asarray(mask, dtype=bool))
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(ds.X_train[:, cols], ds.y_train)
    pred = clf.predict(ds.X_test[:, cols])
    y = ds.y_test
    cm = confusion_matrix(y, pred, labels=range(len(ds.class_names)))
    attack, normal = y != 0, y == 0
    return {
        "n_features": int(len(cols)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "detection_rate": float((pred[attack] != 0).mean()),    # attacks flagged as any attack
        "false_alarm_rate": float((pred[normal] != 0).mean()),  # normal flagged as attack
        "confusion_matrix": cm.tolist(),
    }


def random_search(fit: Fitness, n: int, seed: int = 0):
    """Sanity check: random masks across densities 10%..90%."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = rng.uniform(0.1, 0.9)
        mask = rng.random(fit.n_features) < p
        f, err = fit.evaluate(mask)
        rows.append((f, err, int(mask.sum()), mask))
    return sorted(rows, key=lambda r: r[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="binary", choices=["binary", "multi"])
    ap.add_argument("--n", type=int, default=150, help="random masks to try")
    ap.add_argument("--alpha", type=float, default=0.99)
    args = ap.parse_args()

    ds = load(args.task)
    fit = Fitness(ds, alpha=args.alpha)
    all_mask = np.ones(fit.n_features, bool)

    t0 = time.perf_counter()
    f_all, e_all = fit.evaluate(all_mask)
    t_eval = time.perf_counter() - t0
    fit.evaluate(all_mask)  # second call must hit the cache
    print(f"all-features  fitness={f_all:.5f}  val_err={e_all:.4f}  ({t_eval:.2f}s/eval, cache hit ok={fit.evals == 1})")

    t0 = time.perf_counter()
    rows = random_search(fit, args.n)
    dt = time.perf_counter() - t0
    fs = np.array([r[0] for r in rows])
    errs = np.array([r[1] for r in rows])
    print(f"random search: {args.n} masks in {dt:.1f}s ({dt / fit.evals:.2f}s/eval avg)")
    print(f"  fitness  best={fs.min():.5f}  median={np.median(fs):.5f}  worst={fs.max():.5f}")
    print(f"  val_err  best={errs.min():.4f}  median={np.median(errs):.4f}  worst={errs.max():.4f}")
    print(f"  masks beating all-features on fitness: {(fs < f_all).sum()}/{args.n}")
    print(f"  masks with val_err <= all-features:     {(errs <= e_all).sum()}/{args.n}")
    print("top 5:")
    for f, err, n_sel, mask in rows[:5]:
        print(f"  fitness={f:.5f}  val_err={err:.4f}  n={n_sel:2d}")

    best = rows[0][3]
    print("best random mask features:", [ds.feature_names[i] for i in np.flatnonzero(best)])
    for name, m in (("all-features", all_mask), ("best random", best)):
        t = test_metrics(ds, m)
        print(f"TEST {name:12s} n={t['n_features']:2d} acc={t['accuracy']:.4f} macroF1={t['macro_f1']:.4f} "
              f"DR={t['detection_rate']:.4f} FAR={t['false_alarm_rate']:.4f}")
