"""Baselines for the HHO comparison: binary GA, binary PSO, and filter (MI / chi2) top-k.

All return the same result dict shape as hho.binary_hho (minus per-hawk logs), so run.py
and the website treat every algorithm alike.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.feature_selection import chi2, mutual_info_classif


def _record(F, best_f, best_b, t):
    return {"t": t, "best_fitness": float(best_f), "mean_fitness": float(np.mean(F)),
            "best_n_selected": int(best_b.sum())}


def _result(name, seed, n, T, best_f, best_b, names, history, t_start, **extra):
    return {
        "algorithm": name, "seed": seed, "n_hawks": n, "iterations": T,
        "best_fitness": float(best_f),
        "best_mask": "".join("1" if v else "0" for v in best_b),
        "selected_features": [names[d] for d in np.flatnonzero(best_b)],
        "n_selected": int(best_b.sum()),
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "history": history, **extra,
    }


def _repair(B, rng):
    """No empty subsets: switch on one random feature in any all-zero row."""
    empty = ~B.any(axis=1)
    B[empty, rng.integers(B.shape[1], size=empty.sum())] = True
    return B


def binary_ga(fitness, n_features, pop_size=20, iterations=100, seed=0, feature_names=None,
              crossover_rate=0.9, mutation_rate=None, tournament=3, elites=2):
    """Generational GA: tournament selection, uniform crossover, bit-flip mutation, elitism."""
    rng = np.random.default_rng(seed)
    D, N, T = n_features, pop_size, iterations
    pm = mutation_rate if mutation_rate is not None else 1.0 / D
    names = feature_names or [f"f{d}" for d in range(D)]
    t_start = time.perf_counter()

    P = _repair(rng.random((N, D)) < 0.5, rng)
    F = np.array([fitness(b) for b in P])
    b = int(F.argmin())
    best_f, best_b = F[b], P[b].copy()
    history = [_record(F, best_f, best_b, 0)]

    def pick():
        idx = rng.integers(N, size=tournament)
        return P[idx[F[idx].argmin()]]

    for t in range(T):
        order = np.argsort(F)
        children = [P[i].copy() for i in order[:elites]]
        while len(children) < N:
            a, c = pick(), pick()
            if rng.random() < crossover_rate:
                swap = rng.random(D) < 0.5
                a, c = np.where(swap, c, a), np.where(swap, a, c)
            for child in (a, c):
                child = child ^ (rng.random(D) < pm)
                children.append(child)
        P = _repair(np.array(children[:N]), rng)
        F = np.array([fitness(x) for x in P])
        b = int(F.argmin())
        if F[b] < best_f:
            best_f, best_b = F[b], P[b].copy()
        history.append(_record(F, best_f, best_b, t + 1))

    return _result("GA", seed, N, T, best_f, best_b, names, history, t_start)


def binary_pso(fitness, n_features, swarm_size=20, iterations=100, seed=0, feature_names=None,
               c1=2.0, c2=2.0, w_start=0.9, w_end=0.4, v_max=6.0):
    """Kennedy & Eberhart (1997) binary PSO: sigmoid of velocity gives P(bit = 1)."""
    rng = np.random.default_rng(seed)
    D, N, T = n_features, swarm_size, iterations
    names = feature_names or [f"f{d}" for d in range(D)]
    t_start = time.perf_counter()

    X = _repair(rng.random((N, D)) < 0.5, rng)
    V = rng.uniform(-1, 1, (N, D))
    F = np.array([fitness(x) for x in X])
    pbest, pbest_f = X.copy(), F.copy()
    g = int(F.argmin())
    gbest, gbest_f = X[g].copy(), F[g]
    history = [_record(F, gbest_f, gbest, 0)]

    for t in range(T):
        w = w_start - (w_start - w_end) * t / T
        r1, r2 = rng.random((N, D)), rng.random((N, D))
        Xf = X.astype(float)
        V = w * V + c1 * r1 * (pbest - Xf) + c2 * r2 * (gbest - Xf)
        V = np.clip(V, -v_max, v_max)
        X = _repair(rng.random((N, D)) < 1 / (1 + np.exp(-V)), rng)
        F = np.array([fitness(x) for x in X])
        improved = F < pbest_f
        pbest[improved], pbest_f[improved] = X[improved], F[improved]
        g = int(pbest_f.argmin())
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), pbest_f[g]
        history.append(_record(F, gbest_f, gbest, t + 1))

    return _result("PSO", seed, N, T, gbest_f, gbest, names, history, t_start)


def filter_topk(fitness, X, y, method="mi", seed=0, feature_names=None):
    """Rank features by a filter score, then pick k by the same wrapper fitness on validation.

    Tuning k with the wrapper fitness keeps the comparison fair: the filter gets the same
    objective as the metaheuristics, it just can't search beyond prefixes of the ranking.
    """
    t_start = time.perf_counter()
    D = X.shape[1]
    names = feature_names or [f"f{d}" for d in range(D)]
    if method == "mi":
        scores = mutual_info_classif(X, y, random_state=seed)
    elif method == "chi2":
        scores = np.nan_to_num(chi2(X, y)[0])
    else:
        raise ValueError(method)
    order = np.argsort(-scores)

    history, best_f, best_b = [], np.inf, None
    for k in range(1, D + 1):
        mask = np.zeros(D, bool)
        mask[order[:k]] = True
        f = fitness(mask)
        if f < best_f:
            best_f, best_b = f, mask
        history.append({"t": k, "best_fitness": float(best_f), "mean_fitness": float(f),
                        "best_n_selected": int(best_b.sum())})

    return _result(method.upper(), seed, 1, D, best_f, best_b, names, history, t_start,
                   ranking=[names[d] for d in order], scores=[float(scores[d]) for d in order])
