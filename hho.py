"""Binary Harris Hawks Optimization for feature selection (Heidari et al. 2019).

Each hawk keeps a continuous position X in [LB, UB]^D and a 0/1 feature mask B.
The continuous HHO update rules move X; a transfer function turns that into B:

  S-shaped:  B_d = 1 if rand < sigmoid(X_d)             (position -> probability of 1)
  V-shaped:  flip B_d if rand < |tanh(X_d - X_d_old)|   (step size -> probability of flip)

V-shaped is applied to the displacement rather than the position: as besieging tightens
and moves shrink, hawks stop flipping bits, which is what lets the search settle.

Every iteration is logged (energy, strategy per hawk, masks, best-so-far) so the website
can replay the run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

LB, UB = -4.0, 4.0

# strategy codes; must match site/js/hho2d.js STRATEGY
PERCH_RANDOM, PERCH_MEAN, SOFT, HARD, SOFT_DIVE, HARD_DIVE = range(6)
STRATEGY_NAMES = ["perch_random", "perch_mean", "soft", "hard", "soft_dive", "hard_dive"]


def _levy(rng: np.random.Generator, d: int, beta: float = 1.5) -> np.ndarray:
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, d)
    v = rng.normal(0, 1, d)
    return 0.01 * u / np.abs(v) ** (1 / beta)


class Transfer:
    def __init__(self, kind: str, rng: np.random.Generator):
        if kind not in ("s", "v"):
            raise ValueError("transfer must be 's' or 'v'")
        self.kind, self.rng = kind, rng

    def __call__(self, x_new: np.ndarray, x_old: np.ndarray, b_old: np.ndarray) -> np.ndarray:
        r = self.rng.random(x_new.shape)
        if self.kind == "s":
            return r < 1 / (1 + np.exp(-x_new))
        flip = r < np.abs(np.tanh(x_new - x_old))
        return np.where(flip, ~b_old, b_old)


def binary_hho(fitness, n_features: int, n_hawks: int = 20, iterations: int = 100,
               transfer: str = "s", seed: int = 0, log_masks: bool = True,
               feature_names: list[str] | None = None) -> dict:
    """Minimise `fitness(mask) -> float`. Returns the result dict that run.py / the site consume."""
    rng = np.random.default_rng(seed)
    tf = Transfer(transfer, rng)
    N, D, T = n_hawks, n_features, iterations
    t_start = time.perf_counter()

    X = rng.uniform(LB, UB, (N, D))
    B = rng.random((N, D)) < 0.5
    B[~B.any(axis=1), rng.integers(D)] = True  # no empty masks at start
    F = np.array([fitness(b) for b in B])

    best = int(F.argmin())
    rabbit_x, rabbit_b, rabbit_f = X[best].copy(), B[best].copy(), float(F[best])

    def pack(masks):
        return ["".join("1" if v else "0" for v in m) for m in masks]

    history = [{
        "t": 0, "energy": [2.0] * N, "strategy": [None] * N,
        "best_fitness": rabbit_f, "mean_fitness": float(F.mean()),
        "best_n_selected": int(rabbit_b.sum()),
        **({"masks": pack(B), "best_mask": pack([rabbit_b])[0]} if log_masks else {}),
    }]

    for t in range(T):
        X_mean = X.mean(axis=0)
        energies, strategies = [], []

        for i in range(N):
            E0 = 2 * rng.random() - 1
            E = 2 * E0 * (1 - t / T)
            J = 2 * (1 - rng.random())
            absE = abs(E)
            x_old, b_old = X[i].copy(), B[i].copy()

            if absE >= 1:  # exploration
                if rng.random() >= 0.5:
                    Xr = X[rng.integers(N)]
                    x_new = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * x_old)
                    s = PERCH_RANDOM
                else:
                    x_new = (rabbit_x - X_mean) - rng.random() * (LB + rng.random() * (UB - LB))
                    s = PERCH_MEAN
                x_new = np.clip(x_new, LB, UB)
                b_new = tf(x_new, x_old, b_old)
                f_new = None
            else:  # exploitation
                r = rng.random()
                if r >= 0.5:
                    if absE >= 0.5:
                        x_new = (rabbit_x - x_old) - E * np.abs(J * rabbit_x - x_old)
                        s = SOFT
                    else:
                        x_new = rabbit_x - E * np.abs(rabbit_x - x_old)
                        s = HARD
                    x_new = np.clip(x_new, LB, UB)
                    b_new = tf(x_new, x_old, b_old)
                    f_new = None
                else:  # progressive rapid dives with greedy acceptance
                    ref = x_old if absE >= 0.5 else X_mean
                    s = SOFT_DIVE if absE >= 0.5 else HARD_DIVE
                    Y = np.clip(rabbit_x - E * np.abs(J * rabbit_x - ref), LB, UB)
                    bY = tf(Y, x_old, b_old)
                    fY = fitness(bY)
                    if fY < F[i]:
                        x_new, b_new, f_new = Y, bY, fY
                    else:
                        Z = np.clip(Y + rng.random(D) * _levy(rng, D), LB, UB)
                        bZ = tf(Z, x_old, b_old)
                        fZ = fitness(bZ)
                        if fZ < F[i]:
                            x_new, b_new, f_new = Z, bZ, fZ
                        else:
                            x_new, b_new, f_new = x_old, b_old, F[i]

            if not b_new.any():
                b_new = b_new.copy()
                b_new[rng.integers(D)] = True
                f_new = None
            X[i], B[i] = x_new, b_new
            F[i] = fitness(b_new) if f_new is None else f_new
            energies.append(round(E, 4))
            strategies.append(s)

        best = int(F.argmin())
        if F[best] < rabbit_f:
            rabbit_x, rabbit_b, rabbit_f = X[best].copy(), B[best].copy(), float(F[best])

        history.append({
            "t": t + 1, "energy": energies, "strategy": strategies,
            "best_fitness": rabbit_f, "mean_fitness": float(F.mean()),
            "best_n_selected": int(rabbit_b.sum()),
            **({"masks": pack(B), "best_mask": pack([rabbit_b])[0]} if log_masks else {}),
        })

    names = feature_names or [f"f{d}" for d in range(D)]
    return {
        "algorithm": f"HHO-{transfer.upper()}",
        "seed": seed, "n_hawks": N, "iterations": T, "transfer": transfer,
        "best_fitness": rabbit_f,
        "best_mask": pack([rabbit_b])[0],
        "selected_features": [names[d] for d in np.flatnonzero(rabbit_b)],
        "n_selected": int(rabbit_b.sum()),
        "runtime_s": round(time.perf_counter() - t_start, 2),
        "strategy_names": STRATEGY_NAMES,
        "history": history,
    }


if __name__ == "__main__":
    from data import load
    from fitness import Fitness, test_metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="binary", choices=["binary", "multi"])
    ap.add_argument("--transfer", default="s", choices=["s", "v"])
    ap.add_argument("--hawks", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ds = load(args.task)
    fit = Fitness(ds, seed=args.seed)
    res = binary_hho(fit, len(ds.feature_names), args.hawks, args.iters, args.transfer,
                     args.seed, feature_names=ds.feature_names)
    res["task"] = args.task
    res["fitness_evals"], res["fitness_calls"] = fit.evals, fit.calls
    _, res["best_val_error"] = fit.evaluate(np.array([c == "1" for c in res["best_mask"]]))
    res["test"] = test_metrics(ds, np.array([c == "1" for c in res["best_mask"]]))

    curve = [h["best_fitness"] for h in res["history"]]
    marks = sorted({0, len(curve) // 4, len(curve) // 2, 3 * len(curve) // 4, len(curve) - 1})
    print(f"{res['algorithm']} seed={args.seed}  {res['runtime_s']}s  "
          f"evals={fit.evals} calls={fit.calls} (cache hit {fit.hit_rate:.0%})")
    print("convergence:", "  ".join(f"t{m}={curve[m]:.5f}" for m in marks))
    print(f"best fitness={res['best_fitness']:.5f} val_err={res['best_val_error']:.4f} n={res['n_selected']}")
    print("selected:", res["selected_features"])
    t = res["test"]
    print(f"TEST acc={t['accuracy']:.4f} macroF1={t['macro_f1']:.4f} DR={t['detection_rate']:.4f} FAR={t['false_alarm_rate']:.4f}")

    out = args.out or Path("results/runs") / f"hho_{args.transfer}_{args.task}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print("wrote", out, f"({out.stat().st_size / 1024:.0f} KB)")
