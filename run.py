"""Multi-seed experiment runner: every (algorithm, seed) pair is one job, one process per core.

    python run.py --seeds 30 --workers 4
    python run.py --algos hho_v ga --seeds 3     # subset

Each job writes results/sweep/<algo>_<task>_seed<k>.json and is skipped if that file exists,
so an interrupted sweep resumes where it stopped. A summary CSV is rebuilt at the end.
"""

import os

# One BLAS thread per process; parallelism comes from running seeds side by side.
# Must be set before numpy is imported (Windows workers re-import this module).
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Separate from results/runs (standalone hho.py runs) so the two can never collide.
RUNS = Path(__file__).parent / "results" / "sweep"
ALGOS = ["hho_s", "hho_v", "ga", "pso", "mi", "chi2", "all"]

_ds_cache = {}


def _dataset(task):
    if task not in _ds_cache:
        from data import load
        _ds_cache[task] = load(task)
    return _ds_cache[task]


def run_job(algo, seed, task, n_agents, iterations):
    from baselines import binary_ga, binary_pso, filter_topk
    from fitness import Fitness, test_metrics
    from hho import binary_hho

    ds = _dataset(task)
    fit = Fitness(ds, seed=seed)  # same subsample per seed across algorithms -> paired tests
    D, names = len(ds.feature_names), ds.feature_names

    if algo in ("hho_s", "hho_v"):
        res = binary_hho(fit, D, n_agents, iterations, algo[-1], seed,
                         log_masks=(seed == 0), feature_names=names)
    elif algo == "ga":
        res = binary_ga(fit, D, n_agents, iterations, seed, names)
    elif algo == "pso":
        res = binary_pso(fit, D, n_agents, iterations, seed, names)
    elif algo in ("mi", "chi2"):
        res = filter_topk(fit, fit.X_tr, fit.y_tr, algo, seed, names)
    elif algo == "all":
        t0 = time.perf_counter()
        f = fit(np.ones(D, bool))
        res = {"algorithm": "ALL", "seed": seed, "best_fitness": f, "best_mask": "1" * D,
               "selected_features": list(names), "n_selected": D,
               "runtime_s": round(time.perf_counter() - t0, 2), "history": []}
    else:
        raise ValueError(algo)

    mask = np.array([c == "1" for c in res["best_mask"]])
    res.update(task=task, key=algo, fitness_evals=fit.evals, fitness_calls=fit.calls,
               best_val_error=fit.evaluate(mask)[1], test=test_metrics(ds, mask))
    out = RUNS / f"{algo}_{task}_seed{seed}.json"
    out.write_text(json.dumps(res))
    return algo, seed, res


def summarise(task):
    rows = []
    for p in sorted(RUNS.glob(f"*_{task}_seed*.json")):
        r = json.loads(p.read_text())
        if "key" not in r:
            continue  # standalone hho.py runs without test/key fields
        t = r["test"]
        rows.append({
            "algo": r["key"], "seed": r["seed"], "fitness": r["best_fitness"],
            "val_error": r["best_val_error"], "n_selected": r["n_selected"],
            "test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"],
            "detection_rate": t["detection_rate"], "false_alarm_rate": t["false_alarm_rate"],
            "fitness_evals": r["fitness_evals"], "runtime_s": r["runtime_s"],
        })
    out = RUNS.parent / f"summary_{task}.csv"
    if rows:
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return out, rows


def print_table(rows):
    algos = [a for a in ALGOS if any(r["algo"] == a for r in rows)]
    cols = ["fitness", "n_selected", "test_accuracy", "test_macro_f1", "detection_rate",
            "false_alarm_rate", "fitness_evals", "runtime_s"]
    print(f"{'algo':6s} {'runs':>4s} " + " ".join(f"{c[:16]:>18s}" for c in cols))
    for a in algos:
        sub = [r for r in rows if r["algo"] == a]
        cells = []
        for c in cols:
            v = np.array([r[c] for r in sub], float)
            fmt = "{:.4f} +- {:.4f}" if v.mean() < 1 else "{:.1f} +- {:.1f}"
            cells.append(f"{fmt.format(v.mean(), v.std()):>18s}")
        print(f"{a:6s} {len(sub):4d} " + " ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--task", default="binary", choices=["binary", "multi"])
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = [(a, s) for s in range(args.seed_start, args.seed_start + args.seeds) for a in args.algos
            if not (RUNS / f"{a}_{args.task}_seed{s}.json").exists()]
    print(f"{len(jobs)} jobs to run on {args.workers} workers")

    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_job, a, s, args.task, args.agents, args.iters): (a, s) for a, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            a, s = futs[fut]
            try:
                _, _, r = fut.result()
                print(f"[{i}/{len(jobs)} {time.perf_counter() - t0:7.0f}s] {a:6s} seed {s:2d}  "
                      f"fit={r['best_fitness']:.5f} n={r['n_selected']:2d} "
                      f"test_acc={r['test']['accuracy']:.4f} ({r['runtime_s']:.0f}s)", flush=True)
            except Exception as e:  # keep the sweep going; the job reruns next time
                print(f"[{i}/{len(jobs)}] {a} seed {s} FAILED: {e!r}", flush=True)

    out, rows = summarise(args.task)
    print(f"\nwrote {out}")
    print_table(rows)
