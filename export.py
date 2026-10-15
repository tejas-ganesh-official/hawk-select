"""Bundle experiment outputs into the static JSON files the website reads (site/data/)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import FEATURES

SITE_DATA = Path(__file__).parent / "site" / "data"


def _all_features_test(task: str) -> dict:
    import numpy as np

    from data import load
    from fitness import test_metrics

    ds = load(task)
    return test_metrics(ds, np.ones(len(ds.feature_names), bool))


def export_replay(run_path: Path, out: Path = SITE_DATA / "replay.json") -> None:
    run = json.loads(run_path.read_text())
    if "masks" not in run["history"][0]:
        raise ValueError(f"{run_path} was run without mask logging; cannot replay")
    replay = {
        "algorithm": run["algorithm"],
        "task": run.get("task", "binary"),
        "seed": run["seed"],
        "n_hawks": run["n_hawks"],
        "iterations": run["iterations"],
        "feature_names": FEATURES,
        "strategy_names": run["strategy_names"],
        "selected_features": run["selected_features"],
        "best_fitness": run["best_fitness"],
        "best_val_error": run.get("best_val_error"),
        "fitness_evals": run.get("fitness_evals"),
        "runtime_s": run["runtime_s"],
        "test": run.get("test"),
        "history": [
            {k: h[k] for k in ("t", "energy", "strategy", "best_fitness", "mean_fitness",
                               "best_n_selected", "masks", "best_mask")}
            for h in run["history"]
        ],
    }
    replay["baseline_test"] = _all_features_test(replay["task"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) from {run_path}")


FEATURE_GROUPS = [("basic", 9), ("content", 13), ("traffic", 9), ("host", 10)]
ALGO_LABELS = {"hho_s": "HHO-S", "hho_v": "HHO-V", "ga": "GA", "pso": "PSO",
               "mi": "MI top-k", "chi2": "Chi² top-k", "all": "All features"}
METRICS = ["fitness", "val_error", "n_selected", "accuracy", "macro_f1",
           "detection_rate", "false_alarm_rate", "fitness_evals", "runtime_s"]


def export_dataset(out: Path = SITE_DATA / "dataset.json") -> None:
    """Class balance, attack types, per-feature stats and the feature correlation matrix."""
    import numpy as np
    from sklearn.feature_selection import mutual_info_classif

    from data import ATTACK_CATEGORY, CLASSES_MULTI, TEST_FILE, TRAIN_FILE, _read, load

    raw = {"train": _read(TRAIN_FILE), "test": _read(TEST_FILE)}
    classes, types = {}, {}
    for split, df in raw.items():
        cats = df["label"].map(ATTACK_CATEGORY)
        classes[split] = {c: int((cats == c).sum()) for c in CLASSES_MULTI}
        vc = df.loc[df["label"] != "normal", "label"].value_counts()
        types[split] = [{"name": k, "category": ATTACK_CATEGORY[k], "count": int(v)} for k, v in vc.items()]
    novel = sorted(set(raw["test"]["label"]) - set(raw["train"]["label"]))

    ds = load("binary")
    X, y = ds.X_train, ds.y_train
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), 20000, replace=False)
    mi = mutual_info_classif(X[idx], y[idx], random_state=0)

    groups = [g for g, n in FEATURE_GROUPS for _ in range(n)]
    features = [{"name": n, "group": groups[d], "mi": round(float(mi[d]), 4),
                 "constant": bool(X[:, d].std() == 0)} for d, n in enumerate(FEATURES)]
    data = {
        "n_train": len(raw["train"]), "n_test": len(raw["test"]),
        "classes": classes, "attack_types": types, "novel_test_types": novel,
        "features": features, "groups": [g for g, _ in FEATURE_GROUPS],
        "corr": np.round(corr, 2).tolist(),
    }
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


def export_results(task: str = "binary", sweep: Path = Path("results/sweep"),
                   out: Path = SITE_DATA / "results.json") -> None:
    """Aggregate every sweep run per algorithm: metrics, convergence bands, selection frequency."""
    import numpy as np

    runs: dict[str, list] = {}
    for p in sorted(sweep.glob(f"*_{task}_seed*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["key"], []).append(r)

    algos = []
    for key, label in ALGO_LABELS.items():
        rs = sorted(runs.get(key, []), key=lambda r: r["seed"])
        if not rs:
            continue
        rows = [{
            "seed": r["seed"], "fitness": r["best_fitness"], "val_error": r["best_val_error"],
            "n_selected": r["n_selected"], "accuracy": r["test"]["accuracy"],
            "macro_f1": r["test"]["macro_f1"], "detection_rate": r["test"]["detection_rate"],
            "false_alarm_rate": r["test"]["false_alarm_rate"],
            "fitness_evals": r["fitness_evals"], "runtime_s": r["runtime_s"],
        } for r in rs]
        mat = {m: np.array([row[m] for row in rows], float) for m in METRICS}
        masks = np.array([[c == "1" for c in r["best_mask"]] for r in rs])
        cms = np.array([r["test"]["confusion_matrix"] for r in rs], float)
        cm_norm = (cms / cms.sum(axis=2, keepdims=True)).mean(axis=0)

        entry = {
            "key": key, "label": label, "n_runs": len(rs), "runs": rows,
            "mean": {m: float(v.mean()) for m, v in mat.items()},
            "std": {m: float(v.std()) for m, v in mat.items()},
            "selection_freq": masks.mean(axis=0).round(3).tolist(),
            "confusion": np.round(cm_norm, 4).tolist(),
            "convergence": None,
        }
        if key in ("hho_s", "hho_v", "ga", "pso"):
            curves = np.array([[h["best_fitness"] for h in r["history"]] for r in rs])
            nsel = np.array([[h["best_n_selected"] for h in r["history"]] for r in rs])
            entry["convergence"] = {
                "mean": curves.mean(0).round(6).tolist(), "std": curves.std(0).round(6).tolist(),
                "nsel_mean": nsel.mean(0).round(2).tolist(),
            }
        algos.append(entry)

    from data import CLASSES_MULTI
    data = {"task": task, "feature_names": FEATURES, "algos": algos,
            "class_names": ["normal", "attack"] if task == "binary" else CLASSES_MULTI}
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB) from {sum(len(v) for v in runs.values())} runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["replay", "dataset", "results"],
                    choices=["replay", "dataset", "results"])
    ap.add_argument("--replay", type=Path, default=Path("results/sweep/hho_v_binary_seed0.json"))
    args = ap.parse_args()
    if "replay" in args.what:
        export_replay(args.replay)
    if "dataset" in args.what:
        export_dataset()
    if "results" in args.what:
        export_results()
