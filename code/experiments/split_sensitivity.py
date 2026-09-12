"""Sensitivity of conformal results to the held-out V-SIM split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_analysis import (
    ID_DATASET,
    SHIFTED_DATASETS,
    deterministic_group_split,
    fit_intervals,
    predict_intervals,
    mean_interval_score,
)

MODELS = ("dnsmos", "nisqa", "utmos")
SCORES = ("mahalanobis", "l2", "knn")
FRACTIONS = (0.3, 0.5, 0.7)
SEEDS = tuple(range(20270001, 20270021))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_score(cache_dir: Path, model: str, dataset: str) -> dict[str, np.ndarray]:
    loaded = np.load(cache_dir / "scores" / f"scores_{model}_{dataset}.npz", allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="results/canonical")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_bins", type=int, default=5)
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)
    rows = []

    for model in MODELS:
        id_data = load_score(cache_dir, model, ID_DATASET)
        shifted = {dataset: load_score(cache_dir, model, dataset) for dataset in SHIFTED_DATASETS}
        for fraction in FRACTIONS:
            for seed in SEEDS:
                cal_mask, eval_mask, _ = deterministic_group_split(
                    id_data["filepaths"], fraction, seed
                )
                datasets = {ID_DATASET: {key: value[eval_mask] for key, value in id_data.items()}}
                datasets.update(shifted)
                for score_name in SCORES:
                    fit_args = (
                        id_data["mos_pred"][cal_mask], id_data["mos"][cal_mask],
                        id_data[score_name][cal_mask], args.alpha,
                    )
                    fixed = fit_intervals(*fit_args, None)
                    adaptive = fit_intervals(*fit_args, args.n_bins)
                    for dataset, data in datasets.items():
                        metrics = {}
                        for method, fitted in (("fixed", fixed), ("adaptive", adaptive)):
                            lo, hi = predict_intervals(data["mos_pred"], data[score_name], fitted)
                            metrics[method] = {
                                "coverage": float(np.mean((data["mos"] >= lo) & (data["mos"] <= hi))),
                                "width": float(np.mean(hi - lo)),
                                "interval_score": mean_interval_score(data["mos"], lo, hi, args.alpha),
                            }
                        rows.append({
                            "model": model,
                            "score": score_name,
                            "calibration_fraction": fraction,
                            "seed": seed,
                            "dataset": dataset,
                            "calibration_n": int(cal_mask.sum()),
                            "evaluation_n": len(data["mos"]),
                            **{f"fixed_{key}": value for key, value in metrics["fixed"].items()},
                            **{f"adaptive_{key}": value for key, value in metrics["adaptive"].items()},
                            **{f"delta_{key}": metrics["adaptive"][key] - metrics["fixed"][key]
                               for key in metrics["fixed"]},
                        })
    output = cache_dir / "split_sensitivity.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_sensitivity"] = {
        "seeds": list(SEEDS), "calibration_fractions": list(FRACTIONS),
        "n_bins": args.n_bins,
    }
    manifest["outputs"] = list(dict.fromkeys([*manifest["outputs"], output.name]))
    manifest.setdefault("code_sha256", {})["split_sensitivity.py"] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {name: sha256(cache_dir / name) for name in manifest["outputs"]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[split_sensitivity] Saved {output}")


if __name__ == "__main__":
    main()
