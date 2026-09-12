"""Sensitivity of supervised difficulty baselines to the three-way split."""

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
    fit_intervals,
    predict_intervals,
    mean_interval_score,
)
from supervised_difficulty_baselines import (
    MODELS,
    NEIGHBORS,
    load_scores,
    load_features,
    three_way_split,
    local_residual_difficulty,
    normalized_intervals,
)

BASE_SEED = 20270830


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def interval_metrics(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict[str, float]:
    return {
        "coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "mean_interval_score": mean_interval_score(target, lower, upper, alpha),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=20)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    cache_dir = project_root / "results" / "canonical"
    rows: list[dict] = []

    for model in MODELS:
        print(f"[supervised-split] {model}", flush=True)
        id_scores = load_scores(cache_dir, model, ID_DATASET)
        id_features = load_features(project_root, model, ID_DATASET)
        shifted = {
            dataset: (
                load_scores(cache_dir, model, dataset),
                load_features(project_root, model, dataset),
            )
            for dataset in SHIFTED_DATASETS
        }
        for offset in range(args.n_seeds):
            seed = BASE_SEED + offset
            difficulty_mask, conformal_mask, evaluation_mask = three_way_split(
                id_scores["filepaths"], seed=seed
            )
            train_residuals = np.abs(
                id_scores["mos"][difficulty_mask] - id_scores["mos_pred"][difficulty_mask]
            )
            cal_pred = id_scores["mos_pred"][conformal_mask]
            cal_target = id_scores["mos"][conformal_mask]
            fixed = fit_intervals(
                cal_pred, cal_target, id_scores["mahalanobis"][conformal_mask],
                args.alpha, None,
            )
            unsupervised_cal = {
                "mahalanobis_mondrian": id_scores["mahalanobis"][conformal_mask],
                "predicted_mos_mondrian": cal_pred,
                "mos_extremity_mondrian": np.abs(cal_pred - 3.0),
            }
            fitted_unsupervised = {
                method: fit_intervals(
                    cal_pred, cal_target, difficulty, args.alpha, args.n_bins
                )
                for method, difficulty in unsupervised_cal.items()
            }
            local_cal = {
                neighbors: local_residual_difficulty(
                    id_features[difficulty_mask], train_residuals,
                    id_features[conformal_mask], neighbors,
                )
                for neighbors in NEIGHBORS
            }
            fitted_local = {
                neighbors: fit_intervals(
                    cal_pred, cal_target, difficulty, args.alpha, args.n_bins
                )
                for neighbors, difficulty in local_cal.items()
            }

            datasets = {
                ID_DATASET: (id_scores, id_features, evaluation_mask),
                **{
                    dataset: (data, features, np.ones(len(data["mos"]), dtype=bool))
                    for dataset, (data, features) in shifted.items()
                },
            }
            for dataset, (data, features, mask) in datasets.items():
                target = data["mos"][mask]
                pred = data["mos_pred"][mask]
                fixed_lower, fixed_upper = predict_intervals(
                    pred, data["mahalanobis"][mask], fixed
                )
                fixed_metrics = interval_metrics(
                    target, fixed_lower, fixed_upper, args.alpha
                )
                methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                test_unsupervised = {
                    "mahalanobis_mondrian": data["mahalanobis"][mask],
                    "predicted_mos_mondrian": pred,
                    "mos_extremity_mondrian": np.abs(pred - 3.0),
                }
                for method, fitted in fitted_unsupervised.items():
                    methods[method] = predict_intervals(
                        pred, test_unsupervised[method], fitted
                    )
                for neighbors in NEIGHBORS:
                    local_test = local_residual_difficulty(
                        id_features[difficulty_mask], train_residuals,
                        features[mask], neighbors,
                    )
                    methods[f"local_residual_mondrian_k{neighbors}"] = predict_intervals(
                        pred, local_test, fitted_local[neighbors]
                    )
                    methods[f"local_residual_normalized_k{neighbors}"] = normalized_intervals(
                        cal_target, cal_pred, local_cal[neighbors],
                        pred, local_test, args.alpha,
                    )
                for method, (lower, upper) in methods.items():
                    values = interval_metrics(target, lower, upper, args.alpha)
                    rows.append({
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        **values,
                        "delta_coverage": values["coverage"] - fixed_metrics["coverage"],
                        "delta_width": values["mean_width"] - fixed_metrics["mean_width"],
                        "delta_interval_score": (
                            values["mean_interval_score"]
                            - fixed_metrics["mean_interval_score"]
                        ),
                    })

    output_path = cache_dir / "supervised_split_sensitivity.csv"
    summary_path = cache_dir / "supervised_split_summary.csv"
    raw = pd.DataFrame(rows)
    raw.to_csv(output_path, index=False)

    seed_macro = (
        raw.groupby(["model", "method", "seed"], as_index=False)[
            ["delta_coverage", "delta_width", "delta_interval_score"]
        ].mean()
    )
    summaries = []
    for aggregation, frame, group_columns in (
        ("macro_across_datasets", seed_macro, ["model", "method"]),
        ("per_dataset", raw, ["model", "dataset", "method"]),
    ):
        for keys, group in frame.groupby(group_columns, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys))
            values = group["delta_interval_score"]
            row.update({
                "aggregation": aggregation,
                "n_seeds": int(group["seed"].nunique()),
                "mean_delta_coverage": float(group["delta_coverage"].mean()),
                "mean_delta_width": float(group["delta_width"].mean()),
                "mean_delta_interval_score": float(values.mean()),
                "std_delta_interval_score": float(values.std(ddof=1)),
                "min_delta_interval_score": float(values.min()),
                "max_delta_interval_score": float(values.max()),
                "fraction_improved": float((values < 0).mean()),
            })
            summaries.append(row)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["supervised_split_sensitivity"] = {
        "base_seed": BASE_SEED,
        "n_seeds": args.n_seeds,
        "n_bins": args.n_bins,
        "neighbors": list(NEIGHBORS),
    }
    manifest["outputs"] = list(dict.fromkeys([
        *manifest["outputs"], output_path.name, summary_path.name
    ]))
    manifest.setdefault("code_sha256", {})[
        "supervised_split_sensitivity.py"
    ] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {
        name: sha256(cache_dir / name) for name in manifest["outputs"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[supervised-split] Complete", flush=True)


if __name__ == "__main__":
    main()
