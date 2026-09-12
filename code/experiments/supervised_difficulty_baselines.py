"""Competitive post-hoc adaptive conformal baselines with disjoint data roles.

V-SIM is partitioned by group into

* 25% difficulty-model training
* 25% conformal calibration
* 50% held-out in-distribution evaluation

All interval methods use the same conformal calibration subset. The supervised
local-residual difficulty estimate is trained only on the first partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from bootstrap_ci import extract_groups
from canonical_analysis import (
    ID_DATASET,
    SHIFTED_DATASETS,
    exact_conformal_radius,
    fit_intervals,
    predict_intervals,
    mean_interval_score,
)

MODELS = ("dnsmos", "nisqa", "utmos")
NEIGHBORS = (10, 25, 50)
SEED = 20270819


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_scores(cache_dir: Path, model: str, dataset: str) -> dict[str, np.ndarray]:
    loaded = np.load(cache_dir / "scores" / f"scores_{model}_{dataset}.npz", allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def load_features(project_root: Path, model: str, dataset: str) -> np.ndarray:
    loaded = np.load(
        project_root / "results" / "features" / f"features_{model}_{dataset}.npz",
        allow_pickle=True,
    )
    return loaded["features"].astype(np.float64)


def three_way_split(
    filepaths: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = extract_groups(filepaths)
    unique = np.unique(groups)
    shuffled = np.random.default_rng(seed).permutation(unique)
    n = len(shuffled)
    difficulty_groups = set(shuffled[:n // 4])
    conformal_groups = set(shuffled[n // 4:n // 2])
    difficulty = np.array([group in difficulty_groups for group in groups])
    conformal = np.array([group in conformal_groups for group in groups])
    evaluation = ~(difficulty | conformal)
    return difficulty, conformal, evaluation


def local_residual_difficulty(
    train_features: np.ndarray,
    train_residuals: np.ndarray,
    query_features: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    query_scaled = scaler.transform(query_features)
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    neighbor_model.fit(train_scaled)
    indices = neighbor_model.kneighbors(query_scaled, return_distance=False)
    return train_residuals[indices].mean(axis=1)


def normalized_intervals(
    cal_target: np.ndarray,
    cal_pred: np.ndarray,
    cal_difficulty: np.ndarray,
    test_pred: np.ndarray,
    test_difficulty: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    floor = max(float(np.quantile(cal_difficulty, 0.05)), 1e-3)
    cal_scale = np.maximum(cal_difficulty, floor)
    test_scale = np.maximum(test_difficulty, floor)
    normalized_scores = np.abs(cal_target - cal_pred) / cal_scale
    radius = exact_conformal_radius(normalized_scores, alpha)
    return test_pred - radius * test_scale, test_pred + radius * test_scale


def metrics(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
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
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    cache_dir = project_root / "results" / "canonical"
    interval_rows = []
    ranking_rows = []

    for model in MODELS:
        id_scores = load_scores(cache_dir, model, ID_DATASET)
        id_features = load_features(project_root, model, ID_DATASET)
        difficulty_mask, conformal_mask, evaluation_mask = three_way_split(id_scores["filepaths"])
        train_residuals = np.abs(
            id_scores["mos"][difficulty_mask] - id_scores["mos_pred"][difficulty_mask]
        )
        datasets = {
            ID_DATASET: (id_scores, id_features, evaluation_mask),
            **{
                dataset: (
                    load_scores(cache_dir, model, dataset),
                    load_features(project_root, model, dataset),
                    None,
                )
                for dataset in SHIFTED_DATASETS
            },
        }

        base_cal_args = (
            id_scores["mos_pred"][conformal_mask],
            id_scores["mos"][conformal_mask],
        )
        fixed = fit_intervals(
            *base_cal_args,
            id_scores["mahalanobis"][conformal_mask], args.alpha, None,
        )
        unsupervised_difficulties = {
            "mahalanobis_mondrian": id_scores["mahalanobis"],
            "predicted_mos_mondrian": id_scores["mos_pred"],
            "mos_extremity_mondrian": np.abs(id_scores["mos_pred"] - 3.0),
        }
        fitted_unsupervised = {
            name: fit_intervals(
                *base_cal_args, values[conformal_mask], args.alpha, args.n_bins
            )
            for name, values in unsupervised_difficulties.items()
        }

        for neighbors in NEIGHBORS:
            local_cal = local_residual_difficulty(
                id_features[difficulty_mask], train_residuals,
                id_features[conformal_mask], neighbors,
            )
            local_fitted = fit_intervals(
                *base_cal_args, local_cal, args.alpha, args.n_bins
            )
            for dataset, (data, features, mask) in datasets.items():
                if mask is None:
                    mask = np.ones(len(data["mos"]), dtype=bool)
                target = data["mos"][mask]
                pred = data["mos_pred"][mask]
                if dataset == ID_DATASET:
                    local_test = local_residual_difficulty(
                        id_features[difficulty_mask], train_residuals,
                        features[mask], neighbors,
                    )
                else:
                    local_test = local_residual_difficulty(
                        id_features[difficulty_mask], train_residuals,
                        features[mask], neighbors,
                    )

                methods: dict[str, tuple[np.ndarray, np.ndarray]] = {
                    f"local_residual_mondrian_k{neighbors}": predict_intervals(
                        pred, local_test, local_fitted
                    ),
                    f"local_residual_normalized_k{neighbors}": normalized_intervals(
                        id_scores["mos"][conformal_mask],
                        id_scores["mos_pred"][conformal_mask],
                        local_cal, pred, local_test, args.alpha
                    ),
                }
                if neighbors == NEIGHBORS[0]:
                    methods["fixed"] = predict_intervals(
                        pred, data["mahalanobis"][mask], fixed
                    )
                    test_difficulties = {
                        "mahalanobis_mondrian": data["mahalanobis"][mask],
                        "predicted_mos_mondrian": pred,
                        "mos_extremity_mondrian": np.abs(pred - 3.0),
                    }
                    methods.update({
                        name: predict_intervals(pred, test_difficulties[name], fitted)
                        for name, fitted in fitted_unsupervised.items()
                    })
                for method, (lower, upper) in methods.items():
                    interval_rows.append({
                        "model": model, "dataset": dataset, "method": method,
                        "difficulty_neighbors": neighbors if "local_residual" in method else np.nan,
                        "difficulty_train_n": int(difficulty_mask.sum()),
                        "conformal_calibration_n": int(conformal_mask.sum()),
                        "evaluation_n": int(mask.sum()),
                        **metrics(target, lower, upper, args.alpha),
                    })

                abs_error = np.abs(target - pred)
                threshold = np.quantile(abs_error, 0.75)
                labels = (abs_error >= threshold).astype(int)
                ranking_rows.append({
                    "model": model, "dataset": dataset,
                    "difficulty": f"local_residual_k{neighbors}",
                    "auroc_top_error_quartile": float(roc_auc_score(labels, local_test)),
                    "srcc_with_abs_error": float(pd.Series(local_test).corr(pd.Series(abs_error), method="spearman")),
                })

    interval_path = cache_dir / "supervised_difficulty_intervals.csv"
    ranking_path = cache_dir / "supervised_difficulty_ranking.csv"
    pd.DataFrame(interval_rows).to_csv(interval_path, index=False)
    pd.DataFrame(ranking_rows).to_csv(ranking_path, index=False)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_outputs = [interval_path.name, ranking_path.name]
    manifest["supervised_difficulty"] = {
        "seed": SEED,
        "difficulty_training_fraction": 0.25,
        "conformal_calibration_fraction": 0.25,
        "id_evaluation_fraction": 0.50,
        "neighbors": list(NEIGHBORS),
    }
    manifest["outputs"] = list(dict.fromkeys([*manifest["outputs"], *new_outputs]))
    manifest.setdefault("code_sha256", {})["supervised_difficulty_baselines.py"] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {name: sha256(cache_dir / name) for name in manifest["outputs"]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[supervised_difficulty] Complete")


if __name__ == "__main__":
    main()
