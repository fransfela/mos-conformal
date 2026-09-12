"""Paired bootstrap for competitive post-hoc difficulty baselines.

The three-way V-SIM split and each fitted local-residual difficulty function are
held fixed. Calibration and evaluation groups are resampled independently in
every replicate. All conformal radii and Mondrian bin boundaries are refitted.
The intervals therefore quantify sampling uncertainty conditional on the
trained difficulty estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_ci import extract_groups, build_group_index, bootstrap_group_indices
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

SEED = 20270820


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def make_group_sampler(filepaths: np.ndarray):
    unique, mapping = build_group_index(extract_groups(filepaths))
    return lambda rng: bootstrap_group_indices(unique, mapping, rng)


def metrics(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict[str, float]:
    return {
        "coverage": float(np.mean((target >= lower) & (target <= upper))),
        "mean_width": float(np.mean(upper - lower)),
        "mean_interval_score": mean_interval_score(target, lower, upper, alpha),
    }


def prepare_model(
    project_root: Path, cache_dir: Path, model: str
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    id_scores = load_scores(cache_dir, model, ID_DATASET)
    id_features = load_features(project_root, model, ID_DATASET)
    difficulty_mask, conformal_mask, evaluation_mask = three_way_split(id_scores["filepaths"])
    train_residuals = np.abs(
        id_scores["mos"][difficulty_mask] - id_scores["mos_pred"][difficulty_mask]
    )

    calibration = {key: value[conformal_mask] for key, value in id_scores.items()}
    calibration["predicted_mos"] = calibration["mos_pred"]
    calibration["mos_extremity"] = np.abs(calibration["mos_pred"] - 3.0)
    for neighbors in NEIGHBORS:
        calibration[f"local_residual_k{neighbors}"] = local_residual_difficulty(
            id_features[difficulty_mask], train_residuals,
            id_features[conformal_mask], neighbors,
        )

    datasets: dict[str, dict[str, np.ndarray]] = {}
    for dataset in (ID_DATASET, *SHIFTED_DATASETS):
        if dataset == ID_DATASET:
            data = {key: value[evaluation_mask] for key, value in id_scores.items()}
            features = id_features[evaluation_mask]
        else:
            data = load_scores(cache_dir, model, dataset)
            features = load_features(project_root, model, dataset)
        data["predicted_mos"] = data["mos_pred"]
        data["mos_extremity"] = np.abs(data["mos_pred"] - 3.0)
        for neighbors in NEIGHBORS:
            data[f"local_residual_k{neighbors}"] = local_residual_difficulty(
                id_features[difficulty_mask], train_residuals, features, neighbors,
            )
        datasets[dataset] = data
    return calibration, datasets


def method_intervals(
    method: str,
    calibration: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    cal_pred = calibration["mos_pred"][cal_idx]
    cal_target = calibration["mos"][cal_idx]
    test_pred = test["mos_pred"][test_idx]
    if method == "fixed":
        fitted = fit_intervals(
            cal_pred, cal_target, calibration["mahalanobis"][cal_idx], alpha, None
        )
        return predict_intervals(test_pred, test["mahalanobis"][test_idx], fitted)

    if method.startswith("local_residual_normalized_"):
        suffix = method.removeprefix("local_residual_normalized_")
        key = f"local_residual_{suffix}"
        return normalized_intervals(
            cal_target, cal_pred, calibration[key][cal_idx],
            test_pred, test[key][test_idx], alpha,
        )

    difficulty_by_method = {
        "mahalanobis_mondrian": "mahalanobis",
        "predicted_mos_mondrian": "predicted_mos",
        "mos_extremity_mondrian": "mos_extremity",
    }
    if method.startswith("local_residual_mondrian_"):
        suffix = method.removeprefix("local_residual_mondrian_")
        key = f"local_residual_{suffix}"
    else:
        key = difficulty_by_method[method]
    fitted = fit_intervals(
        cal_pred, cal_target, calibration[key][cal_idx], alpha, n_bins
    )
    return predict_intervals(test_pred, test[key][test_idx], fitted)


def all_methods() -> tuple[str, ...]:
    return (
        "fixed",
        "mahalanobis_mondrian",
        "predicted_mos_mondrian",
        "mos_extremity_mondrian",
        *(f"local_residual_mondrian_k{k}" for k in NEIGHBORS),
        *(f"local_residual_normalized_k{k}" for k in NEIGHBORS),
    )


def percentile(values: list[float]) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def update_manifest(cache_dir: Path, output_path: Path, n_boot: int, n_bins: int) -> None:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["supervised_difficulty_bootstrap"] = {
        "seed": SEED,
        "n_boot": n_boot,
        "n_bins": n_bins,
        "conditioning": (
            "difficulty estimator fixed; calibration and evaluation groups resampled; "
            "radii and bin boundaries refitted"
        ),
    }
    manifest["outputs"] = list(dict.fromkeys([*manifest["outputs"], output_path.name]))
    manifest.setdefault("code_sha256", {})[
        "bootstrap_supervised_baselines.py"
    ] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {
        name: sha256(cache_dir / name) for name in manifest["outputs"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    cache_dir = project_root / "results" / "canonical"
    rng = np.random.default_rng(SEED)
    methods = all_methods()
    rows: list[dict] = []

    for model in MODELS:
        print(f"[supervised-bootstrap] {model}", flush=True)
        calibration, datasets = prepare_model(project_root, cache_dir, model)
        cal_sampler = make_group_sampler(calibration["filepaths"])
        full_cal_idx = np.arange(len(calibration["mos"]))
        for dataset, test in datasets.items():
            test_sampler = make_group_sampler(test["filepaths"])
            full_test_idx = np.arange(len(test["mos"]))
            point: dict[str, dict[str, float]] = {}
            for method in methods:
                lower, upper = method_intervals(
                    method, calibration, test, full_cal_idx, full_test_idx,
                    args.alpha, args.n_bins,
                )
                point[method] = metrics(test["mos"], lower, upper, args.alpha)

            differences = {
                (method, metric): []
                for method in methods if method != "fixed"
                for metric in ("coverage", "mean_width", "mean_interval_score")
            }
            for _ in range(args.n_boot):
                cal_idx = cal_sampler(rng)
                test_idx = test_sampler(rng)
                replicate: dict[str, dict[str, float]] = {}
                for method in methods:
                    lower, upper = method_intervals(
                        method, calibration, test, cal_idx, test_idx,
                        args.alpha, args.n_bins,
                    )
                    replicate[method] = metrics(
                        test["mos"][test_idx], lower, upper, args.alpha
                    )
                for method in methods:
                    if method == "fixed":
                        continue
                    for metric in ("coverage", "mean_width", "mean_interval_score"):
                        differences[(method, metric)].append(
                            replicate[method][metric] - replicate["fixed"][metric]
                        )

            for method in methods:
                if method == "fixed":
                    continue
                for metric in ("coverage", "mean_width", "mean_interval_score"):
                    lo, hi = percentile(differences[(method, metric)])
                    rows.append({
                        "model": model,
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "fixed": point["fixed"][metric],
                        "method_value": point[method][metric],
                        "method_minus_fixed": point[method][metric] - point["fixed"][metric],
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "n_boot": args.n_boot,
                        "conditioning": "fixed_difficulty_estimator",
                    })

    output_path = cache_dir / "bootstrap_supervised_differences.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    update_manifest(cache_dir, output_path, args.n_boot, args.n_bins)
    print("[supervised-bootstrap] Complete", flush=True)


if __name__ == "__main__":
    main()
