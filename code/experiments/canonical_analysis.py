"""Canonical, leakage-aware analysis for the ICASSP MOS reliability study.

This script creates one synchronized result bundle from saved model features.
It does not modify manuscript files. The design uses

* NISQA TRAIN_SIM only for feature-reference statistics and the k-NN bank
* a deterministic 50/50 group split of NISQA VAL_SIM for conformal calibration
  and held-out in-distribution evaluation
* TRAIN_LIVE, VAL_LIVE, TEST_FOR, TEST_LIVETALK, and TEST_P501 as shifted evaluations

Outputs in ``results/canonical`` include cached scores, split membership,
coverage and bin-count sensitivity, within-domain and pooled error detection,
risk-coverage curves, and a provenance manifest with file hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score, roc_curve

from bootstrap_ci import extract_groups
from evaluate import load_features, compute_coverage_width
from fit_mahalanobis import (
    load_model as load_mahal_model,
    mahalanobis_scores,
    l2_scores,
    knn_scores,
)

SEED = 20270817
REFERENCE_DATASET = "nisqa_train_sim"
ID_DATASET = "nisqa_val_sim"
SHIFTED_DATASETS = (
    "nisqa_train_live",
    "nisqa_val_live",
    "nisqa_test_for",
    "nisqa_test_livetalk",
    "nisqa_test_p501",
)
MODELS = ("dnsmos", "nisqa", "utmos")
K_VALUES = (2, 3, 5, 10)
ERROR_THRESHOLDS = (0.5, 0.75, 1.0)
RISK_COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def exact_conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.size == 0:
        raise ValueError("Cannot calibrate an empty residual array")
    rank = min(int(np.ceil((residuals.size + 1) * (1 - alpha))), residuals.size)
    return float(np.partition(residuals, rank - 1)[rank - 1])


def deterministic_group_split(
    filepaths: np.ndarray, calibration_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = extract_groups(filepaths)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    n_cal = int(round(calibration_fraction * len(shuffled)))
    calibration_groups = set(shuffled[:n_cal])
    cal_mask = np.array([g in calibration_groups for g in groups], dtype=bool)
    return cal_mask, ~cal_mask, groups


def fit_intervals(
    calibration_pred: np.ndarray,
    calibration_mos: np.ndarray,
    calibration_score: np.ndarray,
    alpha: float,
    n_bins: int | None,
) -> dict:
    residuals = np.abs(calibration_mos - calibration_pred)
    global_radius = exact_conformal_radius(residuals, alpha)
    if n_bins is None:
        return {"global_radius": global_radius, "edges": None, "radii": None}

    edges = np.quantile(calibration_score, np.linspace(0, 1, n_bins + 1))
    radii = np.empty(n_bins, dtype=np.float64)
    counts = np.empty(n_bins, dtype=int)
    bin_index = np.digitize(calibration_score, edges[1:-1])
    for bin_id in range(n_bins):
        mask = bin_index == bin_id
        counts[bin_id] = int(mask.sum())
        radii[bin_id] = exact_conformal_radius(residuals[mask], alpha)
    return {
        "global_radius": global_radius,
        "edges": edges,
        "radii": radii,
        "counts": counts,
    }


def predict_intervals(pred: np.ndarray, score: np.ndarray, fitted: dict) -> tuple[np.ndarray, np.ndarray]:
    if fitted["edges"] is None:
        half_width = np.full(len(pred), fitted["global_radius"], dtype=np.float64)
    else:
        bin_index = np.digitize(score, fitted["edges"][1:-1])
        half_width = fitted["radii"][bin_index]
    return pred - half_width, pred + half_width


def mean_interval_score(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Mean interval score. Lower is better at the requested coverage level."""
    width = upper - lower
    below = target < lower
    above = target > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - target[below])
    score[above] += (2.0 / alpha) * (target[above] - upper[above])
    return float(np.mean(score))


def safe_auroc(labels: np.ndarray, score: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, score, sample_weight=sample_weight))


def fpr_at_tpr(labels: np.ndarray, score: np.ndarray, target_tpr: float = 0.95) -> float:
    """Smallest empirical FPR at or above the requested TPR."""
    fpr, tpr, _ = roc_curve(labels, score)
    eligible = np.flatnonzero(tpr >= target_tpr)
    return float(fpr[eligible[0]]) if eligible.size else float(fpr[-1])


def balanced_weights(domains: np.ndarray) -> np.ndarray:
    domains = np.asarray(domains)
    counts = pd.Series(domains).value_counts().to_dict()
    weights = np.array([1.0 / counts[d] for d in domains], dtype=np.float64)
    return weights / weights.mean()


def make_score_cache(
    features_dir: Path, mahal_dir: Path, out_dir: Path, model: str,
    subjective_metadata: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, dict[str, np.ndarray]]:
    cache_dir = out_dir / "scores"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mu, precision = load_mahal_model(mahal_dir / f"mahal_{model}.npz")
    reference = load_features(features_dir, model, REFERENCE_DATASET)
    if reference is None:
        raise FileNotFoundError(f"Missing reference features for {model}")
    reference_features = reference["features"]
    datasets = (REFERENCE_DATASET, ID_DATASET, *SHIFTED_DATASETS)
    cached: dict[str, dict[str, np.ndarray]] = {}
    for dataset in datasets:
        output_path = cache_dir / f"scores_{model}_{dataset}.npz"
        data = load_features(features_dir, model, dataset)
        if data is None:
            raise FileNotFoundError(f"Missing features for {model}/{dataset}")
        source_path = features_dir / f"features_{model}_{dataset}.npz"
        cache_is_fresh = output_path.exists() and output_path.stat().st_mtime >= max(
            source_path.stat().st_mtime,
            (mahal_dir / f"mahal_{model}.npz").stat().st_mtime,
        )
        if cache_is_fresh:
            loaded = np.load(output_path, allow_pickle=True)
            payload = {key: loaded[key] for key in loaded.files}
        else:
            payload = {
                "mahalanobis": mahalanobis_scores(data["features"], mu, precision),
                "l2": l2_scores(data["features"], mu),
                "knn": knn_scores(data["features"], reference_features, k=1),
                "mos": data["mos"].astype(np.float64),
                "mos_pred": data["mos_pred"].astype(np.float64),
                "filepaths": data["filepaths"],
            }
        if "mos_std" not in payload or "votes" not in payload:
            metadata = [
                subjective_metadata[(dataset, Path(str(path)).name)]
                for path in data["filepaths"]
            ]
            payload["mos_std"] = np.asarray([item[0] for item in metadata], dtype=np.float64)
            payload["votes"] = np.asarray([item[1] for item in metadata], dtype=np.float64)
            cache_is_fresh = False
        if not cache_is_fresh:
            payload["mos"] = data["mos"].astype(np.float64)
            payload["mos_pred"] = data["mos_pred"].astype(np.float64)
            payload["filepaths"] = data["filepaths"]
        np.savez_compressed(output_path, **payload)
        cached[dataset] = payload
    return cached


def interval_analysis(
    cached: dict[str, dict[str, np.ndarray]],
    model: str,
    cal_mask: np.ndarray,
    eval_mask: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    calibration = cached[ID_DATASET]
    rows = []
    for score_name in ("mahalanobis", "l2", "knn"):
        fixed = fit_intervals(
            calibration["mos_pred"][cal_mask], calibration["mos"][cal_mask],
            calibration[score_name][cal_mask], alpha, None,
        )
        variants = [("fixed", None, fixed)]
        for n_bins in K_VALUES:
            fitted = fit_intervals(
                calibration["mos_pred"][cal_mask], calibration["mos"][cal_mask],
                calibration[score_name][cal_mask], alpha, n_bins,
            )
            variants.append(("adaptive", n_bins, fitted))

        for method, n_bins, fitted in variants:
            for dataset in (ID_DATASET, *SHIFTED_DATASETS):
                data = cached[dataset]
                mask = eval_mask if dataset == ID_DATASET else np.ones(len(data["mos"]), dtype=bool)
                lo, hi = predict_intervals(data["mos_pred"][mask], data[score_name][mask], fitted)
                metrics = compute_coverage_width(data["mos"][mask], lo, hi)
                rows.append({
                    "model": model,
                    "score": score_name,
                    "method": method,
                    "n_bins": 1 if n_bins is None else n_bins,
                    "dataset": dataset,
                    "n": int(mask.sum()),
                    "coverage": metrics["coverage"],
                    "mean_width": metrics["mean_width"],
                    "mean_interval_score": mean_interval_score(data["mos"][mask], lo, hi, alpha),
                    "coverage_error": abs(metrics["coverage"] - (1 - alpha)),
                    "min_cal_bin_n": int(cal_mask.sum()) if n_bins is None else int(fitted["counts"].min()),
                    "max_cal_bin_n": int(cal_mask.sum()) if n_bins is None else int(fitted["counts"].max()),
                })
    return pd.DataFrame(rows)


def assemble_evaluation_pool(
    cached: dict[str, dict[str, np.ndarray]], eval_mask: np.ndarray
) -> pd.DataFrame:
    frames = []
    for dataset in (ID_DATASET, *SHIFTED_DATASETS):
        data = cached[dataset]
        mask = eval_mask if dataset == ID_DATASET else np.ones(len(data["mos"]), dtype=bool)
        frame = pd.DataFrame({
            "dataset": dataset,
            "mos": data["mos"][mask],
            "mos_pred": data["mos_pred"][mask],
            "mahalanobis": data["mahalanobis"][mask],
            "l2": data["l2"][mask],
            "knn": data["knn"][mask],
            "filepath": data["filepaths"][mask].astype(str),
            "mos_std": data["mos_std"][mask],
            "votes": data["votes"][mask],
        })
        frame["mos_extremity"] = np.abs(frame["mos_pred"] - 3.0)
        frame["abs_error"] = np.abs(frame["mos"] - frame["mos_pred"])
        frame["mos_se"] = frame["mos_std"] / np.sqrt(frame["votes"])
        frame["subjective_ci_tcrit"] = student_t.ppf(0.975, frame["votes"] - 1)
        frame["standardized_abs_error"] = frame["abs_error"] / frame["mos_se"].clip(lower=1e-12)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def error_detection_analysis(pool: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    score_names = ("mahalanobis", "l2", "knn", "mos_extremity")

    tasks: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
    for dataset, frame in pool.groupby("dataset", sort=False):
        threshold = float(frame["abs_error"].quantile(0.75))
        tasks.append(("within_domain_top_quartile", dataset,
                      (frame["abs_error"].to_numpy() >= threshold).astype(int), None))

    global_threshold = float(pool["abs_error"].quantile(0.75))
    tasks.append(("pooled_top_quartile", "all",
                  (pool["abs_error"].to_numpy() >= global_threshold).astype(int), None))
    tasks.append(("balanced_pooled_top_quartile", "all",
                  (pool["abs_error"].to_numpy() >= global_threshold).astype(int),
                  balanced_weights(pool["dataset"].to_numpy())))
    for threshold in ERROR_THRESHOLDS:
        tasks.append((f"pooled_abs_error_gt_{threshold:g}", "all",
                      (pool["abs_error"].to_numpy() > threshold).astype(int), None))
    standardized_threshold = float(pool["standardized_abs_error"].quantile(0.75))
    tasks.append(("pooled_standardized_error_top_quartile", "all",
                  (pool["standardized_abs_error"].to_numpy() >= standardized_threshold).astype(int), None))
    tasks.append(("pooled_outside_subjective_95ci", "all",
                  (pool["standardized_abs_error"].to_numpy()
                   > pool["subjective_ci_tcrit"].to_numpy()).astype(int), None))

    for task, dataset, labels, weights in tasks:
        frame = pool if dataset == "all" else pool[pool["dataset"] == dataset]
        if dataset != "all":
            labels = labels[:len(frame)]
        for score_name in score_names:
            auc = safe_auroc(labels, frame[score_name].to_numpy(), weights)
            rows.append({
                "model": model,
                "task": task,
                "dataset": dataset,
                "score": score_name,
                "n": len(frame),
                "positive_rate": float(labels.mean()),
                "auroc_prespecified_direction": auc,
                "auroc_direction_free_diagnostic": max(auc, 1 - auc) if np.isfinite(auc) else auc,
                "preferred_direction": "higher_score_higher_error" if auc >= 0.5 else "higher_score_lower_error",
            })
    return pd.DataFrame(rows)


def risk_coverage_analysis(pool: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    errors_sq = pool["abs_error"].to_numpy() ** 2
    for weighting in ("natural", "domain_balanced"):
        weights = np.ones(len(pool)) if weighting == "natural" else balanced_weights(pool["dataset"].to_numpy())
        for score_name in ("mahalanobis", "l2", "knn", "mos_extremity"):
            score = pool[score_name].to_numpy()
            for direction, order in (
                ("retain_low_score", np.argsort(score)),
                ("retain_high_score_diagnostic", np.argsort(-score)),
            ):
                for coverage in RISK_COVERAGES:
                    keep = order[:max(1, int(round(coverage * len(order))))]
                    rmse = float(np.sqrt(np.average(errors_sq[keep], weights=weights[keep])))
                    rows.append({
                        "model": model,
                        "weighting": weighting,
                        "score": score_name,
                        "direction": direction,
                        "coverage": coverage,
                        "n_retained": len(keep),
                        "rmse": rmse,
                    })
    return pd.DataFrame(rows)


def domain_detection_analysis(
    cached: dict[str, dict[str, np.ndarray]], model: str, eval_mask: np.ndarray
) -> pd.DataFrame:
    id_data = cached[ID_DATASET]
    rows = []
    for dataset in SHIFTED_DATASETS:
        shifted = cached[dataset]
        for score_name in ("mahalanobis", "l2", "knn"):
            labels = np.concatenate([np.zeros(eval_mask.sum()), np.ones(len(shifted["mos"]))])
            scores = np.concatenate([id_data[score_name][eval_mask], shifted[score_name]])
            rows.append({
                "model": model,
                "dataset": dataset,
                "score": score_name,
                "n_id": int(eval_mask.sum()),
                "n_shifted": len(shifted["mos"]),
                "auroc": safe_auroc(labels, scores),
                "fpr_at_95_tpr": fpr_at_tpr(labels, scores),
            })
    return pd.DataFrame(rows)


def git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return None


def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    features_dir = project_root / "results" / "features"
    mahal_dir = project_root / "results" / "mahal"
    out_dir = project_root / "results" / "canonical"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = project_root / "data" / "NISQA_Corpus" / "NISQA_corpus_file.csv"
    metadata_frame = pd.read_csv(metadata_path)
    db_to_dataset = {
        "NISQA_TRAIN_SIM": "nisqa_train_sim",
        "NISQA_TRAIN_LIVE": "nisqa_train_live",
        "NISQA_VAL_SIM": "nisqa_val_sim",
        "NISQA_VAL_LIVE": "nisqa_val_live",
        "NISQA_TEST_FOR": "nisqa_test_for",
        "NISQA_TEST_LIVETALK": "nisqa_test_livetalk",
        "NISQA_TEST_P501": "nisqa_test_p501",
    }
    subjective_metadata = {
        (db_to_dataset[row.db], Path(row.filename_deg).name): (float(row.mos_std), float(row.votes))
        for row in metadata_frame.itertuples()
        if row.db in db_to_dataset
    }

    split_reference = load_features(features_dir, "nisqa", ID_DATASET)
    cal_mask, eval_mask, groups = deterministic_group_split(
        split_reference["filepaths"], args.calibration_fraction, args.seed
    )
    pd.DataFrame({
        "filepath": split_reference["filepaths"].astype(str),
        "group": groups,
        "role": np.where(cal_mask, "calibration", "id_evaluation"),
    }).to_csv(out_dir / "val_sim_split.csv", index=False)

    interval_frames = []
    error_frames = []
    risk_frames = []
    domain_frames = []
    input_files: list[Path] = []
    for model in args.models:
        print(f"[canonical] Caching scores for {model}", flush=True)
        cached = make_score_cache(features_dir, mahal_dir, out_dir, model, subjective_metadata)
        interval_frames.append(interval_analysis(cached, model, cal_mask, eval_mask, args.alpha))
        pool = assemble_evaluation_pool(cached, eval_mask)
        error_frames.append(error_detection_analysis(pool, model))
        risk_frames.append(risk_coverage_analysis(pool, model))
        domain_frames.append(domain_detection_analysis(cached, model, eval_mask))
        input_files.extend(features_dir / f"features_{model}_{d}.npz"
                           for d in (REFERENCE_DATASET, ID_DATASET, *SHIFTED_DATASETS))
        input_files.append(mahal_dir / f"mahal_{model}.npz")
    input_files.append(metadata_path)

    pd.concat(interval_frames, ignore_index=True).to_csv(out_dir / "interval_sensitivity.csv", index=False)
    pd.concat(error_frames, ignore_index=True).to_csv(out_dir / "error_detection.csv", index=False)
    pd.concat(risk_frames, ignore_index=True).to_csv(out_dir / "risk_coverage.csv", index=False)
    pd.concat(domain_frames, ignore_index=True).to_csv(out_dir / "domain_detection.csv", index=False)

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "alpha": args.alpha,
        "calibration_fraction": args.calibration_fraction,
        "reference_dataset": REFERENCE_DATASET,
        "calibration_dataset": f"{ID_DATASET}:deterministic_group_half",
        "id_evaluation_dataset": f"{ID_DATASET}:complementary_group_half",
        "shifted_datasets": list(SHIFTED_DATASETS),
        "models": list(args.models),
        "k_values": list(K_VALUES),
        "error_thresholds": list(ERROR_THRESHOLDS),
        "calibration_n": int(cal_mask.sum()),
        "id_evaluation_n": int(eval_mask.sum()),
        "git_commit": git_commit(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "input_sha256": {str(p.relative_to(project_root)): sha256(p) for p in input_files},
        "model_sha256": {
            "dnsmos": sha256(project_root / "models" / "DNSMOS" / "sig_bak_ovr.onnx"),
            "nisqa": sha256(project_root / "models" / "NISQA" / "weights" / "nisqa.tar"),
            "utmos": sha256(project_root / "models" / "UTMOS" / "epoch=3-step=7459.ckpt"),
        },
        "dnsmos_prediction_mapping": {
            "method": "official polynomial applied to stored clip-mean raw output",
            "limitation": "Exact for single-segment clips. For multi-segment clips, the official implementation maps each segment before averaging.",
            "21_clip_crosscheck_mean_absolute_difference_mos": 0.0037805069,
            "21_clip_crosscheck_95th_percentile_absolute_difference_mos": 0.0168044894,
        },
        "code_sha256": {
            "extract_features.py": sha256(project_root / "code" / "experiments" / "extract_features.py"),
            "canonical_analysis.py": sha256(Path(__file__).resolve()),
        },
        "outputs": [
            "val_sim_split.csv", "interval_sensitivity.csv", "error_detection.csv",
            "risk_coverage.csv", "domain_detection.csv",
        ],
    }
    manifest["output_sha256"] = {
        name: sha256(out_dir / name) for name in manifest["outputs"]
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[canonical] Complete: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--calibration_fraction", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
