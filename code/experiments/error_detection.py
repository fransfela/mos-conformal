"""
error_detection.py — Direct test of whether OOD scores flag unreliable predictions.

Addresses a conflation in the original ablation: AUROC there measures whether
a score can tell which *dataset/domain* a clip came from, not whether it can
tell which *individual predictions* are wrong. This script tests the latter
directly, within each dataset separately (pooling across datasets would let a
domain-vs-typical-error confound masquerade as instance-level reliability).

For each (model, dataset) pair:
  1. Label the top-quartile absolute-error samples as "unreliable" (positive).
  2. Compute AUROC for detecting that label using: Mahalanobis, L2, KNN(k=1),
     and a naive "MOS extremity" baseline (|q_hat - 3.0|, no reference stats
     needed at all) that a hostile reviewer would expect us to compare against.
  3. Bootstrap a CI for each AUROC (grouped by content id, as in bootstrap_ci.py).
  4. Compute a risk-coverage curve: sort by each score descending, abstain on
     the highest-scoring x%, report RMSE on the retained samples, compared
     against a random-abstention reference (mean RMSE over random subsets of
     the same retained size).

# PAPER: direct error-detection / risk-coverage evidence

Usage:
  python error_detection.py \
    --features_dir ../../results/features \
    --mahal_dir    ../../results/mahal \
    --out_dir      ../../results \
    --n_boot 2000
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fit_mahalanobis import load_model as load_mahal_model, mahalanobis_scores, l2_scores, knn_scores
from evaluate import load_features, compute_auroc, _REFERENCE_DATASET, _OOD_DATASETS
from bootstrap_ci import extract_groups, build_group_index, bootstrap_group_indices, percentile_ci, RNG_SEED

ERROR_QUANTILE = 0.75  # top quartile of |error| = "unreliable"
COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
N_RANDOM_REFS = 200  # random-abstention reference draws per coverage level


def high_error_auroc(
    score: np.ndarray, abs_err: np.ndarray, groups: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """AUROC for detecting top-quartile-error samples using `score`."""
    threshold = np.quantile(abs_err, ERROR_QUANTILE)
    label = (abs_err >= threshold).astype(int)
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan"), float("nan"), float("nan")
    point = compute_auroc(score[label == 0], score[label == 1])

    unique_groups, group_map = build_group_index(groups)
    boots = []
    for _ in range(n_boot):
        idx = bootstrap_group_indices(unique_groups, group_map, rng)
        lbl = label[idx]
        if lbl.sum() == 0 or lbl.sum() == len(lbl):
            continue
        boots.append(compute_auroc(score[idx][lbl == 0], score[idx][lbl == 1]))
    lo, hi = percentile_ci(boots)
    return point, lo, hi


def risk_coverage_curve(
    score: np.ndarray, abs_err: np.ndarray, rng: np.random.Generator,
) -> list[dict]:
    """RMSE on the retained (lowest-score) fraction of samples vs. a
    random-abstention reference of the same retained size."""
    n = len(score)
    order = np.argsort(score)  # ascending: retain the lowest-score samples first
    rows = []
    for cov in COVERAGE_LEVELS:
        k = max(1, int(round(cov * n)))
        retained_idx = order[:k]
        rmse_score = float(np.sqrt(np.mean(abs_err[retained_idx] ** 2)))

        random_rmses = []
        for _ in range(N_RANDOM_REFS):
            ridx = rng.choice(n, size=k, replace=False)
            random_rmses.append(np.sqrt(np.mean(abs_err[ridx] ** 2)))
        rmse_random_mean = float(np.mean(random_rmses))

        rows.append({
            "coverage": cov, "n_retained": k,
            "rmse_abstain_high_score": rmse_score,
            "rmse_random_abstention": rmse_random_mean,
        })
    return rows


def run(features_dir: Path, mahal_dir: Path, out_dir: Path, models: list[str], n_boot: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows_auroc = []
    rows_risk = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")
        mahal_path = mahal_dir / f"mahal_{model}.npz"
        if not mahal_path.exists():
            print(f"  [SKIP] No Mahalanobis model: {mahal_path}")
            continue
        mu, sigma_inv = load_mahal_model(mahal_path)

        train_data = load_features(features_dir, model, _REFERENCE_DATASET)
        train_feats = train_data["features"]

        for dataset in sorted(_OOD_DATASETS):
            data = load_features(features_dir, model, dataset)
            if data is None:
                continue
            feats = data["features"]
            mos_true = data["mos"].astype(np.float64)
            mos_pred = data["mos_pred"].astype(np.float64)
            abs_err = np.abs(mos_true - mos_pred)
            groups = extract_groups(data["filepaths"])

            scores = {
                "mahalanobis": mahalanobis_scores(feats, mu, sigma_inv),
                "l2": l2_scores(feats, mu),
                "knn": knn_scores(feats, train_feats, k=1),
                "mos_extremity": np.abs(mos_pred - 3.0),
            }

            print(f"\n  Dataset: {dataset} (N={len(mos_true)}, top-quartile |err| >= "
                  f"{np.quantile(abs_err, ERROR_QUANTILE):.3f})")
            for score_name, score in scores.items():
                point, lo, hi = high_error_auroc(score, abs_err, groups, n_boot, rng)
                print(f"    {score_name:15s} AUROC(high-error)={point:.3f} [{lo:.3f},{hi:.3f}]")
                rows_auroc.append({
                    "model": model, "dataset": dataset, "score": score_name,
                    "auroc_high_error": point, "ci_lo": lo, "ci_hi": hi, "n_boot": n_boot,
                })

            for score_name, score in scores.items():
                for r in risk_coverage_curve(score, abs_err, rng):
                    r.update({"model": model, "dataset": dataset, "score": score_name})
                    rows_risk.append(r)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df_auroc = pd.DataFrame(rows_auroc)
    auroc_path = out_dir / f"error_detection_auroc_{ts}.csv"
    df_auroc.to_csv(auroc_path, index=False)
    print(f"\n[error_detection] Saved → {auroc_path}")

    df_risk = pd.DataFrame(rows_risk)
    risk_path = out_dir / f"risk_coverage_{ts}.csv"
    df_risk.to_csv(risk_path, index=False)
    print(f"[error_detection] Saved → {risk_path}")

    # Summary: mean AUROC per score type across all model/dataset combos
    print("\n--- Mean AUROC(high-error) across all models x datasets ---")
    print(df_auroc.groupby("score")["auroc_high_error"].mean().round(3).to_string())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct error-detection evidence (not domain AUROC).")
    p.add_argument("--features_dir", default="../../results/features")
    p.add_argument("--mahal_dir", default="../../results/mahal")
    p.add_argument("--out_dir", default="../../results")
    p.add_argument("--models", nargs="+", default=["nisqa", "dnsmos", "utmos"])
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run(Path(args.features_dir), Path(args.mahal_dir), out_dir, args.models, args.n_boot)


if __name__ == "__main__":
    main()
