"""
evaluate.py — Evaluation pipeline: compute all metrics reported in the paper.

Metrics computed per (model, dataset) pair:
  Point estimate:  PCC, SRCC  (vs. true MOS)
  OOD detection:   AUROC (ID eval = V-SIM held out; OOD = V-LIV/T-FOR/T-TLK/T-P501)
  Coverage:        empirical marginal coverage at nominal 90%
  Interval width:  mean interval width (fixed and adaptive wrappers)

Outputs (saved to results/):
  # PAPER: Table 2 — PCC / SRCC
  eval_pcc_srcc_<timestamp>.csv

  # PAPER: Table 3 — Coverage and width
  eval_coverage_width_<timestamp>.csv

  # PAPER: Fig auroc — AUROC for OOD detection
  eval_auroc_<timestamp>.csv

  # Combined summary (all metrics)
  eval_summary_<timestamp>.json

Usage:
  python evaluate.py \\
    --features_dir ../../results/features \\
    --mahal_dir    ../../results/mahal \\
    --wrapper_dir  ../../results/conformal \\
    --out_dir      ../../results \\
    --models nisqa dnsmos utmos \\
    --datasets nisqa_train_sim nisqa_train_live nisqa_val_sim nisqa_val_live nisqa_test_for nisqa_test_livetalk nisqa_test_p501
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from fit_mahalanobis import load_model as load_mahal_model, mahalanobis_scores, l2_scores, knn_scores
from conformal_wrapper import ConformalWrapper


# ---------------------------------------------------------------------------
# OOD label assignment
#
# _REFERENCE_DATASET fits the Mahalanobis/L2/KNN reference statistics only.
# _ID_EVAL_DATASET is the held-out in-distribution evaluation set: it is never
# used to fit the reference statistics, so scoring it cannot leak (this fixes
# the k=1 KNN self-match bug where TRAIN_SIM was queried against itself, and
# removes outcome-dependent ID/OOD labeling). TRAIN_LIVE is used only for
# conformal calibration and is excluded from the AUROC ID/OOD pool entirely.
# OOD roles are predeclared and never adjusted based on observed AUROC.
# ---------------------------------------------------------------------------

_REFERENCE_DATASET = "nisqa_train_sim"
_ID_EVAL_DATASET = "nisqa_val_sim"
_OOD_DATASETS = {"nisqa_val_live", "nisqa_test_for",
                 "nisqa_test_livetalk", "nisqa_test_p501"}
_SHIFT_ORDER = [
    "nisqa_train_sim", "nisqa_train_live",
    "nisqa_val_sim", "nisqa_val_live",
    "nisqa_test_for", "nisqa_test_livetalk", "nisqa_test_p501",
]
_CAL_DATASET = "nisqa_train_live"   # calibration set for conformal wrapper

# Baseline names for interval methods
_BASELINE_FIXED_SIGMA = "fixed_sigma"
_BASELINE_CONFORMAL_FIXED = "conformal_fixed"
_BASELINE_CONFORMAL_ADAPT = "conformal_adaptive"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_features(features_dir: Path, model: str, dataset: str) -> dict | None:
    """Load features_<model>_<dataset>.npz; return None if not found."""
    path = features_dir / f"features_{model}_{dataset}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],
        "mos": data["mos"],
        "mos_pred": data["mos_pred"],
        "filepaths": data["filepaths"],
        "dataset": str(data["dataset"][0]),
    }


def load_wrapper(wrapper_dir: Path, model: str, mode: str) -> ConformalWrapper | None:
    """Load calibrated conformal wrapper. mode: 'fixed' or 'adaptive'."""
    path = wrapper_dir / f"wrapper_{model}_{mode}.pkl"
    if not path.exists():
        return None
    return ConformalWrapper.load(path)


# ---------------------------------------------------------------------------
# Naive fixed-±sigma baseline
# ---------------------------------------------------------------------------

def fixed_sigma_intervals(
    mos_pred_cal: np.ndarray,
    mos_true_cal: np.ndarray,
    mos_pred_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Naive interval: q_hat ± std(calibration residuals)."""
    sigma = float(np.std(mos_true_cal - mos_pred_cal, ddof=1))
    lo = (mos_pred_test - sigma).astype(np.float32)
    hi = (mos_pred_test + sigma).astype(np.float32)
    return lo, hi


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_pcc_srcc(mos_pred: np.ndarray, mos_true: np.ndarray) -> dict[str, float]:
    if len(mos_pred) < 2:
        return {"pcc": float("nan"), "srcc": float("nan"), "rmse": float("nan")}
    pcc, _ = pearsonr(mos_pred, mos_true)
    srcc, _ = spearmanr(mos_pred, mos_true)
    rmse = float(np.sqrt(np.mean((mos_pred - mos_true) ** 2)))
    return {"pcc": float(pcc), "srcc": float(srcc), "rmse": rmse}


def compute_auroc(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
) -> float:
    """
    AUROC for OOD detection.
    Label: 0 = in-distribution, 1 = OOD.
    Higher Mahalanobis score -> more likely OOD.
    """
    if len(scores_id) == 0 or len(scores_ood) == 0:
        return float("nan")
    y_true = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    y_score = np.concatenate([scores_id, scores_ood])
    return float(roc_auc_score(y_true, y_score))


def compute_fpr_at_tpr(
    scores_id: np.ndarray,
    scores_ood: np.ndarray,
    tpr_target: float = 0.95,
) -> float:
    """
    FPR at tpr_target TPR for OOD detection (Hendrycks 2017 standard).
    Lower is better.
    """
    if len(scores_id) == 0 or len(scores_ood) == 0:
        return float("nan")
    y_true = np.concatenate([np.zeros(len(scores_id)), np.ones(len(scores_ood))])
    y_score = np.concatenate([scores_id, scores_ood])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Interpolate FPR at the target TPR
    idx = np.searchsorted(tpr, tpr_target, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def compute_ood_srcc(
    ood_scores: np.ndarray,
    abs_errors: np.ndarray,
) -> float:
    """
    Spearman correlation between OOD score and absolute prediction error.
    Quantifies how well d_M(x) tracks when the predictor is wrong.
    """
    if len(ood_scores) < 2:
        return float("nan")
    srcc, _ = spearmanr(ood_scores, abs_errors)
    return float(srcc)


def compute_coverage_width(
    mos_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> dict[str, float]:
    covered = ((mos_true >= lo) & (mos_true <= hi)).mean()
    width = (hi - lo).mean()
    return {"coverage": float(covered), "mean_width": float(width)}


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    features_dir: Path,
    mahal_dir: Path,
    wrapper_dir: Path,
    out_dir: Path,
    models: list[str],
    datasets: list[str],
) -> dict:
    """Run all evaluations and return a nested results dict."""

    results = {}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Accumulate rows for paper tables
    rows_pcc_srcc = []
    rows_coverage = []
    rows_auroc = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # Load Mahalanobis model
        mahal_path = mahal_dir / f"mahal_{model}.npz"
        if not mahal_path.exists():
            print(f"  [SKIP] Mahalanobis model not found: {mahal_path}")
            continue
        mu, sigma_inv = load_mahal_model(mahal_path)

        # Load calibration data (NISQA_TRAIN_LIVE)
        cal_data = load_features(features_dir, model, _CAL_DATASET)
        if cal_data is None:
            print(f"  [SKIP] Calibration features not found for {model}/{_CAL_DATASET}")
            continue

        cal_mahal = mahalanobis_scores(cal_data["features"], mu, sigma_inv)

        # Load wrappers
        wrapper_fixed = load_wrapper(wrapper_dir, model, "fixed")
        wrapper_adapt = load_wrapper(wrapper_dir, model, "adaptive")

        # In-distribution AUROC pool = held-out V-SIM only (never used to fit
        # mu/sigma_inv or the KNN reference bank, so scoring it cannot leak).
        train_sim_data = load_features(features_dir, model, _REFERENCE_DATASET)
        _train_feats_for_knn = train_sim_data["features"] if train_sim_data is not None else None

        id_eval_data = load_features(features_dir, model, _ID_EVAL_DATASET)
        if id_eval_data is not None:
            mahal_id_all = mahalanobis_scores(id_eval_data["features"], mu, sigma_inv)
            l2_id_all = l2_scores(id_eval_data["features"], mu)
            knn_id_all = (
                knn_scores(id_eval_data["features"], _train_feats_for_knn, k=1)
                if _train_feats_for_knn is not None else np.array([])
            )
        else:
            mahal_id_all = np.array([])
            l2_id_all = np.array([])
            knn_id_all = np.array([])

        results[model] = {}

        for dataset in datasets:
            data = load_features(features_dir, model, dataset)
            if data is None:
                print(f"  [SKIP] Features not found: {model}/{dataset}")
                continue

            mos_true = data["mos"].astype(np.float32)
            mos_pred = data["mos_pred"].astype(np.float32)
            features = data["features"]
            mahal = mahalanobis_scores(features, mu, sigma_inv)
            l2 = l2_scores(features, mu)

            print(f"\n  Dataset: {dataset}  (N={len(mos_true)})")

            # --- Point estimate metrics
            pt = compute_pcc_srcc(mos_pred, mos_true)
            print(f"    PCC={pt['pcc']:.3f}  SRCC={pt['srcc']:.3f}  RMSE={pt['rmse']:.3f}")
            rows_pcc_srcc.append({
                "model": model, "dataset": dataset,
                "pcc": pt["pcc"], "srcc": pt["srcc"], "rmse": pt["rmse"],
                "n": len(mos_true),
            })

            # --- OOD detection AUROC, FPR@95TPR, and SRCC: Mahalanobis, L2 and KNN
            if dataset in _OOD_DATASETS and len(mahal_id_all) > 0:
                abs_err = np.abs(mos_true - mos_pred)
                auroc_mah = compute_auroc(mahal_id_all, mahal)
                auroc_l2  = compute_auroc(l2_id_all, l2)
                knn_ood   = knn_scores(features, _train_feats_for_knn, k=1) if _train_feats_for_knn is not None else np.array([])
                auroc_knn = compute_auroc(knn_id_all, knn_ood) if len(knn_id_all) > 0 else float("nan")
                fpr_mah   = compute_fpr_at_tpr(mahal_id_all, mahal)
                fpr_l2    = compute_fpr_at_tpr(l2_id_all, l2)
                fpr_knn   = compute_fpr_at_tpr(knn_id_all, knn_ood) if len(knn_id_all) > 0 else float("nan")
                srcc_mah  = compute_ood_srcc(mahal, abs_err)
                srcc_l2   = compute_ood_srcc(l2, abs_err)
                srcc_knn  = compute_ood_srcc(knn_ood, abs_err) if len(knn_ood) > 0 else float("nan")
                print(f"    AUROC  Mahal={auroc_mah:.3f}  L2={auroc_l2:.3f}  KNN={auroc_knn:.3f}")
                print(f"    FPR95  Mahal={fpr_mah:.3f}  L2={fpr_l2:.3f}  KNN={fpr_knn:.3f}")
                print(f"    SRCC   Mahal={srcc_mah:.3f}  L2={srcc_l2:.3f}  KNN={srcc_knn:.3f}")
                for score_type, auroc_val, fpr_val, srcc_val in [
                    ("mahalanobis", auroc_mah, fpr_mah, srcc_mah),
                    ("l2",          auroc_l2,  fpr_l2,  srcc_l2),
                    ("knn",         auroc_knn, fpr_knn, srcc_knn),
                ]:
                    rows_auroc.append({
                        "model": model, "dataset": dataset,
                        "score_type": score_type, "auroc": auroc_val,
                        "fpr95": fpr_val, "srcc_ood": srcc_val,
                        "n_id": len(mahal_id_all), "n_ood": len(mahal),
                    })

            # --- Coverage and width
            # Baseline: fixed ±sigma
            if cal_data is not None:
                lo_sig, hi_sig = fixed_sigma_intervals(
                    cal_data["mos_pred"], cal_data["mos"], mos_pred
                )
                cov_sig = compute_coverage_width(mos_true, lo_sig, hi_sig)
                rows_coverage.append({
                    "model": model, "dataset": dataset, "method": _BASELINE_FIXED_SIGMA,
                    **cov_sig,
                })
                print(f"    ±sigma  cov={cov_sig['coverage']:.3f}  width={cov_sig['mean_width']:.3f}")

            # Conformal fixed
            if wrapper_fixed is not None:
                lo_cf, hi_cf = wrapper_fixed.predict(mos_pred)
                cov_cf = compute_coverage_width(mos_true, lo_cf, hi_cf)
                rows_coverage.append({
                    "model": model, "dataset": dataset, "method": _BASELINE_CONFORMAL_FIXED,
                    **cov_cf,
                })
                print(f"    Conf.fixed  cov={cov_cf['coverage']:.3f}  width={cov_cf['mean_width']:.3f}")

            # Conformal adaptive
            if wrapper_adapt is not None:
                lo_ca, hi_ca = wrapper_adapt.predict(mos_pred, mahal)
                cov_ca = compute_coverage_width(mos_true, lo_ca, hi_ca)
                rows_coverage.append({
                    "model": model, "dataset": dataset, "method": _BASELINE_CONFORMAL_ADAPT,
                    **cov_ca,
                })
                print(f"    Conf.adapt  cov={cov_ca['coverage']:.3f}  width={cov_ca['mean_width']:.3f}")

            results[model][dataset] = {
                "pcc": pt["pcc"], "srcc": pt["srcc"],
                "n": int(len(mos_true)),
                "mahal_mean": float(mahal.mean()),
                "mahal_std": float(mahal.std()),
            }

    # --- Save tables

    # PAPER: Table 2
    df_pcc = pd.DataFrame(rows_pcc_srcc)
    pcc_path = out_dir / f"eval_pcc_srcc_{timestamp}.csv"
    df_pcc.to_csv(pcc_path, index=False)
    print(f"\n[evaluate] Saved PCC/SRCC table → {pcc_path}")

    # PAPER: Table 3
    df_cov = pd.DataFrame(rows_coverage)
    cov_path = out_dir / f"eval_coverage_width_{timestamp}.csv"
    df_cov.to_csv(cov_path, index=False)
    print(f"[evaluate] Saved coverage/width table → {cov_path}")

    # PAPER: Fig auroc
    df_auroc = pd.DataFrame(rows_auroc)
    auroc_path = out_dir / f"eval_auroc_{timestamp}.csv"
    df_auroc.to_csv(auroc_path, index=False)
    print(f"[evaluate] Saved AUROC table → {auroc_path}")

    # Summary JSON
    summary_path = out_dir / f"eval_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[evaluate] Saved summary → {summary_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate OOD detection and coverage metrics.")
    p.add_argument("--features_dir", default="../../results/features")
    p.add_argument("--mahal_dir", default="../../results/mahal")
    p.add_argument("--wrapper_dir", default="../../results/conformal")
    p.add_argument("--out_dir", default="../../results")
    p.add_argument("--models", nargs="+", default=["nisqa"])
    p.add_argument("--datasets", nargs="+",
                   default=["nisqa_train_sim", "nisqa_train_live", "nisqa_val_sim",
                            "nisqa_val_live", "nisqa_test_for",
                            "nisqa_test_livetalk", "nisqa_test_p501"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_evaluation(
        features_dir=Path(args.features_dir),
        mahal_dir=Path(args.mahal_dir),
        wrapper_dir=Path(args.wrapper_dir),
        out_dir=out_dir,
        models=args.models,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
