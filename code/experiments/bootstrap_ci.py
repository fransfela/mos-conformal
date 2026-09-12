"""
bootstrap_ci.py — Bootstrap confidence intervals for all headline metrics.

Addresses reviewer concern: no variability estimate was reported for AUROC,
FPR@95, SRCC, RMSE, or the fixed-vs-adaptive coverage delta. Clips within the
NISQA Corpus share source content across talkers (e.g. TEST_P501 groups four
talker recordings under one condition id, "cNNNNN", parsed from the filename).
Resampling individual clips would treat these as independent when they are
not, so the bootstrap resamples whole content-id groups rather than rows.

Uses the corrected ID/OOD split from evaluate.py:
  Reference/fit set   : nisqa_train_sim
  In-distribution eval: nisqa_val_sim   (held out, never used to fit anything)
  OOD eval             : nisqa_val_live, nisqa_test_for,
                         nisqa_test_livetalk, nisqa_test_p501

Outputs:
  results/bootstrap_ci_<timestamp>.csv
    Columns: metric, model, dataset, point_estimate, ci_lo, ci_hi, n_boot, n_groups

Usage:
  python bootstrap_ci.py \
    --features_dir ../../results/features \
    --mahal_dir    ../../results/mahal \
    --wrapper_dir  ../../results/conformal \
    --out_dir      ../../results \
    --n_boot 2000
"""

from __future__ import annotations

import argparse
import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fit_mahalanobis import load_model as load_mahal_model, mahalanobis_scores, l2_scores, knn_scores
from conformal_wrapper import ConformalWrapper
from evaluate import (
    load_features,
    load_wrapper,
    compute_auroc,
    compute_fpr_at_tpr,
    compute_pcc_srcc,
    compute_ood_srcc,
    compute_coverage_width,
    _REFERENCE_DATASET,
    _ID_EVAL_DATASET,
    _OOD_DATASETS,
    _CAL_DATASET,
    _SHIFT_ORDER,
)

_GROUP_PATTERNS = (
    re.compile(r"(c\d+)", re.IGNORECASE),
    re.compile(r"([^\\/]+?)_live_phone_", re.IGNORECASE),
)
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Group-aware bootstrap helper
# ---------------------------------------------------------------------------

def extract_groups(filepaths: np.ndarray) -> np.ndarray:
    """Parse the NISQA Corpus content/condition id ('cNNNNN') from each filepath.
    Falls back to a unique per-row id (equivalent to i.i.d. resampling) if no
    match is found, e.g. for corpora without this naming convention."""
    groups = []
    for i, fp in enumerate(filepaths):
        match = next((p.search(str(fp)) for p in _GROUP_PATTERNS if p.search(str(fp))), None)
        groups.append(match.group(1).lower() if match else f"__row{i}")
    return np.array(groups)


def build_group_index(groups: np.ndarray) -> tuple[np.ndarray, dict]:
    """Precompute unique groups and their member row indices once, so repeated
    bootstrap resampling only does array lookups instead of rebuilding the
    group -> rows mapping (which involves an np.where scan) on every call."""
    unique_groups = np.unique(groups)
    group_to_rows = {g: np.where(groups == g)[0] for g in unique_groups}
    return unique_groups, group_to_rows


def bootstrap_group_indices(
    unique_groups: np.ndarray,
    group_to_rows: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample unique groups with replacement and return row indices."""
    sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
    idx = np.concatenate([group_to_rows[g] for g in sampled_groups])
    return idx


def percentile_ci(values: list[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


# ---------------------------------------------------------------------------
# Per-metric bootstrap routines
# ---------------------------------------------------------------------------

def bootstrap_auroc_fpr_srcc(
    id_feats: np.ndarray, id_groups: np.ndarray,
    ood_feats: np.ndarray, ood_groups: np.ndarray, ood_mos: np.ndarray, ood_pred: np.ndarray,
    mu: np.ndarray, sigma_inv: np.ndarray, train_feats: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict[str, dict[str, tuple]]:
    """Bootstrap AUROC/FPR95/SRCC_ood for mahalanobis, l2, and knn scores.

    Per-sample scores depend only on the fixed reference statistics (mu,
    sigma_inv, train_feats), not on which rows are resampled, so they are
    computed once and reused across all bootstrap iterations by indexing
    (recomputing d_M from raw features per resample would be O(n_boot * N * d^2),
    prohibitively slow for UTMOS where d=1024).
    """
    point = {}
    boots = {f"{s}_{m}": [] for s in ("mahalanobis", "l2", "knn") for m in ("auroc", "fpr95", "srcc_ood")}

    def _scores(feats):
        return {
            "mahalanobis": mahalanobis_scores(feats, mu, sigma_inv),
            "l2": l2_scores(feats, mu),
            "knn": knn_scores(feats, train_feats, k=1),
        }

    id_scores_full = _scores(id_feats)
    ood_scores_full = _scores(ood_feats)
    abs_err_full = np.abs(ood_mos - ood_pred)
    for score_type in ("mahalanobis", "l2", "knn"):
        point[f"{score_type}_auroc"] = compute_auroc(id_scores_full[score_type], ood_scores_full[score_type])
        point[f"{score_type}_fpr95"] = compute_fpr_at_tpr(id_scores_full[score_type], ood_scores_full[score_type])
        point[f"{score_type}_srcc_ood"] = compute_ood_srcc(ood_scores_full[score_type], abs_err_full)

    id_unique, id_map = build_group_index(id_groups)
    ood_unique, ood_map = build_group_index(ood_groups)
    for _ in range(n_boot):
        id_idx = bootstrap_group_indices(id_unique, id_map, rng)
        ood_idx = bootstrap_group_indices(ood_unique, ood_map, rng)
        abs_err = np.abs(ood_mos[ood_idx] - ood_pred[ood_idx])
        for score_type in ("mahalanobis", "l2", "knn"):
            id_s = id_scores_full[score_type][id_idx]
            ood_s = ood_scores_full[score_type][ood_idx]
            boots[f"{score_type}_auroc"].append(compute_auroc(id_s, ood_s))
            boots[f"{score_type}_fpr95"].append(compute_fpr_at_tpr(id_s, ood_s))
            boots[f"{score_type}_srcc_ood"].append(compute_ood_srcc(ood_s, abs_err))

    return point, boots


def bootstrap_point_estimate(
    mos_pred: np.ndarray, mos_true: np.ndarray, groups: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> tuple[dict, dict]:
    point = compute_pcc_srcc(mos_pred, mos_true)
    boots = {"srcc": [], "rmse": [], "pcc": []}
    unique_groups, group_map = build_group_index(groups)
    for _ in range(n_boot):
        idx = bootstrap_group_indices(unique_groups, group_map, rng)
        m = compute_pcc_srcc(mos_pred[idx], mos_true[idx])
        boots["srcc"].append(m["srcc"])
        boots["rmse"].append(m["rmse"])
        boots["pcc"].append(m["pcc"])
    return point, boots


def bootstrap_coverage_delta(
    mos_true: np.ndarray, mos_pred: np.ndarray, mahal: np.ndarray, groups: np.ndarray,
    wrapper_fixed: ConformalWrapper, wrapper_adapt: ConformalWrapper,
    n_boot: int, rng: np.random.Generator,
) -> tuple[dict, dict]:
    lo_f, hi_f = wrapper_fixed.predict(mos_pred)
    cov_f = compute_coverage_width(mos_true, lo_f, hi_f)
    lo_a, hi_a = wrapper_adapt.predict(mos_pred, mahal)
    cov_a = compute_coverage_width(mos_true, lo_a, hi_a)
    point = {
        "coverage_fixed": cov_f["coverage"], "coverage_adaptive": cov_a["coverage"],
        "coverage_delta": cov_a["coverage"] - cov_f["coverage"],
    }
    boots = {"coverage_fixed": [], "coverage_adaptive": [], "coverage_delta": []}
    unique_groups, group_map = build_group_index(groups)
    for _ in range(n_boot):
        idx = bootstrap_group_indices(unique_groups, group_map, rng)
        lo_f, hi_f = wrapper_fixed.predict(mos_pred[idx])
        cf = compute_coverage_width(mos_true[idx], lo_f, hi_f)["coverage"]
        lo_a, hi_a = wrapper_adapt.predict(mos_pred[idx], mahal[idx])
        ca = compute_coverage_width(mos_true[idx], lo_a, hi_a)["coverage"]
        boots["coverage_fixed"].append(cf)
        boots["coverage_adaptive"].append(ca)
        boots["coverage_delta"].append(ca - cf)
    return point, boots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_bootstrap(
    features_dir: Path, mahal_dir: Path, wrapper_dir: Path,
    models: list[str], n_boot: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")

        mahal_path = mahal_dir / f"mahal_{model}.npz"
        if not mahal_path.exists():
            print(f"  [SKIP] No Mahalanobis model: {mahal_path}")
            continue
        mu, sigma_inv = load_mahal_model(mahal_path)

        train_data = load_features(features_dir, model, _REFERENCE_DATASET)
        train_feats = train_data["features"]

        id_data = load_features(features_dir, model, _ID_EVAL_DATASET)
        if id_data is None:
            print(f"  [SKIP] No ID eval features: {_ID_EVAL_DATASET}")
            continue
        id_feats = id_data["features"]
        id_groups = extract_groups(id_data["filepaths"])
        print(f"  ID eval ({_ID_EVAL_DATASET}): N={len(id_feats)}, groups={len(np.unique(id_groups))}")

        wrapper_fixed = load_wrapper(wrapper_dir, model, "fixed")
        wrapper_adapt = load_wrapper(wrapper_dir, model, "adaptive")

        # --- Point-estimate metrics (SRCC/RMSE) for all 7 declared subsets
        for dataset in _SHIFT_ORDER:
            data = load_features(features_dir, model, dataset)
            if data is None:
                continue
            groups = extract_groups(data["filepaths"])
            mos_true = data["mos"].astype(np.float64)
            mos_pred = data["mos_pred"].astype(np.float64)
            point, boots = bootstrap_point_estimate(mos_pred, mos_true, groups, n_boot, rng)
            for metric in ("srcc", "rmse", "pcc"):
                lo, hi = percentile_ci(boots[metric])
                rows.append({"metric": metric, "model": model, "dataset": dataset,
                             "point_estimate": point[metric], "ci_lo": lo, "ci_hi": hi,
                             "n_boot": n_boot, "n_groups": len(np.unique(groups))})
            print(f"    {dataset:22s} SRCC={point['srcc']:.3f} "
                  f"[{percentile_ci(boots['srcc'])[0]:.3f},{percentile_ci(boots['srcc'])[1]:.3f}]  "
                  f"RMSE={point['rmse']:.3f} [{percentile_ci(boots['rmse'])[0]:.3f},{percentile_ci(boots['rmse'])[1]:.3f}]")

            # --- Coverage delta (fixed vs adaptive), all 7 subsets
            if wrapper_fixed is not None and wrapper_adapt is not None:
                mahal = mahalanobis_scores(data["features"], mu, sigma_inv)
                point_c, boots_c = bootstrap_coverage_delta(
                    mos_true, mos_pred, mahal, groups, wrapper_fixed, wrapper_adapt, n_boot, rng
                )
                for metric in ("coverage_fixed", "coverage_adaptive", "coverage_delta"):
                    lo, hi = percentile_ci(boots_c[metric])
                    rows.append({"metric": metric, "model": model, "dataset": dataset,
                                 "point_estimate": point_c[metric], "ci_lo": lo, "ci_hi": hi,
                                 "n_boot": n_boot, "n_groups": len(np.unique(groups))})
                print(f"      coverage fixed={point_c['coverage_fixed']:.3f} adapt={point_c['coverage_adaptive']:.3f} "
                      f"delta={point_c['coverage_delta']:+.3f} "
                      f"[{percentile_ci(boots_c['coverage_delta'])[0]:+.3f},{percentile_ci(boots_c['coverage_delta'])[1]:+.3f}]")

        # --- AUROC / FPR95 / SRCC_ood for the 4 predeclared OOD subsets
        for dataset in sorted(_OOD_DATASETS):
            data = load_features(features_dir, model, dataset)
            if data is None:
                continue
            ood_groups = extract_groups(data["filepaths"])
            mos_true = data["mos"].astype(np.float64)
            mos_pred = data["mos_pred"].astype(np.float64)
            point, boots = bootstrap_auroc_fpr_srcc(
                id_feats, id_groups, data["features"], ood_groups, mos_true, mos_pred,
                mu, sigma_inv, train_feats, n_boot, rng,
            )
            for score_type in ("mahalanobis", "l2", "knn"):
                for metric_suffix in ("auroc", "fpr95", "srcc_ood"):
                    key = f"{score_type}_{metric_suffix}"
                    lo, hi = percentile_ci(boots[key])
                    rows.append({"metric": key, "model": model, "dataset": dataset,
                                 "point_estimate": point[key], "ci_lo": lo, "ci_hi": hi,
                                 "n_boot": n_boot, "n_groups": len(np.unique(ood_groups))})
            print(f"    {dataset:22s} "
                  f"Mahal AUROC={point['mahalanobis_auroc']:.3f} "
                  f"[{percentile_ci(boots['mahalanobis_auroc'])[0]:.3f},{percentile_ci(boots['mahalanobis_auroc'])[1]:.3f}]  "
                  f"KNN AUROC={point['knn_auroc']:.3f} "
                  f"[{percentile_ci(boots['knn_auroc'])[0]:.3f},{percentile_ci(boots['knn_auroc'])[1]:.3f}]  "
                  f"Mahal SRCC_ood={point['mahalanobis_srcc_ood']:+.3f} "
                  f"[{percentile_ci(boots['mahalanobis_srcc_ood'])[0]:+.3f},{percentile_ci(boots['mahalanobis_srcc_ood'])[1]:+.3f}]")

    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap CIs for all headline metrics.")
    p.add_argument("--features_dir", default="../../results/features")
    p.add_argument("--mahal_dir", default="../../results/mahal")
    p.add_argument("--wrapper_dir", default="../../results/conformal")
    p.add_argument("--out_dir", default="../../results")
    p.add_argument("--models", nargs="+", default=["nisqa", "dnsmos", "utmos"])
    p.add_argument("--n_boot", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_bootstrap(
        features_dir=Path(args.features_dir),
        mahal_dir=Path(args.mahal_dir),
        wrapper_dir=Path(args.wrapper_dir),
        models=args.models,
        n_boot=args.n_boot,
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"bootstrap_ci_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[bootstrap_ci] Saved → {csv_path}")


if __name__ == "__main__":
    main()
