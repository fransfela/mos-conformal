"""Precision-weighted ensemble disagreement (repo-only follow-up to Idea A/E).

Does not modify canonical_analysis.py, any file under paper/, or any tracked
results/canonical/ output. Writes new files under results/exploratory/ only.

Motivation (diagnosed from Idea A): unweighted leave-one-out disagreement is
weak for NISQA (in-sample, usually the most accurate model, so disagreeing
with the other two often means NISQA is right) and for T-TLK (UTMOS is badly
wrong there, so its prediction contaminates the "other two" reference mean
for whichever model isn't UTMOS).

Fix tested here: weight each "other model" contribution by inverse-variance
precision (1 / calibration-only RMSE^2) instead of an unweighted mean, so a
model with poor calibration-set accuracy contributes less to the reference
used to judge disagreement. Weights are estimated ONLY from the canonical
calibration role (results/canonical/val_sim_split.csv, role == "calibration"),
never from the evaluation or shifted sets, mirroring how the paper fits its
Mahalanobis reference only on T-SIM.

Idea F1 -- error-detection AUROC: precision-weighted vs. unweighted
  leave-one-out disagreement (same protocol as Idea A / error_detection.py).
Idea F2 -- conformal interval stratification: precision-weighted disagreement
  vs. Mahalanobis and unweighted disagreement (same protocol as Idea E).

Usage:
  python exploratory_precision_weighted_ideas.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bootstrap_ci import extract_groups
from error_detection import high_error_auroc
from canonical_analysis import (
    ID_DATASET, SHIFTED_DATASETS, MODELS,
    fit_intervals, predict_intervals, mean_interval_score,
)
from evaluate import compute_coverage_width
from exploratory_advanced_ideas import load_val_sim_roles, build_full_pool
from exploratory_ensemble_ideas import OUT_DIR

SEED = 20270903
ALL_DATASETS = (ID_DATASET, *SHIFTED_DATASETS)
ALPHA = 0.10
N_BINS = 5


def calibration_rmse(pools: dict[str, dict[str, dict]]) -> dict[str, float]:
    """Each model's RMSE on the canonical calibration role of V-SIM only."""
    id_pool = pools[ID_DATASET]
    rmse = {}
    for model in MODELS:
        data = id_pool[model]
        mask = data["role"] == "calibration"
        mos = data["mos"][mask].astype(np.float64)
        pred = data["mos_pred"][mask].astype(np.float64)
        rmse[model] = float(np.sqrt(np.mean((mos - pred) ** 2)))
    return rmse


def add_precision_weighted_disagreement(
    pools: dict[str, dict[str, dict]], rmse_cal: dict[str, float],
) -> None:
    """In-place: adds 'precision_weighted_disagreement' to every model/dataset pool."""
    precision = {m: 1.0 / (rmse_cal[m] ** 2) for m in MODELS}
    for per_model in pools.values():
        preds = np.stack([per_model[m]["mos_pred"].astype(np.float64) for m in MODELS], axis=1)
        for i, model in enumerate(MODELS):
            other_idx = [j for j in range(len(MODELS)) if j != i]
            weights = np.array([precision[MODELS[j]] for j in other_idx])
            weights = weights / weights.sum()
            weighted_other_mean = preds[:, other_idx] @ weights
            per_model[model]["precision_weighted_disagreement"] = np.abs(preds[:, i] - weighted_other_mean)


def run_idea_f1(pools: dict[str, dict[str, dict]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    score_names = ("mahalanobis", "knn", "ensemble_loo_disagreement", "precision_weighted_disagreement")
    for dataset in ALL_DATASETS:
        for model in MODELS:
            data = pools[dataset][model]
            abs_err = np.abs(data["mos"].astype(np.float64) - data["mos_pred"].astype(np.float64))
            groups = extract_groups(data["filepaths"])
            for score_name in score_names:
                point, lo, hi = high_error_auroc(data[score_name].astype(np.float64), abs_err, groups, n_boot, rng)
                rows.append({"model": model, "dataset": dataset, "score": score_name,
                             "auroc": point, "ci_lo": lo, "ci_hi": hi, "n": len(abs_err)})
    return pd.DataFrame(rows)


def run_idea_f2(pools: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    score_names = ("mahalanobis", "ensemble_loo_disagreement", "precision_weighted_disagreement")
    for model in MODELS:
        id_data = pools[ID_DATASET][model]
        cal_mask = id_data["role"] == "calibration"
        eval_mask = id_data["role"] == "id_evaluation"

        for score_name in score_names:
            fixed = fit_intervals(
                id_data["mos_pred"][cal_mask].astype(np.float64), id_data["mos"][cal_mask].astype(np.float64),
                id_data[score_name][cal_mask].astype(np.float64), ALPHA, None,
            )
            adaptive = fit_intervals(
                id_data["mos_pred"][cal_mask].astype(np.float64), id_data["mos"][cal_mask].astype(np.float64),
                id_data[score_name][cal_mask].astype(np.float64), ALPHA, N_BINS,
            )
            for method, fitted in (("fixed", fixed), ("adaptive", adaptive)):
                for dataset in ALL_DATASETS:
                    data = pools[dataset][model]
                    mask = eval_mask if dataset == ID_DATASET else np.ones(len(data["mos"]), dtype=bool)
                    lo, hi = predict_intervals(
                        data["mos_pred"][mask].astype(np.float64), data[score_name][mask].astype(np.float64), fitted,
                    )
                    metrics = compute_coverage_width(data["mos"][mask].astype(np.float64), lo, hi)
                    rows.append({
                        "model": model, "score": score_name, "method": method, "dataset": dataset,
                        "n": int(mask.sum()), "coverage": metrics["coverage"], "mean_width": metrics["mean_width"],
                        "mean_interval_score": mean_interval_score(data["mos"][mask].astype(np.float64), lo, hi, ALPHA),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    roles = load_val_sim_roles()
    pools = {dataset: build_full_pool(dataset, roles) for dataset in ALL_DATASETS}

    rmse_cal = calibration_rmse(pools)
    print("=== Calibration-only RMSE per model (weight basis) ===")
    print(pd.Series(rmse_cal).round(3))
    print()
    add_precision_weighted_disagreement(pools, rmse_cal)

    df_f1 = run_idea_f1(pools, n_boot=2000, rng=rng)
    df_f1.to_csv(OUT_DIR / "idea_f1_precision_weighted_error_detection.csv", index=False)
    print("=== Idea F1: mean error-detection AUROC by score ===")
    print(df_f1.groupby("score")["auroc"].mean().round(3).sort_values(ascending=False))
    print()
    print("=== Idea F1: unweighted vs. precision-weighted disagreement, per model x dataset ===")
    pivot = df_f1[df_f1.score.isin(["ensemble_loo_disagreement", "precision_weighted_disagreement"])]
    print(pivot.pivot_table(index=["model", "dataset"], columns="score", values="auroc").round(3))
    print()

    df_f2 = run_idea_f2(pools)
    df_f2.to_csv(OUT_DIR / "idea_f2_precision_weighted_intervals.csv", index=False)
    print("=== Idea F2: mean interval score by model x score x method (lower is better) ===")
    print(df_f2.groupby(["model", "score", "method"])["mean_interval_score"].mean().round(3))


if __name__ == "__main__":
    main()
