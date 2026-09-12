"""Follow-up exploratory (repo-only) probes building on exploratory_ensemble_ideas.py.

Does not modify canonical_analysis.py, any file under paper/, or any tracked
results/canonical/ output. Writes new files under results/exploratory/ only.

Idea C -- does simple averaging of the three frozen point estimates reduce
  RMSE relative to the best single model? (classic ensembling, not an
  uncertainty signal -- a free accuracy check.)
Idea D -- gradient-boosted combiner: same features and same disjoint
  difficulty/evaluation split as Idea B's logistic combiner, but a
  HistGradientBoostingClassifier instead of logistic regression, to see
  whether a non-linear combiner raises the ceiling found in Idea B.
Idea E -- ensemble-disagreement-stratified conformal interval: repeats the
  paper's Table 3 fixed-vs-adaptive comparison (Mondrian binning, K=5,
  alpha=0.10) but stratifies bins by ensemble_loo_disagreement instead of
  Mahalanobis distance, using the identical calibration/id-evaluation split
  (results/canonical/val_sim_split.csv) as the canonical pipeline.

Usage:
  python exploratory_advanced_ideas.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from bootstrap_ci import extract_groups
from error_detection import high_error_auroc, ERROR_QUANTILE
from canonical_analysis import (
    ID_DATASET, SHIFTED_DATASETS, MODELS,
    fit_intervals, predict_intervals, mean_interval_score,
)
from evaluate import compute_coverage_width
from supervised_difficulty_baselines import three_way_split
from exploratory_ensemble_ideas import CACHE_DIR, OUT_DIR, load_scores, align_by_filepath, FEATURE_NAMES

SEED = 20270903
ALL_DATASETS = (ID_DATASET, *SHIFTED_DATASETS)
ALPHA = 0.10
N_BINS = 5


def load_val_sim_roles() -> dict[str, str]:
    split = pd.read_csv(CACHE_DIR / "val_sim_split.csv")
    return dict(zip(split["filepath"].astype(str), split["role"]))


def build_full_pool(dataset: str, roles: dict[str, str]) -> dict[str, dict[str, np.ndarray]]:
    """Like exploratory_ensemble_ideas.build_pool but keeps ALL V-SIM rows
    (tagged with their calibration/id_evaluation role) instead of filtering."""
    per_model = align_by_filepath({model: load_scores(model, dataset) for model in MODELS})
    preds = np.stack([per_model[m]["mos_pred"].astype(np.float64) for m in MODELS], axis=1)
    for i, model in enumerate(MODELS):
        others_mean = np.delete(preds, i, axis=1).mean(axis=1)
        per_model[model]["ensemble_loo_disagreement"] = np.abs(preds[:, i] - others_mean)
        per_model[model]["ensemble_mean_pred"] = preds.mean(axis=1)
        per_model[model]["mos_extremity"] = np.abs(per_model[model]["mos_pred"].astype(np.float64) - 3.0)
    if dataset == ID_DATASET:
        fp = np.array([str(x) for x in per_model[MODELS[0]]["filepaths"]])
        role = np.array([roles.get(f, "unknown") for f in fp])
        for model in MODELS:
            per_model[model]["role"] = role
    else:
        n = len(per_model[MODELS[0]]["mos"])
        for model in MODELS:
            per_model[model]["role"] = np.full(n, "shifted", dtype=object)
    return per_model


# ---------------------------------------------------------------------------
# Idea C -- ensemble-mean point accuracy
# ---------------------------------------------------------------------------

def run_idea_c(pools: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for dataset in ALL_DATASETS:
        mos = pools[dataset][MODELS[0]]["mos"].astype(np.float64)
        candidates = {m: pools[dataset][m]["mos_pred"].astype(np.float64) for m in MODELS}
        candidates["ensemble_mean"] = pools[dataset][MODELS[0]]["ensemble_mean_pred"]
        for name, pred in candidates.items():
            rmse = float(np.sqrt(np.mean((mos - pred) ** 2)))
            srcc = float(pd.Series(pred).corr(pd.Series(mos), method="spearman"))
            rows.append({"dataset": dataset, "predictor": name, "rmse": rmse, "srcc": srcc})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Idea D -- gradient-boosted combiner
# ---------------------------------------------------------------------------

def run_idea_d(pools: dict[str, dict[str, dict]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        id_data = pools[ID_DATASET][model]
        difficulty_mask, _, evaluation_mask = three_way_split(id_data["filepaths"])
        abs_err_id = np.abs(id_data["mos"].astype(np.float64) - id_data["mos_pred"].astype(np.float64))
        threshold = np.quantile(abs_err_id[difficulty_mask], ERROR_QUANTILE)

        x_train = np.stack([id_data[f].astype(np.float64)[difficulty_mask] for f in FEATURE_NAMES], axis=1)
        y_train = (abs_err_id[difficulty_mask] >= threshold).astype(int)
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=SEED).fit(x_train, y_train)

        for dataset in ALL_DATASETS:
            data = pools[dataset][model]
            mask = evaluation_mask if dataset == ID_DATASET else np.ones(len(data["mos"]), dtype=bool)
            abs_err = np.abs(data["mos"].astype(np.float64) - data["mos_pred"].astype(np.float64))[mask]
            groups = extract_groups(data["filepaths"])[mask]
            x_eval = np.stack([data[f].astype(np.float64)[mask] for f in FEATURE_NAMES], axis=1)
            combined_score = clf.predict_proba(x_eval)[:, 1]
            point, lo, hi = high_error_auroc(combined_score, abs_err, groups, n_boot, rng)
            rows.append({"model": model, "dataset": dataset, "score": "gbm_combiner",
                         "auroc": point, "ci_lo": lo, "ci_hi": hi, "n": len(abs_err),
                         "n_train": int(difficulty_mask.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Idea E -- ensemble-disagreement-stratified conformal interval
# ---------------------------------------------------------------------------

def run_idea_e(pools: dict[str, dict[str, dict]]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        id_data = pools[ID_DATASET][model]
        cal_mask = id_data["role"] == "calibration"
        eval_mask = id_data["role"] == "id_evaluation"

        for score_name in ("mahalanobis", "ensemble_loo_disagreement"):
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

    df_c = run_idea_c(pools)
    df_c.to_csv(OUT_DIR / "idea_c_ensemble_mean_point_accuracy.csv", index=False)
    print("=== Idea C: RMSE, single models vs. simple ensemble mean ===")
    print(df_c.pivot(index="dataset", columns="predictor", values="rmse").round(3))
    print()
    print("=== Idea C: SRCC, single models vs. simple ensemble mean ===")
    print(df_c.pivot(index="dataset", columns="predictor", values="srcc").round(3))
    print()

    df_d = run_idea_d(pools, n_boot=2000, rng=rng)
    df_d.to_csv(OUT_DIR / "idea_d_gbm_combiner_error_detection.csv", index=False)
    print("=== Idea D: mean GBM-combiner AUROC by model (out-of-domain generalization) ===")
    print(df_d.groupby("model")["auroc"].mean().round(3))
    print()
    print(df_d.round(3).to_string(index=False))
    print()

    df_e = run_idea_e(pools)
    df_e.to_csv(OUT_DIR / "idea_e_disagreement_stratified_intervals.csv", index=False)
    print("=== Idea E: mean interval score by model x score x method (lower is better) ===")
    print(df_e.groupby(["model", "score", "method"])["mean_interval_score"].mean().round(3))
    print()
    print("=== Idea E: full table ===")
    print(df_e.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
