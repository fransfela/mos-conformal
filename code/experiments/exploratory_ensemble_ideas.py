"""Exploratory (repo-only) test of two candidate error-detection signals NOT in the paper.

Does not modify canonical_analysis.py, any file under paper/, or any tracked
results/canonical/ output. Reads only from the existing canonical score cache
(results/canonical/scores/*.npz, results/canonical/val_sim_split.csv) and
writes new files under results/exploratory/. This is a decide-later probe, not
a pipeline change.

Idea A -- leave-one-out ensemble disagreement (unsupervised, zero-cost):
  For each frozen predictor, |q_hat_model - mean(q_hat_other_two)| across the
  three already-scored models (DNSMOS, NISQA, UTMOS) on the same clip. No
  fitting required. Evaluated with the exact same within-domain top-quartile
  |error| AUROC + group-bootstrap-CI protocol as error_detection.py, on the
  same V-SIM held-out evaluation half used for the paper's canonical numbers.

Idea B -- supervised logistic combiner:
  Logistic regression over [mahalanobis, l2, knn, mos_extremity,
  ensemble_loo_disagreement], trained ONLY on the 25% "difficulty" group
  partition of V-SIM defined by supervised_difficulty_baselines.three_way_split
  (disjoint from the 50% "evaluation" partition used for in-domain testing),
  then evaluated out-of-domain on every shifted subset. Mirrors the existing
  repo convention for supervised difficulty baselines (train/test disjoint by
  content-group id, never fit on shifted labels).

Canonical paper reference (Session 12/13, chance level for all four):
  mean error-detection AUROC -- mahalanobis 0.478, l2 0.433, knn 0.499,
  mos_extremity 0.421 (see PAPER_STATUS.md / results/canonical/error_detection.csv,
  column auroc_prespecified_direction, task=within_domain_top_quartile).

Usage:
  python exploratory_ensemble_ideas.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bootstrap_ci import extract_groups
from error_detection import high_error_auroc, ERROR_QUANTILE
from canonical_analysis import ID_DATASET, SHIFTED_DATASETS, MODELS
from supervised_difficulty_baselines import three_way_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "results" / "canonical"
OUT_DIR = PROJECT_ROOT / "results" / "exploratory"
SEED = 20270902
ALL_DATASETS = (ID_DATASET, *SHIFTED_DATASETS)
FEATURE_NAMES = ["mahalanobis", "l2", "knn", "mos_extremity", "ensemble_loo_disagreement"]


def load_scores(model: str, dataset: str) -> dict[str, np.ndarray]:
    loaded = np.load(CACHE_DIR / "scores" / f"scores_{model}_{dataset}.npz", allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def load_id_eval_filepaths() -> set[str]:
    """Filepaths in the canonical V-SIM held-out in-domain evaluation half
    (role == 'id_evaluation' in val_sim_split.csv), i.e. never used to fit the
    Mahalanobis/kNN reference or calibrate any conformal interval."""
    split = pd.read_csv(CACHE_DIR / "val_sim_split.csv")
    return set(split.loc[split["role"] == "id_evaluation", "filepath"].astype(str))


def align_by_filepath(per_model: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """Sort each model's rows by filepath so per-clip rows line up across models."""
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for model, data in per_model.items():
        fp = np.array([str(x) for x in data["filepaths"]])
        order = np.argsort(fp)
        aligned[model] = {key: np.asarray(value)[order] for key, value in data.items()}
    ref_fp = np.array([str(x) for x in aligned[MODELS[0]]["filepaths"]])
    for model in MODELS[1:]:
        fp = np.array([str(x) for x in aligned[model]["filepaths"]])
        assert np.array_equal(ref_fp, fp), f"filepath mismatch for {model} vs {MODELS[0]}"
    return aligned


def build_pool(dataset: str, id_eval_filepaths: set[str]) -> dict[str, dict[str, np.ndarray]]:
    per_model = align_by_filepath({model: load_scores(model, dataset) for model in MODELS})
    if dataset == ID_DATASET:
        keep = np.array([str(fp) in id_eval_filepaths for fp in per_model[MODELS[0]]["filepaths"]])
        per_model = {model: {key: value[keep] for key, value in data.items()} for model, data in per_model.items()}

    preds = np.stack([per_model[m]["mos_pred"].astype(np.float64) for m in MODELS], axis=1)  # (N, 3)
    for i, model in enumerate(MODELS):
        others_mean = np.delete(preds, i, axis=1).mean(axis=1)
        per_model[model]["ensemble_loo_disagreement"] = np.abs(preds[:, i] - others_mean)
        per_model[model]["ensemble_std"] = preds.std(axis=1)
        per_model[model]["mos_extremity"] = np.abs(per_model[model]["mos_pred"].astype(np.float64) - 3.0)
    return per_model


def run_idea_a(pools: dict[str, dict[str, dict]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    score_names = ("mahalanobis", "l2", "knn", "mos_extremity", "ensemble_loo_disagreement", "ensemble_std")
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


def run_idea_b(pools: dict[str, dict[str, dict]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        id_data = pools[ID_DATASET][model]
        difficulty_mask, _, evaluation_mask = three_way_split(id_data["filepaths"])
        abs_err_id = np.abs(id_data["mos"].astype(np.float64) - id_data["mos_pred"].astype(np.float64))
        threshold = np.quantile(abs_err_id[difficulty_mask], ERROR_QUANTILE)

        x_train = np.stack([id_data[f].astype(np.float64)[difficulty_mask] for f in FEATURE_NAMES], axis=1)
        y_train = (abs_err_id[difficulty_mask] >= threshold).astype(int)
        scaler = StandardScaler().fit(x_train)
        clf = LogisticRegression(max_iter=1000).fit(scaler.transform(x_train), y_train)

        for dataset in ALL_DATASETS:
            data = pools[dataset][model]
            mask = evaluation_mask if dataset == ID_DATASET else np.ones(len(data["mos"]), dtype=bool)
            abs_err = np.abs(data["mos"].astype(np.float64) - data["mos_pred"].astype(np.float64))[mask]
            groups = extract_groups(data["filepaths"])[mask]
            x_eval = np.stack([data[f].astype(np.float64)[mask] for f in FEATURE_NAMES], axis=1)
            combined_score = clf.decision_function(scaler.transform(x_eval))
            point, lo, hi = high_error_auroc(combined_score, abs_err, groups, n_boot, rng)
            rows.append({"model": model, "dataset": dataset, "score": "logistic_combiner",
                         "auroc": point, "ci_lo": lo, "ci_hi": hi, "n": len(abs_err),
                         "n_train": int(difficulty_mask.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    id_eval_filepaths = load_id_eval_filepaths()
    pools = {dataset: build_pool(dataset, id_eval_filepaths) for dataset in ALL_DATASETS}

    df_a = run_idea_a(pools, n_boot=2000, rng=rng)
    df_a.to_csv(OUT_DIR / "idea_a_ensemble_disagreement_error_detection.csv", index=False)

    df_b = run_idea_b(pools, n_boot=2000, rng=rng)
    df_b.to_csv(OUT_DIR / "idea_b_logistic_combiner_error_detection.csv", index=False)

    print("=== Idea A: mean error-detection AUROC by score (all model x dataset pairs, n={}) ===".format(len(df_a) // 6))
    print(df_a.groupby("score")["auroc"].mean().round(3).sort_values(ascending=False).to_string())
    print()
    print("=== Idea A: ensemble_loo_disagreement, per model x dataset ===")
    print(df_a[df_a.score == "ensemble_loo_disagreement"].round(3).to_string(index=False))
    print()
    print("=== Idea B: mean logistic-combiner AUROC by model (out-of-domain generalization) ===")
    print(df_b.groupby("model")["auroc"].mean().round(3).to_string())
    print()
    print("=== Idea B: full table ===")
    print(df_b.round(3).to_string(index=False))
    print()
    print("Canonical paper baselines for reference (Session 12/13, chance level):")
    print("  mahalanobis=0.478  l2=0.433  knn=0.499  mos_extremity=0.421")


if __name__ == "__main__":
    main()
