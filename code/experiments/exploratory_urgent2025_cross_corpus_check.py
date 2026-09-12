"""Cross-corpus robustness check: does everything found on NISQA Corpus
(Idea A/C: ensemble-mean point accuracy, ensemble-disagreement error
detection) hold up on a genuinely different corpus (URGENT 2025 blind-test
enhancement submissions)? Also applies the canonical NISQA-Corpus-fitted
Mahalanobis/kNN reference to this new corpus, to see how those OOD scores
behave outside the corpus they were calibrated on.

Repo-only robustness check. Does not modify canonical_analysis.py, the
paper, or any results/canonical/ artifact. Reads results/features_pilot/*
(URGENT2025 scaled sample) and results/mahal/* (canonical NISQA-Corpus-fitted
reference). Writes to results/exploratory/ only.

Usage:
  python exploratory_urgent2025_cross_corpus_check.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_ci import extract_groups
from error_detection import high_error_auroc
from fit_mahalanobis import load_model as load_mahal_model, mahalanobis_scores, l2_scores, knn_scores
from evaluate import load_features as load_canonical_features, _REFERENCE_DATASET
from exploratory_ensemble_ideas import OUT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PILOT_DIR = PROJECT_ROOT / "results" / "features_pilot"
MAHAL_DIR = PROJECT_ROOT / "results" / "mahal"
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
DATASET = "urgent2025_sqa_scaled"
MODELS = ("dnsmos", "nisqa", "utmos")
SEED = 20270903


def load_scaled(model: str) -> dict[str, np.ndarray]:
    loaded = np.load(FEATURES_PILOT_DIR / f"features_{model}_{DATASET}.npz", allow_pickle=True)
    return {key: loaded[key] for key in loaded.files}


def align_by_filepath(per_model: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    aligned: dict[str, dict[str, np.ndarray]] = {}
    for model, data in per_model.items():
        fp = np.array([str(x) for x in data["filepaths"]])
        order = np.argsort(fp)
        n = len(fp)
        aligned[model] = {
            key: (np.asarray(value)[order] if np.asarray(value).shape[:1] == (n,) else value)
            for key, value in data.items()
        }
    ref_fp = np.array([str(x) for x in aligned[MODELS[0]]["filepaths"]])
    for model in MODELS[1:]:
        fp = np.array([str(x) for x in aligned[model]["filepaths"]])
        assert np.array_equal(ref_fp, fp), f"filepath mismatch for {model}"
    return aligned


def point_accuracy(pool: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    mos = pool[MODELS[0]]["mos"].astype(np.float64)
    preds = np.stack([pool[m]["mos_pred"].astype(np.float64) for m in MODELS], axis=1)
    ensemble = preds.mean(axis=1)
    rows = []
    for name, pred in {**{m: pool[m]["mos_pred"].astype(np.float64) for m in MODELS}, "ensemble_mean": ensemble}.items():
        rmse = float(np.sqrt(np.mean((mos - pred) ** 2)))
        srcc = float(pd.Series(pred).corr(pd.Series(mos), method="spearman"))
        rows.append({"predictor": name, "rmse": rmse, "srcc": srcc, "n": len(mos)})
    return pd.DataFrame(rows)


def error_detection(pool: dict[str, dict[str, np.ndarray]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    preds = np.stack([pool[m]["mos_pred"].astype(np.float64) for m in MODELS], axis=1)
    rows = []
    for i, model in enumerate(MODELS):
        others_mean = np.delete(preds, i, axis=1).mean(axis=1)
        disagreement = np.abs(preds[:, i] - others_mean)
        mos_extremity = np.abs(pool[model]["mos_pred"].astype(np.float64) - 3.0)
        abs_err = np.abs(pool[model]["mos"].astype(np.float64) - pool[model]["mos_pred"].astype(np.float64))
        groups = extract_groups(pool[model]["filepaths"])
        for score_name, score in (("ensemble_loo_disagreement", disagreement), ("mos_extremity", mos_extremity)):
            point, lo, hi = high_error_auroc(score, abs_err, groups, n_boot, rng)
            rows.append({"model": model, "score": score_name, "auroc": point, "ci_lo": lo, "ci_hi": hi, "n": len(abs_err)})
    return pd.DataFrame(rows)


def canonical_ood_scores(pool: dict[str, dict[str, np.ndarray]], n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the NISQA-Corpus-fitted Mahalanobis/L2/kNN reference to this new
    corpus: domain separability (vs. held-out NISQA V-SIM) and within-corpus
    error-detection AUROC, exactly mirroring the paper's canonical protocol."""
    rows = []
    for model in MODELS:
        mu, sigma_inv = load_mahal_model(MAHAL_DIR / f"mahal_{model}.npz")
        train_feats = load_canonical_features(FEATURES_DIR, model, _REFERENCE_DATASET)["features"]
        id_feats = load_canonical_features(FEATURES_DIR, model, "nisqa_val_sim")["features"]

        urgent_feats = pool[model]["features"].astype(np.float64)
        abs_err = np.abs(pool[model]["mos"].astype(np.float64) - pool[model]["mos_pred"].astype(np.float64))
        groups = extract_groups(pool[model]["filepaths"])

        scores = {
            "mahalanobis": (mahalanobis_scores(id_feats, mu, sigma_inv), mahalanobis_scores(urgent_feats, mu, sigma_inv)),
            "l2": (l2_scores(id_feats, mu), l2_scores(urgent_feats, mu)),
            "knn": (knn_scores(id_feats, train_feats, k=1), knn_scores(urgent_feats, train_feats, k=1)),
        }
        for score_name, (id_score, urgent_score) in scores.items():
            labels = np.concatenate([np.zeros(len(id_score)), np.ones(len(urgent_score))])
            combined = np.concatenate([id_score, urgent_score])
            from sklearn.metrics import roc_auc_score
            domain_auroc = float(roc_auc_score(labels, combined))
            point, lo, hi = high_error_auroc(urgent_score, abs_err, groups, n_boot, rng)
            rows.append({
                "model": model, "score": score_name,
                "domain_auroc_vs_nisqa_vsim": domain_auroc,
                "error_detection_auroc": point, "ci_lo": lo, "ci_hi": hi,
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pool = align_by_filepath({model: load_scaled(model) for model in MODELS})

    df_point = point_accuracy(pool)
    df_point.to_csv(OUT_DIR / "urgent2025_point_accuracy.csv", index=False)
    print("=== URGENT2025 (n=364): point accuracy, single models vs. ensemble mean ===")
    print(df_point.round(3).to_string(index=False))
    print()

    df_err = error_detection(pool, n_boot=2000, rng=rng)
    df_err.to_csv(OUT_DIR / "urgent2025_error_detection.csv", index=False)
    print("=== URGENT2025: error-detection AUROC (ensemble disagreement vs. NISQA-Corpus result of 0.626) ===")
    print(df_err.round(3).to_string(index=False))
    print()

    df_ood = canonical_ood_scores(pool, n_boot=2000, rng=rng)
    df_ood.to_csv(OUT_DIR / "urgent2025_canonical_ood_scores.csv", index=False)
    print("=== URGENT2025: NISQA-Corpus-fitted Mahalanobis/L2/kNN applied out-of-corpus ===")
    print(df_ood.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
