"""
robustness_ledoitwolf_utmos.py -- One-off robustness check for Method Sec. 3.1.

UTMOS features are D=1024-dimensional and the feature reference set is N=10,000
(TRAIN_SIM), giving N/D~9.8. The paper claims this ratio is large enough to skip
Ledoit-Wolf shrinkage. This script refits the UTMOS Mahalanobis reference with
Ledoit-Wolf shrinkage (sklearn), rescoring the exact canonical ID/OOD split and
within-domain error-detection tasks, and reports the AUROC delta against the
empirical-covariance numbers already in results/canonical/*.csv. It reads only
from results/features and results/canonical/val_sim_split.csv and writes
nothing back into results/canonical or results/mahal, so it cannot change any
number already reported in the paper.

Usage:
  python robustness_ledoitwolf_utmos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fit_mahalanobis import fit_mahalanobis, mahalanobis_scores

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "results" / "features"
CANONICAL_DIR = ROOT / "results" / "canonical"

ID_DATASET = "nisqa_val_sim"
SHIFTED_DATASETS = (
    "nisqa_train_live",
    "nisqa_val_live",
    "nisqa_test_for",
    "nisqa_test_livetalk",
    "nisqa_test_p501",
)
REFERENCE_DATASET = "nisqa_train_sim"
MODEL = "utmos"


def load(dataset: str) -> dict[str, np.ndarray]:
    data = np.load(FEATURES_DIR / f"features_{MODEL}_{dataset}.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


def error_detection_auroc(mos: np.ndarray, mos_pred: np.ndarray, score: np.ndarray) -> float:
    abs_error = np.abs(mos - mos_pred)
    threshold = np.quantile(abs_error, 0.75)
    labels = (abs_error >= threshold).astype(int)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, score))


def main() -> None:
    reference = load(REFERENCE_DATASET)
    n_ref, d_ref = reference["features"].shape
    print(f"[robustness] UTMOS reference: N={n_ref}, D={d_ref}, N/D={n_ref / d_ref:.2f}")

    mu_emp, prec_emp = fit_mahalanobis(reference["features"], regularise=False)
    mu_lw, prec_lw = fit_mahalanobis(reference["features"], regularise=True)

    split = pd.read_csv(CANONICAL_DIR / "val_sim_split.csv")
    id_role = dict(zip(split["filepath"].astype(str), split["role"]))

    id_data = load(ID_DATASET)
    eval_mask = np.array(
        [id_role.get(Path(str(fp)).name) == "id_evaluation"
         or id_role.get(str(fp)) == "id_evaluation" for fp in id_data["filepaths"]]
    )
    if eval_mask.sum() == 0:
        # fall back: split.csv stores full paths, features store bare names or vice versa
        name_role = {Path(k).name: v for k, v in id_role.items()}
        eval_mask = np.array(
            [name_role.get(Path(str(fp)).name) == "id_evaluation" for fp in id_data["filepaths"]]
        )
    assert eval_mask.sum() > 0, "Could not align val_sim_split.csv with UTMOS val_sim features"
    print(f"[robustness] ID-evaluation half: {eval_mask.sum()} / {len(eval_mask)} clips")

    def score_all(mu: np.ndarray, prec: np.ndarray, data: dict) -> np.ndarray:
        return mahalanobis_scores(data["features"], mu, prec)

    id_scores_emp = score_all(mu_emp, prec_emp, id_data)
    id_scores_lw = score_all(mu_lw, prec_lw, id_data)

    rows = []
    for dataset in SHIFTED_DATASETS:
        shifted = load(dataset)
        shifted_scores_emp = score_all(mu_emp, prec_emp, shifted)
        shifted_scores_lw = score_all(mu_lw, prec_lw, shifted)

        labels = np.concatenate([np.zeros(eval_mask.sum()), np.ones(len(shifted["mos"]))])
        domain_auroc_emp = roc_auc_score(
            labels, np.concatenate([id_scores_emp[eval_mask], shifted_scores_emp])
        )
        domain_auroc_lw = roc_auc_score(
            labels, np.concatenate([id_scores_lw[eval_mask], shifted_scores_lw])
        )

        error_auroc_emp = error_detection_auroc(shifted["mos"], shifted["mos_pred"], shifted_scores_emp)
        error_auroc_lw = error_detection_auroc(shifted["mos"], shifted["mos_pred"], shifted_scores_lw)

        rows.append({
            "dataset": dataset,
            "domain_auroc_empirical": domain_auroc_emp,
            "domain_auroc_ledoitwolf": domain_auroc_lw,
            "domain_auroc_delta": domain_auroc_lw - domain_auroc_emp,
            "error_auroc_empirical": error_auroc_emp,
            "error_auroc_ledoitwolf": error_auroc_lw,
            "error_auroc_delta": error_auroc_lw - error_auroc_emp,
        })

    # ID-evaluation half is also an error-detection task (V-SIM row in Fig. 2)
    id_error_emp = error_detection_auroc(
        id_data["mos"][eval_mask], id_data["mos_pred"][eval_mask], id_scores_emp[eval_mask]
    )
    id_error_lw = error_detection_auroc(
        id_data["mos"][eval_mask], id_data["mos_pred"][eval_mask], id_scores_lw[eval_mask]
    )
    rows.append({
        "dataset": ID_DATASET,
        "domain_auroc_empirical": float("nan"),
        "domain_auroc_ledoitwolf": float("nan"),
        "domain_auroc_delta": float("nan"),
        "error_auroc_empirical": id_error_emp,
        "error_auroc_ledoitwolf": id_error_lw,
        "error_auroc_delta": id_error_lw - id_error_emp,
    })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print(df.to_string(index=False))
    print(f"\n[robustness] max |domain AUROC delta| = {df['domain_auroc_delta'].abs().max():.4f}")
    print(f"[robustness] max |error AUROC delta|  = {df['error_auroc_delta'].abs().max():.4f}")


if __name__ == "__main__":
    main()
