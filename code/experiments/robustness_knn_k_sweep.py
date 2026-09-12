"""
robustness_knn_k_sweep.py -- k-NN k-sensitivity check under the CURRENT canonical protocol.

Session 12 swept k on an older split (pre-canonical, pre-T-LIV). This script repeats the
sweep using the exact canonical protocol in canonical_analysis.py: T-SIM as the fixed
reference bank, the deterministic group-based V-SIM half (val_sim_split.csv, role ==
"id_evaluation") as the ID class, and all five predeclared shifted subsets (including T-LIV)
as the OOD class. It reads only results/features and results/canonical/val_sim_split.csv and
writes nothing back into results/canonical, so it cannot change any number already reported
in the paper.

Usage:
  python robustness_knn_k_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fit_mahalanobis import knn_scores

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "results" / "features"
CANONICAL_DIR = ROOT / "results" / "canonical"

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
K_VALUES = (1, 5, 10, 20, 50)


def load(model: str, dataset: str) -> dict[str, np.ndarray]:
    data = np.load(FEATURES_DIR / f"features_{model}_{dataset}.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    split = pd.read_csv(CANONICAL_DIR / "val_sim_split.csv")
    eval_names = set(
        Path(str(fp)).name for fp in split.loc[split["role"] == "id_evaluation", "filepath"]
    )

    rows = []
    for model in MODELS:
        reference = load(model, REFERENCE_DATASET)
        id_data = load(model, ID_DATASET)
        eval_mask = np.array(
            [Path(str(fp)).name in eval_names for fp in id_data["filepaths"]]
        )
        assert eval_mask.sum() > 0, f"Empty ID-evaluation mask for {model}"

        for dataset in SHIFTED_DATASETS:
            shifted = load(model, dataset)
            labels = np.concatenate([np.zeros(eval_mask.sum()), np.ones(len(shifted["mos"]))])
            for k in K_VALUES:
                id_scores = knn_scores(id_data["features"][eval_mask], reference["features"], k=k)
                shifted_scores = knn_scores(shifted["features"], reference["features"], k=k)
                scores = np.concatenate([id_scores, shifted_scores])
                auroc = roc_auc_score(labels, scores)
                rows.append({"model": model, "dataset": dataset, "k": k, "auroc": auroc})

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="k", columns="model", values="auroc", aggfunc="mean")
    pivot["mean_over_3_models"] = pivot.mean(axis=1)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print("Per-model mean AUROC (averaged over 5 shifted subsets) by k:")
    print(pivot.to_string())
    print()
    overall = df.groupby("k")["auroc"].mean()
    print("Overall mean AUROC (3 models x 5 shifted subsets = 15 pairs) by k:")
    print(overall.to_string())


if __name__ == "__main__":
    main()
