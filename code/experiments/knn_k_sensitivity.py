"""
knn_k_sensitivity.py — KNN k-sensitivity ablation for OOD detection.

Sweeps k = {1, 5, 10, 20, 50} and computes AUROC + FPR@95TPR for each
(model, OOD dataset, k) combination. Also produces an AUROC-vs-k plot.

# PAPER: Ablation — KNN k sensitivity

Usage:
  python knn_k_sensitivity.py \
    --features_dir ../../results/features \
    --mahal_dir    ../../results/mahal \
    --out_dir      ../../results \
    --fig_dir      ../../paper/figures \
    --models nisqa dnsmos utmos
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fit_mahalanobis import load_model as load_mahal_model, knn_scores
from evaluate import (
    compute_auroc,
    compute_fpr_at_tpr,
    load_features,
    _OOD_DATASETS,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_VALUES = [1, 5, 10, 20, 50]

# ID eval set is held out from the reference bank (V-SIM), not the reference
# bank itself, to avoid self-neighbor leakage (distance-0 self-matches at k=1).
_ID_EVAL_DATASET = "nisqa_val_sim"
_TRAIN_SIM = "nisqa_train_sim"

# Wong colorblind-safe palette
_WONG = {
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "blue":   "#0072B2",
    "vermil": "#D55E00",
}
_MODEL_COLORS = {
    "nisqa":  _WONG["blue"],
    "dnsmos": _WONG["orange"],
    "utmos":  _WONG["green"],
}
_MODEL_MARKERS = {"nisqa": "o", "dnsmos": "s", "utmos": "^"}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_k_sweep(
    features_dir: Path,
    models: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    """Compute AUROC and FPR@95 for each (model, dataset, k)."""

    rows = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")

        # Load training features (KNN reference set)
        train_data = load_features(features_dir, model, _TRAIN_SIM)
        if train_data is None:
            print(f"  [SKIP] No TRAIN_SIM features for {model}")
            continue
        train_feats = train_data["features"]

        # Load in-distribution evaluation features (held out, not the reference bank)
        # (needed for AUROC: ID vs OOD comparison)
        id_data = load_features(features_dir, model, _ID_EVAL_DATASET)
        if id_data is None:
            print(f"  [SKIP] No {_ID_EVAL_DATASET} features for {model}")
            continue
        id_features = id_data["features"]

        for k in k_values:
            print(f"\n  k={k}")
            # ID scores
            knn_id = knn_scores(id_features, train_feats, k=k)

            for ood_ds in sorted(_OOD_DATASETS):
                ood_data = load_features(features_dir, model, ood_ds)
                if ood_data is None:
                    continue
                knn_ood = knn_scores(ood_data["features"], train_feats, k=k)
                auroc = compute_auroc(knn_id, knn_ood)
                fpr95 = compute_fpr_at_tpr(knn_id, knn_ood)
                print(f"    {ood_ds:25s}  AUROC={auroc:.3f}  FPR@95={fpr95:.3f}")
                rows.append({
                    "model": model,
                    "dataset": ood_ds,
                    "k": k,
                    "auroc": auroc,
                    "fpr95": fpr95,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot: AUROC vs k (one line per model, averaged over OOD datasets)
# ---------------------------------------------------------------------------

def plot_auroc_vs_k(
    df: pd.DataFrame,
    out_path: Path,
    k_values: list[int],
) -> None:
    """Line plot: mean AUROC across OOD datasets vs k, one line per model."""

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    fig, ax = plt.subplots(figsize=(3.4, 2.2))

    for model in df["model"].unique():
        sub = df[df["model"] == model]
        avg = sub.groupby("k")["auroc"].mean()
        ax.plot(
            avg.index, avg.values,
            marker=_MODEL_MARKERS.get(model, "o"),
            color=_MODEL_COLORS.get(model, "#333"),
            linewidth=1.2,
            markersize=5,
            label=model.upper(),
        )
        # Annotate each point with 1 decimal
        for ki, vi in zip(avg.index, avg.values):
            ax.annotate(
                f"{vi:.1f}",
                (ki, vi),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=6,
                ha="center",
            )

    ax.set_xlabel("$k$ (number of neighbours)")
    ax.set_ylabel("AUROC (mean over OOD sets)")
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"\n[knn_k_sensitivity] Saved plot → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KNN k-sensitivity ablation.")
    p.add_argument("--features_dir", default="../../results/features")
    p.add_argument("--mahal_dir", default="../../results/mahal")
    p.add_argument("--out_dir", default="../../results")
    p.add_argument("--fig_dir", default="../../paper/figures")
    p.add_argument("--models", nargs="+", default=["nisqa", "dnsmos", "utmos"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = run_k_sweep(
        features_dir=Path(args.features_dir),
        models=args.models,
        k_values=K_VALUES,
    )

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"knn_k_sensitivity_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[knn_k_sensitivity] Saved CSV → {csv_path}")

    # Print summary table (avg AUROC per model × k)
    print("\n--- Average AUROC (over OOD datasets) ---")
    pivot = df.pivot_table(index="model", columns="k", values="auroc", aggfunc="mean")
    print(pivot.round(3).to_string())

    print("\n--- Average FPR@95 (over OOD datasets) ---")
    pivot_fpr = df.pivot_table(index="model", columns="k", values="fpr95", aggfunc="mean")
    print(pivot_fpr.round(3).to_string())

    # Plot
    plot_auroc_vs_k(df, fig_dir / "knn_k_sensitivity.pdf", K_VALUES)


if __name__ == "__main__":
    main()
