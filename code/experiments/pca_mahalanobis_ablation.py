"""
pca_mahalanobis_ablation.py — PCA dimensionality reduction before Mahalanobis OOD scoring.

For UTMOS (d=1024, N=10000), the sample covariance is near-rank-deficient (N/D≈9.8).
PCA to a lower dimension d' before fitting the Gaussian may improve the Mahalanobis
OOD detector by removing noisy directions.

Sweeps d' = {16, 32, 50, 64, 128, 256, None} (None = full dim, no PCA).
Computes AUROC + FPR@95 for each (model, dataset, d') combination.

# PAPER: Ablation — PCA before Mahalanobis

Usage:
  python pca_mahalanobis_ablation.py \
    --features_dir ../../results/features \
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
from sklearn.decomposition import PCA

from fit_mahalanobis import fit_mahalanobis, mahalanobis_scores
from evaluate import (
    compute_auroc,
    compute_fpr_at_tpr,
    load_features,
    _OOD_DATASETS,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

D_PRIME_VALUES = [16, 32, 50, 64, 128, 256, None]  # None = no PCA (full dim)

# ID eval set is held out from the reference bank (V-SIM), not the reference
# bank itself, to avoid in-sample optimism when scoring the fit data.
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
_MODEL_COLORS = {"nisqa": _WONG["blue"], "dnsmos": _WONG["orange"], "utmos": _WONG["green"]}
_MODEL_MARKERS = {"nisqa": "o", "dnsmos": "s", "utmos": "^"}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_pca_sweep(
    features_dir: Path,
    models: list[str],
    d_prime_values: list[int | None],
) -> pd.DataFrame:
    """Compute Mahalanobis AUROC and FPR@95 for each (model, d', OOD dataset)."""

    rows = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")

        # Load TRAIN_SIM features (for PCA fitting + Mahalanobis fitting)
        train_data = load_features(features_dir, model, _TRAIN_SIM)
        if train_data is None:
            print(f"  [SKIP] No TRAIN_SIM features for {model}")
            continue
        train_feats = train_data["features"]
        orig_dim = train_feats.shape[1]
        print(f"  Original feature dim: {orig_dim}")

        # Load held-out in-distribution evaluation features (not the reference bank)
        id_data = load_features(features_dir, model, _ID_EVAL_DATASET)
        if id_data is None:
            continue
        id_features = id_data["features"]

        for d_prime in d_prime_values:
            # Skip d' >= original dim (PCA would be no-op or invalid)
            if d_prime is not None and d_prime >= orig_dim:
                continue

            label = f"d'={d_prime}" if d_prime is not None else f"full (d={orig_dim})"
            print(f"\n  {label}")

            # Fit PCA on TRAIN_SIM
            if d_prime is not None:
                pca = PCA(n_components=d_prime, random_state=42)
                pca.fit(train_feats)
                var_explained = pca.explained_variance_ratio_.sum()
                print(f"    Variance explained: {var_explained:.3f}")
                train_feats_pca = pca.transform(train_feats)
                id_feats_pca = pca.transform(id_features)
            else:
                pca = None
                train_feats_pca = train_feats
                id_feats_pca = id_features

            # Fit Mahalanobis on (possibly reduced) TRAIN_SIM
            mu, sigma_inv = fit_mahalanobis(train_feats_pca, regularise=None)

            # Compute ID Mahalanobis scores
            mahal_id = mahalanobis_scores(id_feats_pca, mu, sigma_inv)

            for ood_ds in sorted(_OOD_DATASETS):
                ood_data = load_features(features_dir, model, ood_ds)
                if ood_data is None:
                    continue
                ood_feats = ood_data["features"]
                if pca is not None:
                    ood_feats = pca.transform(ood_feats)
                mahal_ood = mahalanobis_scores(ood_feats, mu, sigma_inv)

                auroc = compute_auroc(mahal_id, mahal_ood)
                fpr95 = compute_fpr_at_tpr(mahal_id, mahal_ood)
                print(f"    {ood_ds:25s}  AUROC={auroc:.3f}  FPR@95={fpr95:.3f}")
                rows.append({
                    "model": model,
                    "dataset": ood_ds,
                    "d_prime": d_prime if d_prime is not None else orig_dim,
                    "pca": d_prime is not None,
                    "auroc": auroc,
                    "fpr95": fpr95,
                    "var_explained": var_explained if d_prime is not None else 1.0,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot: AUROC vs d' (one line per model)
# ---------------------------------------------------------------------------

def plot_auroc_vs_dprime(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Line plot: mean AUROC across OOD datasets vs d', one line per model."""

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
        avg = sub.groupby("d_prime")["auroc"].mean().sort_index()
        ax.plot(
            avg.index, avg.values,
            marker=_MODEL_MARKERS.get(model, "o"),
            color=_MODEL_COLORS.get(model, "#333"),
            linewidth=1.2,
            markersize=5,
            label=model.upper(),
        )
        for di, vi in zip(avg.index, avg.values):
            ax.annotate(
                f"{vi:.1f}",
                (di, vi),
                textcoords="offset points",
                xytext=(0, 6),
                fontsize=6,
                ha="center",
            )

    ax.set_xlabel("PCA dimension $d'$")
    ax.set_ylabel("AUROC (mean over OOD sets)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(df["d_prime"].unique()))
    ax.set_xticklabels([str(int(x)) for x in sorted(df["d_prime"].unique())])
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"\n[pca_ablation] Saved plot -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA + Mahalanobis ablation.")
    p.add_argument("--features_dir", default="../../results/features")
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

    df = run_pca_sweep(
        features_dir=Path(args.features_dir),
        models=args.models,
        d_prime_values=D_PRIME_VALUES,
    )

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"pca_mahalanobis_ablation_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[pca_ablation] Saved CSV -> {csv_path}")

    # Print summary (avg AUROC per model x d')
    print("\n--- Average AUROC (over OOD datasets) ---")
    pivot = df.pivot_table(index="model", columns="d_prime", values="auroc", aggfunc="mean")
    print(pivot.round(3).to_string())

    print("\n--- Average FPR@95 (over OOD datasets) ---")
    pivot_fpr = df.pivot_table(index="model", columns="d_prime", values="fpr95", aggfunc="mean")
    print(pivot_fpr.round(3).to_string())

    # Variance explained summary
    print("\n--- Variance explained ---")
    var_pivot = df[df["pca"]].drop_duplicates(["model", "d_prime"]).pivot_table(
        index="model", columns="d_prime", values="var_explained", aggfunc="mean"
    )
    print(var_pivot.round(3).to_string())

    # Plot
    plot_auroc_vs_dprime(df, fig_dir / "pca_mahalanobis_ablation.pdf")


if __name__ == "__main__":
    main()
