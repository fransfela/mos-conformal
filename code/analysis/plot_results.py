"""
plot_results.py â€” Generate all paper figures from evaluation CSV outputs.

Figures produced:
  # PAPER: Fig 1 â€” pipeline_diagram.pdf
    (generated as schematic; requires manual refinement)

  # PAPER: Fig 2 â€” mahal_vs_error.pdf
    Scatter: Mahalanobis distance vs. absolute MOS error, per model, per dataset.
    Coloured by dataset (shift level). Shows OOD score correlates with prediction error.

  # PAPER: Fig 3 â€” auroc_bar.pdf
    Bar chart: AUROC for OOD detection vs. dataset (shift level), per model.

  # PAPER: Fig 4 â€” coverage_vs_shift.pdf
    Line plot: empirical coverage vs. dataset order (shift sequence),
    comparing fixed conformal, adaptive conformal, and naive Â±sigma baseline.

Usage:
  python plot_results.py \\
    --features_dir  ../../results/features \\
    --mahal_dir     ../../results/mahal \\
    --eval_dir      ../../results \\
    --out_dir       ../../paper/figures \\
    --models nisqa dnsmos utmos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent / "experiments"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from fit_mahalanobis import load_model as load_mahal_model, mahalanobis_scores


# ---------------------------------------------------------------------------
# Plot style â€” IEEE-compatible
# Figure rules:
#   1. Wong (2011) colorblind-safe palette throughout
#   2. Distinct markers + line styles for B&W printability
#   3. Dark grey (#303030) edge/border on all bars and scatter points
# ---------------------------------------------------------------------------

# Wong colorblind-safe palette
_WONG = {
    "black":    "#000000",
    "orange":   "#E69F00",
    "sky":      "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermil":   "#D55E00",
    "pink":     "#CC79A7",
}
_EDGE = "#303030"   # dark grey border for bars and scatter points

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
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
})

_DATASET_LABELS = {
    "nisqa_train_sim":    "TRAIN SIM",
    "nisqa_train_live":   "TRAIN LIVE",
    "nisqa_val_sim":      "VAL SIM",
    "nisqa_val_live":     "VAL LIVE",
    "nisqa_test_for":     "TEST FOR",
    "nisqa_test_livetalk":"TEST LIVETALK",
    "nisqa_test_p501":    "TEST P.501",
}
_SHIFT_ORDER = [
    "nisqa_train_sim", "nisqa_train_live",
    "nisqa_val_sim", "nisqa_val_live",
    "nisqa_test_for", "nisqa_test_livetalk", "nisqa_test_p501",
]
_IN_DIST = {"nisqa_train_sim", "nisqa_train_live"}

# Per-dataset: color (Wong) + marker shape for B&W differentiability
_DATASET_STYLE = {
    "nisqa_train_sim":    {"color": _WONG["blue"],   "marker": "o",  "hatch": ""},
    "nisqa_train_live":   {"color": _WONG["sky"],    "marker": "s",  "hatch": "///"},
    "nisqa_val_sim":      {"color": _WONG["green"],  "marker": "^",  "hatch": ""},
    "nisqa_val_live":     {"color": _WONG["pink"],   "marker": "D",  "hatch": "///"},
    "nisqa_test_for":     {"color": _WONG["orange"], "marker": "p",  "hatch": "xxx"},
    "nisqa_test_livetalk":{"color": _WONG["vermil"], "marker": "*",  "hatch": "..."},
    "nisqa_test_p501":    {"color": _WONG["black"],  "marker": "X",  "hatch": "---"},
}
# Convenience accessors
_COLORS  = {k: v["color"]  for k, v in _DATASET_STYLE.items()}
_MARKERS = {k: v["marker"] for k, v in _DATASET_STYLE.items()}
_HATCHES = {k: v["hatch"]  for k, v in _DATASET_STYLE.items()}

# Per-model colors (Wong) + hatch for B&W bar differentiation
_MODEL_COLORS = {
    "nisqa":  {"color": _WONG["blue"],   "hatch": ""},
    "dnsmos": {"color": _WONG["vermil"], "hatch": "///"},
    "utmos":  {"color": _WONG["green"],  "hatch": "xxx"},
}

_MODEL_LABELS = {"dnsmos": "DNSMOS", "nisqa": "NISQA", "utmos": "UTMOS"}

# Coverage line-plot: distinct colors + line styles + markers for B&W
_METHOD_STYLES = {
    "fixed_sigma":        {"linestyle": (0, (4, 2)),  "marker": "s",  "color": _WONG["orange"],
                           "markerfacecolor": _WONG["orange"], "markeredgecolor": _EDGE,
                           "markeredgewidth": 0.6},
    "conformal_fixed":    {"linestyle": "-",           "marker": "o",  "color": _WONG["blue"],
                           "markerfacecolor": _WONG["blue"],   "markeredgecolor": _EDGE,
                           "markeredgewidth": 0.6},
    "conformal_adaptive": {"linestyle": (0, (6, 2, 1, 2)), "marker": "^", "color": _WONG["vermil"],
                           "markerfacecolor": _WONG["vermil"], "markeredgecolor": _EDGE,
                           "markeredgewidth": 0.6},
}
_METHOD_LABELS = {
    "fixed_sigma":        r"Fixed $\pm\sigma$",
    "conformal_fixed":    "Conformal (fixed)",
    "conformal_adaptive": "Conformal (adaptive, ours)",
}


# ---------------------------------------------------------------------------
# Fig 2 â€” Mahalanobis distance vs. absolute MOS error
# ---------------------------------------------------------------------------

def plot_mahal_vs_error(
    features_dir: Path,
    mahal_dir: Path,
    out_dir: Path,
    models: list[str],
    datasets: list[str],
) -> None:
    """
    # PAPER: Fig 2
    Scatter plot: d_M(x) vs |q* - q_hat|, coloured by dataset.
    One panel per model.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7.16, 1.9), sharey=False)
    if n_models == 1:
        axes = [axes]
    plt.subplots_adjust(bottom=0.32, wspace=0.35)

    for ax, model in zip(axes, models):
        mahal_path = mahal_dir / f"mahal_{model}.npz"
        if not mahal_path.exists():
            ax.set_title(f"{_MODEL_LABELS.get(model, model)} (no model)")
            continue
        mu, sigma_inv = load_mahal_model(mahal_path)

        for dataset in datasets:
            feat_path = features_dir / f"features_{model}_{dataset}.npz"
            if not feat_path.exists():
                continue
            data = np.load(feat_path, allow_pickle=True)
            feats = data["features"]
            mos_true = data["mos"].astype(np.float32)
            mos_pred = data["mos_pred"].astype(np.float32)

            d_m = mahalanobis_scores(feats, mu, sigma_inv)
            abs_err = np.abs(mos_true - mos_pred)

            ds_style = _DATASET_STYLE.get(dataset, {"color": "grey", "marker": "o"})
            ax.scatter(
                d_m, abs_err,
                s=8, alpha=0.5,
                color=ds_style["color"],
                marker=ds_style["marker"],
                linewidths=0.5,
                edgecolors=_EDGE,
                label=_DATASET_LABELS.get(dataset, dataset),
            )

        ax.set_xlabel(r"Mahalanobis distance $d_M(\mathbf{x})$")
        ax.set_ylabel(r"$|\hat{q} - q^*|$ (MOS)")
        ax.set_title(_MODEL_LABELS.get(model, model))
        ax.grid(True)

    # Single compact legend below all panels â€” one row, abbreviated labels
    _SHORT_LABELS = {
        "nisqa_train_sim":    "T-SIM",
        "nisqa_train_live":   "T-LIV",
        "nisqa_val_sim":      "V-SIM",
        "nisqa_val_live":     "V-LIV",
        "nisqa_test_for":     "T-FOR",
        "nisqa_test_livetalk": "LIVE",
        "nisqa_test_p501":    "P.501",
    }
    # Rebuild handles with short labels
    handles_all, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            short = next((v for k, v in _SHORT_LABELS.items()
                          if _DATASET_LABELS.get(k, k) == li), li)
            if short not in labels_all:
                handles_all.append(hi)
                labels_all.append(short)
    fig.legend(
        handles_all, labels_all,
        loc="lower center",
        ncol=len(labels_all),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.6,
    )
    out = out_dir / "mahal_vs_error.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot_results] Saved -> {out}")

    # --- NISQA-only single-column version for paper Fig 2 ---
    nisqa_model = "nisqa" if "nisqa" in models else models[0]
    fig2, ax2 = plt.subplots(figsize=(3.35, 2.6))
    mahal_path = mahal_dir / f"mahal_{nisqa_model}.npz"
    if mahal_path.exists():
        mu, sigma_inv = load_mahal_model(mahal_path)
        for dataset in datasets:
            feat_path = features_dir / f"features_{nisqa_model}_{dataset}.npz"
            if not feat_path.exists():
                continue
            data = np.load(feat_path, allow_pickle=True)
            feats = data["features"]
            mos_true = data["mos"].astype(np.float32)
            mos_pred = data["mos_pred"].astype(np.float32)
            d_m = mahalanobis_scores(feats, mu, sigma_inv)
            abs_err = np.abs(mos_true - mos_pred)
            ds_style = _DATASET_STYLE.get(dataset, {"color": "grey", "marker": "o"})
            ax2.scatter(
                d_m, abs_err,
                s=6, alpha=0.5,
                color=ds_style["color"],
                marker=ds_style["marker"],
                linewidths=0.4,
                edgecolors=_EDGE,
                label=_DATASET_LABELS.get(dataset, dataset),
            )
        ax2.set_xlabel(r"Mahalanobis distance $d_M(\mathbf{x})$")
        ax2.set_ylabel(r"$|\hat{q} - q^*|$ (MOS)")
        ax2.set_title(_MODEL_LABELS.get(nisqa_model, nisqa_model))
        ax2.grid(True)
        ax2.legend(fontsize=6, ncol=2, loc="upper left")
    fig2.tight_layout()
    out2 = out_dir / "mahal_nisqa.pdf"
    fig2.savefig(out2)
    plt.close(fig2)
    print(f"[plot_results] Saved â†’ {out2}")


# ---------------------------------------------------------------------------
# Fig 3 â€” AUROC bar chart
# ---------------------------------------------------------------------------

def plot_auroc_bar(
    eval_dir: Path,
    out_dir: Path,
    models: list[str],
) -> None:
    """
    # PAPER: Fig 3
    Grouped bar chart: AUROC per (model, dataset), datasets ordered by shift.
    """
    # Find latest AUROC CSV
    csv_files = sorted(eval_dir.glob("eval_auroc_*.csv"))
    if not csv_files:
        print(f"[plot_results] No eval_auroc_*.csv found in {eval_dir}. Skipping Fig 3.")
        return
    df = pd.read_csv(csv_files[-1])

    ood_datasets = [d for d in _SHIFT_ORDER if d not in _IN_DIST]
    x = np.arange(len(ood_datasets))
    width = 0.25
    offsets = np.linspace(-(len(models) - 1) * width / 2, (len(models) - 1) * width / 2, len(models))

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, model in enumerate(models):
        aurocs = []
        for ds in ood_datasets:
            row = df[(df["model"] == model) & (df["dataset"] == ds)]
            aurocs.append(float(row["auroc"].iloc[0]) if len(row) > 0 else float("nan"))
        mstyle = _MODEL_COLORS.get(model, {"color": "grey", "hatch": ""})
        ax.bar(
            x + offsets[i], aurocs, width,
            label=_MODEL_LABELS.get(model, model),
            color=mstyle["color"],
            hatch=mstyle["hatch"],
            edgecolor=_EDGE,
            linewidth=0.7,
        )

    ax.axhline(0.5, color="black", linewidth=0.6, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([_DATASET_LABELS.get(d, d) for d in ood_datasets], rotation=15, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    out = out_dir / "auroc_bar.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot_results] Saved â†’ {out}")


# ---------------------------------------------------------------------------
# Fig 4 â€” Coverage vs. shift level
# ---------------------------------------------------------------------------

def plot_coverage_vs_shift(
    eval_dir: Path,
    out_dir: Path,
    models: list[str],
    nominal_alpha: float = 0.10,
) -> None:
    """
    # PAPER: Fig 4
    Line plot: empirical coverage vs. dataset order per interval method.
    Separate panels per model.
    """
    csv_files = sorted(eval_dir.glob("eval_coverage_width_*.csv"))
    if not csv_files:
        print(f"[plot_results] No eval_coverage_width_*.csv found in {eval_dir}. Skipping Fig 4.")
        return
    df = pd.read_csv(csv_files[-1])

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(2.8 * n_models, 2.4), sharey=True)
    if n_models == 1:
        axes = [axes]

    methods = ["fixed_sigma", "conformal_fixed", "conformal_adaptive"]

    for ax, model in zip(axes, models):
        for method in methods:
            covs = []
            for ds in _SHIFT_ORDER:
                row = df[(df["model"] == model) & (df["dataset"] == ds) & (df["method"] == method)]
                covs.append(float(row["coverage"].iloc[0]) if len(row) > 0 else float("nan"))
            style = _METHOD_STYLES.get(method, {})
            ax.plot(
                range(len(_SHIFT_ORDER)), covs,
                label=_METHOD_LABELS.get(method, method),
                **style,
            )

        # Nominal coverage line
        ax.axhline(1 - nominal_alpha, color="black", linewidth=0.8, linestyle="-",
                   label=f"Nominal ({100*(1-nominal_alpha):.0f}%)")
        ax.set_xticks(range(len(_SHIFT_ORDER)))
        ax.set_xticklabels(
            [_DATASET_LABELS.get(d, d) for d in _SHIFT_ORDER],
            rotation=30, ha="right",
        )
        ax.set_title(_MODEL_LABELS.get(model, model))
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(0.15, 1.05)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    out = out_dir / "coverage_vs_shift.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot_results] Saved â†’ {out}")


# ---------------------------------------------------------------------------
# Fig 4b â€” Coverage vs. shift (averaged across models, single-panel for paper)
# ---------------------------------------------------------------------------

def plot_coverage_avg(
    eval_dir: Path,
    out_dir: Path,
    models: list[str],
    nominal_alpha: float = 0.10,
) -> None:
    """
    # PAPER: Fig 4 (compact)
    Single-panel coverage vs. shift, averaged across all models.
    Suitable for a single-column figure.
    """
    csv_files = sorted(eval_dir.glob("eval_coverage_width_*.csv"))
    if not csv_files:
        print(f"[plot_results] No eval_coverage_width_*.csv in {eval_dir}. Skipping.")
        return
    df = pd.read_csv(csv_files[-1])
    methods = ["fixed_sigma", "conformal_fixed", "conformal_adaptive"]

    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for method in methods:
        covs = []
        for ds in _SHIFT_ORDER:
            vals = df[(df["dataset"] == ds) & (df["method"] == method) &
                      (df["model"].isin(models))]["coverage"]
            covs.append(float(vals.mean()) if len(vals) > 0 else float("nan"))
        style = _METHOD_STYLES.get(method, {})
        ax.plot(
            range(len(_SHIFT_ORDER)), covs,
            label=_METHOD_LABELS.get(method, method),
            **style,
        )

    ax.axhline(1 - nominal_alpha, color="black", linewidth=0.8, linestyle="-",
               label=f"Nominal ({100*(1-nominal_alpha):.0f}%)")
    short_labels = ["T-SIM", "T-LIVE", "V-SIM", "V-LIVE", "FOR", "TALK", "P.5"]
    ax.set_xticks(range(len(_SHIFT_ORDER)))
    ax.set_xticklabels(short_labels, rotation=0)
    ax.set_ylabel("Coverage (avg. 3 models)")
    ax.set_ylim(0.15, 1.05)
    ax.legend(loc="lower left", fontsize=6, ncol=1)
    ax.grid(True)
    fig.tight_layout()
    out = out_dir / "coverage_avg.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot_results] Saved â†’ {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate paper figures from evaluation results.")
    p.add_argument("--features_dir", default="../../results/features")
    p.add_argument("--mahal_dir", default="../../results/mahal")
    p.add_argument("--eval_dir", default="../../results")
    p.add_argument("--out_dir", default="../../paper/figures")
    p.add_argument("--models", nargs="+", default=["nisqa", "dnsmos", "utmos"])
    p.add_argument("--datasets", nargs="+",
                   default=["nisqa_train_sim", "nisqa_train_live", "nisqa_val_sim",
                            "nisqa_val_live", "nisqa_test_for",
                            "nisqa_test_livetalk", "nisqa_test_p501"])
    p.add_argument("--alpha", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[plot_results] Generating figures ...")

    plot_mahal_vs_error(
        features_dir=Path(args.features_dir),
        mahal_dir=Path(args.mahal_dir),
        out_dir=out_dir,
        models=args.models,
        datasets=args.datasets,
    )
    plot_auroc_bar(
        eval_dir=Path(args.eval_dir),
        out_dir=out_dir,
        models=args.models,
    )
    plot_coverage_vs_shift(
        eval_dir=Path(args.eval_dir),
        out_dir=out_dir,
        models=args.models,
        nominal_alpha=args.alpha,
    )
    plot_coverage_avg(
        eval_dir=Path(args.eval_dir),
        out_dir=out_dir,
        models=args.models,
        nominal_alpha=args.alpha,
    )
    print("[plot_results] Done.")


if __name__ == "__main__":
    main()

