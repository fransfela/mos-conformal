"""Generate the two primary paper figures from the canonical analysis bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "experiments"))

from bootstrap_ci import extract_groups, build_group_index, bootstrap_group_indices


MODELS = ("dnsmos", "nisqa", "utmos")
DATASETS = (
    "nisqa_val_sim",
    "nisqa_train_live",
    "nisqa_val_live",
    "nisqa_test_for",
    "nisqa_test_livetalk",
    "nisqa_test_p501",
)
MODEL_LABELS = {"dnsmos": "DNSMOS", "nisqa": "NISQA", "utmos": "UTMOS"}
DATASET_LABELS = {
    "nisqa_val_sim": "V-SIM",
    "nisqa_train_live": "T-LIV",
    "nisqa_val_live": "V-LIV",
    "nisqa_test_for": "T-FOR",
    "nisqa_test_livetalk": "T-TLK",
    "nisqa_test_p501": "T-P501",
}
COLORS = {
    "nisqa_val_sim": "#0072B2",
    "nisqa_train_live": "#56B4E9",
    "nisqa_val_live": "#E69F00",
    "nisqa_test_for": "#009E73",
    "nisqa_test_livetalk": "#CC79A7",
    "nisqa_test_p501": "#D55E00",
}
MODEL_COLORS = {"dnsmos": "#0072B2", "nisqa": "#E69F00", "utmos": "#009E73"}
MODEL_HATCHES = {"dnsmos": "////", "nisqa": "....", "utmos": "xxxx"}
SEED = 20270821
N_BOOT = 2000

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.55,
})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(cache_dir: Path, model: str, dataset: str) -> dict[str, np.ndarray]:
    data = np.load(
        cache_dir / "scores" / f"scores_{model}_{dataset}.npz", allow_pickle=True
    )
    return {key: data[key] for key in data.files}


def evaluation_mask(cache_dir: Path) -> np.ndarray:
    split = pd.read_csv(cache_dir / "val_sim_split.csv")
    return split["role"].eq("id_evaluation").to_numpy()


def group_sampler(filepaths: np.ndarray):
    unique, mapping = build_group_index(extract_groups(filepaths))
    return lambda rng: bootstrap_group_indices(unique, mapping, rng)


def bootstrap_error_detection(cache_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    id_mask = evaluation_mask(cache_dir)
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            data = load(cache_dir, model, dataset)
            mask = id_mask if dataset == "nisqa_val_sim" else np.ones(len(data["mos"]), bool)
            target = data["mos"][mask]
            prediction = data["mos_pred"][mask]
            score = data["mahalanobis"][mask]
            filepaths = data["filepaths"][mask]
            error = np.abs(target - prediction)
            threshold = float(np.quantile(error, 0.75))
            labels = (error >= threshold).astype(int)
            point = float(roc_auc_score(labels, score))
            sampler = group_sampler(filepaths)
            replicates = []
            for _ in range(N_BOOT):
                indices = sampler(rng)
                sampled_error = error[indices]
                sampled_threshold = float(np.quantile(sampled_error, 0.75))
                sampled_labels = (sampled_error >= sampled_threshold).astype(int)
                if np.unique(sampled_labels).size == 2:
                    replicates.append(
                        float(roc_auc_score(sampled_labels, score[indices]))
                    )
            ci_low, ci_high = np.percentile(replicates, [2.5, 97.5])
            rows.append({
                "model": model,
                "dataset": dataset,
                "score": "mahalanobis",
                "task": "within_domain_top_error_quartile",
                "n": int(mask.sum()),
                "auroc": point,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_boot": len(replicates),
            })
    return pd.DataFrame(rows)


def plot_scatter(cache_dir: Path, figure_dir: Path) -> None:
    id_mask = evaluation_mask(cache_dir)
    fig, axes = plt.subplots(
        len(MODELS), len(DATASETS), figsize=(7.16, 5.05),
        sharex=False, sharey=True,
    )
    for row, model in enumerate(MODELS):
        row_distances = []
        for dataset in DATASETS:
            data = load(cache_dir, model, dataset)
            mask = id_mask if dataset == "nisqa_val_sim" else np.ones(len(data["mos"]), bool)
            row_distances.append(data["mahalanobis"][mask])
        row_distance = np.concatenate(row_distances)
        row_limits = (
            float(row_distance.min()) * 0.94,
            float(row_distance.max()) * 1.06,
        )
        for column, dataset in enumerate(DATASETS):
            ax = axes[row, column]
            data = load(cache_dir, model, dataset)
            mask = id_mask if dataset == "nisqa_val_sim" else np.ones(len(data["mos"]), bool)
            error = np.abs(data["mos"][mask] - data["mos_pred"][mask])
            distance = data["mahalanobis"][mask]
            ax.scatter(
                distance, error, s=3.0, alpha=0.20, color="black",
                linewidths=0, rasterized=True,
            )

            # Equal-frequency bins expose the conditional median without imposing
            # a linear relationship or allowing dense regions to dominate visually.
            edges = np.unique(np.quantile(distance, np.linspace(0, 1, 9)))
            centers, medians = [], []
            if len(edges) > 2:
                bins = np.digitize(distance, edges[1:-1])
                for bin_id in range(len(edges) - 1):
                    selected = bins == bin_id
                    if selected.sum() >= 8:
                        centers.append(float(np.median(distance[selected])))
                        medians.append(float(np.median(error[selected])))
                ax.plot(
                    centers, medians, color="black", linewidth=1.05,
                    marker="o", markersize=2.8, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=0.55, zorder=3,
                )
            rho = float(spearmanr(distance, error).statistic)
            ax.text(
                0.04, 0.93, rf"$\rho_s={rho:.2f}$", transform=ax.transAxes,
                ha="left", va="top", fontsize=6.7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
            )
            ax.set_xscale("log")
            ax.set_xlim(*row_limits)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.18, linewidth=0.4)
            ax.tick_params(direction="out", length=2.5, width=0.55)
            ax.tick_params(
                axis="x", which="both", labelbottom=(column == 0)
            )
            if row == 0:
                ax.set_title(DATASET_LABELS[dataset], fontweight="bold", pad=3)
            if column == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n" + r"$|q^*-\hat q|$ (MOS)")
    fig.supxlabel(r"Mahalanobis distance $d_M(\mathbf{x})$ (log scale)", y=0.025)
    for row in range(len(MODELS)):
        for column in range(1, len(DATASETS)):
            plt.setp(axes[row, column].get_xticklabels(), visible=False)
            plt.setp(axes[row, column].get_xticklabels(minor=True), visible=False)
    fig.subplots_adjust(
        left=0.085, right=0.995, top=0.955, bottom=0.105,
        wspace=0.12, hspace=0.13,
    )
    fig.savefig(figure_dir / "mahal_vs_error.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "mahal_vs_error.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_auroc(frame: pd.DataFrame, figure_dir: Path) -> None:
    x = np.arange(len(DATASETS), dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    # IEEE single-column width. Color and hatching provide redundant encoding,
    # so the plot remains distinguishable in grayscale and for color-vision deficiencies.
    fig, ax = plt.subplots(figsize=(3.35, 2.30))
    for offset, model in zip(offsets, MODELS):
        subset = frame[frame["model"] == model].set_index("dataset").loc[list(DATASETS)]
        values = subset["auroc"].to_numpy()
        lower = values - subset["ci_low"].to_numpy()
        upper = subset["ci_high"].to_numpy() - values
        positions = x + offset
        ax.bar(
            positions, values, width=width * 0.9, color=MODEL_COLORS[model],
            hatch=MODEL_HATCHES[model], edgecolor="black", linewidth=0.65,
            label=MODEL_LABELS[model], zorder=2,
        )
        ax.errorbar(
            positions, values, yerr=np.vstack([lower, upper]), fmt="o",
            color="black", markerfacecolor="white", markeredgewidth=0.7,
            markersize=2.8, capsize=1.8, elinewidth=0.75, zorder=3,
        )
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.2, 0.8)
    ax.grid(axis="y", color="#b0b0b0", alpha=0.45, linewidth=0.45)
    ax.tick_params(axis="both", labelsize=7)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=6.5,
              columnspacing=0.9, handlelength=1.5, handletextpad=0.4)
    fig.tight_layout(pad=0.35)
    fig.savefig(figure_dir / "auroc_bar.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "auroc_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def update_manifest(cache_dir: Path, output_path: Path, figure_dir: Path) -> None:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paper_figures"] = {
        "seed": SEED,
        "n_boot": N_BOOT,
        "error_definition": "within-domain top quartile absolute MOS error",
        "error_score": "Mahalanobis distance with prespecified higher-is-worse direction",
        "figure_sha256": {
            name: sha256(figure_dir / name)
            for name in ("mahal_vs_error.pdf", "auroc_bar.pdf")
        },
    }
    manifest["outputs"] = list(dict.fromkeys([*manifest["outputs"], output_path.name]))
    manifest.setdefault("code_sha256", {})[
        "plot_canonical_paper_figures.py"
    ] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {
        name: sha256(cache_dir / name) for name in manifest["outputs"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    cache_dir = PROJECT_ROOT / "results" / "canonical"
    figure_dir = PROJECT_ROOT / "paper" / "figures"
    output_path = cache_dir / "error_detection_domain_bootstrap.csv"
    if output_path.exists():
        result = pd.read_csv(output_path)
        expected = {(model, dataset) for model in MODELS for dataset in DATASETS}
        available = set(zip(result["model"], result["dataset"]))
        cache_complete = expected == available
    else:
        cache_complete = False
    if not cache_complete:
        result = bootstrap_error_detection(cache_dir)
        result.to_csv(output_path, index=False)
    plot_scatter(cache_dir, figure_dir)
    plot_error_auroc(result, figure_dir)
    update_manifest(cache_dir, output_path, figure_dir)
    print("[paper-figures] Complete")


if __name__ == "__main__":
    main()
