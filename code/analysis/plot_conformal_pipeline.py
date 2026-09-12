"""Generate the compact, vector pipeline figure used in the paper."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "fig1_pipeline.pdf"


def box(ax, xy, width, height, text, face, edge="#26364f", linestyle="-", fontsize=5.6):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17243a",
        linespacing=1.25,
    )


def arrow(ax, start, end, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.05,
            color="#26364f",
            linestyle=linestyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.965, "OFFLINE", fontsize=7.2, fontweight="bold", color="#26364f")
    box(
        ax,
        (0.02, 0.69),
        0.42,
        0.20,
        "T-SIM features\nfit $d(\\mathbf{z})$",
        "#dce8f5",
        fontsize=6.2,
    )
    box(
        ax,
        (0.56, 0.69),
        0.42,
        0.20,
        "V-SIM-Cal labels\nfit conformal radii",
        "#f8e7bc",
        fontsize=6.2,
    )

    ax.plot([0.02, 0.98], [0.62, 0.62], color="#8793a5", linewidth=0.8)
    ax.text(0.02, 0.57, "INFERENCE", fontsize=7.2, fontweight="bold", color="#26364f")

    box(ax, (0.01, 0.29), 0.14, 0.18, "Speech\n$\\mathbf{x}$", "#f2f3f5", fontsize=6.0)
    box(
        ax,
        (0.20, 0.25),
        0.25,
        0.26,
        "Frozen predictor\n$\\hat q=f(\\mathbf{x})$\n$\\mathbf{z}=\\phi(\\mathbf{x})$",
        "#dce8f5",
    )
    box(
        ax,
        (0.50, 0.25),
        0.29,
        0.26,
        "Distance score\n$d(\\mathbf{z})\\rightarrow Q(\\mathbf{x})$",
        "#dcefe9",
    )
    box(
        ax,
        (0.84, 0.29),
        0.15,
        0.18,
        "Interval\n$[\\hat q-Q,\\hat q+Q]$",
        "#e3e5e8",
        fontsize=5.2,
    )

    arrow(ax, (0.15, 0.38), (0.20, 0.38))
    arrow(ax, (0.45, 0.38), (0.50, 0.38))
    arrow(ax, (0.79, 0.38), (0.84, 0.38))
    arrow(ax, (0.23, 0.69), (0.59, 0.51), linestyle="--")
    arrow(ax, (0.77, 0.69), (0.70, 0.51), linestyle="--")

    box(
        ax,
        (0.10, 0.035),
        0.80,
        0.12,
        "Tested hypothesis: does $d(\\mathbf{z})$ rank prediction error\nwell enough to choose an utterance-specific radius?",
        "#ffffff",
        edge="#58677c",
        linestyle="--",
        fontsize=5.4,
    )

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
