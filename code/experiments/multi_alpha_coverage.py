"""
multi_alpha_coverage.py — Evaluate conformal coverage at multiple alpha levels.

Calibrates conformal wrappers at alpha = {0.05, 0.10, 0.20} (95%, 90%, 80% nominal)
and evaluates empirical coverage + interval width on all test datasets.

# PAPER: Ablation — multiple coverage levels

Usage:
  python multi_alpha_coverage.py \
    --features_dir ../../results/features \
    --mahal_dir    ../../results/mahal \
    --out_dir      ../../results \
    --models nisqa dnsmos utmos
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_wrapper import ConformalWrapper
from fit_mahalanobis import load_model as load_mahal_model, mahalanobis_scores
from evaluate import load_features, compute_coverage_width, _SHIFT_ORDER, _CAL_DATASET


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALPHA_VALUES = [0.05, 0.10, 0.20]  # → 95%, 90%, 80% nominal coverage


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_multi_alpha(
    features_dir: Path,
    mahal_dir: Path,
    models: list[str],
    datasets: list[str],
    alpha_values: list[float],
) -> pd.DataFrame:
    rows = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")

        # Load Mahalanobis model
        mahal_path = mahal_dir / f"mahal_{model}.npz"
        if not mahal_path.exists():
            print(f"  [SKIP] No Mahalanobis model: {mahal_path}")
            continue
        mu, sigma_inv = load_mahal_model(mahal_path)

        # Load calibration data
        cal_data = load_features(features_dir, model, _CAL_DATASET)
        if cal_data is None:
            print(f"  [SKIP] No calibration data for {model}")
            continue

        cal_mos_true = cal_data["mos"].astype(np.float32)
        cal_mos_pred = cal_data["mos_pred"].astype(np.float32)
        cal_mahal = mahalanobis_scores(cal_data["features"], mu, sigma_inv)

        for alpha in alpha_values:
            nominal = 1.0 - alpha
            print(f"\n  alpha={alpha} (nominal {nominal*100:.0f}%)")

            # Calibrate wrappers at this alpha
            wrapper_fixed = ConformalWrapper(alpha=alpha, adaptive=False)
            wrapper_fixed.calibrate(cal_mos_true, cal_mos_pred, cal_mahal)

            wrapper_adapt = ConformalWrapper(alpha=alpha, adaptive=True, n_bins=5)
            wrapper_adapt.calibrate(cal_mos_true, cal_mos_pred, cal_mahal)

            print(f"    Fixed threshold: {wrapper_fixed._threshold:.3f} MOS")

            for dataset in datasets:
                data = load_features(features_dir, model, dataset)
                if data is None:
                    continue

                mos_true = data["mos"].astype(np.float32)
                mos_pred = data["mos_pred"].astype(np.float32)
                mahal = mahalanobis_scores(data["features"], mu, sigma_inv)

                # Fixed conformal
                lo_f, hi_f = wrapper_fixed.predict(mos_pred)
                cov_f = compute_coverage_width(mos_true, lo_f, hi_f)

                # Adaptive conformal
                lo_a, hi_a = wrapper_adapt.predict(mos_pred, mahal)
                cov_a = compute_coverage_width(mos_true, lo_a, hi_a)

                print(f"    {dataset:25s}  fixed={cov_f['coverage']:.3f}/{cov_f['mean_width']:.2f}"
                      f"  adapt={cov_a['coverage']:.3f}/{cov_a['mean_width']:.2f}")

                for method, cov in [("conformal_fixed", cov_f), ("conformal_adaptive", cov_a)]:
                    rows.append({
                        "model": model,
                        "dataset": dataset,
                        "alpha": alpha,
                        "nominal_coverage": nominal,
                        "method": method,
                        "coverage": cov["coverage"],
                        "mean_width": cov["mean_width"],
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-alpha conformal coverage evaluation.")
    p.add_argument("--features_dir", default="../../results/features")
    p.add_argument("--mahal_dir", default="../../results/mahal")
    p.add_argument("--out_dir", default="../../results")
    p.add_argument("--models", nargs="+", default=["nisqa", "dnsmos", "utmos"])
    p.add_argument("--datasets", nargs="+", default=_SHIFT_ORDER)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_multi_alpha(
        features_dir=Path(args.features_dir),
        mahal_dir=Path(args.mahal_dir),
        models=args.models,
        datasets=args.datasets,
        alpha_values=ALPHA_VALUES,
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"multi_alpha_coverage_{ts}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[multi_alpha] Saved CSV -> {csv_path}")

    # Summary: average coverage per (alpha, method) across models and datasets
    print("\n--- Average coverage (across models × datasets) ---")
    summary = df.pivot_table(
        index=["alpha", "method"],
        values=["coverage", "mean_width"],
        aggfunc="mean",
    ).round(3)
    print(summary.to_string())

    # Summary by alpha on calibration set only
    print("\n--- Calibration set coverage (should match nominal) ---")
    cal = df[df["dataset"] == _CAL_DATASET]
    if not cal.empty:
        cal_summary = cal.pivot_table(
            index=["alpha", "method"],
            values=["coverage", "mean_width"],
            aggfunc="mean",
        ).round(3)
        print(cal_summary.to_string())

    # Summary on strong OOD sets
    ood_strong = df[df["dataset"].isin(["nisqa_test_for", "nisqa_test_livetalk", "nisqa_test_p501"])]
    if not ood_strong.empty:
        print("\n--- Strong OOD average coverage ---")
        ood_summary = ood_strong.pivot_table(
            index=["alpha", "method"],
            values=["coverage", "mean_width"],
            aggfunc="mean",
        ).round(3)
        print(ood_summary.to_string())


if __name__ == "__main__":
    main()
