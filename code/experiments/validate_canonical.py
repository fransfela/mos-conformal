"""Fail-fast integrity checks for the canonical experiment bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from canonical_analysis import exact_conformal_radius


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "results" / "canonical"
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    assert set(manifest["outputs"]) == set(manifest["output_sha256"])
    for name, expected in manifest["output_sha256"].items():
        assert sha256(cache_dir / name) == expected, f"Hash mismatch: {name}"

    split = pd.read_csv(cache_dir / "val_sim_split.csv")
    assert len(split) == manifest["calibration_n"] + manifest["id_evaluation_n"]
    assert set(split["role"]) == {"calibration", "id_evaluation"}
    assert split.groupby("group")["role"].nunique().max() == 1

    assert exact_conformal_radius(np.arange(1, 11), alpha=0.2) == 9

    required_csvs = {
        "interval_sensitivity.csv": ["coverage", "mean_width", "mean_interval_score"],
        "error_detection.csv": ["auroc_prespecified_direction"],
        "risk_coverage.csv": ["rmse"],
        "domain_detection.csv": ["auroc", "fpr_at_95_tpr"],
        "bootstrap_interval_differences.csv": ["ci_lo", "ci_hi"],
        "bootstrap_pooled_error.csv": ["ci_lo", "ci_hi"],
        "split_sensitivity.csv": ["delta_interval_score"],
        "supervised_difficulty_intervals.csv": ["coverage", "mean_interval_score"],
        "supervised_difficulty_ranking.csv": ["auroc_top_error_quartile"],
        "bootstrap_supervised_differences.csv": [
            "method_minus_fixed", "ci_lo", "ci_hi"
        ],
        "supervised_split_sensitivity.csv": [
            "delta_coverage", "delta_width", "delta_interval_score"
        ],
        "supervised_split_summary.csv": [
            "mean_delta_interval_score", "fraction_improved"
        ],
        "locked_method_evaluation.csv": [
            "selected_minus_fixed", "conditional_ci_lo", "conditional_ci_hi"
        ],
        "error_detection_domain_bootstrap.csv": ["auroc", "ci_low", "ci_high"],
    }
    for name, columns in required_csvs.items():
        frame = pd.read_csv(cache_dir / name)
        assert len(frame) > 0, f"Empty result: {name}"
        for column in columns:
            assert column in frame, f"Missing {column} in {name}"
            assert np.isfinite(frame[column]).all(), f"Non-finite {column} in {name}"

    intervals = pd.read_csv(cache_dir / "interval_sensitivity.csv")
    assert intervals["coverage"].between(0, 1).all()
    assert (intervals["mean_width"] >= 0).all()
    error = pd.read_csv(cache_dir / "error_detection.csv")
    assert error["auroc_prespecified_direction"].between(0, 1).all()
    domain = pd.read_csv(cache_dir / "domain_detection.csv")
    assert domain["auroc"].between(0, 1).all()
    assert domain["fpr_at_95_tpr"].between(0, 1).all()

    for model in ("dnsmos", "nisqa", "utmos"):
        for dataset in (
            "nisqa_train_sim", "nisqa_train_live", "nisqa_val_sim", "nisqa_val_live",
            "nisqa_test_for", "nisqa_test_livetalk", "nisqa_test_p501",
        ):
            loaded = np.load(
                cache_dir / "scores" / f"scores_{model}_{dataset}.npz",
                allow_pickle=True,
            )
            n = len(loaded["mos"])
            for key in ("mos_pred", "mahalanobis", "l2", "knn", "mos_std", "votes", "filepaths"):
                assert len(loaded[key]) == n, f"Length mismatch: {model}/{dataset}/{key}"
            for key in ("mos", "mos_pred", "mahalanobis", "l2", "knn", "mos_std", "votes"):
                assert np.isfinite(loaded[key]).all(), f"Non-finite cache: {model}/{dataset}/{key}"

    print("[validate_canonical] All integrity checks passed")


if __name__ == "__main__":
    main()
