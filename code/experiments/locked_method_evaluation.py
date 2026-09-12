"""Select an interval method on V-SIM only and evaluate it unchanged on shifts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from canonical_analysis import ID_DATASET, SHIFTED_DATASETS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "results" / "canonical"
    intervals = pd.read_csv(cache_dir / "supervised_difficulty_intervals.csv")
    bootstrap = pd.read_csv(cache_dir / "bootstrap_supervised_differences.csv")
    rows = []

    for model, model_frame in intervals.groupby("model", sort=False):
        selection = model_frame[model_frame["dataset"] == ID_DATASET]
        selected = selection.loc[selection["mean_interval_score"].idxmin(), "method"]
        fixed_selection_score = float(
            selection.loc[selection["method"] == "fixed", "mean_interval_score"].iloc[0]
        )
        selected_selection_score = float(
            selection.loc[selection["method"] == selected, "mean_interval_score"].iloc[0]
        )
        for dataset in SHIFTED_DATASETS:
            domain = model_frame[model_frame["dataset"] == dataset]
            fixed = domain[domain["method"] == "fixed"].iloc[0]
            chosen = domain[domain["method"] == selected].iloc[0]
            oracle = domain.loc[domain["mean_interval_score"].idxmin()]
            uncertainty = bootstrap[
                (bootstrap["model"] == model)
                & (bootstrap["dataset"] == dataset)
                & (bootstrap["method"] == selected)
                & (bootstrap["metric"] == "mean_interval_score")
            ].iloc[0]
            rows.append({
                "model": model,
                "selection_dataset": ID_DATASET,
                "selected_method": selected,
                "selection_fixed_interval_score": fixed_selection_score,
                "selection_selected_interval_score": selected_selection_score,
                "test_dataset": dataset,
                "fixed_coverage": fixed["coverage"],
                "selected_coverage": chosen["coverage"],
                "fixed_width": fixed["mean_width"],
                "selected_width": chosen["mean_width"],
                "fixed_interval_score": fixed["mean_interval_score"],
                "selected_interval_score": chosen["mean_interval_score"],
                "selected_minus_fixed": (
                    chosen["mean_interval_score"] - fixed["mean_interval_score"]
                ),
                "conditional_ci_lo": uncertainty["ci_lo"],
                "conditional_ci_hi": uncertainty["ci_hi"],
                "oracle_method": oracle["method"],
                "oracle_interval_score": oracle["mean_interval_score"],
                "selection_regret": (
                    chosen["mean_interval_score"] - oracle["mean_interval_score"]
                ),
            })

    output_path = cache_dir / "locked_method_evaluation.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["locked_method_evaluation"] = {
        "selection_rule": "minimum mean interval score on held-out V-SIM",
        "test_domains": list(SHIFTED_DATASETS),
        "bootstrap_conditioning": "selected method held fixed",
    }
    manifest["outputs"] = list(dict.fromkeys([*manifest["outputs"], output_path.name]))
    manifest.setdefault("code_sha256", {})[
        "locked_method_evaluation.py"
    ] = sha256(Path(__file__).resolve())
    manifest["output_sha256"] = {
        name: sha256(cache_dir / name) for name in manifest["outputs"]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[locked-method] Complete")


if __name__ == "__main__":
    main()
