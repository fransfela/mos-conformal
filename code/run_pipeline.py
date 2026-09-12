"""
run_pipeline.py — Orchestrate the full experiment pipeline end-to-end.

Runs the following steps in order:
  1.  Load dataset CSVs (DNS or NISQA corpus, depending on EXPERIMENT_MODE)
  2.  Extract features for each model × dataset
  3.  Fit Mahalanobis model on in-distribution features (per model)
  4.  Score all datasets with Mahalanobis distance (per model)
  5.  Calibrate conformal wrappers on calibration dataset (per model)
  6.  Run evaluation (PCC, SRCC, AUROC, coverage, width)
  7.  Generate paper figures
  8.  Print summary table to stdout

All intermediate results are saved to results/ so individual steps can
be re-run without re-running the full pipeline.

Prerequisites — set these paths before running:
  DATA_ROOTS dict below (paths to each dataset on your local filesystem)
  MODEL_PATHS dict below (paths to each model checkpoint / directory)
  EXPERIMENT_MODE: "nisqa" (NISQA corpus, default) or "dns" (DNS Challenge)

Usage:
  python run_pipeline.py [--skip_data_prep] [--skip_features] [--steps 1,2,3]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (relative to this script — do not change)
# ---------------------------------------------------------------------------

_CODE_ROOT   = Path(__file__).parent                      # code/
_PROJ_ROOT   = _CODE_ROOT.parent                          # 2027 ICASSP/
_RESULTS     = _PROJ_ROOT / "results"                     # results/
_PAPER_FIG   = _PROJ_ROOT / "paper" / "figures"           # paper/figures/
_DATA_PREP   = _CODE_ROOT / "data_prep"                   # code/data_prep/
_EXPERIMENTS = _CODE_ROOT / "experiments"                  # code/experiments/
_ANALYSIS    = _CODE_ROOT / "analysis"                    # code/analysis/

# ---------------------------------------------------------------------------
# !! CONFIGURE THESE PATHS BEFORE RUNNING !!
# ---------------------------------------------------------------------------

# Experiment mode: "nisqa" uses the NISQA Corpus (already downloaded);
#                  "dns"   uses DNS Challenge 2020–2024 (requires separate download).
EXPERIMENT_MODE = "nisqa"

DATA_ROOTS = {
    # Paper datasets (DNS = in-distribution → OOD ladder)
    "dns2020":       Path(r"C:\data\DNS2020"),
    "dns2021":       Path(r"C:\data\DNS2021"),
    "dns2022":       Path(r"C:\data\DNS2022"),
    "dns2024":       Path(r"C:\data\DNS2024"),
    "urgent2026":    Path(r"C:\data\URGENT2026"),
    "odaq":          Path(r"C:\data\ODAQ"),
    # NISQA corpus (downloaded to data/ — used when EXPERIMENT_MODE="nisqa")
    "nisqa_corpus":  _PROJ_ROOT / "data" / "NISQA_Corpus",
}

MODEL_PATHS = {
    "dnsmos": _PROJ_ROOT / "models" / "DNSMOS",       # directory with sig_bak_ovr.onnx
    "nisqa":  _PROJ_ROOT / "models" / "NISQA",        # cloned NISQA repo (weights/nisqa.tar inside)
    "utmos":  _PROJ_ROOT / "models" / "UTMOS",         # UTMOS demo dir with .ckpt
}

_MODELS = ["dnsmos", "nisqa", "utmos"]

# Dataset sequences per experiment mode.
# First entry  = in-distribution (Mahalanobis fit)
# Second entry = calibration     (conformal calibration)
# Rest         = OOD test sets   (evaluation ladder)
_DATASET_SEQUENCES = {
    "dns": ["dns2020", "dns2021", "dns2022", "dns2024", "urgent2026", "odaq"],
    "nisqa": [
        "nisqa_train_sim",    # in-distribution (simulated conditions)
        "nisqa_train_live",   # calibration     (live recordings, mild shift)
        "nisqa_val_sim",      # test: mild OOD
        "nisqa_val_live",     # test: mild OOD
        "nisqa_test_for",     # test: moderate OOD (foreign conditions)
        "nisqa_test_livetalk", # test: strong OOD
        "nisqa_test_p501",    # test: strong OOD (ITU-T P.501)
    ],
}

_DATASETS      = _DATASET_SEQUENCES[EXPERIMENT_MODE]
_TRAIN_DATASET = _DATASETS[0]   # in-distribution
_CAL_DATASET   = _DATASETS[1]   # calibration

# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], step: str) -> None:
    print(f"\n{'─'*60}")
    print(f"STEP: {step}")
    print(f"CMD : {' '.join(str(c) for c in cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[WARN] Step '{step}' exited with code {result.returncode}. "
              "Check output above. Continuing pipeline.", file=sys.stderr)


def py(script: Path, args: list[str], step: str) -> None:
    run([sys.executable, str(script)] + args, step)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_data_prep() -> None:
    """Step 1: Load all datasets to CSV (mode-dependent)."""
    _RESULTS.mkdir(parents=True, exist_ok=True)

    if EXPERIMENT_MODE == "nisqa":
        # NISQA corpus — generate all 7 subsets at once
        corpus_root = DATA_ROOTS["nisqa_corpus"]
        if corpus_root.exists():
            py(_DATA_PREP / "load_nisqa_corpus.py", [
                "--root", str(corpus_root),
                "--subset", "all",
                "--out_dir", str(_RESULTS),
            ], "load_nisqa_corpus all")
        else:
            print(f"[SKIP] NISQA corpus not found: {corpus_root}")

    else:  # dns mode
        # DNS years
        for year, ds_name in [(2020, "dns2020"), (2021, "dns2021"), (2022, "dns2022"), (2024, "dns2024")]:
            if DATA_ROOTS[ds_name].exists():
                py(_DATA_PREP / "load_dns.py", [
                    "--root", str(DATA_ROOTS[ds_name]),
                    "--year", str(year),
                    "--out", str(_RESULTS / f"{ds_name}.csv"),
                ], f"load_dns {year}")
            else:
                print(f"[SKIP] {ds_name} root not found: {DATA_ROOTS[ds_name]}")

        # URGENT 2026
        if DATA_ROOTS["urgent2026"].exists():
            py(_DATA_PREP / "load_urgent.py", [
                "--root", str(DATA_ROOTS["urgent2026"]),
                "--out", str(_RESULTS / "urgent2026.csv"),
            ], "load_urgent")
        else:
            print(f"[SKIP] URGENT 2026 root not found: {DATA_ROOTS['urgent2026']}")

        # ODAQ
        if DATA_ROOTS["odaq"].exists():
            py(_DATA_PREP / "load_odaq.py", [
                "--root", str(DATA_ROOTS["odaq"]),
                "--out", str(_RESULTS / "odaq.csv"),
            ], "load_odaq")
        else:
            print(f"[SKIP] ODAQ root not found: {DATA_ROOTS['odaq']}")


def step_extract_features() -> None:
    """Step 4: Extract penultimate-layer features for all model × dataset combinations."""
    feat_dir = _RESULTS / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for model in _MODELS:
        if not MODEL_PATHS[model].exists():
            print(f"[SKIP] Model path not found: {MODEL_PATHS[model]}")
            continue
        for dataset in _DATASETS:
            csv = _RESULTS / f"{dataset}.csv"
            if not csv.exists():
                print(f"[SKIP] Data CSV not found: {csv}")
                continue
            py(_EXPERIMENTS / "extract_features.py", [
                "--model", model,
                "--model_path", str(MODEL_PATHS[model]),
                "--data_csv", str(csv),
                "--out_dir", str(feat_dir),
            ], f"extract_features {model}/{dataset}")


def step_fit_mahalanobis() -> None:
    """Step 3: Fit Mahalanobis model on in-distribution training features."""
    mahal_dir = _RESULTS / "mahal"
    mahal_dir.mkdir(parents=True, exist_ok=True)

    for model in _MODELS:
        feat_path = _RESULTS / "features" / f"features_{model}_{_TRAIN_DATASET}.npz"
        if not feat_path.exists():
            print(f"[SKIP] Training features not found: {feat_path}")
            continue
        py(_EXPERIMENTS / "fit_mahalanobis.py", [
            "--features", str(feat_path),
            "--model", model,
            "--out_dir", str(mahal_dir),
        ], f"fit_mahalanobis {model}")


def step_calibrate_conformal() -> None:
    """Step 5: Calibrate conformal wrappers on calibration dataset."""
    conf_dir = _RESULTS / "conformal"
    conf_dir.mkdir(parents=True, exist_ok=True)

    for model in _MODELS:
        pred_csv = _RESULTS / "features" / f"features_{model}_{_CAL_DATASET}.csv"
        mahal_csv = _RESULTS / "mahal" / f"mahal_{model}_{_CAL_DATASET}_scores.csv"

        if not pred_csv.exists():
            print(f"[SKIP] Calibration feature CSV not found: {pred_csv}")
            continue
        if not mahal_csv.exists():
            print(f"[INFO] Mahalanobis scores for {_CAL_DATASET}/{model} not found. "
                  "Run score_all_datasets step first.")
            continue

        py(_EXPERIMENTS / "conformal_wrapper.py", [
            "--pred_csv", str(pred_csv),
            "--mahal_csv", str(mahal_csv),
            "--alpha", "0.10",
            "--n_bins", "5",
            "--out", str(conf_dir / f"wrapper_{model}.pkl"),
        ], f"calibrate_conformal {model}")


def step_evaluate() -> None:
    """Step 8: Run evaluation metrics."""
    py(_EXPERIMENTS / "evaluate.py", [
        "--features_dir", str(_RESULTS / "features"),
        "--mahal_dir",    str(_RESULTS / "mahal"),
        "--wrapper_dir",  str(_RESULTS / "conformal"),
        "--out_dir",      str(_RESULTS),
        "--models"] + _MODELS + [
        "--datasets"] + _DATASETS,
        "evaluate",
    )


def step_plot() -> None:
    """Step 9: Generate paper figures."""
    _PAPER_FIG.mkdir(parents=True, exist_ok=True)
    py(_ANALYSIS / "plot_results.py", [
        "--features_dir", str(_RESULTS / "features"),
        "--mahal_dir",    str(_RESULTS / "mahal"),
        "--eval_dir",     str(_RESULTS),
        "--out_dir",      str(_PAPER_FIG),
        "--models"] + _MODELS + [
        "--datasets"] + _DATASETS,
        "plot_results",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STEPS = {
    "data_prep":         step_data_prep,
    "extract_features":  step_extract_features,
    "fit_mahalanobis":   step_fit_mahalanobis,
    "calibrate":         step_calibrate_conformal,
    "evaluate":          step_evaluate,
    "plot":              step_plot,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full ICASSP 2027 experiment pipeline.")
    p.add_argument(
        "--steps", nargs="+", choices=list(_STEPS),
        default=list(_STEPS),
        help="Which steps to run (default: all).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print("ICASSP 2027 — OOD MOS Detection Experiment Pipeline")
    print("=" * 60)
    for step_name in args.steps:
        _STEPS[step_name]()
    print("\n[run_pipeline] Pipeline complete.")


if __name__ == "__main__":
    main()
