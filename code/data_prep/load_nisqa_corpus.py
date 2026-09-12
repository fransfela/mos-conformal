"""
load_nisqa_corpus.py — Load NISQA Corpus (Mittag et al., INTERSPEECH 2021).

The NISQA Corpus (https://github.com/gabrielmittag/NISQA) is available from
Zenodo at https://zenodo.org/record/4728081. After extraction, the root
directory contains:

  NISQA_Corpus/
    NISQA_corpus_file.csv    ← file-level MOS (14 432 rows, 7 subsets)
    NISQA_corpus_con.csv     ← condition-level MOS (averaged)
    NISQA_TRAIN_SIM/deg/     ← 10 000 simulated clips
    NISQA_TRAIN_LIVE/deg/    ← 1 020 live-recorded clips
    NISQA_VAL_SIM/deg/       ← 2 500 simulated (validation)
    NISQA_VAL_LIVE/deg/      ← 200  live (validation)
    NISQA_TEST_FOR/deg/      ← 240  foreign conditions
    NISQA_TEST_LIVETALK/deg/ ← 232  live-talk conditions
    NISQA_TEST_P501/deg/     ← 240  ITU-T P.501 conditions

The script outputs a CSV with columns:
  filepath    : str   — absolute path to degraded audio file
  condition   : str   — NISQA condition description
  mos         : float — file-level mean opinion score
  dataset     : str   — e.g. "nisqa_train_sim", "nisqa_test_for"
  split       : str   — "train" | "calibration" | "test"

Role assignments (used by run_pipeline.py):
  nisqa_train_sim   → in-distribution  (train / fit Mahalanobis)
  nisqa_train_live  → calibration      (conformal calibration)
  nisqa_val_sim     → mild OOD test
  nisqa_val_live    → mild OOD test
  nisqa_test_for    → moderate OOD test
  nisqa_test_livetalk → strong OOD test
  nisqa_test_p501   → strong OOD test (ITU-T standardised)

Usage:
  # Generate one CSV per subset:
  python load_nisqa_corpus.py --root /path/to/NISQA_Corpus --subset nisqa_train_sim --out ../../results/nisqa_train_sim.csv

  # Generate ALL subsets at once (recommended):
  python load_nisqa_corpus.py --root /path/to/NISQA_Corpus --subset all --out_dir ../../results
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Map command-line subset names → db column values in the CSV
_SUBSET_MAP = {
    "nisqa_train_sim":    "NISQA_TRAIN_SIM",
    "nisqa_train_live":   "NISQA_TRAIN_LIVE",
    "nisqa_val_sim":      "NISQA_VAL_SIM",
    "nisqa_val_live":     "NISQA_VAL_LIVE",
    "nisqa_test_for":     "NISQA_TEST_FOR",
    "nisqa_test_livetalk": "NISQA_TEST_LIVETALK",
    "nisqa_test_p501":    "NISQA_TEST_P501",
}

# Role of each subset in the OOD experiment
_SUBSET_SPLIT = {
    "nisqa_train_sim":    "train",
    "nisqa_train_live":   "calibration",
    "nisqa_val_sim":      "test",
    "nisqa_val_live":     "test",
    "nisqa_test_for":     "test",
    "nisqa_test_livetalk": "test",
    "nisqa_test_p501":    "test",
}


def load_subset(root: Path, subset_name: str) -> pd.DataFrame:
    """
    Load one NISQA corpus subset into a normalised DataFrame.

    Parameters
    ----------
    root        : path to the NISQA_Corpus directory
    subset_name : one of the keys in _SUBSET_MAP (e.g. "nisqa_train_sim")

    Returns
    -------
    DataFrame with columns: filepath, condition, mos, dataset, split
    """
    csv_path = root / "NISQA_corpus_file.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"NISQA_corpus_file.csv not found under {root}. "
            "Download and extract the NISQA Corpus from https://zenodo.org/record/4728081"
        )

    db_name = _SUBSET_MAP[subset_name]
    df_all = pd.read_csv(csv_path)
    df = df_all[df_all["db"] == db_name].copy()

    if df.empty:
        raise ValueError(f"No rows found for db='{db_name}' in {csv_path}")

    # Resolve absolute paths: filepath_deg is relative to root
    df["filepath"] = df["filepath_deg"].apply(
        lambda p: str(root / p)
    )

    # Verify at least some files exist (catch wrong root path early)
    sample = df["filepath"].iloc[0]
    if not os.path.isfile(sample):
        raise FileNotFoundError(
            f"Audio file not found: {sample}\n"
            f"Check that --root points to the extracted NISQA_Corpus directory."
        )

    df["condition"] = df["con_description"].fillna(df["con"].astype(str))
    df["dataset"]   = subset_name
    df["split"]     = _SUBSET_SPLIT[subset_name]

    return df[["filepath", "condition", "mos", "dataset", "split"]].reset_index(drop=True)


def load_all(root: Path) -> dict[str, pd.DataFrame]:
    """Load all subsets. Returns {subset_name: DataFrame}."""
    results = {}
    for name in _SUBSET_MAP:
        try:
            df = load_subset(root, name)
            results[name] = df
            print(f"  {name:25s}: {len(df):5d} clips, MOS {df['mos'].mean():.2f} ± {df['mos'].std():.2f}")
        except Exception as exc:
            print(f"  [WARN] Could not load {name}: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load NISQA Corpus into normalised CSV(s) for the OOD pipeline."
    )
    p.add_argument(
        "--root", required=True,
        help="Path to the extracted NISQA_Corpus directory."
    )
    p.add_argument(
        "--subset", default="all",
        choices=["all"] + list(_SUBSET_MAP.keys()),
        help="Which subset to load. Use 'all' to generate one CSV per subset (requires --out_dir)."
    )
    p.add_argument(
        "--out", default=None,
        help="Output CSV path (single subset only)."
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Output directory for all subsets (used with --subset all)."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.root)

    if args.subset == "all":
        if args.out_dir is None:
            print("ERROR: --out_dir is required when --subset all is used.", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading all NISQA corpus subsets from {root} ...")
        subsets = load_all(root)
        for name, df in subsets.items():
            out_csv = out_dir / f"{name}.csv"
            df.to_csv(out_csv, index=False)
            print(f"  → {out_csv}")
    else:
        if args.out is None:
            print("ERROR: --out is required when loading a single subset.", file=sys.stderr)
            sys.exit(1)
        out_csv = Path(args.out)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        print(f"Loading {args.subset} from {root} ...")
        df = load_subset(root, args.subset)
        df.to_csv(out_csv, index=False)
        print(f"  {len(df)} clips → {out_csv}")


if __name__ == "__main__":
    main()
