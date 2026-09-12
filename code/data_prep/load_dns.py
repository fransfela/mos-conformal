"""
load_dns.py — Load DNS Challenge MOS data (2020 / 2021 / 2022 / 2024).

Each DNS Challenge release follows roughly the same structure:
  <root>/
    MOS_testset/        (or subjective_testset/)
      <condition>/
        *.wav           (degraded / enhanced speech)
    results/
      <mos_scores>.csv  (or P.835 ratings in various formats)

The exact layout differs across years. This script normalises all
releases into a common pandas DataFrame with columns:
  filepath   : str   — absolute path to audio file
  condition  : str   — enhancement system / condition label
  mos        : float — mean opinion score (overall / P.835 OVRL)
  dataset    : str   — e.g. "dns2020", "dns2021", "dns2022", "dns2024"
  split      : str   — "train" | "calibration" | "test" (assigned externally)

Usage:
  python load_dns.py --root /data/DNS2020 --year 2020 --out ../../results/dns2020.csv
  python load_dns.py --root /data/DNS2021 --year 2021 --out ../../results/dns2021.csv
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Year-specific loaders
# ---------------------------------------------------------------------------

def _load_2020(root: Path) -> pd.DataFrame:
    """
    DNS Challenge 2020 (INTERSPEECH 2020 challenge).
    MOS CSV expected at <root>/MOS_testset/MOS_score_testset.csv
    or matching pattern *MOS*.csv under root.
    """
    csv_candidates = sorted(root.rglob("*MOS*.csv")) + sorted(root.rglob("*mos*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"No MOS CSV found under {root}. "
            "Expected a file matching *MOS*.csv or *mos*.csv."
        )
    df_raw = pd.read_csv(csv_candidates[0])
    # Normalise column names (DNS 2020 uses 'filename', 'MOS', 'Condition')
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]
    col_map = {
        "filename": "filepath",
        "file": "filepath",
        "audio": "filepath",
        "mos": "mos",
        "mean_mos": "mos",
        "ovrl_mos": "mos",
        "condition": "condition",
        "system": "condition",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})
    # Resolve absolute paths
    if "filepath" in df_raw.columns:
        df_raw["filepath"] = df_raw["filepath"].apply(
            lambda p: str(root / p) if not os.path.isabs(p) else p
        )
    if "condition" not in df_raw.columns:
        df_raw["condition"] = "unknown"
    return df_raw[["filepath", "condition", "mos"]]


def _load_2021(root: Path) -> pd.DataFrame:
    """
    DNS Challenge 2021 (ICASSP 2021 challenge).
    Same approach as 2020 but may use DNSMOS P.835 ratings.
    """
    return _load_generic(root, year=2021)


def _load_2022(root: Path) -> pd.DataFrame:
    return _load_generic(root, year=2022)


def _load_2024(root: Path) -> pd.DataFrame:
    return _load_generic(root, year=2024)


def _load_generic(root: Path, year: int) -> pd.DataFrame:
    """
    Generic loader for DNS 2021 / 2022 / 2024.
    Searches for any CSV under root containing MOS scores.
    """
    csv_candidates = (
        sorted(root.rglob("*P835*.csv"))
        + sorted(root.rglob("*MOS*.csv"))
        + sorted(root.rglob("*mos*.csv"))
        + sorted(root.rglob("*score*.csv"))
    )
    if not csv_candidates:
        raise FileNotFoundError(
            f"[DNS {year}] No MOS CSV found under {root}. "
            "Please supply the subjective test results CSV."
        )
    df_raw = pd.read_csv(csv_candidates[0])
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    # Map common column variants to canonical names
    col_map = {
        "filename": "filepath",
        "file": "filepath",
        "audio_file": "filepath",
        "filepath": "filepath",
        "ovrl_mos": "mos",
        "overall_mos": "mos",
        "mos": "mos",
        "mean_mos": "mos",
        "condition": "condition",
        "system": "condition",
        "model": "condition",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    required = {"filepath", "mos"}
    missing = required - set(df_raw.columns)
    if missing:
        raise KeyError(
            f"[DNS {year}] Required columns {missing} not found. "
            f"Available columns: {list(df_raw.columns)}. "
            "Check col_map in load_dns.py and add the correct mapping."
        )

    if "condition" not in df_raw.columns:
        df_raw["condition"] = "unknown"

    df_raw["filepath"] = df_raw["filepath"].apply(
        lambda p: str(root / p) if not os.path.isabs(str(p)) else str(p)
    )
    return df_raw[["filepath", "condition", "mos"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {2020: _load_2020, 2021: _load_2021, 2022: _load_2022, 2024: _load_2024}


def load_dns(root: str | Path, year: int) -> pd.DataFrame:
    """
    Load a DNS Challenge dataset and return a normalised DataFrame.

    Parameters
    ----------
    root : path to the DNS Challenge release root directory
    year : 2020 | 2021 | 2022 | 2024

    Returns
    -------
    pd.DataFrame with columns: filepath, condition, mos, dataset
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"DNS {year} root not found: {root}")
    if year not in _LOADERS:
        raise ValueError(f"Unsupported year {year}. Supported: {list(_LOADERS)}")

    df = _LOADERS[year](root)
    df = df.dropna(subset=["filepath", "mos"]).copy()
    df["mos"] = df["mos"].astype(float)
    df["dataset"] = f"dns{year}"
    df["split"] = "unassigned"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load DNS Challenge MOS data to CSV.")
    p.add_argument("--root", required=True, help="Path to DNS Challenge root directory.")
    p.add_argument("--year", required=True, type=int, choices=[2020, 2021, 2022, 2024])
    p.add_argument("--out", required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_dns(args.root, args.year)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[load_dns] Saved {len(df)} samples → {out}")


if __name__ == "__main__":
    main()
