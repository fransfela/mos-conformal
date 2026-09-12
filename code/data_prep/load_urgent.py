"""
load_urgent.py — Load URGENT 2026 blind test set MOS data.

URGENT 2026 challenge: https://urgent-challenge.github.io/urgent2026/
Expected structure:
  <root>/
    test/
      <system_id>/
        *.wav
    results/
      mos_scores.csv    (or similar — adapt col_map below)

Output DataFrame columns:
  filepath  : str   — absolute path to audio file
  condition : str   — system / team identifier
  mos       : float — mean opinion score (overall)
  dataset   : str   — "urgent2026"
  split     : str   — "test" (all URGENT data is held-out OOD)

Usage:
  python load_urgent.py --root /data/URGENT2026 --out ../../results/urgent2026.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def load_urgent(root: str | Path) -> pd.DataFrame:
    """
    Load URGENT 2026 blind test set.

    Parameters
    ----------
    root : path to the URGENT 2026 root directory

    Returns
    -------
    pd.DataFrame with columns: filepath, condition, mos, dataset, split
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"URGENT 2026 root not found: {root}")

    # Search for MOS CSV
    csv_candidates = (
        sorted(root.rglob("*mos*.csv"))
        + sorted(root.rglob("*MOS*.csv"))
        + sorted(root.rglob("*score*.csv"))
        + sorted(root.rglob("*result*.csv"))
    )
    if not csv_candidates:
        raise FileNotFoundError(
            f"No MOS CSV found under {root}. "
            "Download the URGENT 2026 subjective test results and place under <root>/results/."
        )

    df_raw = pd.read_csv(csv_candidates[0])
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    col_map = {
        "filename": "filepath",
        "file": "filepath",
        "audio_file": "filepath",
        "filepath": "filepath",
        "mos": "mos",
        "ovrl_mos": "mos",
        "overall_mos": "mos",
        "mean_mos": "mos",
        "system": "condition",
        "condition": "condition",
        "team": "condition",
        "model": "condition",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    required = {"filepath", "mos"}
    missing = required - set(df_raw.columns)
    if missing:
        raise KeyError(
            f"[URGENT 2026] Required columns {missing} not found. "
            f"Available columns: {list(df_raw.columns)}. "
            "Update col_map in load_urgent.py."
        )

    if "condition" not in df_raw.columns:
        df_raw["condition"] = "unknown"

    df_raw["filepath"] = df_raw["filepath"].apply(
        lambda p: str(root / p) if not Path(str(p)).is_absolute() else str(p)
    )
    df_raw = df_raw.dropna(subset=["filepath", "mos"]).copy()
    df_raw["mos"] = df_raw["mos"].astype(float)
    df_raw["dataset"] = "urgent2026"
    df_raw["split"] = "test"
    return df_raw[["filepath", "condition", "mos", "dataset", "split"]].reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load URGENT 2026 MOS data to CSV.")
    p.add_argument("--root", required=True, help="Path to URGENT 2026 root directory.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_urgent(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[load_urgent] Saved {len(df)} samples → {out}")


if __name__ == "__main__":
    main()
