"""
load_odaq.py — Load the ODAQ (Object-based audio quality) dataset.

Reference: Torcoli et al. (2021), IEEE Trans. Audio, Speech, Lang. Process.
Dataset: https://zenodo.org/record/5500022  (or later Zenodo version)

Expected structure after download:
  <root>/
    audio/
      <item_id>/
        <condition>.wav
    scores/
      ODAQ_scores.csv   (or ODAQ_MOS.csv)

ODAQ uses a MUSHRA-like continuous quality scale (0–100). We normalise
to a 1–5 MOS scale by the linear transform: mos = 1 + (score / 100) * 4
so that 0 → 1.0 and 100 → 5.0.

Output DataFrame columns:
  filepath  : str   — absolute path to audio file
  condition : str   — ODAQ processing condition label
  mos       : float — normalised MOS on [1, 5] scale
  mos_raw   : float — original ODAQ score on [0, 100] scale
  dataset   : str   — "odaq"
  split     : str   — "ood" (ODAQ is always cross-domain OOD)

Usage:
  python load_odaq.py --root /data/ODAQ --out ../../results/odaq.csv
"""

import argparse
from pathlib import Path

import pandas as pd


_ODAQ_SCALE_MIN = 0.0
_ODAQ_SCALE_MAX = 100.0
_MOS_MIN = 1.0
_MOS_MAX = 5.0


def _normalise_to_mos(score: float) -> float:
    """Linear mapping: ODAQ [0, 100] → MOS [1, 5]."""
    return _MOS_MIN + (score - _ODAQ_SCALE_MIN) / (
        _ODAQ_SCALE_MAX - _ODAQ_SCALE_MIN
    ) * (_MOS_MAX - _MOS_MIN)


def load_odaq(root: str | Path) -> pd.DataFrame:
    """
    Load ODAQ dataset and return a normalised DataFrame.

    Parameters
    ----------
    root : path to the ODAQ root directory

    Returns
    -------
    pd.DataFrame with columns: filepath, condition, mos, mos_raw, dataset, split
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ODAQ root not found: {root}")

    csv_candidates = (
        sorted(root.rglob("*ODAQ*.csv"))
        + sorted(root.rglob("*odaq*.csv"))
        + sorted(root.rglob("*score*.csv"))
        + sorted(root.rglob("*mos*.csv"))
        + sorted(root.rglob("*MOS*.csv"))
    )
    if not csv_candidates:
        raise FileNotFoundError(
            f"No score CSV found under {root}. "
            "Expected ODAQ_scores.csv or similar under <root>/scores/."
        )

    df_raw = pd.read_csv(csv_candidates[0])
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    col_map = {
        "filename": "filepath",
        "file": "filepath",
        "audio": "filepath",
        "filepath": "filepath",
        "item": "filepath",
        "mean_score": "mos_raw",
        "score": "mos_raw",
        "mean_mos": "mos_raw",
        "mos": "mos_raw",
        "condition": "condition",
        "processing": "condition",
        "method": "condition",
        "codec": "condition",
    }
    df_raw = df_raw.rename(columns={k: v for k, v in col_map.items() if k in df_raw.columns})

    required = {"filepath", "mos_raw"}
    missing = required - set(df_raw.columns)
    if missing:
        raise KeyError(
            f"[ODAQ] Required columns {missing} not found. "
            f"Available columns: {list(df_raw.columns)}. "
            "Update col_map in load_odaq.py."
        )

    if "condition" not in df_raw.columns:
        df_raw["condition"] = "unknown"

    df_raw["filepath"] = df_raw["filepath"].apply(
        lambda p: str(root / p) if not Path(str(p)).is_absolute() else str(p)
    )
    df_raw = df_raw.dropna(subset=["filepath", "mos_raw"]).copy()
    df_raw["mos_raw"] = df_raw["mos_raw"].astype(float)

    # Detect scale: if all values are in [1, 5] already, skip normalisation
    if df_raw["mos_raw"].max() <= 5.0 and df_raw["mos_raw"].min() >= 1.0:
        df_raw["mos"] = df_raw["mos_raw"]
    else:
        df_raw["mos"] = df_raw["mos_raw"].apply(_normalise_to_mos)

    df_raw["dataset"] = "odaq"
    df_raw["split"] = "ood"
    return df_raw[
        ["filepath", "condition", "mos", "mos_raw", "dataset", "split"]
    ].reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load ODAQ dataset MOS data to CSV.")
    p.add_argument("--root", required=True, help="Path to ODAQ root directory.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_odaq(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[load_odaq] Saved {len(df)} samples → {out}")


if __name__ == "__main__":
    main()
