"""Tiny, safe pilot: validate the URGENT2025-SQA pipeline end-to-end on a
handful of clips before committing to any larger download.

Streams (does NOT bulk-download the 36 GB dataset) urgent-challenge/
urgent2025-sqa's blind_test_mos split via the `datasets` library, saves only
the first N_PILOT clips that have a non-null human MOS to disk, and writes a
data_csv compatible with the existing extract_features.py loader (columns:
filepath, mos, dataset). Does not touch any canonical/paper artifact.

Usage:
  python exploratory_urgent2025_pilot_download.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_AUDIO_DIR = PROJECT_ROOT / "data" / "URGENT2025_SQA_pilot" / "audio"
OUT_CSV = PROJECT_ROOT / "results" / "urgent2025_sqa_pilot.csv"
N_PILOT = 24


def main() -> None:
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    stream = load_dataset("urgent-challenge/urgent2025-sqa", split="blind_test_mos", streaming=True)
    # Decode raw bytes ourselves via soundfile; torchcodec (the datasets 4.x
    # default audio decoder) requires an FFmpeg shared-library setup this
    # machine does not have.
    stream = stream.cast_column("audio", Audio(decode=False))

    rows = []
    for row in stream:
        if row.get("mos") is None:
            continue
        sample_id = row["sample_id"]
        raw_audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
        out_path = OUT_AUDIO_DIR / f"{sample_id}.flac"
        sf.write(out_path, raw_audio, sr)
        rows.append({
            "filepath": str(out_path),
            "sample_id": sample_id,
            "system_id": row["system_id"],
            "mos": float(row["mos"]),
            "dataset": "urgent2025_sqa_pilot",
            "ref_dnsmos_ovrl": row.get("dnsmos_ovrl"),
            "ref_nisqa_mos": row.get("nisqa_mos"),
            "ref_utmos": row.get("utmos"),
        })
        print(f"  [{len(rows)}/{N_PILOT}] {sample_id} mos={row['mos']:.3f} sr={sr}")
        if len(rows) >= N_PILOT:
            break

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(rows)} clips -> {OUT_AUDIO_DIR}")
    print(f"Manifest -> {OUT_CSV}")


if __name__ == "__main__":
    main()
