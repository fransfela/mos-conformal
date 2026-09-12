"""Scaled (still bounded) pilot: deterministic, diverse sample from
urgent-challenge/urgent2025-sqa's blind_test_mos split, across systems.

Single streaming pass (no bulk download of the 36 GB dataset): for each row,
a clip is kept only if SHA-256(seed|sample_id) falls under a fixed inclusion
probability, calibrated so the expected total is ~N_TARGET. This mirrors the
"rank by SHA-256 of seed|clip_id, never look at the label" selection already
used for the 2024 pilot (results/urgent2024_mos_external.manifest.json) --
deterministic and reproducible, and does not peek at MOS when selecting.

Usage:
  python exploratory_urgent2025_scaled_download.py
"""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_AUDIO_DIR = PROJECT_ROOT / "data" / "URGENT2025_SQA_scaled" / "audio"
OUT_CSV = PROJECT_ROOT / "results" / "urgent2025_sqa_scaled.csv"
SEED = 20270903
N_TARGET = 300
ESTIMATED_SPLIT_SIZE = 23_400  # per the urgent2025-sqa dataset card
INCLUSION_PROB = N_TARGET / ESTIMATED_SPLIT_SIZE
HASH_MOD = 1_000_000


def include(sample_id: str) -> bool:
    h = int(hashlib.sha256(f"{SEED}|{sample_id}".encode()).hexdigest(), 16) % HASH_MOD
    return h < int(INCLUSION_PROB * HASH_MOD)


def main() -> None:
    OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[selection] seed={SEED} inclusion_prob={INCLUSION_PROB:.5f} target~{N_TARGET}")

    stream = load_dataset("urgent-challenge/urgent2025-sqa", split="blind_test_mos", streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))

    rows = []
    n_seen = 0
    t0 = time.time()
    for row in stream:
        n_seen += 1
        if row.get("mos") is None or not include(row["sample_id"]):
            if n_seen % 2000 == 0:
                print(f"  ... scanned {n_seen} rows, kept {len(rows)}, {time.time()-t0:.0f}s elapsed")
            continue
        raw_audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=False)
        out_path = OUT_AUDIO_DIR / f"{row['sample_id']}.flac"
        sf.write(out_path, raw_audio, sr)
        rows.append({
            "filepath": str(out_path),
            "sample_id": row["sample_id"],
            "system_id": row["system_id"],
            "mos": float(row["mos"]),
            "dataset": "urgent2025_sqa_scaled",
            "ref_dnsmos_ovrl": row.get("dnsmos_ovrl"),
            "ref_nisqa_mos": row.get("nisqa_mos"),
            "ref_utmos": row.get("utmos"),
        })
        if len(rows) % 25 == 0:
            print(f"  [{len(rows)}] scanned {n_seen} rows, {time.time()-t0:.0f}s elapsed")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nScanned {n_seen} rows in {time.time()-t0:.0f}s")
    print(f"Saved {len(df)} clips -> {OUT_AUDIO_DIR}")
    print(f"Manifest -> {OUT_CSV}")
    print(df["system_id"].value_counts())


if __name__ == "__main__":
    main()
