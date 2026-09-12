"""Download a deterministic, label-blind subset of URGENT 2024 MOS.

The public Hugging Face release contains 6,900 enhanced-speech clips from 23
systems.  To keep external validation computationally tractable, this script
selects ``n_per_system`` clips per system by a fixed SHA-256 ranking of clip
IDs.  MOS values are never used for selection.

The resulting CSV follows the schema consumed by ``extract_features.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


DATASET = "urgent-challenge/urgent2024_mos"
CONFIG = "default"
SPLIT = "test"
TOTAL_ROWS = 6900
PAGE_SIZE = 100
SEED = 20270818


def _get_json(url: str, attempts: int = 10) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt + 1 == attempts:
                raise
            retry_after = int(exc.headers.get("Retry-After", "30"))
            time.sleep(max(retry_after, 2**attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _get_page(offset: int, length: int, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"rows_{offset:04d}_{length}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = _get_json(_rows_url(offset, length))
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    time.sleep(0.5)
    return payload


def _download(url: str, target: Path, attempts: int = 5) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(target)
            return
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)


def _rows_url(offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return f"https://datasets-server.huggingface.co/rows?{query}"


def _system_id(clip_id: str) -> str:
    return clip_id.split("-", maxsplit=1)[0]


def _rank(clip_id: str) -> str:
    return hashlib.sha256(f"{SEED}|{clip_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n_per_system", type=int, default=10)
    args = parser.parse_args()

    if args.n_per_system < 1:
        raise ValueError("--n_per_system must be positive")

    rows: list[dict] = []
    cache_dir = args.root / "metadata_pages"
    for offset in range(0, TOTAL_ROWS, PAGE_SIZE):
        length = min(PAGE_SIZE, TOTAL_ROWS - offset)
        payload = _get_page(offset, length, cache_dir)
        for item in payload["rows"]:
            row = item["row"]
            audio = row["audio"]
            if not audio:
                raise RuntimeError(f"No audio URL for row {item['row_idx']}")
            rows.append(
                {
                    "row_idx": int(item["row_idx"]),
                    "id": row["id"],
                    "condition": _system_id(row["id"]),
                    "mos": float(row["mos"]),
                    "raw_ratings": json.dumps(row["raw_ratings"]),
                    "audio_url": audio[0]["src"],
                }
            )
        print(f"Metadata: {len(rows)}/{TOTAL_ROWS}")

    if len(rows) != TOTAL_ROWS:
        raise RuntimeError(f"Expected {TOTAL_ROWS} rows, received {len(rows)}")

    by_system: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_system[row["condition"]].append(row)

    selected: list[dict] = []
    for system, system_rows in sorted(by_system.items()):
        ranked = sorted(system_rows, key=lambda row: (_rank(row["id"]), row["id"]))
        if len(ranked) < args.n_per_system:
            raise RuntimeError(f"System {system} has only {len(ranked)} rows")
        selected.extend(ranked[: args.n_per_system])

    audio_dir = args.root / "audio"
    output_rows = []
    for index, row in enumerate(sorted(selected, key=lambda item: item["id"]), start=1):
        filepath = (audio_dir / f"{row['id']}.flac").resolve()
        _download(row["audio_url"], filepath)
        output_rows.append(
            {
                "filepath": str(filepath),
                "condition": row["condition"],
                "mos": row["mos"],
                "dataset": "urgent2024_mos_external",
                "split": "external_test",
                "id": row["id"],
                "row_idx": row["row_idx"],
                "raw_ratings": row["raw_ratings"],
                "selection_rank_sha256": _rank(row["id"]),
            }
        )
        print(f"Audio: {index}/{len(selected)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    selected_ids = "\n".join(row["id"] for row in output_rows).encode("utf-8")
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "source_total_rows": TOTAL_ROWS,
        "selection": "lowest SHA-256 rank of seed|clip_id within each system",
        "seed": SEED,
        "n_systems": len(by_system),
        "n_per_system": args.n_per_system,
        "n_selected": len(output_rows),
        "selected_ids_sha256": hashlib.sha256(selected_ids).hexdigest(),
        "csv_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "labels_used_for_selection": False,
        "role": "external_test_only",
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Saved {len(output_rows)} clips from {len(by_system)} systems to {args.out} "
        f"(seed={SEED}, n_per_system={args.n_per_system})"
    )


if __name__ == "__main__":
    main()
