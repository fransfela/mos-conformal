"""
download_resume.py — Resumable download with automatic retry.

Usage:
    python download_resume.py <url> <destination>

Retries indefinitely with exponential back-off until the file is complete.
"""

import os
import sys
import time
import urllib.request
import urllib.error

URL  = "https://zenodo.org/record/4728081/files/NISQA_Corpus.zip"
DEST = r"c:\Users\rffela\OneDrive - GN Store Nord\Documents\Jabra\2026\Development\2027 ICASSP\data\NISQA_Corpus.zip"

CHUNK   = 1 * 1024 * 1024   # 1 MB chunks
MAX_RETRIES = 999
INITIAL_WAIT = 10            # seconds before first retry


def get_remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def download(url: str, dest: str) -> None:
    attempt = 0
    wait = INITIAL_WAIT

    remote_size = get_remote_size(url)
    if remote_size:
        print(f"Remote size: {remote_size / 1e9:.2f} GB")

    while True:
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0

        if remote_size and existing >= remote_size:
            print(f"Download complete: {existing / 1e9:.2f} GB")
            return

        attempt += 1
        print(f"\n[Attempt {attempt}] Resuming from {existing / 1e9:.2f} GB ...")

        headers = {}
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as resp, \
                 open(dest, "ab") as fh:
                downloaded = existing
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / remote_size * 100) if remote_size else 0
                    print(f"\r  {downloaded / 1e9:.2f} GB  ({pct:.1f}%)", end="", flush=True)
            print()
            # Verify completion
            final = os.path.getsize(dest)
            if remote_size and final >= remote_size:
                print(f"Done: {final / 1e9:.2f} GB")
                return
        except (urllib.error.URLError, OSError, ConnectionResetError) as exc:
            print(f"\n  Connection error: {exc}")
            if attempt >= MAX_RETRIES:
                print("Max retries reached. Aborting.")
                sys.exit(1)
            print(f"  Waiting {wait}s before retry ...")
            time.sleep(wait)
            wait = min(wait * 2, 120)   # cap at 2 min


if __name__ == "__main__":
    url  = sys.argv[1] if len(sys.argv) > 1 else URL
    dest = sys.argv[2] if len(sys.argv) > 2 else DEST
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    download(url, dest)
