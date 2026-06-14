#!/usr/bin/env python
"""Download RAGTruth (build-time network OK) and build the claim cache. Disk-guarded.

One-command reproduction step: `python scripts/10_data.py`. Idempotent (skips existing
files). All later stages read the cache offline.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ragtruth"
BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
FILES = ["source_info.jsonl", "response.jsonl"]
MIN_FREE_GB = 4


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if free_gb(DATA) < MIN_FREE_GB:
        print(f"FATAL: free disk {free_gb(DATA):.1f}GB < {MIN_FREE_GB}GB floor", file=sys.stderr)
        return 17
    for f in FILES:
        dst = DATA / f
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[data] {f} present ({dst.stat().st_size/1e6:.1f} MB)")
            continue
        url = f"{BASE}/{f}"
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"[data] saved {f} ({dst.stat().st_size/1e6:.1f} MB)")

    # build + summarize the parsed cache
    from praman.data import load_records, make_splits
    recs, report = load_records(use_cache=False)
    sp = make_splits(recs)
    print("[data] split sizes:", {k: v["n"] for k, v in sp.summary().items()})
    print(f"[data] free disk now {free_gb(DATA):.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
