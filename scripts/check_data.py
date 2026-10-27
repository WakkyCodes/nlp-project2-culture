from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Path shim: make `src/` importable when run as a file ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import build_dataset, parse_filename  # noqa: E402
from config.paths import XH_DIR, EN_DIR, METADATA_CSV, SPLITS_DIR  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Verify Project 2 corpus loader.")
    p.add_argument("--xh-dir",   type=Path, default=XH_DIR)
    p.add_argument("--en-dir",   type=Path, default=EN_DIR if EN_DIR.exists() else None)
    p.add_argument("--metadata", type=Path, default=METADATA_CSV)
    p.add_argument("--seed",     type=int,  default=42)
    args = p.parse_args()

    print(f"[INFO] Xh dir   : {args.xh_dir}")
    print(f"[INFO] En dir   : {args.en_dir}")
    print(f"[INFO] Metadata : {args.metadata}")

    # 0. Basic existence checks (fail fast with actionable messages)
    if not args.xh_dir.exists():
        print(f"[FAIL] Xh dir does not exist: {args.xh_dir}")
        return 1
    if not args.metadata.exists():
        print(f"[FAIL] Metadata CSV missing: {args.metadata}")
        return 1

    # 1. Filename sanity
    bad = [f.name for f in args.xh_dir.glob("*.txt") if parse_filename(f) is None]
    if bad:
        print(f"[FAIL] {len(bad)} non-conforming filenames, e.g. {bad[:5]}")
        return 1
    print(f"[INFO] All {len(list(args.xh_dir.glob('*.txt')))} Xh filenames parse correctly.")

    # 2. Build dataset
    try:
        bundle = build_dataset(args.xh_dir, args.en_dir, args.metadata, args.seed)
    except Exception as e:
        print(f"[FAIL] build_dataset raised: {e}")
        return 1

    records, splits, stats = bundle["records"], bundle["splits"], bundle["stats"]

       # 3. Empty files — WARN, expected in historical corpora don't fail
    empty = [r.filepath for r in records if not r.text.strip()]
    if empty:
        print(f"[WARN] {len(empty)} empty files survived load: {empty[:5]}")
    else:
        print("[INFO] No empty files in loaded records.")

    # 4. Issue-disjoint splits
    seen = {}
    for name, recs in splits.items():
        for r in recs:
            if r.issue_id in seen and seen[r.issue_id] != name:
                print(f"[FAIL] leakage: {r.issue_id} in {seen[r.issue_id]} & {name}")
                return 1
            seen[r.issue_id] = name

    # 5. Newspaper coverage per split
    for name, recs in splits.items():
        papers = sorted({r.newspaper for r in recs})
        print(f"[INFO] {name:5s}: {len(recs):4d} docs, newspapers={papers}")

    # 6. Stats
    print(f"[INFO] Issues    : {stats['n_issues']}")
    print(f"[INFO] Words     : {stats['n_words_total']:,}")
    print(f"[INFO] Date range: {stats['date_min']} → {stats['date_max']}")
    print(f"[INFO] Overlap 1884–1888:")
    for k, v in stats.get("overlap_1884_1888", {}).items():
        print(f"         {k}: {v}")

    print("[PASS] All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())