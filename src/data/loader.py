
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# `[NEWSPAPER]_[YYYY-MM-DD]_[PAGE].txt` e.g. IMVO_1884-11-03_p001.txt
FILENAME_RE = re.compile(
    r"^(?P<newspaper>[A-Za-z]+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<page>p\d+)\.txt$"
)

OVERLAP_START = pd.Timestamp("1884-01-01")
OVERLAP_END = pd.Timestamp("1888-12-31")

DEFAULT_SPLIT = (0.70, 0.15, 0.15)  # train, val, test


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    """One parsed ``.txt`` file plus its metadata."""
    filepath: str
    newspaper: str
    issue_id: str
    date: str
    page: str
    language: str
    text: str
    n_words: int
    # Page-level metadata (joined from LanguageSplitTracker):
    tot_blocks: Optional[int] = None
    xh_pct: Optional[float] = None
    xh_blocks: Optional[int] = None
    en_pct: Optional[float] = None
    en_blocks: Optional[int] = None
    tot_wordcount: Optional[int] = None
    xho_eng_wordcount: Optional[int] = None
    xho_wordcount: Optional[int] = None
    eng_wordcount: Optional[int] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(path: Path) -> Optional[Dict[str, str]]:
    """Return metadata dict from a filename, or ``None`` if it doesn't match."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "newspaper": g["newspaper"].upper(),
        "date": g["date"],
        "page": g["page"],
        "issue_id": f"{g['newspaper'].upper()}_{g['date']}",
    }


# ---------------------------------------------------------------------------
# File-level loading
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    """Read a text file with UTF-8 and a fallback for OCR quirk encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    logger.warning("Falling back to latin-1 for %s (possible mojibake).", path)
    return path.read_text(encoding="latin-1", errors="replace")


def _word_count(text: str) -> int:
    """Simple whitespace token count (whitespace split; robust to OCR)."""
    return len(text.split())


def load_txt_folder(
    folder: Path,
    language: str,
    skip_empty: bool = True,
) -> List[DocumentRecord]:
    """
    Load every ``.txt`` file in ``folder`` into :class:`DocumentRecord` objects.

    Empty or whitespace-only files are skipped by default (historical corpora
    frequently contain blank pages and OCR failures — see ``skip_empty``).
    """
    records: List[DocumentRecord] = []
    if not folder.exists():
        logger.warning("Folder not found: %s", folder)
        return records

    n_empty = 0
    for path in sorted(folder.glob("*.txt")):
        meta = parse_filename(path)
        if meta is None:
            logger.warning("Skipping non-conforming filename: %s", path.name)
            continue
        text = _read_text(path)
        if skip_empty and not text.strip():
            n_empty += 1
            continue
        records.append(
            DocumentRecord(
                filepath=str(path),
                newspaper=meta["newspaper"],
                issue_id=meta["issue_id"],
                date=meta["date"],
                page=meta["page"],
                language=language,
                text=text,
                n_words=_word_count(text),
            )
        )

    if n_empty:
        logger.info(
            "Skipped %d empty %s files in %s (blank pages / OCR failures).",
            n_empty, language, folder.name,
        )
    logger.info("Loaded %d %s documents from %s", len(records), language, folder)
    return records


# ---------------------------------------------------------------------------
# Metadata join
# ---------------------------------------------------------------------------

def load_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Load ``LanguageSplitTracker70-94.csv`` (edition-level metadata).

    Expected columns (case-insensitive): Edition, Tot_Blocks, Xh%, En%,
    Xho_Wordcount, Eng_Wordcount. Extra columns are tolerated.
    """
    if not csv_path.exists():
        logger.warning("Metadata CSV not found: %s (continuing without join).", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded metadata: %d rows, columns=%s", len(df), list(df.columns))
    return df


def _normalise_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names from the tracker sheet.

    Tracker columns (as of 2024 dataset):
        Edition, Tot_Blocks, Xh%, Xh_Blocks, En%, En_Blocks,
        Tot_Wordcount, Xho+Eng_Wordcount, Xho_Wordcount, Eng_Wordcount
    """
    if df.empty:
        return df
    rename = {
        "edition":           "edition",
        "tot_blocks":        "tot_blocks",
        "xh%":               "xh_pct",
        "xh_blocks":         "xh_blocks",
        "en%":               "en_pct",
        "en_blocks":         "en_blocks",
        "tot_wordcount":     "tot_wordcount",
        "xho+eng_wordcount": "xho_eng_wordcount",
        "xho_wordcount":     "xho_wordcount",
        "eng_wordcount":     "eng_wordcount",
    }
    lower = {c.lower(): c for c in df.columns}
    out = df.rename(columns={lower[k]: v for k, v in rename.items() if k in lower})
    return out

def attach_metadata(records: List[DocumentRecord], meta_df: pd.DataFrame) -> None:
    """
    Join **page-level** metadata from ``LanguageSplitTracker70-94.csv`` onto
    records in-place.

    The tracker's ``Edition`` column is the *filename*
    (e.g. ``IMVO_1884-11-03_p001.txt``), so we index by filename stem
    (``IMVO_1884-11-03_p001``) to match ``Path(rec.filepath).stem``.

    Records whose filename is absent from the tracker are left with
    ``None`` in all metadata fields (they are still valid documents).
    """
    if meta_df.empty:
        logger.warning("attach_metadata: metadata DataFrame is empty — nothing to join.")
        return

    norm = _normalise_metadata(meta_df).copy()

    # Derive join key: strip the trailing '.txt' from the Edition column.
    # Use `.astype(str)` so numeric-ish editions don't get mangled.
    norm["_key"] = (
        norm["edition"].astype(str).str.strip().str.replace(r"\.txt$", "", regex=True)
    )
    norm = norm.drop(columns=["edition"]).drop_duplicates(subset="_key").set_index("_key")

    logger.info(
        "attach_metadata: %d tracker rows, %d unique filenames after dedup.",
        len(meta_df), len(norm),
    )

    matched = 0
    for rec in records:
        key = Path(rec.filepath).stem
        if key not in norm.index:
            continue
        row = norm.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rec.tot_blocks       = _safe_int(row.get("tot_blocks"))
        rec.xh_pct           = _safe_float(row.get("xh_pct"))
        rec.xh_blocks        = _safe_int(row.get("xh_blocks"))
        rec.en_pct           = _safe_float(row.get("en_pct"))
        rec.en_blocks        = _safe_int(row.get("en_blocks"))
        rec.tot_wordcount    = _safe_int(row.get("tot_wordcount"))
        rec.xho_eng_wordcount= _safe_int(row.get("xho_eng_wordcount"))
        rec.xho_wordcount    = _safe_int(row.get("xho_wordcount"))
        rec.eng_wordcount    = _safe_int(row.get("eng_wordcount"))
        matched += 1

    logger.info(
        "attach_metadata: joined metadata onto %d / %d records (%.1f%%).",
        matched, len(records), 100.0 * matched / max(len(records), 1),
    )

def _safe_int(v) -> Optional[int]:
    try:
        return None if pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v) -> Optional[float]:
    try:
        return None if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Issue-level split
# ---------------------------------------------------------------------------

def split_issues(
    records: List[DocumentRecord],
    ratios: Tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = 42,
) -> Dict[str, List[DocumentRecord]]:
    """
    Split records into train/val/test at the *issue* level, stratified by
    newspaper, so that no issue appears in more than one split.

    Parameters
    ----------
    records : list[DocumentRecord]
    ratios : (train, val, test)
    seed : int
        Deterministic shuffle seed.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"``.
    """
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    # Unique (newspaper, issue_id) pairs
    issue_map: Dict[str, str] = {}
    for rec in records:
        issue_map[rec.issue_id] = rec.newspaper  # issue_id encodes newspaper

    issues_df = (
        pd.Series(issue_map, name="newspaper")
        .rename_axis("issue_id")
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    assignments: Dict[str, str] = {}

    for _, group in issues_df.groupby("newspaper", sort=True):
        ids = group["issue_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(1, int(round(n * ratios[0]))) if n >= 3 else max(1, n - 1)
        n_val = max(1, int(round(n * ratios[1]))) if n >= 3 else max(0, n - n_train - 1)
        # Guard: ensure at least one test issue when n >= 3
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)
        for i, issue in enumerate(ids):
            if i < n_train:
                assignments[issue] = "train"
            elif i < n_train + n_val:
                assignments[issue] = "val"
            else:
                assignments[issue] = "test"

    splits: Dict[str, List[DocumentRecord]] = {"train": [], "val": [], "test": []}
    for rec in records:
        splits[assignments.get(rec.issue_id, "train")].append(rec)

    # Verification: no issue leakage
    seen: Dict[str, str] = {}
    for name, recs in splits.items():
        for r in recs:
            prev = seen.get(r.issue_id)
            if prev is not None and prev != name:
                raise RuntimeError(
                    f"Data leakage detected: issue {r.issue_id} in "
                    f"{prev} and {name}."
                )
            seen[r.issue_id] = name

    logger.info(
        "Split: train=%d docs, val=%d docs, test=%d docs",
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(
    records: List[DocumentRecord],
) -> Dict[str, object]:
    """Return a dictionary of basic corpus statistics."""
    if not records:
        return {"n_documents": 0}

    df = pd.DataFrame([r.to_dict() for r in records])
    df["date_ts"] = pd.to_datetime(df["date"], errors="coerce")

    stats: Dict[str, object] = {
        "n_documents": int(len(df)),
        "n_issues": int(df["issue_id"].nunique()),
        "n_words_total": int(df["n_words"].sum()),
        "date_min": df["date_ts"].min().strftime("%Y-%m-%d") if df["date_ts"].notna().any() else None,
        "date_max": df["date_ts"].max().strftime("%Y-%m-%d") if df["date_ts"].notna().any() else None,
        "per_newspaper": {},
        "per_language": {},
    }

    for np_name, g in df.groupby("newspaper"):
        stats["per_newspaper"][np_name] = {
            "n_documents": int(len(g)),
            "n_issues": int(g["issue_id"].nunique()),
            "n_words": int(g["n_words"].sum()),
            "mean_words_per_doc": float(g["n_words"].mean()),
        }

    for lang, g in df.groupby("language"):
        stats["per_language"][lang] = {
            "n_documents": int(len(g)),
            "n_issues": int(g["issue_id"].nunique()),
            "n_words": int(g["n_words"].sum()),
        }

    # 1884–1888 overlap
    overlap = df[(df["date_ts"] >= OVERLAP_START) & (df["date_ts"] <= OVERLAP_END)]
    if not overlap.empty:
        by_np = overlap.groupby("newspaper").agg(
            n_documents=("issue_id", "size"),
            n_issues=("issue_id", "nunique"),
            first_date=("date_ts", "min"),
            last_date=("date_ts", "max"),
        )
        stats["overlap_1884_1888"] = {
            np_name: {
                "n_documents": int(row["n_documents"]),
                "n_issues": int(row["n_issues"]),
                "first_date": row["first_date"].strftime("%Y-%m-%d"),
                "last_date": row["last_date"].strftime("%Y-%m-%d"),
            }
            for np_name, row in by_np.iterrows()
        }
        # Joint range
        stats["overlap_1884_1888"]["joint_range"] = {
            "start": overlap["date_ts"].min().strftime("%Y-%m-%d"),
            "end": overlap["date_ts"].max().strftime("%Y-%m-%d"),
        }
    else:
        stats["overlap_1884_1888"] = {}

    return stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset(
    xh_dir: Path,
    en_dir: Optional[Path],
    metadata_csv: Path,
    split_seed: int = 42,
) -> Dict[str, object]:
    """
    End-to-end Person 1 pipeline.

    Returns a dict::

        {
            "records": list[DocumentRecord],   # all documents
            "splits":  {train, val, test},     # list[DocumentRecord]
            "stats":   {...},
            "metadata_df": pd.DataFrame,
        }
    """
    records: List[DocumentRecord] = []
    records += load_txt_folder(xh_dir, language="xh")
    if en_dir is not None:
        records += load_txt_folder(en_dir, language="en")

    meta_df = load_metadata(metadata_csv)
    attach_metadata(records, meta_df)

    splits = split_issues(records, seed=split_seed)
    stats = compute_statistics(records)

    return {
        "records": records,
        "splits": splits,
        "stats": stats,
        "metadata_df": meta_df,
    }


def dump_splits(splits: Dict[str, List[DocumentRecord]], out_dir: Path) -> None:
    """Persist each split as a CSV manifest for downstream stages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        manifest = pd.DataFrame([r.to_dict() for r in recs])
        if manifest.empty:
            logger.warning("Split '%s' is empty — skipping.", name)
            continue
        # Do not persist raw text in the manifest (privacy / size).
        manifest = manifest.drop(columns=["text"], errors="ignore")
        manifest.to_csv(out_dir / f"{name}_manifest.csv", index=False)
        logger.info("Wrote %s (%d rows)", out_dir / f"{name}_manifest.csv", len(manifest))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Load & split Project 2 corpus.")
    p.add_argument("--xh-dir", type=Path, required=True)
    p.add_argument("--en-dir", type=Path, default=None)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("results/splits"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bundle = build_dataset(args.xh_dir, args.en_dir, args.metadata, args.seed)
    dump_splits(bundle["splits"], args.out_dir)
    print(json.dumps(bundle["stats"], indent=2, default=str))