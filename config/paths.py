"""

Single source of truth for all project paths.
Imported by scripts and src modules so no one hard-codes relative paths.
"""
from pathlib import Path

# Project root = two levels above this file (config/paths.py → project/)
ROOT = Path(__file__).resolve().parents[1]

# Input data
DATA_DIR     = ROOT / "data"
XH_DIR       = DATA_DIR / "XhTxt"
EN_DIR       = DATA_DIR / "EnTxt"
METADATA_CSV = DATA_DIR / "LanguageSplitTracker70-94.csv"

# Outputs
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR    = RESULTS_DIR / "logs"
SPLITS_DIR  = RESULTS_DIR / "splits"   # created by loader.dump_splits()

# Ensure output dirs exist on import
for d in (FIGURES_DIR, LOGS_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)