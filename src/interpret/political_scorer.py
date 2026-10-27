from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

# Optional dependencies with graceful fallback
try:
    import regex as _regex  # type: ignore
    _HAS_REGEX = True
except ImportError:
    import re as _regex  # type: ignore
    _HAS_REGEX = False
    logger.warning("`regex` not installed — falling back to `re` (click consonants still handled).")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False
    logger.warning("matplotlib/seaborn not installed — plotting disabled.")

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(x, **kw):  # type: ignore
        return x


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

@dataclass
class PoliticalLexicon:
    """Container for the isiXhosa political lexicon."""
    terms_by_theme: Dict[str, List[str]]
    all_terms: List[str]
    case_insensitive: bool = True
    max_edit_distance: int = 1
    min_length_for_fuzzy: int = 5
    version: str = "unknown"

    @classmethod
    def from_yaml(cls, path: Path) -> "PoliticalLexicon":
        with path.open(encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        terms_by_theme: Dict[str, List[str]] = {}
        all_terms: set = set()

        for theme, cfg in doc.get("themes", {}).items():
            terms = [t.strip() for t in cfg.get("terms", []) if t.strip()]
            terms_by_theme[theme] = terms
            all_terms.update(terms)

        ocr = doc.get("ocr", {}) or {}
        return cls(
            terms_by_theme=terms_by_theme,
            all_terms=sorted(all_terms),
            case_insensitive=bool(ocr.get("case_insensitive", True)),
            max_edit_distance=int(ocr.get("max_edit_distance", 1)),
            min_length_for_fuzzy=int(ocr.get("min_length_for_fuzzy", 5)),
            version=str(doc.get("version", "unknown")),
        )

    def summary(self) -> Dict:
        return {
            "version": self.version,
            "n_terms_total": len(self.all_terms),
            "n_themes": len(self.terms_by_theme),
            "terms_per_theme": {k: len(v) for k, v in self.terms_by_theme.items()},
            "ocr_case_insensitive": self.case_insensitive,
            "ocr_max_edit_distance": self.max_edit_distance,
        }


# ---------------------------------------------------------------------------
# Tokenisation (isiXhosa-aware)
# ---------------------------------------------------------------------------

# isiXhosa uses Latin letters plus click digraphs (c, q, x before h / other
# vowels). We tokenise on any non-letter, but preserve intra-word hyphens
# (common in 19th-c. print: 'u-rhulumente').
_TOKEN_RE = _regex.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", _regex.UNICODE)


def tokenise(text: str) -> List[str]:
    """Return alphabetic tokens, preserving intra-word hyphens."""
    if not text:
        return []
    # NFKC normalisation collapses OCR ligatures (ﬁ → fi) into ASCII
    text = unicodedata.normalize("NFKC", text)
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalise_term(term: str, case_insensitive: bool) -> str:
    t = unicodedata.normalize("NFKC", term)
    return t.lower() if case_insensitive else t


def _bounded_levenshtein(a: str, b: str, max_d: int) -> int:
    """Return Levenshtein distance, capped at ``max_d + 1`` for speed."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    if a == b:
        return 0
    # Banded DP
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[lb]


def _compile_lexicon(
    lex: PoliticalLexicon,
) -> Tuple[set, Dict[str, int], List[str]]:
    """
    Return (exact_match_set, fuzzy_match_candidates, all_terms).
    Exact matches are O(1); fuzzy matching is only attempted on tokens longer
    than ``min_length_for_fuzzy``.
    """
    exact = {_normalise_term(t, lex.case_insensitive) for t in lex.all_terms}
    fuzzy_candidates = [
        _normalise_term(t, lex.case_insensitive)
        for t in lex.all_terms
        if len(t) >= lex.min_length_for_fuzzy
    ]
    return exact, {t: i for i, t in enumerate(fuzzy_candidates)}, fuzzy_candidates


def score_document(
    text: str,
    lex: PoliticalLexicon,
    *,
    exact_set: Optional[set] = None,
    fuzzy_index: Optional[Dict[str, int]] = None,
    fuzzy_terms: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Return a dict of political-term statistics for one document.

    Keys:
        n_tokens              : total tokens
        n_political_exact     : exact lexicon matches
        n_political_fuzzy     : additional matches found via edit-distance 1
        n_political_total     : exact + fuzzy
        political_density     : n_political_total / n_tokens
        political_per_1000    : n_political_total / n_tokens * 1000
        mean_word_length      : characters per token (OCR-noise proxy)
        non_alpha_ratio       : share of non-alphabetic characters (OCR proxy)
    """
    if exact_set is None or fuzzy_index is None or fuzzy_terms is None:
        exact_set, fuzzy_index, fuzzy_terms = _compile_lexicon(lex)

    tokens = tokenise(text)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return {
            "n_tokens": 0, "n_political_exact": 0, "n_political_fuzzy": 0,
            "n_political_total": 0, "political_density": 0.0,
            "political_per_1000": 0.0,
            "mean_word_length": 0.0, "non_alpha_ratio": 0.0,
        }

    exact_hits = 0
    fuzzy_hits = 0
    for tok in tokens:
        norm = _normalise_term(tok, lex.case_insensitive)
        if norm in exact_set:
            exact_hits += 1
            continue
        if (
            lex.max_edit_distance > 0
            and len(norm) >= lex.min_length_for_fuzzy
        ):
            for cand in fuzzy_terms:
                if _bounded_levenshtein(norm, cand, lex.max_edit_distance) <= lex.max_edit_distance:
                    fuzzy_hits += 1
                    break

    total_hits = exact_hits + fuzzy_hits
    # OCR proxies
    mean_wl = float(np.mean([len(t) for t in tokens]))
    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = max(len(text), 1)
    non_alpha = 1.0 - (alpha_chars / total_chars)

    return {
        "n_tokens": n_tokens,
        "n_political_exact": exact_hits,
        "n_political_fuzzy": fuzzy_hits,
        "n_political_total": total_hits,
        "political_density": total_hits / n_tokens,
        "political_per_1000": (total_hits / n_tokens) * 1000.0,
        "mean_word_length": mean_wl,
        "non_alpha_ratio": non_alpha,
    }

def _log_odds_score(n_hits: int, n_tokens: int, n_total_hits: int, n_total_tokens: int,
                    alpha: float = 0.5) -> float:
    """
    log-odds ratio with informative Dirichlet prior.

    Length-aware by construction: a 1-token doc with 1 hit is shrunk toward zero,
    while a 1000-token doc with 20 hits gets a substantial score. Returns 0 for
    docs with no tokens.
    """
    if n_tokens == 0 or n_total_tokens == 0:
        return 0.0
    p_doc   = (n_hits + alpha) / (n_tokens + 2 * alpha)
    p_corpus = (n_total_hits + alpha) / (n_total_tokens + 2 * alpha)
    return float(math.log(p_doc / (1.0 - p_doc)) - math.log(p_corpus / (1.0 - p_corpus)))
# ---------------------------------------------------------------------------
# Robustness controls
# ---------------------------------------------------------------------------

@dataclass
class ControlResult:
    name: str
    statistic: float
    p_value: float
    n: int
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            "control": self.name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "n": self.n,
            "interpretation": self.interpretation,
        }
    
def _score_col(df: pd.DataFrame) -> str:
    """Return the preferred per-document score column, with fallbacks."""
    for col in ("political_log_odds", "political_density"):
        if col in df.columns:
            return col
    raise KeyError("No political score column found.")

def control_ocr_noise(df: pd.DataFrame) -> ControlResult:
    """Spearman: political_density vs mean_word_length (OCR-degradation proxy)."""
    d = df.dropna(subset=[_score_col(df), "mean_word_length"])
    if len(d) < 100:
            return ControlResult(unicodedata.name, float("nan"), float("nan"), len(d),"INSUFFICIENT DATA.")
    rho, p = stats.spearmanr(d[_score_col(df)], d["mean_word_length"])
    interp = (
        "WEAK/NULL — political score is not driven by word-length artefacts."
        if abs(rho) < 0.10 or p > 0.05
        else "WARNING — political score correlates with word length; investigate OCR confounding."
    )
    return ControlResult("OCR: political_density vs mean_word_length", float(rho), float(p), len(d), interp)


def control_ocr_noise_chars(df: pd.DataFrame) -> ControlResult:
    """Spearman: political_density vs non_alpha_ratio (second OCR proxy)."""
    d = df.dropna(subset=[_score_col(df), "non_alpha_ratio"])
    if len(d) < 100:
                return ControlResult(unicodedata.name, float("nan"), float("nan"), len(d),"INSUFFICIENT DATA.")
    rho, p = stats.spearmanr(d[_score_col(df)], d["non_alpha_ratio"])
    interp = (
        "WEAK/NULL — political score is not driven by OCR character noise."
        if abs(rho) < 0.10 or p > 0.05
        else "WARNING — political score correlates with non-alpha ratio."
    )
    return ControlResult("OCR: political_density vs non_alpha_ratio", float(rho), float(p), len(d), interp)


def control_era(df: pd.DataFrame) -> ControlResult:
    """Spearman: political_density vs year — is score just time-drift?"""
    d = df.copy()
    if len(d) < 100:
                return ControlResult(unicodedata.name, float("nan"), float("nan"), len(d),"INSUFFICIENT DATA.")
    d["year"] = pd.to_datetime(d["date"], errors="coerce").dt.year
    d = d.dropna(subset=[_score_col(df), "year"])
    rho, p = stats.spearmanr(d[_score_col(df)], d["year"])
    interp = (
        "WEAK/NULL — political score varies beyond simple era drift."
        if abs(rho) < 0.15 or p > 0.05
        else "NOTE — political score has a temporal trend; document it explicitly."
    )
    return ControlResult("Era: political_density vs year", float(rho), float(p), len(d), interp)


def control_length(df: pd.DataFrame) -> ControlResult:
    """Spearman: political_density vs n_words — is density just length?"""
    d = df.dropna(subset=[_score_col(df), "n_words"])
    if len(d) < 100:
                return ControlResult(unicodedata.name, float("nan"), float("nan"), len(d),"INSUFFICIENT DATA.")
    rho, p = stats.spearmanr(d[_score_col(df)], d["n_words"])
    interp = (
        "WEAK/NULL — density is not a length artefact."
        if abs(rho) < 0.15 or p > 0.05
        else "NOTE — density correlates with document length."
    )
    return ControlResult("Length: political_density vs n_words", float(rho), float(p), len(d), interp)


def control_newspaper(df: pd.DataFrame) -> ControlResult:
    """Mann–Whitney U between IMVO and ISIG political_density."""
    d = df.dropna(subset=[_score_col(df), "newspaper"])
    if len(d) < 100:
            return ControlResult(unicodedata.name, float("nan"), float("nan"), len(d),"INSUFFICIENT DATA.")
    groups = d.groupby("newspaper")[_score_col(df)]
    if groups.ngroups != 2:
        return ControlResult("Source: IMVO vs ISIG", float("nan"), float("nan"),
                             len(d), "SKIPPED — need both newspapers present.")
    a, b = groups.get_group(sorted(groups.groups)[0]), groups.get_group(sorted(groups.groups)[1])
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    med_a, med_b = float(np.median(a)), float(np.median(b))
    interp = (
        f"DIFFERENT — median {med_a:.4f} vs {med_b:.4f} (p={p:.4g}); "
        "editorial divergence is present in the data."
        if p < 0.05
        else f"SIMILAR — median {med_a:.4f} vs {med_b:.4f} (p={p:.4g}); "
             "no detectable editorial divergence at issue level."
    )
    return ControlResult(
        f"Source: {sorted(groups.groups)[0]} vs {sorted(groups.groups)[1]}",
        float(u), float(p), len(d), interp,
    )
def control_language_purity(df: pd.DataFrame) -> ControlResult:
    """Spearman: political_density vs xh_pct (language-mix confound)."""
    d = df.dropna(subset=[_score_col(df), "xh_pct"])
    if len(d) < 100:
                return ControlResult(unicodedata.name, float("nan"), float("nan"), len(d),"INSUFFICIENT DATA.")
    if len(d) < 10:
        return ControlResult("Language purity: political_density vs xh_pct",
                             float("nan"), float("nan"), len(d),
                             "SKIPPED — xh_pct not populated in manifest.")
    rho, p = stats.spearmanr(d[_score_col(df)], d["xh_pct"])
    interp = (
        "WEAK/NULL — score is not a proxy for isiXhosa-ness."
        if abs(rho) < 0.15 or p > 0.05
        else "WARNING — political score tracks isiXhosa proportion; add xh_pct as a covariate."
    )
    return ControlResult("Language purity: political_density vs xh_pct",
                         float(rho), float(p), len(d), interp)


def run_all_controls(df: pd.DataFrame) -> List[ControlResult]:
    return [
        control_ocr_noise(df),
        control_ocr_noise_chars(df),
        control_era(df),
        control_length(df),
        control_newspaper(df),
        control_language_purity(df),
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe_plot(fn):
    """Decorator: never let plotting crash the pipeline."""
    def wrapper(*a, **kw):
        if not _HAS_PLOT:
            logger.warning("Skipping plot %s (matplotlib unavailable).", fn.__name__)
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:
            logger.warning("Plot %s failed: %s", fn.__name__, e)
            return None
    return wrapper


@_safe_plot
def plot_density_by_newspaper(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(7, 4))
    sns.violinplot(data=df, x="newspaper", y=_score_col(df), cut=0, inner="quart")
    plt.title("Political-term density by newspaper")
    plt.ylabel(_score_col(df))
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


@_safe_plot
def plot_density_over_time(df: pd.DataFrame, out: Path) -> None:
    d = df.copy()
    d["year"] = pd.to_datetime(d["date"], errors="coerce").dt.year
    d = d.dropna(subset=["year"])
    plt.figure(figsize=(8, 4))
    for np_name, g in d.groupby("newspaper"):
        yearly = g.groupby("year")[_score_col(df)].mean()
        plt.plot(yearly.index, yearly.values, marker="o", label=np_name)
    plt.title("Mean " + _score_col(df) + " per year")
    plt.xlabel("Year"); plt.ylabel("mean " + _score_col(df))
    plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()


@_safe_plot
def plot_ocr_control(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="mean_word_length", y=_score_col(df),
                    hue="newspaper", alpha=0.35, s=12)
    plt.title("OCR control: " + _score_col(df) + " vs mean word length")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


@_safe_plot
def plot_control_correlations(df: pd.DataFrame, out: Path) -> None:
    cols = [_score_col(df), "mean_word_length", "non_alpha_ratio", "n_words"]
    corr = df[cols].corr(method="spearman")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Spearman correlations (robustness)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def score_corpus(
    manifests: Iterable[Path],
    lexicon_path: Path,
    text_root: Path,
    results_dir: Path,
) -> pd.DataFrame:
    """
    Score every document referenced in the split manifests and return a
    tidy ``DataFrame`` with one row per document.

    ``manifests`` are the CSVs produced by ``loader.dump_splits`` — they
    contain ``filepath``, ``newspaper``, ``issue_id``, ``date``, ``page``,
    ``language``, ``n_words``. We re-read the raw text from ``text_root``
    so this function does not depend on in-memory state.
    """
    lex = PoliticalLexicon.from_yaml(lexicon_path)
    logger.info("Lexicon v%s: %d terms across %d themes.",
                lex.version, len(lex.all_terms), len(lex.terms_by_theme))

    exact_set, fuzzy_index, fuzzy_terms = _compile_lexicon(lex)

    MIN_TOKENS = 200  # length floor — see controls analysis

    rows: List[Dict] = []
    skipped_short = 0
    for manifest in manifests:
        if not manifest.exists():
            logger.warning("Manifest missing: %s", manifest)
            continue
        split_name = manifest.stem.replace("_manifest", "")
        df_m = pd.read_csv(manifest)

        # IsiXhosa-only — see language-purity control
        if "language" in df_m.columns:
            df_m = df_m[df_m["language"] == "xh"].copy()
        logger.info("Scoring %d isiXhosa documents from %s split.", len(df_m), split_name)

        # in score_corpus, right after language filter
        n_before = len(df_m)
        if "xh_pct" in df_m.columns:
            df_m = df_m[df_m["xh_pct"].fillna(100) >= 90].copy()  # treat NaN as pure
            logger.info("xh_pct filter: retained %d/%d (%.1f%%) documents.",
                len(df_m), n_before, 100 * len(df_m) / max(n_before, 1))

        for _, r in tqdm(df_m.iterrows(), total=len(df_m), desc=f"{split_name}"):
            text_path = Path(r["filepath"])
            if not text_path.exists():
                text_path = text_root / text_path.name
            if not text_path.exists():
                logger.warning("Text file missing: %s", text_path)
                continue
            text = text_path.read_text(encoding="utf-8", errors="replace")
            stats_d = score_document(
                text, lex,
                exact_set=exact_set, fuzzy_index=fuzzy_index, fuzzy_terms=fuzzy_terms,
            )
            if stats_d["n_tokens"] < MIN_TOKENS:
                skipped_short += 1
                continue
            rows.append({
                "split": split_name,
                "filepath": str(text_path),
                "newspaper": r.get("newspaper"),
                "issue_id": r.get("issue_id"),
                "date": r.get("date"),
                "page": r.get("page"),
                "language": r.get("language"),
                "n_words": r.get("n_words"),
                "xh_pct": r.get("xh_pct"),
                **stats_d,
            })

    logger.info("Skipped %d docs below MIN_TOKENS=%d floor.", skipped_short, MIN_TOKENS)
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No documents scored.")
        return pd.DataFrame(), pd.DataFrame()

    # Corpus-level priors for log-odds
    total_hits   = int(df["n_political_total"].sum())
    total_tokens = int(df["n_tokens"].sum())

    df["political_log_odds"] = df.apply(
        lambda r: _log_odds_score(r["n_political_total"], r["n_tokens"],
                                  total_hits, total_tokens),
        axis=1,
    )

    # Issue-level aggregation (now includes log-odds)
    issue_agg = (
        df.groupby(["issue_id", "newspaper", "date"], as_index=False)
          .agg(
              n_documents=("filepath", "size"),
              n_words=("n_words", "sum"),
              n_tokens=("n_tokens", "sum"),
              n_political_total=("n_political_total", "sum"),
              political_density=("political_density", "mean"),
              political_log_odds=("political_log_odds", "mean"),
              mean_word_length=("mean_word_length", "mean"),
              non_alpha_ratio=("non_alpha_ratio", "mean"),
              xh_pct=("xh_pct", "mean"),
          )
    )
    issue_agg["political_per_1000"] = (
        issue_agg["n_political_total"] / issue_agg["n_tokens"].replace(0, np.nan)
    ) * 1000.0

    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "political_scores_document.csv", index=False)
    issue_agg.to_csv(results_dir / "political_scores_issue.csv", index=False)
    logger.info("Wrote %s and %s",
                results_dir / "political_scores_document.csv",
                results_dir / "political_scores_issue.csv")

    return df, issue_agg


def run_controls_and_plots(doc_df: pd.DataFrame, figures_dir: Path) -> List[ControlResult]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    controls = run_all_controls(doc_df)

    # Persist controls
    ctrl_df = pd.DataFrame([c.to_dict() for c in controls])
    ctrl_df.to_csv(figures_dir.parent / "political_controls.csv", index=False)
    logger.info("Wrote robustness controls to %s",
                figures_dir.parent / "political_controls.csv")

    # Plots
    plot_density_by_newspaper(doc_df, figures_dir / "density_by_newspaper.png")
    plot_density_over_time(doc_df,   figures_dir / "density_over_time.png")
    plot_ocr_control(doc_df,         figures_dir / "control_ocr.png")
    plot_control_correlations(doc_df, figures_dir / "control_correlations.png")

    # Human-readable summary
    for c in controls:
        logger.info("[%s] rho/U=%.4f p=%.4g n=%d — %s",
                    c.name, c.statistic, c.p_value, c.n, c.interpretation)
    return controls


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Score corpus for political-term density and run robustness controls.")
    p.add_argument("--manifests", nargs="+", type=Path, required=True,
                   help="train/val/test manifest CSVs produced by loader.dump_splits")
    p.add_argument("--lexicon", type=Path, required=True)
    p.add_argument("--text-root", type=Path, required=True,
                   help="Root containing XhTxt/ (used to resolve relative paths)")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    doc_df, issue_df = score_corpus(
        manifests=args.manifests,
        lexicon_path=args.lexicon,
        text_root=args.text_root,
        results_dir=args.results_dir,
    )

    controls = run_controls_and_plots(
        doc_df,
        figures_dir=args.results_dir / "figures",
    )

    print(json.dumps({
        "n_documents_scored": int(len(doc_df)),
        "n_issues_scored": int(issue_df["issue_id"].nunique()),
        "controls": [c.to_dict() for c in controls],
    }, indent=2))
    