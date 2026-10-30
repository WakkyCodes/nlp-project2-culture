"""Issue-aware TF-IDF + Logistic Regression for IMVO vs ISIG (1884--1888).

Place at src/models/newspaper_classifier.py, then from repository root run:
    python -m src.models.newspaper_classifier

Validation is reported by default. The held-out test split is ONLY evaluated
with --evaluate-test after modelling decisions have been finalised.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from config.paths import ROOT, RESULTS_DIR, SPLITS_DIR, XH_DIR

CLASSES = ["IMVO", "ISIG"]
# Sensitivity control, not a linguistically validated stopword list.
# These also occur in legitimate prose. Report the masking choice.
#SOURCE_WORDS = re.compile(r"\b(?:imvo|zabantsundu|isigidimi|sigidimi)\b", re.IGNORECASE)
# Expanded source-name masking for sensitivity analysis.
# Some terms also have legitimate meanings in isiXhosa.
SOURCE_WORDS = re.compile(
    r"\b(?:imvo|zabantsundu|isigidimi|sigidimi|xosa|samaxosa|lamaxosa)\b",
    re.IGNORECASE
)

def read_page(filepath: str) -> str:
    p = Path(filepath)
    candidates = ([p] if p.is_absolute() else [ROOT / p]) + [XH_DIR / p.name]
    for candidate in candidates:
        if candidate.is_file():
            for encoding in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    return candidate.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    pass
    raise FileNotFoundError(f"Cannot locate OCR page: {filepath}")


def load_split(name: str, date_min: str, date_max: str) -> pd.DataFrame:
    path = SPLITS_DIR / f"{name}_manifest.csv"
    frame = pd.read_csv(path)
    required = {"filepath", "newspaper", "issue_id", "date", "language", "xh_pct", "n_words"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
    purity = pd.to_numeric(frame.xh_pct.astype(str).str.rstrip("%"), errors="coerce")
    if not purity.dropna().empty and purity.dropna().max() <= 1:
        purity *= 100
    length = pd.to_numeric(frame.n_words, errors="coerce")
    date = pd.to_datetime(frame.date, errors="coerce")
    keep = (frame.language.eq("xh") & frame.newspaper.isin(CLASSES)
            & purity.ge(90) & length.ge(200) & date.between(date_min, date_max))
    frame = frame.loc[keep].copy()
    frame["text"] = frame.filepath.map(read_page)
    frame = frame.loc[frame.text.str.strip().ne("")].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No eligible documents in {path}")
    print(f"{name}: {len(frame)} pages, {frame.issue_id.nunique()} issues; "
          f"IMVO={sum(frame.newspaper == 'IMVO')}, ISIG={sum(frame.newspaper == 'ISIG')}")
    return frame


def issue_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal total weight per issue and equal total weight per newspaper."""
    n_pages = frame.groupby(["newspaper", "issue_id"])["issue_id"].transform("size")
    n_issues = frame.groupby("newspaper").issue_id.nunique()
    weight = 1.0 / (n_pages.to_numpy() * frame.newspaper.map(n_issues).to_numpy())
    return weight / np.mean(weight)


def issue_predictions(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    pred = frame[["newspaper", "issue_id"]].copy()
    pred["p_isig"] = probabilities
    # Give each issue one vote, no matter how many pages it contains.
    issues = pred.groupby(["newspaper", "issue_id"], as_index=False).p_isig.mean()
    issues["predicted"] = np.where(issues.p_isig >= 0.5, "ISIG", "IMVO")
    return issues


def metrics(issues: pd.DataFrame) -> dict:
    true, predicted = issues.newspaper, issues.predicted
    cm = confusion_matrix(true, predicted, labels=CLASSES)
    return {"n_issues": int(len(issues)),
            "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
            "macro_f1": float(f1_score(true, predicted, labels=CLASSES, average="macro", zero_division=0)),
            "confusion_matrix_IMVO_ISIG": cm.tolist()}


def vectorizer_for(ngrams: tuple[int, int]) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
                           ngram_range=ngrams, min_df=2, max_df=0.95,
                           max_features=30000, sublinear_tf=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-test", action="store_true", help="Only after hyperparameters are frozen")
    parser.add_argument("--no-mask-source", action="store_true", help="Ablation: retain explicit newspaper-name words")
    parser.add_argument("--strict-overlap", action="store_true", help="Limit dates to 1884-11-03 through 1888-12-01")
    args = parser.parse_args()
    first, last = (("1884-11-03", "1888-12-01") if args.strict_overlap
                   else ("1884-01-01", "1888-12-31"))
    train, val, test = [load_split(n, first, last) for n in ("train", "val", "test")]
    splits = [set(f.issue_id) for f in (train, val, test)]
    if any(splits[i] & splits[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Issue IDs overlap across manifests: data leakage")

    if not args.no_mask_source:
        for frame in (train, val, test):
            frame["text"] = frame.text.map(lambda x: SOURCE_WORDS.sub(" ", x))
        print("Source-name tokens masked (not a complete masthead/boilerplate removal).")

    train_y = (train.newspaper == "ISIG").astype(int)
    weights = issue_sample_weights(train)
    out = RESULTS_DIR / "classifier"
    out.mkdir(parents=True, exist_ok=True)

    trials = []
    best = None
    for ngrams in [(1, 1), (1, 2)]:
        vectorizer = vectorizer_for(ngrams)
        x_train = vectorizer.fit_transform(train.text)
        x_val = vectorizer.transform(val.text)
        for C in [0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=3000, random_state=42)
            clf.fit(x_train, train_y, sample_weight=weights)
            issues = issue_predictions(val, clf.predict_proba(x_val)[:, 1])
            evaluation = metrics(issues)
            row = {"ngram_range": str(ngrams), "C": C, **{k: v for k, v in evaluation.items() if k != 'confusion_matrix_IMVO_ISIG'}}
            trials.append(row)
            print(f"ngrams={ngrams}, C={C:g} | val issue macro-F1={evaluation['macro_f1']:.3f} | "
                  f"balanced accuracy={evaluation['balanced_accuracy']:.3f}")
            score = (evaluation["macro_f1"], evaluation["balanced_accuracy"])
            if best is None or score > best[0]:
                best = (score, ngrams, C, vectorizer, clf, issues, evaluation)

    pd.DataFrame(trials).to_csv(out / "validation_tuning.csv", index=False)
    assert best is not None
    _, ngrams, C, vectorizer, clf, val_issues, val_eval = best
    val_issues.to_csv(out / "validation_issue_predictions.csv", index=False)
    vocab = vectorizer.get_feature_names_out()
    coefficients = clf.coef_[0]
    order = np.argsort(coefficients)
    terms = pd.DataFrame({"term": vocab, "coefficient_for_ISIG": coefficients})
    terms.iloc[order[:30]].sort_values("coefficient_for_ISIG").to_csv(out / "top_IMVO_features.csv", index=False)
    terms.iloc[order[-30:]].sort_values("coefficient_for_ISIG", ascending=False).to_csv(out / "top_ISIG_features.csv", index=False)
    summary = {"selected_ngrams": list(ngrams), "selected_C": C,
               "source_terms_masked": not args.no_mask_source,
               "date_range": [first, last], "validation_issue_metrics": val_eval}
    print("\nSelected on validation:", summary)
    if args.evaluate_test:
        x_test = vectorizer.transform(test.text)
        test_issues = issue_predictions(test, clf.predict_proba(x_test)[:, 1])
        test_issues.to_csv(out / "test_issue_predictions.csv", index=False)
        summary["test_issue_metrics"] = metrics(test_issues)
        print("FINAL test issue metrics:", summary["test_issue_metrics"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote classifier results to {out}")


if __name__ == "__main__":
    main()
