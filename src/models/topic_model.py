"""NMF topic modelling for the 1884–1888 isiXhosa newspaper comparison.

Place at src/models/topic_model.py and run from the repository root:
    python -m src.models.topic_model

Uses the existing issue-disjoint CSV manifests; does not change the loader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from config.paths import ROOT, RESULTS_DIR, SPLITS_DIR, XH_DIR


def read_page(filepath: str) -> str:
    """Locate text referenced by a manifest, including stale absolute paths."""
    path = Path(filepath)
    options = [path] if path.is_absolute() else [ROOT / path]
    options.append(XH_DIR / path.name)
    for candidate in options:
        if candidate.is_file():
            for encoding in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    return candidate.read_text(encoding=encoding)
                except UnicodeDecodeError:
                    continue
    raise FileNotFoundError(f"Cannot locate page: {filepath}")


def load_split(split: str) -> pd.DataFrame:
    manifest = SPLITS_DIR / f"{split}_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing {manifest}; run Person 1's loader first.")
    df = pd.read_csv(manifest)
    needed = {"filepath", "newspaper", "issue_id", "date", "language", "xh_pct", "n_words"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"{manifest} is missing: {sorted(missing)}")

    purity = pd.to_numeric(df["xh_pct"].astype(str).str.rstrip("%"), errors="coerce")
    # Normalise proportions if the metadata is recorded between 0 and 1.
    if purity.dropna().max() <= 1.0:
        purity *= 100
    length = pd.to_numeric(df["n_words"], errors="coerce")
    dates = pd.to_datetime(df["date"], errors="coerce")
    selected = (
        df["language"].eq("xh")
        & df["newspaper"].isin(["IMVO", "ISIG"])
        & purity.ge(90)
        & length.ge(200)
        & dates.between("1884-01-01", "1888-12-31")
    )
    df = df.loc[selected].copy()
    if df.empty:
        raise ValueError(f"No eligible isiXhosa pages in {manifest}")
    df["text"] = df["filepath"].map(read_page)
    df = df.loc[df["text"].str.strip().ne("")].reset_index(drop=True)
    print(f"{split}: {len(df)} eligible pages in {df.issue_id.nunique()} issues")
    return df


def relative_reconstruction_error(x, weights: np.ndarray, topic_words: np.ndarray) -> float:
    """Compute ||X - WH|| / ||X|| without creating a dense document-term matrix."""
    x_norm2 = float(x.multiply(x).sum())
    cross = float(np.sum(weights * (x @ topic_words.T)))
    model_norm2 = float(np.sum((weights.T @ weights) * (topic_words @ topic_words.T)))
    sse = max(0.0, x_norm2 - 2 * cross + model_norm2)
    return float(np.sqrt(sse / x_norm2)) if x_norm2 else float("nan")


def topic_diversity(components: np.ndarray, top_n: int = 10) -> float:
    top_ids = np.argsort(components, axis=1)[:, -top_n:]
    return len(set(top_ids.ravel().tolist())) / top_ids.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and compare newspaper topics with NMF.")
    parser.add_argument("--topics", default="5,8,10,12,15", help="Comma-separated topic counts")
    parser.add_argument("--min-df", type=int, default=3)
    args = parser.parse_args()

    train, val, test = (load_split(name) for name in ("train", "val", "test"))
    issue_sets = [set(df.issue_id) for df in (train, val, test)]
    if any(issue_sets[i] & issue_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("Issue leakage detected among train/val/test manifests")

    # Experimental isiXhosa stopwords
    
    EXPERIMENTAL_STOPWORDS = [
        "nje", "ngu", "ku", "kodwa",
        "xa", "ngo", "le", "kute",
        "kona", "mr", "sir"
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        min_df=args.min_df,
        max_df=0.90,
        max_features=20000,
        sublinear_tf=True,
        stop_words=EXPERIMENTAL_STOPWORDS,
    )
    x_train = vectorizer.fit_transform(train.text)
    x_val = vectorizer.transform(val.text)
    x_test = vectorizer.transform(test.text)
    vocabulary = vectorizer.get_feature_names_out()
    print(f"TF-IDF vocabulary: {len(vocabulary):,} training-derived terms")

    counts = sorted(set(int(s.strip()) for s in args.topics.split(",")))
    if any(n < 2 or n > min(x_train.shape) for n in counts):
        raise ValueError(f"Topic counts must be 2..{min(x_train.shape)}")
    metrics = []
    models = {}
    for n in counts:
        nmf = NMF(n_components=n, init="nndsvda", random_state=42, max_iter=600)
        nmf.fit(x_train)
        w_val = nmf.transform(x_val)
        error = relative_reconstruction_error(x_val, w_val, nmf.components_)
        diversity = topic_diversity(nmf.components_)
        metrics.append({"n_topics": n, "validation_relative_error": error,
                        "top10_diversity": diversity})
        models[n] = nmf
        print(f"{n:2d} topics | val reconstruction error: {error:.4f} | diversity: {diversity:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tuning = pd.DataFrame(metrics).sort_values("n_topics")
    tuning.to_csv(RESULTS_DIR / "topic_tuning.csv", index=False)
    # Parsimonious selection: smallest model within 3% of the best validation error.
    threshold = 1.03 * tuning.validation_relative_error.min()
    selected_k = int(tuning.loc[tuning.validation_relative_error <= threshold, "n_topics"].min())
    model = models[selected_k]
    print(f"Selected {selected_k} topics; inspect interpretability before finalising.")

    top_rows = []
    for i, weights in enumerate(model.components_):
        highest = weights.argsort()[::-1][:12]
        print(f"Topic {i + 1}: {', '.join(vocabulary[highest])}")
        for rank, idx in enumerate(highest, 1):
            top_rows.append({"topic": i + 1, "rank": rank,
                             "word": vocabulary[idx], "weight": weights[idx]})
    pd.DataFrame(top_rows).to_csv(RESULTS_DIR / "topic_top_words.csv", index=False)

    page_outputs = []
    topic_cols = [f"topic_{i+1}" for i in range(selected_k)]
    for label, frame, matrix in (("train", train, x_train), ("val", val, x_val), ("test", test, x_test)):
        raw = model.transform(matrix)
        normalised = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
        scores = pd.concat([
            frame[["filepath", "newspaper", "issue_id", "date"]].reset_index(drop=True),
            pd.DataFrame(normalised, columns=topic_cols),
        ], axis=1)
        scores.insert(0, "split", label)
        page_outputs.append(scores)
    pages = pd.concat(page_outputs, ignore_index=True)
    pages.to_csv(RESULTS_DIR / "topic_page_scores.csv", index=False)

    # Average pages within each issue first so a long issue is not overweighted.
    issues = pages.groupby(["split", "newspaper", "issue_id"], as_index=False)[topic_cols].mean()
    issues.to_csv(RESULTS_DIR / "topic_issue_scores.csv", index=False)
    test_issues = issues.loc[issues.split.eq("test")]
    source_means = test_issues.groupby("newspaper")[topic_cols].mean().T
    source_means.index = [f"Topic {i + 1}" for i in range(selected_k)]
    source_means.to_csv(RESULTS_DIR / "topic_newspaper_test.csv", index_label="topic")

    ax = source_means.plot(kind="bar", figsize=(max(9, selected_k * 0.85), 5), rot=0)
    ax.set_ylabel("Mean normalised topic weight per test issue")
    ax.set_xlabel("Unsupervised NMF topic")
    ax.set_title("Topic composition by newspaper (held-out issues)")
    ax.legend(title="Newspaper")
    plt.tight_layout()
    figure_dir = RESULTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / "nmf_topic_comparison.png", dpi=160)
    plt.close()
    print(f"Wrote topic results and figure under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
