
from sklearn.linear_model import LogisticRegression

from src.models.newspaper_classifier import (
    load_split,
    SOURCE_WORDS,
    vectorizer_for,
    issue_sample_weights,
    issue_predictions,
    metrics,
)

# Use the same historical overlap as the classifier.
START = "1884-11-03"
END = "1888-12-01"

train = load_split("train", START, END)
val = load_split("val", START, END)

# Apply the same expanded source-name masking.
for frame in (train, val):
    frame["text"] = frame["text"].map(
        lambda text: SOURCE_WORDS.sub(" ", text)
    )

# Reproduce the selected classifier.
vectorizer = vectorizer_for((1, 1))
X_train = vectorizer.fit_transform(train["text"])

model = LogisticRegression(
    C=0.1,
    max_iter=3000,
    random_state=42
)

model.fit(
    X_train,
    (train["newspaper"] == "ISIG").astype(int),
    sample_weight=issue_sample_weights(train)
)

# Identify pages 3 and 4.
is_page34 = val["filepath"].str.contains(
    r"_p00[34]\.txt$",
    case=False,
    regex=True
)

# Retain only issues with an eligible page 3 or 4.
matched_ids = val.loc[
    is_page34, ["newspaper", "issue_id"]
].drop_duplicates()

matched = val.merge(
    matched_ids,
    on=["newspaper", "issue_id"],
    how="inner"
)

page34 = matched[
    matched["filepath"].str.contains(
        r"_p00[34]\.txt$",
        case=False,
        regex=True
    )
].copy()

# Evaluate both conditions on the same issues.
for name, dataset in [
    ("All pages from matched issues", matched),
    ("Only pages 3 and 4", page34),
]:
    X = vectorizer.transform(dataset["text"])
    probabilities = model.predict_proba(X)[:, 1]

    predictions = issue_predictions(
        dataset,
        probabilities
    )

    print(f"\n{name}")
    print("Issues:", len(predictions))
    print("Macro-F1:", round(metrics(predictions)["macro_f1"], 3))
    print(
        "Balanced accuracy:",
        round(metrics(predictions)["balanced_accuracy"], 3)
    )
    print(
        "Confusion matrix:",
        metrics(predictions)["confusion_matrix_IMVO_ISIG"]
    )
