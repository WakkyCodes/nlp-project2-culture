
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.newspaper_classifier import (
    load_split,
    SOURCE_WORDS,
    issue_sample_weights,
    issue_predictions,
    metrics,
)

START = "1884-11-03"
END = "1888-12-01"

def extract_features(text):
    words = re.findall(r"(?u)[^\W\d_]+", text)
    chars = [c for c in text if not c.isspace()]

    mean_length = (
        np.mean([len(w) for w in words])
        if words else 0.0
    )

    non_alpha_ratio = (
        sum(not c.isalpha() for c in chars)
        / max(len(chars), 1)
    )

    return [mean_length, non_alpha_ratio]


train = load_split("train", START, END)
val = load_split("val", START, END)

# Match the classifier's expanded source masking.
for frame in (train, val):
    frame["text"] = frame["text"].map(
        lambda t: SOURCE_WORDS.sub(" ", t)
    )

X_train = np.array(
    [extract_features(t) for t in train["text"]]
)
X_val = np.array(
    [extract_features(t) for t in val["text"]]
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

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

probabilities = model.predict_proba(X_val)[:, 1]
predictions = issue_predictions(val, probabilities)

result = metrics(predictions)

print("\nOCR-proxy-only validation baseline")
print("Issues:", result["n_issues"])
print("Macro-F1:", round(result["macro_f1"], 3))
print(
    "Balanced accuracy:",
    round(result["balanced_accuracy"], 3)
)
print(
    "Confusion matrix:",
    result["confusion_matrix_IMVO_ISIG"]
)
