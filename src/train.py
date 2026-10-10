"""Train and persist three lightweight text classification models."""

from __future__ import annotations

import json
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.evaluate import (
    CLASSIFIERS_PATH,
    EVALUATION_DATA_PATH,
    build_report,
)
from src.get_data import PROJECT_ROOT, ensure_dataset
from src.preprocess import get_dataset_source, load_dataset

MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def train() -> dict[str, Any]:
    ensure_dataset()
    dataset = load_dataset()
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)
    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=42
        ),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(random_state=42), method="sigmoid", cv=3
        ),
    }
    for name, classifier in classifiers.items():
        classifier.fit(x_train, y_train)
        print(f"Trained {name}.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH, compress=3)
    joblib.dump(classifiers, CLASSIFIERS_PATH, compress=3)
    joblib.dump(
        {"x_test": x_test, "y_test": y_test.to_numpy()},
        EVALUATION_DATA_PATH,
        compress=3,
    )

    report = build_report()
    best_name = report["best_model"]
    joblib.dump(classifiers[best_name], MODEL_PATH, compress=3)
    metadata = {
        "best_model": best_name,
        "dataset_source": get_dataset_source(),
        "dataset_rows": int(len(dataset)),
        "ham_count": int((dataset["label"] == 0).sum()),
        "spam_count": int((dataset["label"] == 1).sum()),
        "test_samples": int(len(y_test)),
        "feature_count": int(len(vectorizer.get_feature_names_out())),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Saved {best_name}, vectorizer, and evaluation artifacts to {MODELS_DIR}."
    )
    return report


def main() -> None:
    train()


if __name__ == "__main__":
    main()
