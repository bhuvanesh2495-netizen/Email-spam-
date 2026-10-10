"""Fast cached email classification and an interactive command-line interface."""

from __future__ import annotations

import json
from functools import lru_cache
from math import exp
from typing import Any

import joblib
from src.preprocess import clean_text
from src.train import MODEL_PATH, MODELS_DIR, VECTORIZER_PATH

MAX_EMAIL_CHARS = 20_000


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[Any, Any, str]:
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(
            "Model artifacts are missing. Run 'python -m src.train' first."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata_path = MODELS_DIR / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read model metadata at {metadata_path}.") from error
    return model, vectorizer, str(metadata.get("best_model", "unknown"))


def classify_email(text: str) -> dict[str, str | float]:
    """Classify one message and return its predicted class and confidence."""
    if not isinstance(text, str):
        raise TypeError("Email text must be a string.")
    if not text.strip():
        raise ValueError("Email text must not be empty.")
    if len(text) > MAX_EMAIL_CHARS:
        raise ValueError(f"Email text must be at most {MAX_EMAIL_CHARS} characters.")

    model, vectorizer, _ = _load_artifacts()
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = int(model.predict(features)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
    elif hasattr(model, "decision_function"):
        margin = float(model.decision_function(features).reshape(-1)[0])
        confidence = float(1.0 / (1.0 + exp(-min(abs(margin), 60.0))))
    else:
        confidence = 1.0
    return {
        "label": "Spam" if prediction == 1 else "Ham",
        "confidence": round(min(max(confidence, 0.0), 1.0), 4),
    }


def get_top_spam_keywords(limit: int = 20) -> list[dict[str, float | str]]:
    """Return the strongest spam-associated features from the selected model."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    model, vectorizer, _ = _load_artifacts()
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, "coef_"):
        scores = model.coef_[0]
    elif hasattr(model, "feature_log_prob_"):
        scores = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    elif hasattr(model, "calibrated_classifiers_"):
        coefficients = [
            calibrated.estimator.coef_[0]
            for calibrated in model.calibrated_classifiers_
            if hasattr(calibrated.estimator, "coef_")
        ]
        if not coefficients:
            raise TypeError("The calibrated model does not expose interpretable features.")
        scores = sum(coefficients) / len(coefficients)
    else:
        raise TypeError("The selected model does not expose interpretable features.")
    indexes = scores.argsort()[-limit:][::-1]
    return [
        {"keyword": str(feature_names[index]), "score": float(scores[index])}
        for index in indexes
    ]


def main() -> None:
    print("Email Spam Classifier (type 'quit' to exit)")
    while True:
        try:
            message = input("\nEmail text: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.strip().lower() in {"quit", "exit"}:
            break
        try:
            result = classify_email(message)
        except (TypeError, ValueError, FileNotFoundError, RuntimeError) as error:
            print(f"Error: {error}")
            continue
        print(f"{result['label']} (confidence: {result['confidence']:.1%})")


if __name__ == "__main__":
    main()
