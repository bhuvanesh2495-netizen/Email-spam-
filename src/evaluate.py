"""Evaluation and reporting for the trained spam classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.get_data import PROJECT_ROOT
from src.preprocess import get_dataset_source

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EVALUATION_DATA_PATH = MODELS_DIR / "evaluation_data.joblib"
CLASSIFIERS_PATH = MODELS_DIR / "classifiers.joblib"


def calculate_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": (
            float(false_positive / (false_positive + true_negative))
            if false_positive + true_negative
            else 0.0
        ),
        "false_positives": false_positive,
        "confusion_matrix": matrix.tolist(),
    }


def build_report() -> dict[str, Any]:
    if not EVALUATION_DATA_PATH.exists() or not CLASSIFIERS_PATH.exists():
        raise FileNotFoundError(
            "Training artifacts are missing. Run 'python -m src.train' first."
        )

    evaluation_data = joblib.load(EVALUATION_DATA_PATH)
    classifiers = joblib.load(CLASSIFIERS_PATH)
    x_test, y_test = evaluation_data["x_test"], evaluation_data["y_test"]
    model_metrics = {
        name: calculate_metrics(y_test, model.predict(x_test))
        for name, model in classifiers.items()
    }
    best_model = min(
        model_metrics,
        key=lambda name: (
            -model_metrics[name]["f1"],
            model_metrics[name]["false_positive_rate"],
            name,
        ),
    )
    rows = [
        {
            "model": name,
            **{key: value for key, value in values.items() if key != "confusion_matrix"},
        }
        for name, values in model_metrics.items()
    ]
    comparison = pd.DataFrame(rows).sort_values(
        ["f1", "false_positive_rate"], ascending=[False, True]
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)

    report = {
        "dataset_source": get_dataset_source(),
        "best_model": best_model,
        "test_samples": int(len(y_test)),
        "models": model_metrics,
    }
    (OUTPUTS_DIR / "results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFalse-positive rate is the share of ham messages incorrectly labeled spam.")
    print(f"Selected model: {best_model}")
    print(f"Saved results to {OUTPUTS_DIR}")
    return report


def main() -> None:
    build_report()


if __name__ == "__main__":
    main()
