"""Flask web application for interactive spam classification."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

from src.get_data import PROJECT_ROOT
from src.predict import MAX_EMAIL_CHARS, classify_email, get_top_spam_keywords
from src.train import MODEL_PATH, MODELS_DIR, VECTORIZER_PATH, train

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REQUIRED_PLOTS = (
    "top_spam_keywords.png",
    "wordclouds.png",
    "confusion_matrix.png",
    "model_comparison.png",
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


def ensure_artifacts() -> None:
    if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
        train()
    missing_plots = any(not (OUTPUTS_DIR / name).is_file() for name in REQUIRED_PLOTS)
    if missing_plots:
        from src.visualize import visualize

        visualize()


def _comparison_rows() -> list[dict[str, str]]:
    comparison_path = OUTPUTS_DIR / "model_comparison.csv"
    if not comparison_path.is_file():
        return []
    with comparison_path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


@app.get("/")
def index() -> str:
    metadata_path = MODELS_DIR / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return render_template(
        "index.html",
        keywords=get_top_spam_keywords(12),
        comparison=_comparison_rows(),
        best_model=metadata.get("best_model", "Not available"),
        dataset_source=metadata.get("dataset_source", "Unknown"),
        plots=REQUIRED_PLOTS,
    )


@app.post("/predict")
def predict():
    try:
        payload = request.get_json()
    except BadRequest:
        return jsonify({"error": "Request body must contain valid JSON."}), 400
    except UnsupportedMediaType:
        return jsonify({"error": "Content-Type must be application/json."}), 415
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    text = payload.get("text", payload.get("email"))
    if not isinstance(text, str):
        return jsonify({"error": "Provide email text as a string in 'text'."}), 400
    if not text.strip():
        return jsonify({"error": "Email text must not be empty."}), 400
    if len(text) > MAX_EMAIL_CHARS:
        return (
            jsonify(
                {"error": f"Email text must be at most {MAX_EMAIL_CHARS} characters."}
            ),
            413,
        )
    try:
        return jsonify(classify_email(text))
    except (FileNotFoundError, RuntimeError) as error:
        app.logger.exception("Classification artifacts are unavailable.")
        return jsonify({"error": str(error)}), 500


@app.get("/api/keywords")
def keywords():
    return jsonify({"keywords": get_top_spam_keywords(20)})


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request body is too large."}), 413


ensure_artifacts()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
