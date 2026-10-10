"""Dataset loading and NLP preprocessing shared by training and prediction."""

from __future__ import annotations

import json
import re
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.get_data import METADATA_PATH, ensure_dataset

_NLTK_RESOURCES = {
    "corpora/stopwords": "stopwords",
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/wordnet.zip": "wordnet",
}
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+")
_PUNCTUATION_PATTERN = re.compile(r"[^a-zA-Z\s]")


def ensure_nltk_resources() -> None:
    """Download required NLTK resources and fail explicitly if unavailable."""
    for resource_path, package in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(
                    f"Could not download NLTK resource '{package}'. "
                    "Check network access and retry."
                )
            try:
                nltk.data.find(resource_path)
            except LookupError as error:
                raise RuntimeError(
                    f"NLTK reported success but resource '{package}' is unavailable."
                ) from error


@lru_cache(maxsize=1)
def _text_tools() -> tuple[frozenset[str], WordNetLemmatizer]:
    ensure_nltk_resources()
    return frozenset(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalize an email or SMS into lowercase, lemmatized word tokens."""
    stop_words, lemmatizer = _text_tools()
    normalized = _URL_PATTERN.sub(" ", text.lower())
    normalized = _NUMBER_PATTERN.sub(" ", normalized)
    normalized = _PUNCTUATION_PATTERN.sub(" ", normalized)
    tokens = word_tokenize(normalized, preserve_line=True)
    return " ".join(
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    )


def load_dataset() -> pd.DataFrame:
    """Load, validate, normalize, and clean the project dataset."""
    dataset_path = ensure_dataset()
    frame = pd.read_csv(dataset_path, encoding="latin-1")
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    if {"v1", "v2"}.issubset(normalized_columns):
        frame = frame.rename(
            columns={
                normalized_columns["v1"]: "label",
                normalized_columns["v2"]: "text",
            }
        )
    elif {"label", "text"}.issubset(normalized_columns):
        frame = frame.rename(
            columns={
                normalized_columns["label"]: "label",
                normalized_columns["text"]: "text",
            }
        )
    elif len(frame.columns) >= 2:
        frame = frame.rename(columns={frame.columns[0]: "label", frame.columns[1]: "text"})
    else:
        raise ValueError("Dataset must contain label and text columns (v1 and v2).")

    frame = frame.loc[:, ["label", "text"]].dropna().copy()
    frame["label"] = frame["label"].astype(str).str.strip().str.lower().map(
        {"ham": 0, "spam": 1, "0": 0, "1": 1}
    )
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame.dropna(subset=["label"])
    frame = frame[frame["text"].ne("")].drop_duplicates(subset=["label", "text"])
    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam examples.")
    frame["label"] = frame["label"].astype("int8")
    ensure_nltk_resources()
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].ne("")].reset_index(drop=True)
    if frame["label"].nunique() != 2:
        raise ValueError("Preprocessing left fewer than two message classes.")
    return frame


def get_dataset_source() -> str:
    """Return dataset provenance recorded by the acquisition step."""
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    return str(metadata.get("source", "unknown"))


if __name__ == "__main__":
    data = load_dataset()
    print(f"Loaded {len(data)} unique messages.")
    print(data["label"].value_counts().rename(index={0: "ham", 1: "spam"}))
