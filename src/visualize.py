"""Generate model interpretation and evaluation charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from src.get_data import PROJECT_ROOT
from src.preprocess import load_dataset
from src.predict import get_top_spam_keywords
from src.train import MODEL_PATH, VECTORIZER_PATH

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_PATH = OUTPUTS_DIR / "results.json"

sns.set_theme(style="whitegrid")


def visualize() -> list[Path]:
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Run 'python -m src.train' before visualization.")
    if not RESULTS_PATH.exists():
        raise FileNotFoundError("Run 'python -m src.evaluate' before visualization.")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    keyword_data = get_top_spam_keywords()
    keywords = [item["keyword"] for item in keyword_data]
    scores = [item["score"] for item in keyword_data]
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(keywords[::-1], scores[::-1], color="#ef5350")
    axis.set_title("Top spam-indicating keywords")
    axis.set_xlabel("Spam association score")
    fig.tight_layout()
    path = OUTPUTS_DIR / "top_spam_keywords.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    data = load_dataset()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for label, axis, color in ((1, axes[0], "Reds"), (0, axes[1], "Greens")):
        corpus = " ".join(data.loc[data["label"] == label, "text"].tolist())
        cloud = WordCloud(
            width=900,
            height=550,
            background_color="white",
            colormap=color,
            max_words=100,
            collocations=False,
        ).generate(corpus)
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_title("Spam messages" if label else "Ham messages")
        axis.axis("off")
    fig.tight_layout()
    path = OUTPUTS_DIR / "wordclouds.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    with RESULTS_PATH.open(encoding="utf-8") as source:
        report = json.load(source)
    comparison = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")
    best = report["best_model"]
    matrix = report["models"][best]["confusion_matrix"]
    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=axis,
    )
    axis.set(title=f"Confusion matrix — {best}", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    path = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    fig, axis = plt.subplots(figsize=(10, 6))
    comparison.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", ax=axis, ylim=(0, 1.05), color=["#5c6bc0", "#26a69a", "#ffa726", "#ab47bc"]
    )
    axis.set_title("Classifier performance comparison")
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    axis.legend(loc="lower right")
    fig.tight_layout()
    path = OUTPUTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    generated.append(path)

    print("Generated visualizations:")
    for path in generated:
        print(f"  {path}")
    return generated


def main() -> None:
    visualize()


if __name__ == "__main__":
    main()
