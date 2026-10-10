"""Dataset acquisition and deterministic demo-data fallback."""

from __future__ import annotations

import csv
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "spam.csv"
METADATA_PATH = DATA_DIR / "dataset_metadata.json"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

_SPAM_TEMPLATES = [
    "Congratulations {name}, you have won {prize}! Claim your free gift now at {url}",
    "URGENT: your {account} account is suspended. Verify your details immediately: {url}",
    "Get {discount}% off {product} today only. Shop now at {url}",
    "You have been selected for a {prize}. Reply YES or visit {url} to claim",
    "Your bank detected unusual activity. Confirm your account securely at {url}",
    "Limited time offer: free {product} for the first {count} customers. Call {phone}",
    "Final notice! Claim your cash reward of {amount} before it expires. {url}",
    "Exclusive deal for {name}: buy one {product} and get one free. {url}",
    "We could not deliver your parcel. Pay a small redelivery fee here: {url}",
    "You qualify for a low-rate loan of {amount}. Apply today: {url}",
    "Your mobile number won a {prize}. Call {phone} now to collect",
    "Act now to earn extra income from home. Sign up for free: {url}",
    "Security alert: reset your password now to avoid account closure: {url}",
    "Get a free ringtone and {amount} in airtime credit. Text WIN to {phone}",
    "Hot singles near you want to chat. Join for free at {url}",
]
_HAM_TEMPLATES = [
    "Hi {name}, are we still meeting at {time} {day}?",
    "I will be {place} around {time}. Let me know if that works for you.",
    "Can you please send me the {document} before the meeting?",
    "Thanks for your help with {topic}. I really appreciate it.",
    "Mum asked if you can call her when you get home.",
    "The team review has moved to {day} at {time}. See you there.",
    "I picked up {item} on my way back. Do you need anything else?",
    "Good morning {name}, I have attached the notes from yesterday.",
    "Running a little late, but I should be there in about {count} minutes.",
    "Would you like to get lunch at {place} after work today?",
    "Please remember to bring your {item} when you visit this weekend.",
    "The project update looks good. I will share it with the team.",
    "I am on the train now. I will text you when I arrive.",
    "Happy birthday! Hope you have a lovely day with everyone.",
    "The appointment is confirmed for {day} at {time}.",
    "Could you review the attached {document} and send me your thoughts?",
]
_VALUES = {
    "name": ["Alex", "Sam", "Jordan", "Priya", "Taylor", "Jamie", "Casey"],
    "prize": ["£500", "$1000", "a new phone", "a holiday", "a gift card"],
    "url": [
        "http://claim-now.example",
        "https://secure-prize.example",
        "http://account-check.example",
        "https://offer-today.example",
    ],
    "account": ["bank", "email", "mobile", "online"],
    "discount": ["30", "50", "70", "80"],
    "product": ["tickets", "headphones", "a tablet", "fashion", "subscriptions"],
    "count": ["5", "10", "15", "20"],
    "phone": ["08001234567", "09001234567", "07700900123"],
    "amount": ["£2000", "$500", "£10000", "$250"],
    "time": ["9:30", "10:00", "12:15", "2 pm", "6:45"],
    "day": ["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"],
    "place": ["the station", "the office", "town", "the cafe"],
    "document": ["report", "agenda", "draft", "presentation"],
    "topic": ["the presentation", "the schedule", "the code", "the event"],
    "item": ["keys", "laptop", "groceries", "charger"],
}


def _write_metadata(source: str, rows: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(
        json.dumps({"source": source, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def _download_uci_dataset() -> int:
    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "email-spam-classifier/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("SMSSpamCollection")),
            None,
        )
        if member is None:
            raise ValueError("The UCI archive does not contain SMSSpamCollection.")
        content = archive.read(member).decode("latin-1")

    rows: list[tuple[str, str]] = []
    for row in csv.reader(io.StringIO(content), delimiter="\t"):
        if len(row) >= 2 and row[0].strip().lower() in {"ham", "spam"}:
            rows.append((row[0].strip().lower(), row[1]))
    if not rows:
        raise ValueError("The UCI dataset contained no valid labeled messages.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="latin-1") as output:
        writer = csv.writer(output)
        writer.writerow(["v1", "v2"])
        writer.writerows(rows)
    _write_metadata("real: UCI SMS Spam Collection", len(rows))
    return len(rows)


def _generate_fallback_dataset() -> int:
    rng = random.Random(228)
    rows: list[tuple[str, str]] = []

    def make_message(template: str) -> str:
        return template.format(
            **{key: rng.choice(values) for key, values in _VALUES.items()}
        )

    for _ in range(390):
        template = rng.choice(_SPAM_TEMPLATES)
        message = make_message(template)
        if rng.random() < 0.45:
            message = rng.choice(["", "!!! ", "URGENT: "]) + message
        rows.append(("spam", message))

    for _ in range(2610):
        template = rng.choice(_HAM_TEMPLATES)
        message = make_message(template)
        if rng.random() < 0.35:
            message += rng.choice([" Thanks!", " See you soon.", ""])
        rows.append(("ham", message))

    rng.shuffle(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="", encoding="latin-1") as output:
        writer = csv.writer(output)
        writer.writerow(["v1", "v2"])
        writer.writerows(rows)
    _write_metadata("synthetic: generated fallback demo messages", len(rows))
    print(
        "WARNING: The UCI dataset could not be downloaded. A synthetic fallback "
        "dataset was generated; all model metrics are demo-only and do not "
        "represent real-world performance."
    )
    return len(rows)


def ensure_dataset() -> Path:
    """Return the project dataset, downloading UCI data or generating a fallback."""
    if DATA_PATH.is_file():
        if not METADATA_PATH.exists():
            try:
                import pandas as pd

                rows = len(pd.read_csv(DATA_PATH, encoding="latin-1"))
            except (OSError, ValueError, ImportError):
                rows = 0
            _write_metadata("existing: user-provided dataset (source not verified)", rows)
        return DATA_PATH

    try:
        count = _download_uci_dataset()
        print(f"Downloaded {count} messages from the UCI SMS Spam Collection.")
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"WARNING: Unable to download the UCI dataset: {error}")
        _generate_fallback_dataset()
    return DATA_PATH


if __name__ == "__main__":
    print(ensure_dataset())
