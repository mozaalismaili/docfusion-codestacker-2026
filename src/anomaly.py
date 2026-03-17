import os
import json
import csv
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import re


def extract_features(record: dict) -> list:
    """
    Convert a receipt record into a feature vector for the ML model.
    record should have: vendor, date, total, raw_lines
    """
    features = []

    # Feature 1: is vendor missing?
    features.append(0 if record.get("vendor") else 1)

    # Feature 2: is date missing?
    features.append(0 if record.get("date") else 1)

    # Feature 3: is total missing?
    features.append(0 if record.get("total") else 1)

    # Feature 4: total amount value (0 if missing)
    total = record.get("total")
    try:
        total_val = float(str(total).replace(",", ".")) if total else 0.0
    except ValueError:
        total_val = 0.0
    features.append(total_val)

    # Feature 5: log of total (helps with large value outliers)
    features.append(np.log1p(total_val))

    # Feature 6: number of OCR text lines
    raw_lines = record.get("raw_lines", [])
    features.append(len(raw_lines))

    # Feature 7: average line length
    if raw_lines:
        avg_len = sum(len(l) for l in raw_lines) / len(raw_lines)
    else:
        avg_len = 0.0
    features.append(avg_len)

    # Feature 8: number of amount-like patterns in text
    amount_pattern = r"\b\d{1,6}\.\d{2}\b"
    amount_count = sum(
        len(re.findall(amount_pattern, line))
        for line in raw_lines
    )
    features.append(amount_count)

    # Feature 9: does total appear multiple times? (suspicious)
    if total and total_val > 0:
        total_str = f"{total_val:.2f}"
        total_occurrences = sum(total_str in line for line in raw_lines)
    else:
        total_occurrences = 0
    features.append(total_occurrences)

    # Feature 10: number of lines with numbers
    numeric_lines = sum(
        1 for line in raw_lines
        if re.search(r"\d", line)
    )
    features.append(numeric_lines)

    return features


def load_training_data(train_dir: str, extract_fn):
    """
    Load training data from train_dir.
    train_dir should contain train.jsonl and images/
    extract_fn is the extract_fields function from extractor.py
    """
    train_jsonl = os.path.join(train_dir, "train.jsonl")
    images_dir  = os.path.join(train_dir, "images")

    X, y = [], []

    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            img_id    = record["id"]
            is_forged = int(record.get("is_forged", 0))

            # Find the image file
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(images_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                continue

            # Extract fields from image
            extracted = extract_fn(img_path)

            # Build feature vector
            features = extract_features(extracted)
            X.append(features)
            y.append(is_forged)

    return np.array(X), np.array(y)


def train_model(X: np.ndarray, y: np.ndarray):
    """Train a Random Forest classifier with class balancing."""
    # Balance the dataset since forged receipts are only ~16%
    X_genuine = X[y == 0]
    y_genuine = y[y == 0]
    X_forged  = X[y == 1]
    y_forged  = y[y == 1]

    # Upsample forged class to match genuine count
    X_forged_up, y_forged_up = resample(
        X_forged, y_forged,
        replace=True,
        n_samples=len(X_genuine),
        random_state=42
    )

    X_balanced = np.vstack([X_genuine, X_forged_up])
    y_balanced = np.hstack([y_genuine, y_forged_up])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_scaled, y_balanced)

    return clf, scaler


def predict_forgery(clf, scaler, record: dict) -> int:
    """Predict if a receipt is forged. Returns 0 or 1."""
    features = extract_features(record)
    X = np.array([features])
    X_scaled = scaler.transform(X)
    return int(clf.predict(X_scaled)[0])


def save_model(clf, scaler, model_dir: str):
    """Save model and scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "clf.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {model_dir}")


def load_model(model_dir: str):
    """Load model and scaler from disk."""
    with open(os.path.join(model_dir, "clf.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler


if __name__ == "__main__":
    # Quick test using Find-It-Again data directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from extractor import extract_fields

    finditagain_root = Path("data/finditagain")
    img_dir          = finditagain_root / "train"
    label_file       = finditagain_root / "train.txt"

    print("Building features from Find-It-Again train set...")
    X, y = [], []

    import csv
    with open(label_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = [(row[0], int(row[3])) for row in reader if len(row) >= 4 and row[3].strip().isdigit()]

    for filename, is_forged in rows[:100]:  # test on first 100 for speed
        img_path = img_dir / filename
        if not img_path.exists():
            continue
        extracted = extract_fields(str(img_path))
        features  = extract_features(extracted)
        X.append(features)
        y.append(is_forged)
        if len(X) % 20 == 0:
            print(f"  Processed {len(X)} images...")

    X = np.array(X)
    y = np.array(y)

    print(f"\nTotal samples: {len(y)}, Forged: {y.sum()}, Genuine: {(y==0).sum()}")

    clf, scaler = train_model(X, y)

    # Quick evaluation
    from sklearn.metrics import classification_report
    X_scaled = scaler.transform(X)
    y_pred   = clf.predict(X_scaled)
    print("\nTraining set performance:")
    print(classification_report(y, y_pred, target_names=["Genuine", "Forged"]))

    # Save model
    save_model(clf, scaler, "models/anomaly")