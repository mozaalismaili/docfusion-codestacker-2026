import os
import sys
import json
import csv
import pickle
import numpy as np
import re
from pathlib import Path


# ─── Image Feature Extractor─────────────────────────────

def get_image_features(image_path: str) -> np.ndarray:
    """
    Extract rich image features using OpenCV and scikit-image.
    No deep learning needed — uses classical computer vision.
    
    Features extracted:
    - Noise and sharpness analysis
    - Local Binary Patterns (texture)
    - Frequency domain (FFT) analysis
    - Edge and gradient statistics
    - Block inconsistency (detects copy-paste regions)
    - Compression artifact analysis
    """
    import cv2
    from skimage.feature import local_binary_pattern

    try:
        img = cv2.imread(image_path)
        if img is None:
            return np.zeros(50)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        features = []

        # --- Sharpness & Noise ---
        # Laplacian variance: low = blurry, high = sharp
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        features.append(lap.var())
        features.append(lap.mean())

        # --- Brightness Statistics ---
        features.append(float(gray.mean()))
        features.append(float(gray.std()))
        features.append(float(gray.min()))
        features.append(float(gray.max()))

        # --- Edge Analysis (Canny) ---
        edges = cv2.Canny(gray, 50, 150)
        features.append(float(edges.mean()))
        features.append(float(edges.std()))

        # --- Gradient Analysis (Sobel) ---
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features.append(gradient_mag.mean())
        features.append(gradient_mag.std())
        features.append(gradient_mag.max())

        # --- Local Binary Patterns (Texture) ---
        # LBP detects texture inconsistencies caused by copy-paste
        radius   = 3
        n_points = 8 * radius
        lbp      = local_binary_pattern(gray, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)
        features.extend(lbp_hist.tolist())  # 26 features

        # --- Frequency Domain Analysis (FFT) ---
        # Forged images often have unusual frequency patterns
        fft        = np.fft.fft2(gray)
        fft_shift  = np.fft.fftshift(fft)
        magnitude  = np.abs(fft_shift)
        log_mag    = np.log1p(magnitude)
        features.append(log_mag.mean())
        features.append(log_mag.std())
        features.append(log_mag.max())

        # --- Block Inconsistency Analysis ---
        # Split image into 4 blocks and compare statistics
        # Copy-paste forgeries create blocks with different noise levels
        blocks = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:]
        ]
        block_means = [b.mean() for b in blocks]
        block_stds  = [b.std()  for b in blocks]
        block_vars  = [b.var()  for b in blocks]

        features.append(max(block_means) - min(block_means))  # brightness inconsistency
        features.append(max(block_stds)  - min(block_stds))   # texture inconsistency
        features.append(max(block_vars)  - min(block_vars))   # variance inconsistency
        features.append(np.std(block_means))
        features.append(np.std(block_stds))

        # --- JPEG Artifact Analysis ---
        # Re-encode image at low quality and measure difference
        # Forged images often have double-compression artifacts
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
        _, encoded    = cv2.imencode(".jpg", gray, encode_params)
        decoded       = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        diff          = np.abs(gray.astype(float) - decoded.astype(float))
        features.append(diff.mean())
        features.append(diff.std())
        features.append(diff.max())
        # Ensure exactly 48 image features always
        result = np.array(features, dtype=np.float32)
        if len(result) < 48:
            result = np.pad(result, (0, 48 - len(result)))
        return result[:48]

    except Exception as e:
        return np.zeros(48, dtype=np.float32)



# ─── Text Feature Extractor ──────────────────────────────────────────────────

def extract_text_features(record: dict) -> list:
    """Extract 10 text-based features from a receipt record."""
    features = []

    features.append(0 if record.get("vendor") else 1)
    features.append(0 if record.get("date") else 1)
    features.append(0 if record.get("total") else 1)

    total = record.get("total")
    try:
        total_val = float(str(total).replace(",", ".")) if total else 0.0
    except ValueError:
        total_val = 0.0
    features.append(total_val)
    features.append(np.log1p(total_val))

    raw_lines = record.get("raw_lines", [])
    features.append(len(raw_lines))

    avg_len = sum(len(l) for l in raw_lines) / len(raw_lines) if raw_lines else 0.0
    features.append(avg_len)

    amount_pattern = r"\b\d{1,6}\.\d{2}\b"
    amount_count   = sum(len(re.findall(amount_pattern, line)) for line in raw_lines)
    features.append(amount_count)

    if total and total_val > 0:
        total_str         = f"{total_val:.2f}"
        total_occurrences = sum(total_str in line for line in raw_lines)
    else:
        total_occurrences = 0
    features.append(total_occurrences)

    numeric_lines = sum(1 for line in raw_lines if re.search(r"\d", line))
    features.append(numeric_lines)

    return features


def extract_features(record: dict, image_path: str = None) -> np.ndarray:
    """Combine text features + image features. Always 58 features total."""
    text_features = np.array(extract_text_features(record), dtype=np.float32)  # 10

    if image_path and os.path.exists(image_path):
        image_features = get_image_features(image_path)  # 48
    else:
        image_features = np.zeros(48, dtype=np.float32)

    combined = np.concatenate([text_features, image_features])

    # Safety check — always return exactly 58 features
    if len(combined) < 58:
        combined = np.pad(combined, (0, 58 - len(combined)))
    return combined[:58]


# ─── Model Training ──────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray):
    """Train using XGBoost with scale_pos_weight for imbalance."""
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    # Calculate imbalance ratio
    n_genuine = (y == 0).sum()
    n_forged  = (y == 1).sum()
    scale     = n_genuine / n_forged
    print(f"Class ratio: {scale:.1f}x (genuine/forged)")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale,  # handles imbalance automatically
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    clf.fit(X_scaled, y)

    return clf, scaler


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_forgery(clf, scaler, record: dict, image_path: str = None) -> int:
    """Predict if a receipt is forged. Returns 0 or 1."""
    features = extract_features(record, image_path)
    X        = features.reshape(1, -1)
    X_scaled = scaler.transform(X)
    return int(clf.predict(X_scaled)[0])


# ─── Save / Load ─────────────────────────────────────────────────────────────

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


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_score
    sys.path.insert(0, str(Path(__file__).parent))
    from extractor import extract_fields

    finditagain_root = Path("data/finditagain")

    print("Building features from Find-It-Again train set...")
    X_train, y_train = [], []

    with open(finditagain_root / "train.txt", encoding="utf-8") as f:
        reader     = csv.reader(f)
        next(reader)
        train_rows = [
            (row[0], int(row[3]))
            for row in reader
            if len(row) >= 4 and row[3].strip().isdigit()
        ]

    for filename, is_forged in train_rows:
        img_path = finditagain_root / "train" / filename
        if not img_path.exists():
            continue
        extracted = extract_fields(str(img_path))
        features  = extract_features(extracted, str(img_path))
        X_train.append(features)
        y_train.append(is_forged)
        if len(X_train) % 50 == 0:
            print(f"  Processed {len(X_train)} train images...")

    print("\nBuilding features from Find-It-Again val set...")
    X_val, y_val = [], []

    with open(finditagain_root / "val.txt", encoding="utf-8") as f:
        reader   = csv.reader(f)
        next(reader)
        val_rows = [
            (row[0], int(row[3]))
            for row in reader
            if len(row) >= 4 and row[3].strip().isdigit()
        ]

    for filename, is_forged in val_rows:
        img_path = finditagain_root / "val" / filename
        if not img_path.exists():
            continue
        extracted = extract_fields(str(img_path))
        features  = extract_features(extracted, str(img_path))
        X_val.append(features)
        y_val.append(is_forged)
        if len(X_val) % 50 == 0:
            print(f"  Processed {len(X_val)} val images...")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val   = np.array(X_val)
    y_val   = np.array(y_val)

    print(f"\nTrain: {len(y_train)} samples, {y_train.sum()} forged")
    print(f"Val  : {len(y_val)} samples, {y_val.sum()} forged")

    clf, scaler = train_model(X_train, y_train)

    X_val_scaled = scaler.transform(X_val)
    y_pred       = clf.predict(X_val_scaled)

    print("\nValidation set performance:")
    print(classification_report(y_val, y_pred, target_names=["Genuine", "Forged"]))

    scores = cross_val_score(
        clf, scaler.transform(X_train), y_train, cv=5, scoring="f1"
    )
    print(f"Cross-validation F1: {scores.mean():.2f} (+/- {scores.std():.2f})")

    save_model(clf, scaler, "models/anomaly")