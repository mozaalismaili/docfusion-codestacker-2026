import os
import sys
import json
import pickle
import numpy as np
import re

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


class DocFusionSolution:

    def train(self, train_dir: str, work_dir: str) -> str:
        """Train model on data in train_dir and save to work_dir."""
        from extractor import extract_fields
        from anomaly import extract_features, train_model, save_model

        model_dir  = os.path.join(work_dir, "model")
        train_jsonl = os.path.join(train_dir, "train.jsonl")
        images_dir  = os.path.join(train_dir, "images")

        X, y = [], []

        with open(train_jsonl, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        for record in records:
            img_id    = record["id"]
            is_forged = int(record.get("is_forged", 0))

            # Find image file
            img_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = os.path.join(images_dir, img_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is None:
                continue

            extracted = extract_fields(img_path)
            features  = extract_features(extracted, img_path)
            X.append(features)
            y.append(is_forged)

        if len(X) == 0:
            # No training data found, save dummy model
            os.makedirs(model_dir, exist_ok=True)
            return model_dir

        X = np.array(X)
        y = np.array(y)

        clf, scaler = train_model(X, y)
        save_model(clf, scaler, model_dir)

        return model_dir

    def predict(self, model_dir: str, data_dir: str, out_path: str) -> None:
        """Run inference and write predictions to out_path."""
        from extractor import extract_fields
        from anomaly import extract_features, load_model, predict_forgery

        test_jsonl = os.path.join(data_dir, "test.jsonl")
        images_dir = os.path.join(data_dir, "images")

        # Load model if available
        clf, scaler = None, None
        clf_path    = os.path.join(model_dir, "clf.pkl")
        if os.path.exists(clf_path):
            clf, scaler = load_model(model_dir)

        with open(test_jsonl, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        with open(out_path, "w", encoding="utf-8") as out_f:
            for record in records:
                img_id = record["id"]

                # Find image file
                img_path = None
                for ext in [".jpg", ".png", ".jpeg"]:
                    candidate = os.path.join(images_dir, img_id + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break

                # Extract fields
                if img_path:
                    extracted = extract_fields(img_path)
                else:
                    extracted = {
                        "vendor": None,
                        "date": None,
                        "total": None,
                        "raw_lines": []
                    }

                # Predict forgery
                if clf and scaler and img_path:
                    features  = extract_features(extracted, img_path)
                    is_forged = predict_forgery(clf, scaler, extracted)
                else:
                    is_forged = 0

                prediction = {
                    "id":        img_id,
                    "vendor":    extracted.get("vendor"),
                    "date":      extracted.get("date"),
                    "total":     extracted.get("total"),
                    "is_forged": is_forged
                }

                out_f.write(json.dumps(prediction) + "\n")