import json
import re
from pathlib import Path
from extractor import extract_fields

def normalize_total(val):
    if val is None:
        return None
    return re.sub(r"[^\d\.]", "", str(val)).strip()

def normalize_vendor(val):
    if val is None:
        return None
    return val.upper().strip()

def normalize_date(val):
    if val is None:
        return None
    return val.strip()

def evaluate_sroie(img_dir: Path, entities_dir: Path):
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    results = {"vendor": 0, "date": 0, "total": 0, "count": 0}

    for img_path in images:
        label_path = entities_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path, encoding="utf-8") as f:
            ground_truth = json.load(f)

        predicted = extract_fields(str(img_path))
        results["count"] += 1

        # Check vendor
        gt_vendor = normalize_vendor(ground_truth.get("company", ""))
        pr_vendor = normalize_vendor(predicted.get("vendor", ""))
        if gt_vendor and pr_vendor and gt_vendor in pr_vendor or (pr_vendor and pr_vendor in gt_vendor):
            results["vendor"] += 1

        # Check date
        gt_date = normalize_date(ground_truth.get("date", ""))
        pr_date = normalize_date(predicted.get("date", ""))
        if gt_date and pr_date and gt_date == pr_date:
            results["date"] += 1

        # Check total
        gt_total = normalize_total(ground_truth.get("total", ""))
        pr_total = normalize_total(predicted.get("total", ""))
        if gt_total and pr_total and gt_total == pr_total:
            results["total"] += 1

        if results["count"] % 50 == 0:
            print(f"Processed {results['count']} receipts...")

    count = results["count"]
    print(f"\nResults on {count} receipts:")
    print(f"Vendor accuracy : {results['vendor']/count*100:.1f}%")
    print(f"Date accuracy   : {results['date']/count*100:.1f}%")
    print(f"Total accuracy  : {results['total']/count*100:.1f}%")


if __name__ == "__main__":
    img_dir      = Path("data/sroie/train/img")
    entities_dir = Path("data/sroie/train/entities")
    evaluate_sroie(img_dir, entities_dir)