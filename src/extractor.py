import re
from pathlib import Path
from paddleocr import PaddleOCR

# Initialize OCR once globally
ocr_engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)


def run_ocr(image_path: str) -> list:
    """Run PaddleOCR on an image and return list of text lines."""
    result = ocr_engine.ocr(str(image_path), cls=False)
    lines = []
    if result and result[0]:
        for line in result[0]:
            text = line[1][0].strip()
            if text:
                lines.append(text)
    return lines


def extract_total(lines: list) -> str:
    """Extract total amount from OCR lines."""
    total_keywords = r"\b(total|grand total|jumlah|amt due|balance due|net total|amount due|to pay)\b"
    amount_pattern = r"\b(\d{1,6}\.\d{2})\b"

    # First pass: find a line with total keyword + amount
    # Search from bottom up since total is usually at the bottom
    for line in reversed(lines):
        if re.search(total_keywords, line, re.IGNORECASE):
            match = re.search(amount_pattern, line)
            if match:
                return match.group(1)

    # Second pass: collect all decimal amounts and return the largest
    # but skip lines that look like addresses or phone numbers
    skip_line = r"(jalan|taman|lorong|lot |no\.|tel|phone|fax|floor|level)"
    amounts = []
    for line in lines:
        if re.search(skip_line, line, re.IGNORECASE):
            continue
        for match in re.finditer(amount_pattern, line):
            try:
                val = float(match.group(1))
                if 0 < val < 50000:
                    amounts.append(val)
            except ValueError:
                continue

    if amounts:
        return f"{max(amounts):.2f}"

    return None


def extract_date(lines: list) -> str:
    """Extract date from OCR lines."""
    # Order matters: try most specific patterns first
    date_patterns = [
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b",   # 25/12/2018
        r"\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",             # 2018-12-25
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2})\b",    # 25/12/18
        r"\b(\d{1,2}\s+\w{3,9}\s+\d{4})\b",               # 25 Dec 2018
        r"\b(\w{3,9}\s+\d{1,2},?\s+\d{4})\b",             # Dec 25, 2018
    ]
    date_line_keywords = r"\b(date|tarikh|dt)\b"

    # First pass: lines with date keyword
    for line in lines:
        if re.search(date_line_keywords, line, re.IGNORECASE):
            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1)

    # Second pass: all lines
    for line in lines:
        for pattern in date_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1)

    return None


def extract_vendor(lines: list) -> str:
    """Extract vendor name from OCR lines."""
    # Lines to always skip
    skip_patterns = (
        r"(tel|phone|fax|http|www|receipt|invoice|date|time|reg\.|gst|tax"
        r"|cashier|served by|thank|please|welcome|jalan|taman|lorong"
        r"|lot |wisma|plaza|level|floor|\d{6,}|@|no\.)"
    )

    # Strong company name indicators
    company_indicators = (
        r"\b(sdn[\s\.]?bhd|plt|llp|corp|inc|ltd|enterprise|trading|"
        r"shop|store|market|mall|pharmacy|restaurant|cafe|holdings|group)\b"
    )

    # First pass: find company indicator in first 10 lines
    for line in lines[:10]:
        if re.search(skip_patterns, line, re.IGNORECASE):
            continue
        if re.search(company_indicators, line, re.IGNORECASE):
            return re.sub(r"\s+", " ", line).strip()

    # Second pass: first line that is mostly letters and not a person name
    # Person names are usually short (2-3 words, all lowercase or title case)
    # Company names usually have uppercase or special chars
    for line in lines[:8]:
        if len(line) < 3:
            continue
        if re.search(skip_patterns, line, re.IGNORECASE):
            continue
        letters = sum(c.isalpha() for c in line)
        digits  = sum(c.isdigit() for c in line)
        if letters < len(line) * 0.4:
            continue
        if letters <= digits:
            continue
        # Prefer lines with uppercase (company names tend to be uppercase)
        upper = sum(c.isupper() for c in line if c.isalpha())
        if upper > letters * 0.4:
            return line.strip()

    # Third pass: just return first non-skipped line with letters
    for line in lines[:8]:
        if len(line) < 3:
            continue
        if re.search(skip_patterns, line, re.IGNORECASE):
            continue
        if re.search(r"[A-Za-z]", line):
            return line.strip()

    return None


def extract_fields(image_path: str) -> dict:
    """Main function: given an image path, return extracted fields."""
    lines = run_ocr(image_path)
    return {
        "vendor":    extract_vendor(lines),
        "date":      extract_date(lines),
        "total":     extract_total(lines),
        "raw_lines": lines
    }


if __name__ == "__main__":
    possible_paths = [
        Path("data/sroie/train/img"),
        Path("data/sroie/train/images"),
    ]

    img_dir = None
    for p in possible_paths:
        if p.exists():
            imgs = list(p.glob("*.jpg")) + list(p.glob("*.png"))
            if imgs:
                img_dir = p
                break

    if img_dir is None:
        print("Could not find SROIE images. Folders found:")
        for f in Path("data/sroie").rglob("*"):
            print(f)
    else:
        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        for test_image in imgs[:5]:
            result = extract_fields(str(test_image))
            print(f"Image : {test_image.name}")
            print(f"Vendor: {result['vendor']}")
            print(f"Date  : {result['date']}")
            print(f"Total : {result['total']}")
            print("-" * 40)