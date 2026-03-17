import sys
import os
import re
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="DocFusion", page_icon="📄", layout="wide")
st.title("DocFusion — Intelligent Document Processing")
st.markdown("""
This is the **cloud demo** of DocFusion. It uses EasyOCR for text extraction.
The full pipeline with PaddleOCR runs locally for better accuracy.
""")

# Load EasyOCR lazily
@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

# Load anomaly model lazily
@st.cache_resource
def load_anomaly_model():
    import pickle
    model_dir = "models/anomaly"
    clf_path  = os.path.join(model_dir, "clf.pkl")
    if not os.path.exists(clf_path):
        return None, None
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler


def run_easyocr(image_path: str, reader) -> list:
    results = reader.readtext(image_path)
    return [text for _, text, conf in results if conf > 0.3]


def extract_vendor(lines):
    skip = r"(tel|phone|fax|http|www|date|time|tax|gst|receipt|cashier)"
    company = r"\b(sdn|bhd|plt|enterprise|trading|shop|store|market|corp|ltd)\b"
    for line in lines[:10]:
        if re.search(company, line, re.IGNORECASE):
            return line.strip()
    for line in lines[:6]:
        if len(line) < 3:
            continue
        if re.search(skip, line, re.IGNORECASE):
            continue
        if re.search(r"[A-Za-z]", line):
            return line.strip()
    return None


def extract_date(lines):
    patterns = [
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b",
        r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2})\b",
        r"\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b",
    ]
    for line in lines:
        for p in patterns:
            m = re.search(p, line)
            if m:
                return m.group(1)
    return None


def extract_total(lines):
    total_kw = r"\b(total|amount|grand total|jumlah)\b"
    amount   = r"\b(\d{1,6}\.\d{2})\b"
    for line in reversed(lines):
        if re.search(total_kw, line, re.IGNORECASE):
            m = re.search(amount, line)
            if m:
                return m.group(1)
    amounts = []
    for line in lines:
        for m in re.finditer(amount, line):
            try:
                amounts.append(float(m.group(1)))
            except ValueError:
                continue
    return f"{max(amounts):.2f}" if amounts else None


def extract_features_simple(vendor, date, total, lines):
    total_val = 0.0
    try:
        total_val = float(total) if total else 0.0
    except ValueError:
        pass
    return np.array([
        0 if vendor else 1,
        0 if date else 1,
        0 if total else 1,
        total_val,
        np.log1p(total_val),
        len(lines),
        sum(len(l) for l in lines) / len(lines) if lines else 0,
        sum(len(re.findall(r"\b\d{1,6}\.\d{2}\b", l)) for l in lines),
        sum(1 for l in lines if re.search(r"\d", l)),
        sum(1 for l in lines if re.search(r"(total|amount)", l, re.IGNORECASE)),
    ], dtype=np.float32)


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("About")
st.sidebar.markdown("""
**DocFusion** is an intelligent document processing pipeline built for the 
Rihal CodeStacker 2026 ML Challenge.

**Pipeline:**
- EasyOCR for text extraction (cloud)
- PaddleOCR for full accuracy (local)
- Random Forest for forgery detection

**By:** Moza Amur  
**University:** Sultan Qaboos University  
**Specialization:** AI & Data Science
""")

with st.spinner("Loading OCR engine — this may take a moment on first run..."):
    reader = load_ocr()
    st.sidebar.success("OCR engine ready")

clf, scaler = load_anomaly_model()
if clf:
    st.sidebar.success("Anomaly model loaded")
else:
    st.sidebar.warning("No anomaly model found")

# ── Main UI ──────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a receipt image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Receipt Image")
        img = Image.open(temp_path)
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("Extracted Fields")
        with st.spinner("Running OCR..."):
            lines = run_easyocr(temp_path, reader)

        vendor = extract_vendor(lines)
        date   = extract_date(lines)
        total  = extract_total(lines)

        st.write("**Vendor:**", vendor or "Not found")
        st.write("**Date:**",   date   or "Not found")
        st.write("**Total:**",  total  or "Not found")

        st.subheader("Anomaly Detection")
        if clf and scaler:
            features = extract_features_simple(
                vendor, date, total, lines
            )
            # Pad features to match training size
            if len(features) < scaler.n_features_in_:
                features = np.pad(
                    features,
                    (0, scaler.n_features_in_ - len(features))
                )
            features  = features[:scaler.n_features_in_]
            X_scaled  = scaler.transform(features.reshape(1, -1))
            is_forged = int(clf.predict(X_scaled)[0])
            if is_forged == 1:
                st.error("FORGED — This receipt appears suspicious")
            else:
                st.success("GENUINE — This receipt appears authentic")
        else:
            st.info("Train the model first to enable anomaly detection.")

        st.subheader("Raw OCR Lines")
        st.code("\n".join(lines))

    os.remove(temp_path)