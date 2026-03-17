import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from PIL import Image

# Try to load OCR — may not be available on cloud
try:
    from extractor import extract_fields
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Try to load anomaly model
try:
    from anomaly import extract_features, load_model, predict_forgery
    ANOMALY_AVAILABLE = True
except Exception:
    ANOMALY_AVAILABLE = False

st.set_page_config(page_title="DocFusion", page_icon="📄", layout="wide")
st.title("DocFusion — Intelligent Document Processing")
st.markdown("Upload a receipt image to extract fields and detect forgery.")

if not OCR_AVAILABLE:
    st.warning("OCR engine not available in this environment. Running in demo mode.")

# Load model
model_dir  = "models/anomaly"
clf, scaler = None, None
if ANOMALY_AVAILABLE and os.path.exists(os.path.join(model_dir, "clf.pkl")):
    clf, scaler = load_model(model_dir)
    st.sidebar.success("Anomaly model loaded")
else:
    st.sidebar.warning("Anomaly model not available.")

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

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

        if OCR_AVAILABLE:
            with st.spinner("Running OCR and extraction..."):
                extracted = extract_fields(temp_path)
            st.write("**Vendor:**", extracted.get("vendor") or "Not found")
            st.write("**Date:**",   extracted.get("date")   or "Not found")
            st.write("**Total:**",  extracted.get("total")  or "Not found")

            st.subheader("Anomaly Detection")
            if clf and scaler and ANOMALY_AVAILABLE:
                features  = extract_features(extracted, temp_path)
                is_forged = predict_forgery(clf, scaler, extracted)
                if is_forged == 1:
                    st.error("FORGED — This receipt appears suspicious")
                else:
                    st.success("GENUINE — This receipt appears authentic")
            else:
                st.info("Anomaly model not available in this environment.")

            st.subheader("Raw OCR Lines")
            st.code("\n".join(extracted.get("raw_lines", [])))
        else:
            st.info("""
            This is a demo deployment. The full version with OCR runs locally.
            
            To run the full version:
            1. Clone the repository
            2. Install requirements: pip install -r requirements.txt
            3. Run: streamlit run app.py
            """)

    os.remove(temp_path)
