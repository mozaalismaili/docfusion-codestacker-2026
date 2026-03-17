import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from PIL import Image
from extractor import extract_fields
from anomaly import extract_features, load_model, predict_forgery

st.set_page_config(page_title="DocFusion", page_icon="📄", layout="wide")

st.title("DocFusion — Intelligent Document Processing")
st.markdown("Upload a receipt image to extract fields and detect forgery.")

# Load model
model_dir = "models/anomaly"
clf, scaler = None, None
if os.path.exists(os.path.join(model_dir, "clf.pkl")):
    clf, scaler = load_model(model_dir)
    st.sidebar.success("Anomaly model loaded")
else:
    st.sidebar.warning("No model found. Run src/anomaly.py first.")

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save temp file
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
        with st.spinner("Running OCR and extraction..."):
            extracted = extract_fields(temp_path)

        st.write("**Vendor:**", extracted.get("vendor") or "Not found")
        st.write("**Date:**",   extracted.get("date")   or "Not found")
        st.write("**Total:**",  extracted.get("total")  or "Not found")

        st.subheader("Anomaly Detection")
        if clf and scaler:
            features  = extract_features(extracted, temp_path)
            is_forged = predict_forgery(clf, scaler, extracted)
            if is_forged == 1:
                st.error("FORGED — This receipt appears suspicious")
            else:
                st.success("GENUINE — This receipt appears authentic")
        else:
            st.info("Train the model first to enable anomaly detection.")

        st.subheader("Raw OCR Lines")
        st.code("\n".join(extracted.get("raw_lines", [])))

    os.remove(temp_path)