import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SkinCancer-AI", page_icon="🎗️", layout="centered")

# --- CLINICAL DATABASE ---
CLASSES = [
    "Actinic Keratoses", "Basal Cell Carcinoma (BCC)", "Benign Nevi", 
    "Dermatofibroma", "Melanoma", "Squamous Cell Carcinoma (SCC)"
]

CLINICAL_ADVICE = {
    "Melanoma": {
        "Risk": "CRITICAL",
        "Advice": "Urgent specialist consultation required. May require surgical excision and biopsy.",
        "Doctor": "REQUIRED IMMEDIATELY"
    },
    "Basal Cell Carcinoma (BCC)": {
        "Risk": "HIGH",
        "Advice": "Slow-growing but invasive. Standard treatment involves Mohs surgery or cryotherapy.",
        "Doctor": "REQUIRED within 7-10 days"
    },
    "Squamous Cell Carcinoma (SCC)": {
        "Risk": "HIGH",
        "Advice": "Can spread if left untreated. Requires biopsy and possible minor surgery.",
        "Doctor": "REQUIRED"
    },
    "Actinic Keratoses": {
        "Risk": "MODERATE",
        "Advice": "Pre-cancerous sun damage. Often treated with topical creams or liquid nitrogen.",
        "Doctor": "RECOMMENDED"
    },
    "Dermatofibroma": {
        "Risk": "LOW",
        "Advice": "Non-cancerous fibrous growth. Typically harmless and requires no treatment.",
        "Doctor": "NOT URGENT"
    },
    "Benign Nevi": {
        "Risk": "LOW",
        "Advice": "Common mole. Monitor for any changes in shape or color (ABCD rule).",
        "Doctor": "NOT URGENT (Monitor only)"
    },
    "Inconclusive / Likely Benign": {
        "Risk": "UNKNOWN",
        "Advice": "The AI is uncertain about this texture. Likely non-malignant or image quality issue.",
        "Doctor": "RECOMMENDED for peace of mind"
    }
}

# --- LOAD MODEL (CACHED) ---
@st.cache_resource
def get_model():
    # Force legacy loading and ignore training configuration (compile=False)
    # this is the most stable way to load .h5 files on Streamlit Cloud
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'skincancer_model.h5')
    return tf.keras.models.load_model(model_path, compile=False)

# --- PREPROCESSING (IMPROVED) ---
def preprocess_lesion(image):
    # 1. Convert to RGB and resize
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    
    # 2. Standardize instead of just dividing by 255
    # This helps the model see features even if lighting is poor
    img = (img - np.mean(img)) / (np.std(img) + 1e-7)
    
    return np.expand_dims(img, axis=0)

# --- STREAMLIT UI ---
st.title("🎗️ Enhanced Skin Cancer Detection")
st.write("Deep Learning System for Early Malignancy Classification")

uploaded_file = st.file_uploader("Upload Dermoscopic Image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Use columns for a cleaner layout
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded Lesion Image', use_container_width=True)
    
    with st.spinner('Analyzing cellular patterns...'):
        try:
            # Step 1: Load Model (Only happens once)
            model = get_model()
            
            # Step 2: Preprocess current image
            processed_img = preprocess_lesion(image)
            
            # Step 3: RUN PREDICTION (Fresh every time)
            prediction = model.predict(processed_img)
            
            # Step 4: Interpret results
            conf_val = np.max(prediction)
            
            # Threshold logic to prevent "Always BCC" guesses
            if conf_val < 0.50:
                result = "Inconclusive / Likely Benign"
                confidence = conf_val * 100
            else:
                real_idx = np.argmax(prediction) 
                result = CLASSES[real_idx]
                confidence = conf_val * 100

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            result = "Error"
            confidence = 0

    with col2:
        if result != "Error":
            st.header(f"Diagnosis: {result}")
            st.progress(min(int(confidence), 100))
            st.write(f"**AI Confidence Score:** {confidence:.2f}%")

    # --- CLINICAL GUIDANCE SECTION ---
    if result != "Error":
        info = CLINICAL_ADVICE.get(result, CLINICAL_ADVICE["Inconclusive / Likely Benign"])
        
        st.divider()
        st.subheader("📋 Clinical Guidance")
        
        if info['Risk'] in ["CRITICAL", "HIGH"]:
            st.error(f"**Severity:** {info['Risk']}")
        elif info['Risk'] == "MODERATE":
            st.warning(f"**Severity:** {info['Risk']}")
        else:
            st.success(f"**Severity:** {info['Risk']}")
            
        st.write(f"**Treatment Info:** {info['Advice']}")
        st.info(f"**Professional Consultation:** {info['Doctor']}")

        if info['Doctor'] != "NOT URGENT (Monitor only)":
            st.warning("⚠️ **Dermatologist Referral:** Please schedule a physical biopsy to confirm these results.")

st.markdown("---")
st.caption("Disclaimer: This tool is an AI assistant based on the ISIC dataset. It is not a final medical diagnosis.")
