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

# --- LOAD MODEL ---
@st.cache_resource
def load_skincancer_model():
    # Path fix for Streamlit Cloud
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'skincancer_model.h5')
    # compile=False avoids version mismatch errors between your laptop and the server
    return tf.keras.models.load_model(model_path, compile=False)

# --- PREPROCESSING ---
def preprocess_lesion(image):
    # Ensure RGB and resize to match model input (usually 224x224)
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    # CRITICAL: Scale pixels to [0, 1] to prevent "BCC bias"
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# --- STREAMLIT UI ---
st.title("🎗️ Enhanced Skin Cancer Detection")
st.write("Deep Learning System for Early Malignancy Classification")

uploaded_file = st.file_uploader("Upload Dermoscopic Image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Lesion Image', use_column_width=True)
    
    with st.spinner('Analyzing cellular patterns...'):
        try:
            # 1. Use the cached model loader
            model = load_skincancer_model()
            
            # 2. Process image
            processed_img = preprocess_lesion(image)
            
            # 3. Get raw prediction probabilities
            prediction = model.predict(processed_img)
            
            # 4. Debug: Show probabilities in app (Comment this out for final presentation)
            # st.write("Raw Probabilities:", prediction)

            # 5. Extract results with Confidence Threshold
            THRESHOLD = 0.55  # If AI is less than 55% sure, we call it Inconclusive
            conf_val = np.max(prediction)
            
            if conf_val < THRESHOLD:
                result = "Inconclusive / Likely Benign"
                confidence = conf_val * 100
            else:
                real_idx = np.argmax(prediction) 
                result = CLASSES[real_idx]
                confidence = conf_val * 100

        except Exception as e:
            st.error(f"Error processing model: {e}")
            result = "Error"
            confidence = 0

    # --- DISPLAY RESULTS ---
    if result != "Error":
        st.header(f"Diagnosis: {result}")
        st.progress(int(confidence))
        st.write(f"**AI Confidence Score:** {confidence:.2f}%")

        # --- CLINICAL GUIDANCE LOGIC ---
        info = CLINICAL_ADVICE.get(result, {"Risk": "Unknown", "Advice": "Consult professional.", "Doctor": "Required"})
        
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
            st.warning("⚠️ PROMPT: Please schedule a visit with a dermatologist to confirm this AI screening with a physical biopsy.")

st.markdown("---")
st.caption("Disclaimer: This tool is an AI assistant based on the ISIC dataset. It is not a final medical diagnosis.")
