import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

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
    "Benign Nevi": {
        "Risk": "LOW",
        "Advice": "Common mole. Monitor for any changes in shape or color (ABCD rule).",
        "Doctor": "NOT URGENT (Monitor only)"
    }
}

# --- LOAD MODEL (Trained on ISIC Dataset) ---
@st.cache_resource
def load_skincancer_model():
    
    model = tf.keras.models.load_model('model/skincancer_model.h5')
    return model

# --- PREPROCESSING ---
def preprocess_lesion(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
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
        # --- ADD THE NEW CODE HERE ---
        
        # 1. Load the model (Make sure the path is correct!)
        model = tf.keras.models.load_model('model/skincancer_model.h5')
        
        # 2. Prepare the image
        processed_img = preprocess_lesion(image)
        
        # 3. Get actual prediction
        prediction = model.predict(processed_img)
        
        # 4. Extract the result
        real_idx = np.argmax(prediction) 
        confidence = np.max(prediction) * 100
        result = CLASSES[real_idx]

        

    # --- DISPLAY RESULTS ---
    st.header(f"Diagnosis: {result}")
    st.progress(int(confidence))
    st.write(f"**AI Confidence Score:** {confidence:.2f}%")

    # --- TREATMENT & DOCTOR MEET LOGIC ---
    info = CLINICAL_ADVICE.get(result, {"Risk": "Unknown", "Advice": "Consult professional.", "Doctor": "Required"})
    
    st.divider()
    st.subheader("📋 Clinical Guidance")
    
    if info['Risk'] in ["CRITICAL", "HIGH"]:
        st.error(f"**Severity:** {info['Risk']}")
    else:
        st.success(f"**Severity:** {info['Risk']}")
        
    st.write(f"**Treatment Info:** {info['Advice']}")
    st.info(f"**Professional Consultation:** {info['Doctor']}")

    if info['Doctor'] != "NOT URGENT (Monitor only)":
        st.warning("⚠️ PROMPT: Please schedule a visit with a dermatologist to confirm this AI screening with a physical biopsy.")

st.markdown("---")
st.caption("Disclaimer: This tool is an AI assistant based on the ISIC dataset and JPInfotech methodology. It is not a final medical diagnosis.")
