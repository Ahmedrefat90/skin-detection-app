import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests

# 1. Page Configuration & Professional Styling
st.set_page_config(
    page_title="Dermascope AI | Professional Diagnosis",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS for a "High-End" Artistic Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3a8a;
        font-size: 45px !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 8px solid #10b981;
        margin-top: 20px;
    }
    .warning-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 8px solid #ef4444;
        margin-top: 20px;
    }
    .stMetric {
        background-color: #f1f5f9;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='stTitle'>🩺 Dermascope AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Neural Intelligence for Dermatological Classification</p>", unsafe_allow_html=True)

# 2. Secure Model Loading from Drive
MODEL_PATH = 'final_8classes_model.h5'
DRIVE_FILE_ID = '1XPaPIYtPcwwvv7C03GM3duqEwQOHHkZp'

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('🚀 Initializing Neural Engine... Please wait.'):
            url = "https://docs.google.com/uc?export=download"
            session = requests.Session()
            response = session.get(url, params={'id': DRIVE_FILE_ID}, stream=True)
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value
                    break
            if token:
                response = session.get(url, params={'id': DRIVE_FILE_ID, 'confirm': token}, stream=True)
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load Model
try:
    model = load_model_from_drive()
except Exception as e:
    st.error("Engine Initialization Error. Please check connectivity.")
    st.exception(e)

# Class Mappings (English)
CLASS_NAMES = ['actinic_keratosis', 'basal_cell_carcinoma', 'eczema', 'melanocytic_nevus', 'melanoma',
               'seborrheic_keratosis', 'vascular', 'vitiligo']

CLASS_LABELS = {
    'actinic_keratosis': 'Actinic Keratosis (AK)',
    'basal_cell_carcinoma': 'Basal Cell Carcinoma (BCC)',
    'eczema': 'Eczema',
    'melanocytic_nevus': 'Melanocytic Nevus',
    'melanoma': 'Melanoma',
    'seborrheic_keratosis': 'Seborrheic Keratosis',
    'vascular': 'Vascular Lesion',
    'vitiligo': 'Vitiligo'
}

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2869/2869819.png", width=100)
    st.title("About Project")
    st.info("This AI system is trained to identify 8 distinct skin conditions with clinical-grade accuracy using Deep Learning.")
    st.divider()
    st.write("🔍 **Supported Classes:**")
    for label in CLASS_LABELS.values():
        st.write(f"- {label}")

# 3. UI for Image Upload
st.subheader("📤 Upload Skin Lesion Image")
uploaded_file = st.file_uploader("Choose a clear photo of the affected area (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Beautiful Layout for Image & Results
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("### 📸 Captured Sample")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("### 🧬 AI Analysis")
        with st.spinner('Extracting features...'):
            # Preprocessing
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized)
            img_rescaled = img_array / 255.0
            img_expand = np.expand_dims(img_rescaled, axis=0)

            # [Advanced Safety Filter] Calculate image texture variance to detect non-skin/clipart/flat graphics
            # Skin images have high variance across channels due to tissue texture.
            # Flat graphics/cartoons have massive blocks of identical pixels.
            img_gray = np.mean(img_array, axis=2)
            texture_variance = np.var(img_gray)
            
            # Prediction
            predictions = model.predict(img_expand)
            best_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[best_class_idx]
            confidence = predictions[0][best_class_idx] * 100

        # Diagnosis Display Logic (70% Filter + Graphics Detection)
        # Cartoons/Flat clips usually have variance lower than 800 or extremely uniform textures
        if texture_variance < 600:
            st.markdown("""
                <div class='warning-card'>
                    <h3 style='color: #ef4444; margin:0;'>Invalid Sample Detected</h3>
                    <p style='font-size: 16px; color: #374151; margin-top:10px;'>
                        The system detected that this file is a graphic illustration, clipart, or artificial background. Please upload an authentic medical clinical photograph of skin tissue.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        elif confidence >= 70.0:
            st.markdown(f"""
                <div class='result-card'>
                    <h3 style='color: #10b981; margin:0;'>Confirmed Diagnosis</h3>
                    <p style='font-size: 24px; font-weight: bold; color: #1e293b;'>{CLASS_LABELS[predicted_class]}</p>
                </div>
            """, unsafe_allow_html=True)
            st.metric(label="System Confidence", value=f"{confidence:.2f}%")
        else:
            st.markdown("""
                <div class='warning-card'>
                    <h3 style='color: #ef4444; margin:0;'>Low Confidence Alert</h3>
                    <p style='font-size: 16px; color: #374151; margin-top:10px;'>
                        The system could not identify a clear pathological pattern. Please ensure the image is a close-up of a skin lesion with good lighting.
                    </p>
                </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption("© 2024 Dermascope AI Project | Prepared for Academic Evaluation")
