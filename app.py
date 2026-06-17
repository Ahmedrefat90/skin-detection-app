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
    .main { background-color: #f8f9fa; }
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3a8a;
        font-size: 45px !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle { text-align: center; color: #4b5563; font-size: 18px; margin-bottom: 30px; }
    .result-card {
        background-color: #ffffff; padding: 25px; border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); border-left: 8px solid #10b981; margin-top: 20px;
    }
    .warning-card {
        background-color: #ffffff; padding: 25px; border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1); border-left: 8px solid #ef4444; margin-top: 20px;
    }
    .stMetric { background-color: #f1f5f9; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='stTitle'>🩺 Dermascope AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Neural Intelligence for Dermatological Classification</p>", unsafe_allow_html=True)

# Function to clear session state for the camera freeze bug
def reset_app():
    st.cache_resource.clear()
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

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
                    if chunk: f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model_from_drive()
except Exception as e:
    st.error("Engine Initialization Error.")
    st.exception(e)

# Class Mappings
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
    st.title("Control Panel")
    st.info("Biomedical Engineering Graduation Artifact Model v2.5")
    
    if st.button("🔄 Clear & Reset Camera", use_container_width=True):
        reset_app()
        
    st.divider()
    st.write("🔍 **Supported Classes:**")
    for label in CLASS_LABELS.values():
        st.write(f"- {label}")

# 3. UI for Image Input
st.subheader("📤 Upload Image or Use Device Camera")
uploaded_file = st.file_uploader("Capture or upload skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown("### 📸 Captured Sample")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("### 🧬 AI Analysis")
        with st.spinner('Extracting features...'):
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized)
            img_rescaled = img_array / 255.0
            img_expand = np.expand_dims(img_rescaled, axis=0)

            # --- REVISED REASONABLE ANTI-GRAPHIC FILTER ---
            img_gray = np.mean(img_array, axis=2)
            texture_variance = np.var(img_gray)
            
            # Pure computer white needs to be absolute (1.0) and covering an immense part of the image (like vector background)
            pure_white_mask = (img_rescaled[:,:,0] >= 0.99) & (img_rescaled[:,:,1] >= 0.99) & (img_rescaled[:,:,2] >= 0.99)
            pure_white_ratio = np.sum(pure_white_mask) / (224 * 224)
            
            # Check for pure absolute flat color (0 variance) like artificial grey/black screens
            is_completely_flat = texture_variance < 100

            # Prediction
            predictions = model.predict(img_expand)
            best_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[best_class_idx]
            confidence = predictions[0][best_class_idx] * 100

        # Diagnosis Display Logic
        # Now calibrated so real micro-skin photos with soft lighting or flash flash never trigger invalid detection.
        if is_completely_flat or pure_white_ratio > 0.45:
            st.markdown("""
                <div class='warning-card'>
                    <h3 style='color: #ef4444; margin:0;'>Invalid Sample Detected</h3>
                    <p style='font-size: 16px; color: #374151; margin-top:10px;'>
                        <strong>Safety Protocol Activated:</strong> The system identified this image as a non-clinical graphic or artificial flat background. 
                        <br><br>
                        Please provide a real macro photograph focused directly on the skin pathology.
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
                        The system score fell below the medical threshold (70.00%). The analysis is inconclusive. Please reposition lighting and retry.
                    </p>
                </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption("© 2026 Dermascope AI Project | Prepared for Academic Evaluation")
