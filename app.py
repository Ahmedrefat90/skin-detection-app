import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="Dermascope AI | Professional Diagnosis",
    page_icon="🩺",
    layout="centered"
)

# Professional Presentation Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #1e3a8a;
        font-size: 42px !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle { text-align: center; color: #4b5563; font-size: 16px; margin-bottom: 25px; }
    .result-card {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08); border-left: 6px solid #10b981; margin-top: 15px;
    }
    .warning-card {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08); border-left: 6px solid #ef4444; margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='stTitle'>🩺 Dermascope AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced Neural Intelligence for Dermatological Classification</p>", unsafe_allow_html=True)

# Model Loading Config
MODEL_PATH = 'final_8classes_model.h5'
DRIVE_FILE_ID = '1XPaPIYtPcwwvv7C03GM3duqEwQOHHkZp'

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('🚀 Fetching Neural Weights from Secure Storage...'):
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
    model_loaded = True
except Exception as e:
    st.error("Engine failure loading weights.")
    model_loaded = False

# Mapping Definitions
CLASS_NAMES = ['actinic_keratosis', 'basal_cell_carcinoma', 'eczema', 'melanocytic_nevus', 'melanoma',
               'seborrheic_keratosis', 'vascular', 'vitiligo']

CLASS_LABELS = {
    'actinic_keratosis': 'Actinic Keratosis (AK) / التَّقَرُّن الإشعاعي',
    'basal_cell_carcinoma': 'Basal Cell Carcinoma (BCC) / سرطان الخلايا البازالية',
    'eczema': 'Eczema / الإكزيما',
    'melanocytic_nevus': 'Melanocytic Nevus / الوحمة الميلانينية',
    'melanoma': 'Melanoma / الميلانوما (سرطان الجلد)',
    'seborrheic_keratosis': 'Seborrheic Keratosis / التقرن الدهني',
    'vascular': 'Vascular Lesion / الآفات الوعائية',
    'vitiligo': 'Vitiligo / البهاق'
}

if model_loaded:
    uploaded_file = st.file_uploader("Choose a clear photo of the affected area (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown("### 📸 Captured Sample")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("### 🧬 AI Analysis")
            with st.spinner('Processing pathology metrics...'):
                size = (224, 224)
                image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                img_array = np.asarray(image_resized)
                
                if img_array.ndim == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                # 1. حساب التباين النسيجي القديم (بدون cv2)
                img_gray = np.mean(img_array, axis=2)
                texture_variance = float(np.var(img_gray))

                # 2. فلتر الجلد الرياضي العبقري باستخدام numpy فقط (بدون cv2)
                # بنقيس بكسلات قنوات الألوان الـ RGB ونشوف هل تقع في نطاق الجلد البشري الطبيعي
                R = img_array[:, :, 0]
                G = img_array[:, :, 1]
                B = img_array[:, :, 2]
                
                # معادلة رياضية عالمية لتحديد بكسلات الجلد في فضاء RGB
                skin_mask = (R > 95) & (G > 40) & (B > 20) & ((np.maximum(R, np.maximum(G, B)) - np.minimum(R, np.minimum(G, B))) > 15) & (np.abs(R - G) > 15) & (R > G) & (R > B)
                
                skin_pixels = np.sum(skin_mask)
                total_pixels = img_array.shape[0] * img_array.shape[1]
                skin_ratio = float(skin_pixels / total_pixels)

            # 🛑 حماية ذكية: لو النسيج ممسوح جداً أو نسبة بكسلات الجلد البشري في الصورة أقل من 15% (الروشتة هتطلع صفر جلد)
            if texture_variance < 10.0 or skin_ratio < 0.15:
                st.markdown("""
                    <div class='warning-card'>
                        <h3 style='color: #ef4444; margin:0;'>Invalid Sample Detected</h3>
                        <p style='font-size: 14px; color: #374151; margin-top:8px;'>
                            The system detected that this file lacks clinical skin biomarkers or is an artificial flat surface. Please upload an authentic dermatological medical photograph.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.caption(f"Safety Framework Status -> Texture Var: {texture_variance:.2f} | Skin Biomarkers: {skin_ratio*100:.1f}%")
            
            else:
                # إذا كانت الصورة جلد حقيقي، يشتغل الموديل ويشخص فوراً بامتياز
                with st.spinner('Running neural classification...'):
                    img_rescaled = img_array / 255.0
                    img_expand = np.expand_dims(img_rescaled, axis=0)
                    
                    predictions = model.predict(img_expand)
                    best_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[best_class_idx]
                    confidence = float(predictions[0][best_class_idx] * 100)

                if confidence >= 50.0:
                    st.markdown(f"""
                        <div class='result-card'>
                            <h3 style='color: #10b981; margin:0;'>Confirmed Diagnosis</h3>
                            <p style='font-size: 20px; font-weight: bold; color: #1e293b; margin-top:5px;'>{CLASS_LABELS[predicted_class]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="System Confidence", value=f"{confidence:.2f}%")
                else:
                    st.markdown(f"""
                        <div class='warning-card'>
                            <h3 style='color: #ef4444; margin:0;'>Low Confidence Alert</h3>
                            <p style='font-size: 14px; color: #374151; margin-top:8px;'>
                                The system score ({confidence:.2f}%) fell below the safety clinical threshold (50.00%). 
                                The analysis is inconclusive. Please reposition lighting, avoid shadows, and retry.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

st.divider()
st.caption("© 2026 Dermascope AI Project | Prepared for Academic Evaluation")
