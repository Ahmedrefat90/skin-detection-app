import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests
import cv2

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
                # 1. تجهيز الصورة وتغيير أبعادها لـ المصفوفة
                size = (224, 224)
                image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                img_array = np.asarray(image_resized)
                
                # تصحيح نوع البيانات للتأكد من توافق cv2 و numpy وعدم حدوث إيرور
                if img_array.ndim == 3 and img_array.shape[2] == 4:  # لو الصورة PNG شفافة بنشيل طبقة الـ Alpha
                    img_array = img_array[:, :, :3]
                
                img_array = img_array.astype(np.uint8)

                # 2. حساب التباين النسيجي (Texture Variance)
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                texture_variance = float(np.var(img_gray))

                # 3. حساب نسبة الجلد (Skin Pixel Ratio) بالفضاء اللوني HSV بشكل آمن 100%
                img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([25, 255, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
                skin_pixels = int(np.sum(skin_mask == 255))
                total_pixels = int(img_array.shape[0] * img_array.shape[1])
                skin_ratio = float(skin_pixels / total_pixels)

            # 🛑 أولاً: فحص فلاتر الأمان قبل استدعاء الموديل منعا للـ Error أو التهنيج
            if texture_variance < 15.0 or skin_ratio < 0.12:
                st.markdown("""
                    <div class='warning-card'>
                        <h3 style='color: #ef4444; margin:0;'>Invalid Sample Detected</h3>
                        <p style='font-size: 14px; color: #374151; margin-top:8px;'>
                            The system detected that this file lacks clinical skin biomarkers or is an artificial flat surface. Please upload an authentic dermatological medical photograph.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.caption(f"Safety Check -> Texture Var: {texture_variance:.2f} | Skin Ratio: {skin_ratio*100:.1f}%")
            
            else:
                # طالما الصورة صالحة، نبدأ نتوقع بالموديل بأمان
                with st.spinner('Running neural classification...'):
                    img_rescaled = img_array / 255.0
                    img_expand = np.expand_dims(img_rescaled, axis=0)
                    
                    predictions = model.predict(img_expand)
                    best_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[best_class_idx]
                    confidence = float(predictions[0][best_class_idx] * 100)

                # ✅ عرض النتيجة الحقيقية
                if confidence >= 50.0:
                    st.markdown(f"""
                        <div class='result-card'>
                            <h3 style='color: #10b981; margin:0;'>Confirmed Diagnosis</h3>
                            <p style='font-size: 20px; font-weight: bold; color: #1e293b; margin-top:5px;'>{CLASS_LABELS[predicted_class]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.metric(label="System Confidence", value=f"{confidence:.2f}%")
                
                # ⚠️ حالة ضعف الثقة في الصورة الجلدية
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
