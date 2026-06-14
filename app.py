import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import requests

# 1. إعدادات الصفحة
st.set_page_config(page_title="Dermascope AI", page_icon="🩺", layout="centered")

st.title("🩺 نظام تشخيص أمراض الجلد بالذكاء الاصطناعي")
st.write("مرحباً بك في المنصة الذكية المعتمدة لتصنيف الأمراض الجلدية.")

# 2. تحميل الموديل الذكي من الجوجل درايف
MODEL_PATH = 'final_8classes_model.h5'
DRIVE_FILE_ID = '1XPaPIYtPcwwvv7C03GM3duqEwQOHHkZp'

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('🔄 جاري تحميل ملف الموديل الذكي لأول مرة... برجاء الانتظار ثواني'):
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
                for chunk in response.iter_content(chunk_size=65536):  # سرعنا التحميل هنا
                    if chunk:
                        f.write(chunk)
                        
    # التحميل المباشر الآمن
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# استدعاء الموديل
try:
    model = load_model_from_drive()
except Exception as e:
    st.error("حدث خطأ أثناء تهيئة الموديل في السيرفر.")
    st.exception(e)

# الكلاسات المعتمدة
CLASS_NAMES = ['actinic_keratosis', 'basal_cell_carcinoma', 'eczema', 'melanocytic_nevus', 'melanoma',
               'seborrheic_keratosis', 'vascular', 'vitiligo']

CLASS_ARABIC = {
    'actinic_keratosis': 'التَقَرّن الإشعاعي (Actinic Keratosis)',
    'basal_cell_carcinoma': 'سرطان الخلايا القاعدة (Basal Cell Carcinoma)',
    'eczema': 'الإكزيما (Eczema)',
    'melanocytic_nevus': 'الوحمة الميلانينية (Melanocytic Nevus)',
    'melanoma': 'الميلانوما / سرطان الجلد الأسود (Melanoma)',
    'seborrheic_keratosis': 'التقرن الدهني (Seborrheic Keratosis)',
    'vascular': 'الأمراض الوعائية الجلدية (Vascular)',
    'vitiligo': 'البهاق (Vitiligo)'
}

# 3. واجهة المستخدم لرفع الصور
uploaded_file = st.file_uploader("قم برفع صورة الجلد المصاب للكشف عن المرض...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📸 الصورة المرفوعة للتحليل', use_column_width=True)

    with st.spinner('⏳ جاري استخراج الـ Features وتحليل الأوزان...'):
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image_resized)
        img_rescaled = img_array / 255.0
        img_expand = np.expand_dims(img_rescaled, axis=0)

        predictions = model.predict(img_expand)
        best_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[best_class_idx]
        confidence = predictions[0][best_class_idx] * 100

    st.success(f"### 🎯 التشخيص المتوقع: {CLASS_ARABIC[predicted_class]}")
    st.info(f"##### 📊 نسبة دقة التشخيص (Confidence): {confidence:.2f}%")
