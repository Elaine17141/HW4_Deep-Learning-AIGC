import os
import time
import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# =========================
# 基本設定（與 myna_app.py 對齊）
# =========================

category_en = ["crested_myna", "javan_myna", "common_myna"]
category_zh = ["土八哥", "白尾八哥", "家八哥"]

BASE_DIR = "myna"
MODEL_PATH = "myna_model.h5"

TITLE = "八哥辨識器（AI Demo）"
DESCRIPTION = "請選擇範例圖片或自行上傳八哥照片，系統將即時辨識其種類。"

# =========================
# Streamlit 頁面設定
# =========================

st.set_page_config(page_title=TITLE, layout="centered")
st.title(TITLE)
st.write(DESCRIPTION)

st.markdown(
    """
### 模型資訊
- **Backbone:** ResNet50V2（ImageNet Pretrained）
- **分類類別:** 土八哥 / 白尾八哥 / 家八哥
- **輸入尺寸:** 224 × 224 RGB
- **任務類型:** 多分類影像辨識（Transfer Learning）
"""
)

# =========================
# 載入模型
# =========================

@st.cache_resource
def load_ai_model():
    return load_model(MODEL_PATH)

model = load_ai_model()

# warm-up
dummy = np.zeros((1, 224, 224, 3))
dummy = preprocess_input(dummy)
model.predict(dummy)

# =========================
# 範例圖片清單
# =========================

example_images = []

for cat in category_en:
    cat_dir = os.path.join(BASE_DIR, cat)
    if os.path.exists(cat_dir):
        for fname in os.listdir(cat_dir):
            example_images.append(os.path.join(cat_dir, fname))

# =========================
# UI：選擇圖片來源
# =========================

st.subheader("選擇圖片來源")

source = st.radio(
    "請選擇使用方式",
    ["使用範例圖片", "自行上傳圖片"]
)

img = None

if source == "使用範例圖片":
    selected_image = st.selectbox(
        "選擇一張範例圖片",
        example_images
    )

    if selected_image:
        img = Image.open(selected_image).convert("RGB")

elif source == "自行上傳圖片":
    uploaded_file = st.file_uploader(
        "上傳八哥圖片",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")

# =========================
# 推論流程（與 myna_app.py 對齊）
# =========================

if img is not None:
    st.image(img, caption="輸入圖片", use_container_width=True)

    img_resized = img.resize((224, 224))
    x = np.array(img_resized).reshape((1, 224, 224, 3))
    x = preprocess_input(x)

    start_time = time.time()
    prediction = model.predict(x).flatten()
    end_time = time.time()

    idx = np.argsort(prediction)[::-1]

    st.subheader("AI 辨識結果（信心分數）")

    for i in idx:
        st.write(f"{category_zh[i]}：{prediction[i]:.4f}")

    st.caption(f"推論時間(s)：{end_time - start_time:.3f}")
