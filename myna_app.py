# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import gradio as gr
import PIL.Image as Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# =========================
# 基本設定
# =========================

category_en = "crested_myna,javan_myna,common_myna"
category_zh = "土八哥,白尾八哥,家八哥"

categories = category_en.split(",")
labels = category_zh.split(",")
N = len(categories)

BASE_DIR = "./myna"
MODEL_PATH = "myna_model.h5"

TITLE = "八哥辨識器（AI Demo）"
DESCRIPTION = "請上傳一張八哥照片，系統將即時辨識其種類並顯示信心分數。"

# =========================
# 載入模型
# =========================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("找不到 myna_model.h5，請先完成模型訓練")

model = load_model(MODEL_PATH)

# warm-up（避免第一次推論很慢）
dummy = np.zeros((1, 224, 224, 3))
dummy = preprocess_input(dummy)
model.predict(dummy)

# =========================
# 工具函式
# =========================

def resize_image(inp):
    img = Image.fromarray(inp)
    img = img.convert("RGB")
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    return np.array(img)

def classify_image(inp):
    if inp is None:
        return {}

    start_time = time.time()

    img_array = resize_image(inp)
    x = img_array.reshape((1, 224, 224, 3))
    x = preprocess_input(x)

    prediction = model.predict(x).flatten()
    idx = np.argsort(prediction)[::-1]

    end_time = time.time()

    result = {}
    for i in idx:
        result[labels[i]] = float(prediction[i])

    result["推論時間(s)"] = round(end_time - start_time, 3)
    return result

# =========================
# 範例圖片
# =========================

sample_images = []
for cat in categories:
    cat_dir = os.path.join(BASE_DIR, cat)
    if os.path.exists(cat_dir):
        for fname in os.listdir(cat_dir):
            sample_images.append(os.path.join(cat_dir, fname))

# =========================
# Gradio UI
# =========================

gr.Markdown(
    """
### 模型資訊
- **Backbone:** ResNet50V2（ImageNet Pretrained）
- **分類類別:** 土八哥 / 白尾八哥 / 家八哥
- **輸入尺寸:** 224 × 224 RGB
- **任務類型:** 多分類影像辨識（Transfer Learning）
"""
)

image = gr.Image(label="上傳八哥圖片")
label = gr.Label(label="AI 辨識結果（信心分數）")

app = gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    title=TITLE,
    description=DESCRIPTION,
    examples=sample_images
)

app.launch(debug=True)
