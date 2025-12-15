import numpy as np
import streamlit as st
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input

category_zh = ["土八哥", "白尾八哥", "家八哥"]
N = len(category_zh)

st.set_page_config(page_title="八哥辨識器", layout="centered")
st.title("八哥辨識器")
st.write("請上傳一張八哥照片，系統將預測其種類")

model = load_model("myna_model.h5")

uploaded_file = st.file_uploader("上傳圖片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="輸入圖片", use_container_width=True)

    img = img.resize((224, 224))
    x = np.array(img).reshape((1, 224, 224, 3))
    x = preprocess_input(x)

    pred = model.predict(x)[0]
    idx = np.argsort(pred)[::-1]

    st.subheader("辨識結果")
    for i in idx:
        st.write(f"{category_zh[i]}：{pred[i]:.2f}")
