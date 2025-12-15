八哥影像辨識系統（Myna Bird Classification AI Demo）

本專案為一個基於 遷移式學習（Transfer Learning） 的影像辨識系統，使用深度學習模型辨識三種常見八哥鳥類，並延伸實作完整的互動式 AI Demo。系統不僅完成模型訓練與推論，也提供可實際操作的 Web 介面，展示模型在實務應用情境下的行為與效能。

專案目標

使用少量訓練資料完成多分類影像辨識任務

採用 ImageNet 預訓練模型以提升小資料集表現

將模型推論流程重構為可部署的 AI Demo

提供互動式介面，展示模型預測結果與推論效能

辨識類別

本系統可辨識以下三種八哥鳥類：

土八哥（Crested Myna）

白尾八哥（Javan Myna）

家八哥（Common Myna）

使用模型與方法

Backbone Model：ResNet50V2（ImageNet Pretrained）

學習策略：Transfer Learning

訓練方式：凍結卷積層，僅訓練最終分類層

輸入尺寸：224 × 224 RGB

輸出形式：多分類機率分佈（Softmax）

專案結構
HW4_Deep-Learning-AIGC/
 ├─ myna_app.py        # Gradio 互動式 AI Demo
 ├─ app.py             # Streamlit 互動式 AI Demo（可部署）
 ├─ myna_model.h5      # 訓練完成之模型檔
 ├─ requirements.txt   # 執行所需套件
 └─ myna/              # 本地訓練用資料集（未上傳）

延伸 Demo 功能說明

相較於原始教學範例，本專案進一步延伸實作以下功能：

將 Notebook 重構為獨立 Python 應用程式

分離模型訓練與推論流程，避免重複訓練

建立完整 Gradio Web Demo，支援圖片上傳與範例圖片選擇

建立等價 Streamlit Web Demo，支援雲端部署

顯示各類別預測信心分數（由高至低排序）

即時量測並顯示端到端推論時間（單位：秒）

加入模型 warm-up 機制，改善首次推論延遲

強化影像前處理流程以提升系統穩定性

執行方式
1. 安裝套件
pip install -r requirements.txt

2. 執行 Gradio Demo（本機展示）
python myna_app.py


執行後，於瀏覽器開啟顯示之本地網址即可操作。

3. 執行 Streamlit Demo（部署用）
streamlit run app.py


執行後，於瀏覽器開啟 http://localhost:8501 即可操作。

資料集說明

由於資料集檔案較大，本專案未將影像資料直接上傳至 GitHub。
資料僅用於本地端模型訓練，並於報告中提供原始資料來源連結。

備註

本專案同時提供 Gradio 與 Streamlit 兩種等價 Demo，兩者共用相同模型與推論流程，僅在 UI 框架與展示方式上有所差異，以因應不同展示與部署需求。
