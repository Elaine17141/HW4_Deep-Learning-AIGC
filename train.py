import os
import numpy as np

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

category_en = "crested_myna,javan_myna,common_myna"
categories = category_en.split(",")
N = len(categories)

BASE_DIR = "./myna"
MODEL_PATH = "myna_model.h5"

data = []
target = []

for i in range(N):
    thedir = os.path.join(BASE_DIR, categories[i])
    for fname in os.listdir(thedir):
        img = load_img(os.path.join(thedir, fname), target_size=(224, 224))
        data.append(img_to_array(img))
        target.append(i)

data = preprocess_input(np.array(data))
y = to_categorical(target, N)

resnet = ResNet50V2(include_top=False, pooling="avg")
resnet.trainable = False

model = Sequential([
    resnet,
    Dense(N, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(data, y, batch_size=10, epochs=10)
model.save(MODEL_PATH)
