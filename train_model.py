import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

print("🚀 Training script started")

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X_data.npy")
y = np.load("y_labels.npy")

# Normalize
X = X / 255.0

num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🔵 Loading Xception base model")

base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 🔒 Freeze base model to reduce memory
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation="relu")(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("🧠 Training started")

model.fit(
    X_train,
    y_train,
    epochs=2,          # reduced
    batch_size=4,      # VERY IMPORTANT
    validation_data=(X_test, y_test)
)

model.save("xception_model.h5")
print("✅ Model saved successfully")
