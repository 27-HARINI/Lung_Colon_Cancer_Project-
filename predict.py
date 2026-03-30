from tensorflow.keras.models import load_model
import cv2
import numpy as np

print("Loading trained model...")

model = load_model("xception_model.h5")

# Load image
img = cv2.imread("test_images/normal_case_10.jpg")

# Resize image
img = cv2.resize(img, (224,224))

# Normalize
img = img / 255.0

# Expand dimensions
img = np.expand_dims(img, axis=0)

print("Predicting...")

prediction = model.predict(img)

print("Prediction result:", prediction)

classes = ["Colon Normal", "Colon Cancer", "Lung Cancer", "Lung Normal"]

result = classes[np.argmax(prediction)]

print("Prediction:", result)