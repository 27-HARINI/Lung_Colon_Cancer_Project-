from tensorflow.keras.models import load_model
import cv2
import numpy as np

print("Loading models...")

# Load trained models
xception_model = load_model("xception_model.h5")
resnet_model = load_model("resnet_model.h5")

# Correct class order (alphabetical order used by training generator)
classes = [
    "Colon Cancer",
    "Colon Normal",
    "Lung Benign",
    "Lung Malignant",
    "Lung Normal"
]

# Load test image (PASTE IMAGE PATH HERE)
img = cv2.imread(r"C:\Users\abira\OneDrive\Desktop\Lung_Colon_Cancer_Project\test_images\colonn_1029.jpeg")

# Check if image loaded
if img is None:
    print("Error: Image not found. Check file path.")
    exit()

# Preprocess image
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

print("Predicting using Xception...")
pred1 = xception_model.predict(img)

print("Predicting using ResNet...")
pred2 = resnet_model.predict(img)

# Print predictions for debugging
print("Xception prediction:", pred1)
print("ResNet prediction:", pred2)

# Ensemble prediction (average)
final_pred = (pred1 + pred2) / 2

print("Ensemble prediction:", final_pred)

# Get final class
result = classes[np.argmax(final_pred)]

print("Final Prediction:", result)