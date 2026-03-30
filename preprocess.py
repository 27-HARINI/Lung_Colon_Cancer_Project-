import os
import cv2
import numpy as np

# Image size
IMG_SIZE = 224

# Dataset path
DATASET_PATH = "dataset"

# Folder names EXACTLY as in your system
categories = {
    "lung": {
        "lung-normal": 0,
        "lung cancer- bengin": 1,
        "lung cancer-Malignant": 2
    },
    "colon": {
        "colon normal": 3,
        "colon-cancer": 4
    }
}

data = []
labels = []

def preprocess_images():
    for organ in categories:
        for folder, label in categories[organ].items():
            folder_path = os.path.join(DATASET_PATH, organ, folder)
            print(f"Reading: {folder_path}")

            if not os.path.exists(folder_path):
                print(f"❌ Folder not found: {folder_path}")
                continue

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0  # normalize
                    data.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error reading {img_path}")

    print("✅ Preprocessing completed")

# Run preprocessing
preprocess_images()

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

print("Data shape:", X.shape)
print("Labels shape:", y.shape)

# Save data
np.save("X_data.npy", X)
np.save("y_labels.npy", y)

print("✅ Data saved successfully")
