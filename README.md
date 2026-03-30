# AI-Based Lung and Colon Cancer Detection

##  Project Overview
This project is an AI-based system designed to detect lung and colon cancer from histopathology images using deep learning techniques. It uses an ensemble of Xception and ResNet models to improve classification accuracy and reliability.

---

##  Objectives
- Detect cancer from medical images
- Classify into multiple categories
- Improve prediction accuracy using ensemble learning
- Provide visual explanation using Grad-CAM

---

##  Dataset
- Histopathology image dataset
- Classes:
  - Colon Cancer
  - Colon Normal
  - Lung Benign
  - Lung Malignant
  - Lung Normal

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit

---

##  Models Used
- Xception Model
- ResNet Model
- Ensemble Learning (Average Prediction)

---

##  Workflow
1. Image Upload
2. Image Preprocessing (Resize, Normalize)
3. Model Prediction (Xception + ResNet)
4. Ensemble Prediction
5. Confidence Check
6. Final Classification
7. Grad-CAM Visualization

---

## Features
- Multi-class classification
- Ensemble model for better accuracy
- Confidence-based validation
- Invalid image detection
- Grad-CAM visualization for explainability

---

##  How to Run
pip install -r requirements.txt
streamlit run app.py
