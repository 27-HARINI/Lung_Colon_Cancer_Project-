import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- LOAD MODELS ----------------
xception_model = load_model("xception_model.h5")
resnet_model = load_model("resnet_model.h5")

classes = [
    "Colon Cancer",
    "Colon Normal",
    "Lung Benign",
    "Lung Malignant",
    "Lung Normal"
]

# ---------------- UI ----------------
st.title("AI-Based Lung and Colon Cancer Detection")
st.write("Upload a histopathology image for cancer detection.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

# ---------------- MAIN ----------------
if uploaded_file is not None:

    # ✅ Safe image load
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except:
        st.error("❌ Error loading image.")
        st.stop()

    st.image(image, caption="Uploaded Image", width=300)

    # ✅ Preprocess
    img = image.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ Predictions
    pred_xception = xception_model.predict(img_array)
    pred_resnet = resnet_model.predict(img_array)

    final_pred = (pred_xception + pred_resnet) / 2

    class_index = np.argmax(final_pred)
    confidence = np.max(final_pred) * 100

    # ✅ Entropy
    entropy = -np.sum(final_pred * np.log(final_pred + 1e-10))

    st.subheader("Prediction Result")

    # ---------------- FINAL VALIDATION ----------------

    # ❌ Reject non-medical image (IMPORTANT FIX)
    if confidence < 50 or entropy > 1.1:
        st.error("❌ Invalid Image! Please upload a proper medical image.")
        st.stop()

    # ⚠️ Low confidence but valid
    elif confidence < 70:
        st.warning("⚠️ Low confidence prediction. Result may not be reliable.")
        st.write("Predicted:", classes[class_index])
        st.write("Confidence:", round(confidence,2), "%")

    # ✅ High confidence
    else:
        st.success(classes[class_index])
        st.write("Confidence:", round(confidence,2), "%")

    # ✅ Confidence bar
    st.progress(int(confidence))

    # ---------------- GRAD-CAM (ONLY VALID IMAGES) ----------------
    try:
        last_conv_layer = xception_model.get_layer("block14_sepconv2_act")

        grad_model = tf.keras.models.Model(
            inputs=xception_model.input,
            outputs=[last_conv_layer.output, xception_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap,0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        original_img = cv2.cvtColor(
            np.array(image.resize((224,224))),
            cv2.COLOR_RGB2BGR
        )

        heatmap = cv2.resize(heatmap,(224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = cv2.addWeighted(original_img,0.6,heatmap,0.4,0)

        st.subheader("Grad-CAM Visualization")
        st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    except:
        st.warning("Grad-CAM could not be generated.")