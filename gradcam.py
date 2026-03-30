import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("xception_model.h5")

# Image path
img_path = "test_images/colonn_1005.jpeg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict class
preds = model.predict(img_array)
class_index = np.argmax(preds)

print("Predicted class index:", class_index)

# Get last convolution layer
last_conv_layer = model.get_layer("block14_sepconv2_act")

# Create GradCAM model
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

# Compute gradients
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

# Compute pooled gradients
pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]

# Weight the feature maps
heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

# Normalize heatmap
heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
heatmap = heatmap.numpy()

# Load original image using OpenCV
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (224,224))

# Resize heatmap
heatmap = cv2.resize(heatmap, (224,224))

# Convert heatmap to color
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Overlay heatmap on image
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# Show result
cv2.imshow("GradCAM Result", superimposed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Also display using matplotlib
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Grad-CAM Visualization")
plt.axis("off")
plt.show()
