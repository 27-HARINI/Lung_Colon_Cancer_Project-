import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from tensorflow.keras.utils import save_img

dataset_path = "dataset"
target_count = 90

datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    fill_mode="nearest"
)

for class_name in os.listdir(dataset_path):

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    current_count = len(images)

    print(f"\nProcessing {class_name} ({current_count} images)")

    if current_count >= target_count:
        print("Already enough images.")
        continue

    i = 0

    while len(os.listdir(class_path)) < target_count:

        img_name = images[i % current_count]
        img_path = os.path.join(class_path, img_name)

        img = load_img(img_path)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        aug_iter = datagen.flow(img_array, batch_size=1)

        aug_img = next(aug_iter)[0].astype("uint8")

        new_name = f"aug_{i}_{img_name}"
        save_img(os.path.join(class_path, new_name), aug_img)

        i += 1

    print("Final count:", len(os.listdir(class_path)))