import os
import tensorflow as tf
import numpy as np
from PIL import Image

SOURCE_DIR = "/home/masn24nf/SlowFashion/Dataset_Main"
DEST_DIR = "/home/masn24nf/SlowFashion/Augmented_Dataset"
IMAGE_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 3


def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    image = tf.image.random_brightness(image, max_delta=brightness)
    image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
    image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
    image = tf.image.random_hue(image, max_delta=hue)
    return image

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.map_fn(color_jitter, x)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

os.makedirs(DEST_DIR, exist_ok=True)

def augment_and_save_image(image_path, save_dir, base_name):
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_tensor = tf.expand_dims(img_array, 0)

    for i in range(AUGMENTATIONS_PER_IMAGE):
        augmented = data_augmentation(img_tensor, training=True)
        aug_img = tf.squeeze(augmented, axis=0).numpy().astype(np.uint8)
        aug_img_pil = Image.fromarray(aug_img)
        aug_img_pil.save(os.path.join(save_dir, f"{base_name}_aug_{i}.jpg"))

def process_dataset(source_root, dest_root):
    for class_folder in os.listdir(source_root):
        class_path = os.path.join(source_root, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        dest_class_path = os.path.join(dest_root, class_folder)
        os.makedirs(dest_class_path, exist_ok=True)

        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, img_file)
            base_name = os.path.splitext(img_file)[0]
            augment_and_save_image(img_path, dest_class_path, base_name)

if __name__ == "__main__":
    process_dataset(SOURCE_DIR, DEST_DIR)
