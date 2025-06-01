import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

# Config
IMAGE_SIZE = (224, 224)
IMAGE_PATH = "/home/masn24nf/SlowFashion/Input/Input.jpeg"
MODEL_PATH = "/home/masn24nf/SlowFashion/saved_models_ResNet50/group_classifier.keras"
LAYER_NAME = "conv5_block3_out"
SAVE_DIR = "/home/masn24nf/SlowFashion/GradCAM_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

# Preprocess
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0), img

def make_gradcam_heatmap(img_array, full_model):

    base_model = full_model.get_layer("resnet50")

    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    else:
        raise ValueError("No Conv2D layer found.")

    conv_model = tf.keras.Model(inputs=base_model.input, outputs=last_conv_layer.output)

    conv_output = conv_model(img_array)

    x = conv_output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = full_model.get_layer("dense")(x)
    predictions = full_model.get_layer("dense_1")(x)

    with tf.GradientTape() as tape:
        tape.watch(conv_output)
        x = conv_output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = full_model.get_layer("dense")(x)
        predictions = full_model.get_layer("dense_1")(x)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_output)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()



def save_and_display_gradcam(original_img, heatmap, alpha=0.4, save_path=None):
    img = np.array(original_img).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * alpha + img

    if save_path:
        cv2.imwrite(save_path, superimposed_img.astype(np.uint8))

img_array, original_img = preprocess_image(IMAGE_PATH)
heatmap = make_gradcam_heatmap(img_array, model)

filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
save_file_path = os.path.join(SAVE_DIR, f"{filename}_gradcam.jpg")

save_and_display_gradcam(original_img, heatmap, save_path=save_file_path)
