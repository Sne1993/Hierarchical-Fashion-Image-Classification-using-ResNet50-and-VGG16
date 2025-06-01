from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess

IMAGE_SIZE = (224, 224)
SAVE_DIR = "saved_models_VGG16"
#SAVE_DIR = "saved_models_ResNet50"
DATASET_PATH = "/home/masn24nf/SlowFashion/Dataset_Main"

# Load models
main_model = load_model(os.path.join(SAVE_DIR, "group_classifier.keras"), safe_mode=False)
clothing_model = load_model(os.path.join(SAVE_DIR, "clothing_classifier.keras"), safe_mode=False)
shoes_model = load_model(os.path.join(SAVE_DIR, "shoes_classifier.keras"), safe_mode=False)
bags_model = load_model(os.path.join(SAVE_DIR, "bags_classifier.keras"), safe_mode=False)


main_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=32
)
clothing_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Clothing"),
    image_size=IMAGE_SIZE,
    batch_size=32
)
shoes_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Shoes"),
    image_size=IMAGE_SIZE,
    batch_size=32
)
bags_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Bags"),
    image_size=IMAGE_SIZE,
    batch_size=32
)

def load_and_preprocess(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = vgg_preprocess(img_array)
    return np.expand_dims(img_array, axis=0)

def hierarchical_predict(image_path, threshold=0.8):
    img_tensor = load_and_preprocess(image_path)

    # Predict main category
    main_probs = main_model.predict(img_tensor)[0]
    pred_main = np.argmax(main_probs)
    main_class = main_dataset.class_names[pred_main]

    # Predict sub-category only if confident enough
    if main_class == "Clothing":
        sub_probs = clothing_model.predict(img_tensor)[0]
        sub_pred = np.argmax(sub_probs)
        sub_conf = sub_probs[sub_pred]
        sub_class = clothing_ds.class_names[sub_pred] if sub_conf >= threshold else "Uncertain"
    elif main_class == "Shoes":
        sub_probs = shoes_model.predict(img_tensor)[0]
        sub_pred = np.argmax(sub_probs)
        sub_conf = sub_probs[sub_pred]
        sub_class = shoes_ds.class_names[sub_pred] if sub_conf >= threshold else "Uncertain"
    elif main_class == "Bags":
        sub_probs = bags_model.predict(img_tensor)[0]
        sub_pred = np.argmax(sub_probs)
        sub_conf = sub_probs[sub_pred]
        sub_class = bags_ds.class_names[sub_pred] if sub_conf >= threshold else "Uncertain"
    else:
        sub_class = "Uncertain"

    return main_class, sub_class



# Predict
image_path = "/home/masn24nf/SlowFashion/Input/Input.jpeg"
main_cat, sub_cat = hierarchical_predict(image_path)
print("Main Category:", main_cat)
print("Subclass:", sub_cat)
