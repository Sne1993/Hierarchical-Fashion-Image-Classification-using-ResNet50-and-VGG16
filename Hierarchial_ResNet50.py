import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = "/home/masn24nf/SlowFashion/Augmented_Dataset"
SAVE_DIR = os.path.abspath("saved_models_ResNet50")
os.makedirs(SAVE_DIR, exist_ok=True)

main_train = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

main_val = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

clothing_train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Clothing"),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

clothing_val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Clothing"),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

shoes_train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Shoes"),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

shoes_val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Shoes"),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

bags_train = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Bags"),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

bags_val = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "Bags"),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

main_class_names = main_train.class_names
clothing_class_names = clothing_train.class_names
shoes_class_names = shoes_train.class_names
bags_class_names = bags_train.class_names

AUTOTUNE = tf.data.AUTOTUNE


main_train = main_train.prefetch(AUTOTUNE)
clothing_train = clothing_train.prefetch(AUTOTUNE)
shoes_train = shoes_train.prefetch(AUTOTUNE)
bags_train = bags_train.prefetch(AUTOTUNE)

main_val = main_val.prefetch(AUTOTUNE)
clothing_val = clothing_val.prefetch(AUTOTUNE)
shoes_val = shoes_val.prefetch(AUTOTUNE)
bags_val = bags_val.prefetch(AUTOTUNE)

def get_model(num_classes):
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

main_model = get_model(num_classes=3)
clothing_model = get_model(num_classes=len(clothing_class_names))
shoes_model = get_model(num_classes=len(shoes_class_names))
bags_model = get_model(num_classes=len(bags_class_names))

print("\nMain Category training")
main_model.fit(main_train, validation_data=main_val, epochs=5)

print("\nClothing Category training")
clothing_model.fit(clothing_train, validation_data=clothing_val, epochs=5)

print("\nShoes Category training")
shoes_model.fit(shoes_train, validation_data=shoes_val, epochs=5)

print("\nBags Category training")
bags_model.fit(bags_train, validation_data=bags_val, epochs=5)

def load_and_preprocess(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def hierarchical_predict(image_path, threshold=0.8):
    img_tensor = load_and_preprocess(image_path)
    main_probs = main_model.predict(img_tensor)[0]
    pred_main = np.argmax(main_probs)
    main_class = main_class_names[pred_main]

    if main_class == "Clothing":
        sub_probs = clothing_model.predict(img_tensor)[0]
        sub_pred = np.argmax(sub_probs)
        sub_class = clothing_class_names[sub_pred] if sub_probs[sub_pred] >= threshold else "Uncertain"
    elif main_class == "Shoes":
        sub_probs = shoes_model.predict(img_tensor)[0]
        sub_pred = np.argmax(sub_probs)
        sub_class = shoes_class_names[sub_pred] if sub_probs[sub_pred] >= threshold else "Uncertain"
    elif main_class == "Bags":
        sub_probs = bags_model.predict(img_tensor)[0]
        sub_pred = np.argmax(sub_probs)
        sub_class = bags_class_names[sub_pred] if sub_probs[sub_pred] >= threshold else "Uncertain"
    else:
        sub_class = "Uncertain"

    return main_class, sub_class

# Save models
main_model.save(os.path.join(SAVE_DIR, "group_classifier.keras"))
clothing_model.save(os.path.join(SAVE_DIR, "clothing_classifier.keras"))
shoes_model.save(os.path.join(SAVE_DIR, "shoes_classifier.keras"))
bags_model.save(os.path.join(SAVE_DIR, "bags_classifier.keras"))

def evaluate_model(model, dataset, class_names):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

print("\nMain Model Evaluation:")
evaluate_model(main_model, main_val, main_class_names)

print("\nClothing Model Evaluation:")
evaluate_model(clothing_model, clothing_val, clothing_class_names)

print("\nShoes Model Evaluation:")
evaluate_model(shoes_model, shoes_val, shoes_class_names)

print("\nBags Model Evaluation:")
evaluate_model(bags_model, bags_val, bags_class_names)

# Predict
image_path = "/home/masn24nf/SlowFashion/Input/Input.jpeg"
main_cat, sub_cat = hierarchical_predict(image_path)
print("Main Category:", main_cat)
print("Subclass:", sub_cat)
