This repository provides a hierarchical classification system for fashion product images using deep learning models. It includes code for data augmentation and two model architectures—VGG16 and ResNet50—for classifying images into main categories (Clothing, Shoes, Bags) and further into subcategories within each main group.

Contents
data_aug.py: Script for augmenting the dataset with various image transformations.

Hierarchial_VGG16.py: Hierarchical classification pipeline using VGG16 as the base model.

Hierarchial_ResNet50.py: Hierarchical classification pipeline using ResNet50 as the base model.

/SlowFashion/
    Dataset_Main/           # Original dataset (not included)
    Augmented_Dataset/      # Augmented dataset (output of data_aug.py)
    Input/                  # Input images for prediction
    saved_models_VGG16/     # Saved VGG16 models
    saved_models_ResNet50/  # Saved ResNet50 models
data_aug.py
Hierarchial_VGG16.py
Hierarchial_ResNet50.py
