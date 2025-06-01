import os
from PIL import Image
import numpy as np

data_dir = '/home/masn24nf/SlowFashion/Dataset_Main'

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def process_image_file(filepath):
    try:
        with Image.open(filepath) as img:
            img = img.convert("RGBA")
            return np.array(img)
    except Exception as e:
        os.remove(filepath)
        return None

class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

dataset = {class_name: [] for class_name in class_names}

for class_name in class_names:
    class_dir = os.path.join(data_dir, class_name)
    for root, _, files in os.walk(class_dir):
        for file in files:
            if is_image_file(file):
                file_path = os.path.join(root, file)
                image = process_image_file(file_path)
                if image is not None:
                    dataset[class_name].append(image)
            else:
                full_path = os.path.join(root, file)
                os.remove(full_path)

for class_name, images in dataset.items():
    print(f"Class '{class_name}' has {len(images)} valid images.")
