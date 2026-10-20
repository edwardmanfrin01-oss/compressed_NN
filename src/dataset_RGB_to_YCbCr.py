
import tarfile
import pickle
import os
import json

from PIL import Image
from torchvision import transforms


DATASET_PATH = "data/cifar-10_resized/test"
OUTPUT_DIR = "data/cifar-10_resized_YCbCr/test"
LABELS_FILE = "data/cifar-10_resized/cifar-10_labels.json"


# CIFAR-10 classes

CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


all_labels = []

image_index = 0

for image in os.listdir(DATASET_PATH):
    img_rgb = Image.open(os.path.join(DATASET_PATH, image)).convert('RGB')
    img_ycbcr = img_rgb.convert('YCbCr')

    # Save as lossless jpg
    output_path = os.path.join(
        OUTPUT_DIR,
        f"image_{image_index:05d}.JPEG"  # Use .jpg extension for YCbCr images
    )

    img_ycbcr.save(
        output_path,
        format="JPEG"
    )

    label = json.load(open(LABELS_FILE, "r"))[image_index]["label"]

    # Store the label associated with this image
    all_labels.append({
        "file": f"image_{image_index:05d}.JPEG",
        "label": label,
        "class": CLASSES[label]
    })

    image_index += 1
    if image_index > 100:
        break  # Limit to 100 images for testing purposes

print(f"  {image_index} images processed")


# print details
print("\nProcessing completed!")
print(f"Total images: {image_index}")
print(f"Images saved to: {OUTPUT_DIR}")
print(f"Labels saved to: {LABELS_FILE}")