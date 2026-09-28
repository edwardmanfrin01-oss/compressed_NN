"""Resize of images from 32x32x3 to 224x224x3"""

import tarfile
import pickle
import os
import json

from PIL import Image
from torchvision import transforms


DATASET_PATH = "data/cifar-10-python.tar.gz"
OUTPUT_DIR = "data/cifar-10_resized"
LABELS_FILE = "data/resized/labels.json"

BATCH_NAMES = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5"
]


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

resize = transforms.Resize(
    (224, 224)
)

# output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)


all_labels = []

with tarfile.open(DATASET_PATH, "r:gz") as tar:

    image_index = 0

    for batch_name in BATCH_NAMES:

        print(f"Processing {batch_name}...")

        member_name = f"cifar-10-batches-py/{batch_name}"
        member = tar.getmember(member_name)

        file_object = tar.extractfile(member)

        if file_object is None:
            raise RuntimeError(
                f"Unable to read {batch_name}"
            )

        batch = pickle.load(
            file_object,
            encoding="bytes"
        )

        data = batch[b"data"]
        labels = batch[b"labels"]

        # Process one image at a time
        for image_data, label in zip(data, labels):

            # CIFAR-10 stores images as:
            # (3, 32, 32)
            image_data = image_data.reshape(3, 32, 32)

            # Convert to:
            # (32, 32, 3)
            image_data = image_data.transpose(1, 2, 0)

            # Convert NumPy array to PIL RGB image
            image = Image.fromarray(image_data)

            # Resize from 32x32 to 224x224
            resized_image = resize(image)

            # Save as lossless PNG
            output_path = os.path.join(
                OUTPUT_DIR,
                f"image_{image_index:05d}.png"
            )

            resized_image.save(
                output_path,
                format="PNG"
            )

            # Store the label associated with this image
            all_labels.append({
                "file": f"image_{image_index:05d}.png",
                "label": label,
                "class": CLASSES[label]
            })

            image_index += 1

        print(f"  {len(data)} images processed")


# Save labels
with open(LABELS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        all_labels,
        f,
        indent=2
    )


# print details
print("\nProcessing completed!")
print(f"Total images: {image_index}")
print(f"Images saved to: {OUTPUT_DIR}")
print(f"Labels saved to: {LABELS_FILE}")