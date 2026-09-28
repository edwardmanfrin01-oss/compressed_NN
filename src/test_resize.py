"""Test the resize of a single image, from 32x32x3 to 224x224x3"""
import tarfile
import pickle
import io
import os

from PIL import Image
from torchvision import transforms


DATASET_PATH = "data/cifar-10-python.tar.gz"
OUTPUT_DIR = "data/resized"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "image_00000.png")


with tarfile.open(DATASET_PATH, "r:gz") as tar:

    # CIFAR-10 training set is divided in data_batch_1 ... data_batch_5.
    member = tar.getmember("cifar-10-batches-py/data_batch_1")

    # Extract the file from the memory
    file_object = tar.extractfile(member)

    if file_object is None:
        raise RuntimeError("Impossibile leggere data_batch_1")

    batch = pickle.load(file_object, encoding="bytes")


# Get the first image

data = batch[b"data"]
labels = batch[b"labels"]

image_data = data[0]
label = labels[0]

print("Original image:")
print(f"  Shape array: {image_data.shape}")
print(f"  Label: {label}")


# CIFAR-10 images are saved as 1D vector (32x32x3 = 3072 elements)
# so they must be reshaped in 32x32x3

image_data = image_data.reshape(3, 32, 32)
image_data = image_data.transpose(1, 2, 0)

image = Image.fromarray(image_data)


print(f"  PIL mode: {image.mode}")
print(f"  PIL size: {image.size}")


# Apply the resize 32x32 -> 224x224

resize = transforms.Resize(
    (224, 224)
)

resized_image = resize(image)


print("\nNew Image:")
print(f"  PIL mode: {resized_image.mode}")
print(f"  PIL size: {resized_image.size}")


# Save as PNG lossless

os.makedirs(OUTPUT_DIR, exist_ok=True)

resized_image.save(
    OUTPUT_FILE,
    format="PNG"
)


# Final file

file_size = os.path.getsize(OUTPUT_FILE)

print("\nFile saved:")
print(f"  Path: {OUTPUT_FILE}")
print(f"  Size: {file_size:,} bytes")