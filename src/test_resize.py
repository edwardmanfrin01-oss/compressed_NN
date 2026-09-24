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

    # Il training set CIFAR-10 è diviso in data_batch_1 ... data_batch_5.
    member = tar.getmember("cifar-10-batches-py/data_batch_1")

    # Estraiamo il file in memoria, senza scriverlo su disco.
    file_object = tar.extractfile(member)

    if file_object is None:
        raise RuntimeError("Impossibile leggere data_batch_1")

    batch = pickle.load(file_object, encoding="bytes")


# ============================================================
# 2. Recuperiamo la prima immagine
# ============================================================

data = batch[b"data"]
labels = batch[b"labels"]

image_data = data[0]
label = labels[0]

print("Informazioni immagine originale:")
print(f"  Shape array: {image_data.shape}")
print(f"  Label: {label}")


# CIFAR-10 memorizza ogni immagine come:
#
# [R R R ... R | G G G ... G | B B B ... B]
#
# quindi dobbiamo trasformarla da:
#
# (3072,)
#
# a:
#
# (3, 32, 32)

image_data = image_data.reshape(3, 32, 32)

# PIL invece vuole generalmente:
# (height, width, channels)
#
# quindi:
# (3, 32, 32) -> (32, 32, 3)

image_data = image_data.transpose(1, 2, 0)

image = Image.fromarray(image_data)


print(f"  PIL mode: {image.mode}")
print(f"  PIL size: {image.size}")


# ============================================================
# 3. Resize 32x32 -> 224x224
# ============================================================

resize = transforms.Resize(
    (224, 224)
)

resized_image = resize(image)


print("\nInformazioni immagine ridimensionata:")
print(f"  PIL mode: {resized_image.mode}")
print(f"  PIL size: {resized_image.size}")


# ============================================================
# 4. Salviamo come PNG lossless
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

resized_image.save(
    OUTPUT_FILE,
    format="PNG"
)


# ============================================================
# 5. Informazioni sul file prodotto
# ============================================================

file_size = os.path.getsize(OUTPUT_FILE)

print("\nFile salvato:")
print(f"  Path: {OUTPUT_FILE}")
print(f"  Size: {file_size:,} bytes")