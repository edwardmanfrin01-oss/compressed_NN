import os
from PIL import Image
from timeit import default_timer as timer


DATASET_PATH = "data/cifar-10_resized/train"
OUTPUT_DIR = "data/cifar-10_resized_YCbCr/train"


# output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

image_index = 0

start_time = timer()
for image in os.listdir(DATASET_PATH):

    if not image.lower().endswith(".png"):
        continue
    if not os.path.isfile(os.path.join(DATASET_PATH, image)):
        continue
    
    with Image.open(os.path.join(DATASET_PATH, image)) as source:
        img_rgb = source.convert("RGB")
        img_ycbcr = img_rgb.convert("YCbCr")

    # Store the three channels Y, Cb and Cr as three RGB channels in a png file
    y, cb, cr = img_ycbcr.split()
    img_png = Image.merge("RGB", (y, cb, cr))

    output_name = os.path.splitext(image)[0] + ".png"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    img_png.save(output_path, format="PNG")

    image_index += 1
    if image_index % 1000 == 0:
        print(f"  {image_index} images processed")

end_time = timer() - start_time

print(f"  {image_index} images processed")
print(f"Processing time: {end_time:.1f}s")


# print details
print("\nProcessing completed!")
print(f"Total images: {image_index}")
print(f"Images saved to: {OUTPUT_DIR}")