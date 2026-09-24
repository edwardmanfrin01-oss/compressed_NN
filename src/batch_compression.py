from pathlib import Path
import subprocess
import time

# Folders
input_dir = Path("../data/cifar-10_resized")
output_dir = Path("../data/cifar-10_compressed")
gdcompress_bin = "./target/release/gdcompress"   # Percorso dell'eseguibile Rust (usa gdcompress.exe se sei su Windows)

output_dir.mkdir(parents=True, exist_ok=True)

image_files = list(input_dir.glob("*.png"))
total_images = len(image_files)

print(f"Found {total_images} images to be compressed...")

start_time = time.time()

for idx, img_path in enumerate(image_files, 1):
    output_path = output_dir / f"{img_path.stem}.igd"
    
    # Command for running the EntroGD compression
    cmd = [gdcompress_bin, str(img_path), "-o", str(output_path)]
    
    try:
        # Run the compression command
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if idx % 500 == 0 or idx == total_images:
            print(f"Completed {idx}/{total_images} images")
            print(f"{(idx * 100)/total_images}% Completed...")
    except subprocess.CalledProcessError as e:
        print(f"Error during the compression of image {img_path.name}: {e}")

# Keep track of the time
elapsed_time = time.time() - start_time
print(f"Total compression time: {elapsed_time:.2f}")