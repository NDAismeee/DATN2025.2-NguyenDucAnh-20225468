from pathlib import Path
import shutil
import random

input_dir = Path("/mnt/disk3/anhnd2468/MagicBathyNet/puck_lagoon/img/aerial_filtered")
output_dir = Path("/mnt/disk3/anhnd2468/MagicBathyNet/puck_lagoon/img/aerial_test")
output_dir.mkdir(parents=True, exist_ok=True)

num_images = 100
random_seed = 42

tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

random.seed(random_seed)

selected_files = random.sample(tif_files, min(num_images, len(tif_files)))

for file_path in selected_files:
    shutil.copy2(file_path, output_dir / file_path.name)

print("Done.")
print(f"Total tif files found: {len(tif_files)}")
print(f"Copied files: {len(selected_files)}")
print(f"Output folder: {output_dir}")