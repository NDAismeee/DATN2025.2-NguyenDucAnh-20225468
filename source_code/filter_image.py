from pathlib import Path
import shutil
import numpy as np
import tifffile
from PIL import Image

input_dir = Path("/mnt/disk3/anhnd2468/MagicBathyNet/puck_lagoon/img/aerial")
output_dir = Path("/mnt/disk3/anhnd2468/MagicBathyNet/puck_lagoon/img/aerial_filtered")
output_dir.mkdir(parents=True, exist_ok=True)

image_exts = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

copied_count = 0
skipped_count = 0

for img_path in input_dir.iterdir():
    if img_path.suffix.lower() not in image_exts:
        continue

    try:
        if img_path.suffix.lower() in [".tif", ".tiff"]:
            img = tifffile.imread(img_path)
        else:
            img = np.array(Image.open(img_path))

        # Pixel đen:
        # - ảnh grayscale: value == 0
        # - ảnh RGB: cả 3 kênh đều == 0
        if img.ndim == 2:
            black_mask = img == 0
        else:
            black_mask = np.all(img == 0, axis=-1)

        black_ratio = black_mask.mean()

        # Giữ ảnh nếu vùng đen <= 20%
        if black_ratio <= 0.20:
            shutil.copy2(img_path, output_dir / img_path.name)
            copied_count += 1
            print(f"Copied: {img_path.name} | black ratio = {black_ratio:.2%}")
        else:
            skipped_count += 1
            print(f"Skipped: {img_path.name} | black ratio = {black_ratio:.2%}")

    except Exception as e:
        print(f"Error reading {img_path.name}: {e}")

print("\nDone.")
print(f"Copied images: {copied_count}")
print(f"Skipped images: {skipped_count}")