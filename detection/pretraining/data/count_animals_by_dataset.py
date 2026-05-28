import json
import os


def count_images_by_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    ohio_count = 0
    california_count = 0
    other_count = 0
    total_images = len(data.get("images", []))

    for image in data.get("images", []):
        file_name = image.get("file", "").lower()

        has_animal = False
        for detection in image.get("detections", []):
            category = detection.get("category")
            conf = detection.get("conf", 0)
            if category == "1" and 0.8 <= conf <= 1.0:
                has_animal = True
                break

        if has_animal:
            if "ohio" in file_name:
                ohio_count += 1
            elif "california" in file_name:
                california_count += 1
            else:
                other_count += 1

    return ohio_count, california_count, other_count, total_images


# Resolve dynamic path
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINING_DIR = os.path.dirname(DATA_DIR)
output_file = os.path.join(PRETRAINING_DIR, "output.json")

if os.path.exists(output_file):
    ohio, california, other, total = count_images_by_dataset(output_file)
    print(f"Results for {os.path.basename(output_file)}:")
    print(f"  Total images in file: {total}")
    print(f"  Ohio dataset animal images (conf 0.8-1.0): {ohio}")
    print(f"  California dataset animal images (conf 0.8-1.0): {california}")
    print(f"  Other dataset animal images (conf 0.8-1.0): {other}")
else:
    print(f"File {output_file} not found.")
