import json
import os
import shutil
from PIL import Image


def build_cat_map(categories):
    """Map category strictly to 'Frog' or 'Mouse' based on name, ignoring snakes."""
    cat_map = {}
    for cat in categories:
        name = cat["name"].lower()
        if "frog" in name or "toad" in name:
            cat_map[cat["id"]] = "Frog"
        elif ("mouse" in name or "rat" in name or "muskrat" in name) and (
            "snake" not in name
        ):
            cat_map[cat["id"]] = "Mouse"
    return cat_map


def parse_dataset(dataset_json_path):
    """Returns a dict mapping the image basename to either 'Frog' or 'Mouse'."""
    with open(dataset_json_path, "r") as f:
        data = json.load(f)

    cat_map = build_cat_map(data["categories"])

    # Map image IDs to their basenames
    img_id_to_basename = {}
    for img in data["images"]:
        img_id_to_basename[img["id"]] = os.path.basename(img["file_name"])

    basename_to_label = {}
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        if cat_id in cat_map:
            img_id = ann["image_id"]
            basename = img_id_to_basename.get(img_id)
            if basename:
                basename_to_label[basename] = cat_map[cat_id]

    return basename_to_label


def main():
    base_dir = "/home/Joshua/Downloads/leopard_toad_identification/dataset"
    ohio_json_path = os.path.join(
        base_dir, "ohio_small_animals_subset", "filtered_annotations.json"
    )
    cali_json_path = os.path.join(
        base_dir, "california_small_animals_subset", "filtered_annotations.json"
    )
    output_json_path = os.path.join(base_dir, "output.json")

    # Target structure
    dataset_dir = os.path.join(base_dir, "dataset4")
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print("Loading Ohio annotations...")
    ohio_mapping = parse_dataset(ohio_json_path)

    print("Loading California annotations...")
    cali_mapping = parse_dataset(cali_json_path)

    # Combine mapping (basenames should be fairly robust, but could theoretically overlap.
    # For now we'll combine them.)
    global_mapping = {**ohio_mapping, **cali_mapping}

    print("Loading MegaDetector predictions...")
    with open(output_json_path, "r") as f:
        md_output = json.load(f)

    cali_count = 0
    ohio_count = 0

    print("Beginning extraction...")
    for item in md_output.get("images", []):
        file_path = item["file"]
        basename = os.path.basename(file_path)

        is_cali = "california_small_animals_subset" in file_path
        is_ohio = "ohio_small_animals_subset" in file_path

        if not is_cali and not is_ohio:
            continue

        # Check if we have a valid ground-truth mapping for this image
        if basename not in global_mapping:
            continue

        label = global_mapping[basename]

        # Check valid detections (Megadetector animal = '1', conf >= 0.8)
        valid_detections = []
        for det in item.get("detections", []):
            if det["category"] == "1" and det["conf"] >= 0.25:
                valid_detections.append(det)

        if not valid_detections:
            continue

        try:
            with Image.open(file_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Skipping {file_path}. Could not read image: {e}")
            continue

        # Create YOLO annotation txt file
        # Format req: string_name absolute_xmin absolute_ymin absolute_xmax absolute_ymax
        txt_basename = os.path.splitext(basename)[0] + ".txt"
        txt_path = os.path.join(labels_dir, txt_basename)

        with open(txt_path, "w") as f:
            for det in valid_detections:
                # md format: [xmin, ymin, width, height] as normalized values
                nx, ny, nw, nh = det["bbox"]
                xmin = nx * width
                ymin = ny * height
                xmax = (nx + nw) * width
                ymax = (ny + nh) * height
                f.write(f"{label} {xmin} {ymin} {xmax} {ymax}\n")

        # Move image file
        dest_img_path = os.path.join(images_dir, basename)
        if not os.path.exists(dest_img_path):
            shutil.move(file_path, dest_img_path)

        if is_cali:
            cali_count += 1
        elif is_ohio:
            ohio_count += 1

    print(
        f"Done. Processed {cali_count} California images and {ohio_count} Ohio images."
    )


if __name__ == "__main__":
    main()
