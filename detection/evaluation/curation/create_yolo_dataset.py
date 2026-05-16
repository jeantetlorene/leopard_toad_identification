import pandas as pd
import os
import shutil
from PIL import Image
import hashlib


def get_unique_name(img_path):
    # Use a short hash of the full path to avoid collisions for same filenames in different dirs
    path_hash = hashlib.md5(img_path.encode()).hexdigest()[:8]
    base_name = os.path.basename(img_path)
    return f"{path_hash}-{base_name}"


def convert_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    x = x_center * dw
    y = y_center * dh
    w = width * dw
    h = height * dh
    return x, y, w, h


def process_split(eval_csv, final_csv, split_name, base_dir, sample_classes_txt):
    print(f"Processing split: {split_name}")
    # Load DataFrames
    eval_df = pd.read_csv(eval_csv)
    final_df = pd.read_csv(final_csv)

    # Filter for correct and missed
    correct_images = eval_df[eval_df["evaluation"] == "Correct"]["image_path"].tolist()
    missed_images = eval_df[eval_df["evaluation"] == "Missed Animal"][
        "image_path"
    ].tolist()

    # Create directories for correct YOLO dataset
    split_dir = os.path.join(base_dir, split_name)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Copy classes.txt
    if os.path.exists(sample_classes_txt):
        shutil.copy2(sample_classes_txt, os.path.join(split_dir, "classes.txt"))

    # Create directory for missed images
    missed_dir = os.path.join(base_dir, "missed", split_name)
    os.makedirs(missed_dir, exist_ok=True)

    print(
        f"  Found {len(correct_images)} correct image entries and {len(missed_images)} missed images."
    )

    mapping = []

    # Process missed images
    for img_path in missed_images:
        if not pd.isna(img_path) and os.path.exists(img_path):
            img_name = get_unique_name(img_path)
            mapping.append(
                {
                    "unique_name": img_name,
                    "original_path": img_path,
                    "split": split_name,
                    "type": "missed",
                }
            )
            dest_path = os.path.join(missed_dir, img_name)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
        elif not pd.isna(img_path):
            print(f"  Missed image path not found or invalid: {img_path}")

    # Process correct images
    correct_df = final_df[final_df["image_path"].isin(correct_images)]
    grouped = correct_df.groupby("image_path")

    print(f"  Exporting {len(grouped)} unique images for {split_name} split...")

    for img_path, group in grouped:
        if not pd.isna(img_path) and os.path.exists(img_path):
            img_name = get_unique_name(img_path)
            mapping.append(
                {
                    "unique_name": img_name,
                    "original_path": img_path,
                    "split": split_name,
                    "type": "correct",
                }
            )
            dest_img_path = os.path.join(images_dir, img_name)

            # Read image to get dimensions
            try:
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"  Failed to read image {img_path}: {e}")
                continue

            # Copy image
            if not os.path.exists(dest_img_path):
                shutil.copy2(img_path, dest_img_path)

            # Create label file
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)

            with open(label_path, "w") as f:
                for _, row in group.iterrows():
                    class_id = int(row["class_id"])
                    xmin, ymin, xmax, ymax = (
                        row["xmin"],
                        row["ymin"],
                        row["xmax"],
                        row["ymax"],
                    )
                    x, y, w, h = convert_to_yolo(
                        xmin, ymin, xmax, ymax, img_width, img_height
                    )
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    w = max(0.0, min(1.0, w))
                    h = max(0.0, min(1.0, h))
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        elif not pd.isna(img_path):
            print(f"  Correct image path not found or invalid: {img_path}")

    correct_images_without_boxes = set(correct_images) - set(grouped.groups.keys())
    for img_path in correct_images_without_boxes:
        if not pd.isna(img_path) and os.path.exists(img_path):
            img_name = get_unique_name(img_path)
            mapping.append(
                {
                    "unique_name": img_name,
                    "original_path": img_path,
                    "split": split_name,
                    "type": "correct_empty",
                }
            )
            dest_img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(dest_img_path):
                shutil.copy2(img_path, dest_img_path)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            open(label_path, "w").close()

    return mapping


if __name__ == "__main__":
    base_repo_dir = "/home/Joshua/Downloads/leopard_toad_identification"
    evaluation_dir = os.path.join(base_repo_dir, "detection", "evaluation")
    consensus_dir = os.path.join(evaluation_dir, "consensus_predictions")
    output_base_dir = os.path.join(evaluation_dir, "data")
    sample_classes_txt = os.path.join(
        base_repo_dir,
        "detection",
        "active learning",
        "data",
        "detect_1",
        "test",
        "classes.txt",
    )

    # Clear existing directories for a fresh start
    for d in ["val", "test", "missed"]:
        dir_path = os.path.join(output_base_dir, d)
        if os.path.exists(dir_path):
            print(f"Cleaning up {dir_path}...")
            shutil.rmtree(dir_path)

    all_mappings = []

    # Process Val
    val_eval = os.path.join(consensus_dir, "val_consensus_final_evaluations.csv")
    val_final = os.path.join(consensus_dir, "val_consensus_final.csv")
    if os.path.exists(val_eval) and os.path.exists(val_final):
        all_mappings.extend(
            process_split(
                val_eval, val_final, "val", output_base_dir, sample_classes_txt
            )
        )
    else:
        print("Val files missing")

    # Process Test
    test_eval = os.path.join(consensus_dir, "test_consensus_final_evaluations.csv")
    test_final = os.path.join(consensus_dir, "test_consensus_final.csv")
    if os.path.exists(test_eval) and os.path.exists(test_final):
        all_mappings.extend(
            process_split(
                test_eval, test_final, "test", output_base_dir, sample_classes_txt
            )
        )
    else:
        print("Test files missing")

    # Save mapping file
    if all_mappings:
        mapping_df = pd.DataFrame(all_mappings)
        mapping_path = os.path.join(output_base_dir, "image_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Saved image mapping to {mapping_path}")
