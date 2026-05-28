import os
import shutil
import sys

# Resolve parent directory to import central_config
PIPELINES_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_LEARNING_DIR = os.path.dirname(PIPELINES_DIR)
if ACTIVE_LEARNING_DIR not in sys.path:
    sys.path.append(ACTIVE_LEARNING_DIR)

from central_config import CLASSES


def load_original_classes(dataset_dir):
    """Loads the original classes from the dataset's classes.txt file."""
    paths_to_try = [
        os.path.join(dataset_dir, "train", "classes.txt"),
        os.path.join(dataset_dir, "classes.txt"),
        os.path.join(dataset_dir, "val", "classes.txt"),
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
            return classes

    # Fallback to central config CLASSES if classes.txt is not found
    return [name for name in CLASSES.values()]


def resolve_target_classes(original_classes, target_classes=None, class_mapping=None):
    """
    Resolves the final target classes and a mapping from original class IDs to target class IDs.
    """
    if target_classes is None:
        resolved_classes = list(original_classes)
        id_mapping = {i: i for i in range(len(original_classes))}
        return resolved_classes, id_mapping

    resolved_classes = list(target_classes)
    id_mapping = {}

    for orig_idx, orig_name in enumerate(original_classes):
        if class_mapping is not None and orig_name in class_mapping:
            target_name = class_mapping[orig_name]
        else:
            if orig_name in target_classes:
                target_name = orig_name
            else:
                target_name = None  # Background

        if target_name is not None and target_name in resolved_classes:
            target_idx = resolved_classes.index(target_name)
            id_mapping[orig_idx] = target_idx

    return resolved_classes, id_mapping


def map_split(dataset_dir, dataset_dir_mapped, split, id_mapping, resolved_classes):
    """Creates a mapped dataset split with symlinked images and remapped labels."""
    custom_img_dir = os.path.join(dataset_dir, split, "images")
    custom_lbl_dir = os.path.join(dataset_dir, split, "labels")
    yolo_img_dir = os.path.join(dataset_dir, "images", split)
    yolo_lbl_dir = os.path.join(dataset_dir, "labels", split)

    if os.path.exists(custom_img_dir):
        src_img_dir = custom_img_dir
        src_lbl_dir = custom_lbl_dir
    elif os.path.exists(yolo_img_dir):
        src_img_dir = yolo_img_dir
        src_lbl_dir = yolo_lbl_dir
    else:
        return False  # Split doesn't exist

    dest_img_dir = os.path.join(dataset_dir_mapped, split, "images")
    dest_lbl_dir = os.path.join(dataset_dir_mapped, split, "labels")
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_lbl_dir, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_files = [
        f
        for f in os.listdir(src_img_dir)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    for img_file in img_files:
        # 1. Symlink image
        src_img_path = os.path.abspath(os.path.join(src_img_dir, img_file))
        dest_img_path = os.path.abspath(os.path.join(dest_img_dir, img_file))
        if os.path.exists(dest_img_path):
            os.remove(dest_img_path)
        os.symlink(src_img_path, dest_img_path)

        # 2. Map and write label
        base_name = os.path.splitext(img_file)[0]
        src_lbl_path = os.path.join(src_lbl_dir, base_name + ".txt")
        dest_lbl_path = os.path.join(dest_lbl_dir, base_name + ".txt")

        mapped_lines = []
        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, "r") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        orig_cls_id = int(parts[0])
                        if orig_cls_id in id_mapping:
                            target_cls_id = id_mapping[orig_cls_id]
                            mapped_lines.append(
                                f"{target_cls_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                            )

        with open(dest_lbl_path, "w") as f_out:
            f_out.writelines(mapped_lines)

    # Write classes.txt inside the split folder
    with open(os.path.join(dataset_dir_mapped, split, "classes.txt"), "w") as f_cls:
        for name in resolved_classes:
            f_cls.write(name + "\n")

    return True
