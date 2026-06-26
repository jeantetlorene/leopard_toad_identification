import os
import argparse

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")


def is_background_image(image_name, labels_dir):
    """
    Checks if an image is a background image (has no annotations).
    An image is considered a background if its label file does not exist,
    or if it exists but is completely empty (no bounding boxes).
    """
    base_name, _ = os.path.splitext(image_name)
    label_name = base_name + ".txt"
    label_path = os.path.join(labels_dir, label_name)

    if not os.path.isfile(label_path):
        return True

    try:
        with open(label_path, "r") as f:
            content = f.read().strip()
        # If content is empty after stripping whitespace/newlines, it has no boxes
        return not content
    except Exception as e:
        print(f"Warning: Error reading label file '{label_path}': {e}")
        # If we fail to read, assume it is not a background to prevent accidental deletion
        return False


def clean_dataset_split(dataset_path, dry_run=False, extensions=None):
    """
    Finds and deletes background images and labels from a YOLO dataset split.
    """
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist.")
        return 0

    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    try:
        all_files = os.listdir(images_dir)
        image_files = [f for f in all_files if f.endswith(extensions)]
        image_files.sort()
    except Exception as e:
        print(f"Error scanning directory '{images_dir}': {e}")
        return 0

    print(f"Scanning {len(image_files)} image files in '{images_dir}'...")
    if dry_run:
        print("--- RUNNING IN DRY RUN MODE (No files will be deleted) ---")

    deleted_count = 0
    for img_name in image_files:
        if is_background_image(img_name, labels_dir):
            base_name, _ = os.path.splitext(img_name)
            image_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, base_name + ".txt")

            action_str = "Would remove" if dry_run else "Removing"
            print(f"{action_str} background image & label: {img_name}")

            if not dry_run:
                try:
                    os.remove(image_path)
                    if os.path.isfile(label_path):
                        os.remove(label_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting files for '{img_name}': {e}")
            else:
                deleted_count += 1

    mode_str = "identified (dry-run)" if dry_run else "successfully deleted"
    print(f"\nCleanup complete. Total {deleted_count} background files {mode_str}.")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Clean background/empty images and labels from a YOLO dataset split."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Absolute path to the dataset split (e.g. /home/Joshua/.../dataset/test)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Find and list background images without performing any deletion.",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=list(IMAGE_EXTENSIONS),
        help="List of image extensions to scan (default: jpg, jpeg, png, bmp)",
    )

    args = parser.parse_args()

    clean_dataset_split(
        dataset_path=args.dataset_path,
        dry_run=args.dry_run,
        extensions=tuple(args.extensions),
    )


if __name__ == "__main__":
    main()
