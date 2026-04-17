import pandas as pd
import os
import shutil
import random
from PIL import Image


def get_year_camera(path, subfolder):
    parts = path.split("/")
    year = "unknown"
    for p in parts:
        if p in ["2023", "2024", "2025", "2026"]:
            year = p
            break
    camera = str(subfolder).split("/")[0] if pd.notnull(subfolder) else "unknown"
    return year, camera


def process_dataset(csv_path, bg_csv_path, bg_cam, out_dir):
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    cls_src = "/home/Joshua/Downloads/leopard_toad_identification/dataset/detect_2/yolo_ready_dataset/classes.txt"
    notes_src = "/home/Joshua/Downloads/leopard_toad_identification/dataset/detect_2/yolo_ready_dataset/notes.json"
    if os.path.exists(cls_src):
        shutil.copy(cls_src, out_dir)
    if os.path.exists(notes_src):
        shutil.copy(notes_src, out_dir)

    df = pd.read_csv(csv_path)
    fg_paths = df["image_path"].unique().tolist()

    val_df = pd.read_csv(bg_csv_path)
    bg_df = val_df[val_df["evaluation"] == "Incorrect"]

    def is_target_bg(subf):
        cam = str(subf).split("/")[0]
        return cam == bg_cam

    valid_bg_df = bg_df[bg_df["subfolder"].apply(is_target_bg)]
    bg_paths = valid_bg_df["image_path"].unique().tolist()

    def process_image(p, data_df, is_bg):
        if not os.path.exists(p):
            print(f"Warning: Image does not exist: {p}")
            return False

        subf = data_df[data_df["image_path"] == p].iloc[0]["subfolder"]
        y, c = get_year_camera(p, subf)
        base_name = os.path.basename(p)

        new_name = f"{c}_{y}_{base_name}"
        img_dest = os.path.join(img_out, new_name)
        txt_name = os.path.splitext(new_name)[0] + ".txt"
        lbl_dest = os.path.join(lbl_out, txt_name)

        shutil.copy(p, img_dest)

        with open(lbl_dest, "w") as f:
            if not is_bg:
                boxes = data_df[data_df["image_path"] == p]
                try:
                    with Image.open(p) as img:
                        width, height = img.size
                except Exception as e:
                    print(f"Failed to open image {p}: {e}")
                    return False

                for _, row in boxes.iterrows():
                    cls_id = int(row["class_id"])
                    xmin, ymin = row["xmin"], row["ymin"]
                    xmax, ymax = row["xmax"], row["ymax"]
                    w = xmax - xmin
                    h = ymax - ymin
                    x_center = xmin + w / 2.0
                    y_center = ymin + h / 2.0

                    x_center /= width
                    y_center /= height
                    w /= width
                    h /= height

                    x_center = min(max(x_center, 0.0), 1.0)
                    y_center = min(max(y_center, 0.0), 1.0)
                    w = min(max(w, 0.0), 1.0)
                    h = min(max(h, 0.0), 1.0)

                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        return True

    print(f"\nProcessing output directory: {out_dir}")
    print(
        f"Found Foreground images: {len(fg_paths)}, Background images: {len(bg_paths)}"
    )

    fg_count = 0
    for p in fg_paths:
        if process_image(p, df, False):
            fg_count += 1

    bg_count = 0
    for p in bg_paths:
        if process_image(p, val_df, True):
            bg_count += 1

    print(f"Done for {out_dir}: Successfully saved {fg_count} fg, {bg_count} bg")


def main():
    bg_csv_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/results/detect_2/validation_dataset.csv"

    # Process VAL
    process_dataset(
        csv_path="/home/Joshua/Downloads/leopard_toad_identification/detection/active learning/data/val.csv",
        bg_csv_path=bg_csv_path,
        bg_cam="4R",
        out_dir="/home/Joshua/Downloads/leopard_toad_identification/detection/active learning/data/detect_1/val",
    )

    # Process TEST
    process_dataset(
        csv_path="/home/Joshua/Downloads/leopard_toad_identification/detection/active learning/data/test.csv",
        bg_csv_path=bg_csv_path,
        bg_cam="5Z",
        out_dir="/home/Joshua/Downloads/leopard_toad_identification/detection/active learning/data/detect_1/test",
    )


if __name__ == "__main__":
    main()
