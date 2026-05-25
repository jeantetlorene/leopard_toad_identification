import os
import cv2
import csv
from pathlib import Path
import torch
from ultralytics import RTDETR
from tqdm import tqdm
import concurrent.futures

# --- Class Map and Thresholds ---
CLASSES = {
    0: "Other_Amphibian",
    1: "Small_Mammal",
    2: "Western_Leopard_Toad"
}

# Optimal validation analytical thresholds based on F1-Score maximization
THRESHOLDS = {0: 0.2, 1: 0.2, 2: 0.25}


def apply_clahe(im):
    """
    Applies CLAHE preprocessing in LAB space.
    Input: BGR image numpy array
    Output: BGR image numpy array with CLAHE applied
    """
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    im_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return im_clahe


def _process_image(img_path):
    """
    Reads an image, applies BGR CLAHE preprocessing.
    """
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None, img_path
        input_img = apply_clahe(img_bgr)
        return input_img, img_path
    except Exception:
        return None, img_path


def process_folder(
    input_folder,
    output_folder,
    model,
    img_size,
    batch_size,
    device,
    all_writer
):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Directory {input_path} does not exist. Skipping.")
        return []

    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    excluded_cameras = {"4R", "5Z"}

    # Find immediate subdirectories and the root folder
    immediate_subfolders = [d for d in input_path.iterdir() if d.is_dir()]
    targets = [input_path] + immediate_subfolders

    folder_detections = []

    for target_dir in targets:
        # Determine image paths and CSV filename
        if target_dir == input_path:
            images = [
                f
                for f in target_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            csv_name = f"{input_path.name}_root.csv"
        else:
            images = [
                f
                for f in target_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            csv_name = f"{target_dir.name}.csv"

        # Filter out images from the test/val cameras (4R, 5Z)
        images = [
            f
            for f in images
            if not any(cam in f.parts for cam in excluded_cameras)
        ]

        if not images:
            continue

        csv_path = output_path / csv_name
        print(f"Found {len(images)} unlabeled images in '{target_dir.name}'. Saving predictions to {csv_name}...")

        with open(csv_path, mode="w", newline="") as f_out:
            writer = csv.writer(f_out)
            headers = [
                "image_path",
                "image_name",
                "subfolder",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
            writer.writerow(headers)

            with tqdm(total=len(images), desc=f"Processing {target_dir.name}") as pbar:
                for i in range(0, len(images), batch_size):
                    batch_img_paths = images[i : i + batch_size]
                    batch_input_imgs = []
                    valid_img_paths = []

                    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                        results = list(executor.map(_process_image, batch_img_paths))

                    for input_img, img_path in results:
                        if input_img is None:
                            print(f"Warning: Could not read image {img_path}. Skipping.")
                        else:
                            batch_input_imgs.append(input_img)
                            valid_img_paths.append(img_path)

                    if not batch_input_imgs:
                        pbar.update(len(batch_img_paths))
                        continue

                    # Run batch inference with lowest confidence threshold using FP16 half precision on GPU
                    batch_results = model.predict(
                        batch_input_imgs,
                        conf=0.001,
                        imgsz=img_size,
                        verbose=False,
                        device=device,
                        half=True,  # FP16 Half precision for 2-4x speedup
                        batch=len(batch_input_imgs)  # Push tensor batch directly
                    )

                    for img_path, result in zip(valid_img_paths, batch_results):
                        if target_dir == input_path:
                            subfolder_name = "root"
                        else:
                            subfolder_name = str(img_path.parent.relative_to(input_path))

                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            class_name = CLASSES.get(cls_id, model.names[cls_id])
                            conf = float(box.conf[0])

                            # Apply class-specific optimal validation threshold
                            if conf >= THRESHOLDS.get(cls_id, 0.25):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                row_data = [
                                    str(img_path),
                                    img_path.name,
                                    subfolder_name,
                                    cls_id,
                                    class_name,
                                    f"{conf:.4f}",
                                    round(x1, 1),
                                    round(y1, 1),
                                    round(x2, 1),
                                    round(y2, 1),
                                ]
                                writer.writerow(row_data)
                                all_writer.writerow(row_data)
                                folder_detections.append(row_data)

                    pbar.update(len(batch_img_paths))

    return folder_detections


def main():
    MODEL_PATH = "/home/Joshua/Downloads/leopard_toad_identification/detection/active learning/rtdetr_clahe/runs/cycle_2_pretrained_phase2/weights/best.pt"
    OUTPUT_ROOT = "/home/Joshua/Downloads/leopard_toad_identification/detection/results/detect_rtdetr_cycle2_clahe_pretrained"
    IMG_SIZE = 640
    BATCH_SIZE = 128  # Safe and highly optimized batch size
    DEVICE = 0 if torch.cuda.is_available() else "cpu"

    print("\n=========================================")
    print(f"LOADING BEST MODEL: RT-DETR CLAHE (Cycle 2, Pretrained)")
    print(f"DEVICE: {DEVICE}")
    print(f"BATCH SIZE: {BATCH_SIZE}")
    print("=========================================")

    model = RTDETR(MODEL_PATH)

    years = {
        "2023": "/srv/shared_leopard_toad/2023",
        "2024": "/srv/shared_leopard_toad/2024",
        "2025": "/srv/shared_leopard_toad/2025/Documents",
    }

    # Consolidated combined predictions file path
    unified_csv_path = os.path.join(OUTPUT_ROOT, "all_unlabeled_predictions.csv")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with open(unified_csv_path, mode="w", newline="") as f_all:
        all_writer = csv.writer(f_all)
        headers = [
            "image_path",
            "image_name",
            "subfolder",
            "class_id",
            "class_name",
            "confidence",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ]
        all_writer.writerow(headers)

        grand_total_boxes = 0

        for year, base_input_dir in years.items():
            if not os.path.exists(base_input_dir):
                print(f"Year directory {base_input_dir} not found. Skipping.")
                continue

            # Gather all folders in the year directory
            folders = sorted([d.name for d in Path(base_input_dir).iterdir() if d.is_dir()])
            
            for folder in folders:
                in_dir = os.path.join(base_input_dir, folder)
                out_dir = os.path.join(OUTPUT_ROOT, year, folder)

                print(f"\n--> Starting on folder: {year} / {folder}")
                detections = process_folder(
                    in_dir,
                    out_dir,
                    model,
                    IMG_SIZE,
                    BATCH_SIZE,
                    DEVICE,
                    all_writer
                )
                grand_total_boxes += len(detections)

    print("\n=========================================")
    print("ALL BATCH INFERENCE COMPLETED!")
    print(f"Saved folder-specific CSVs in: {OUTPUT_ROOT}")
    print(f"Saved unified CSV at: {unified_csv_path}")
    print(f"Total detection boxes saved: {grand_total_boxes}")
    print("=========================================")


if __name__ == "__main__":
    main()
