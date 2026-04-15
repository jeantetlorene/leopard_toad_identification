import os
import cv2
import csv
from pathlib import Path
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
from tqdm import tqdm
import concurrent.futures


def apply_clahe_preprocessing(image_rgb):
    """
    Applies CLAHE preprocessing.
    Input: RGB Numpy array
    Output: RGB Numpy array with CLAHE applied
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final_img


def _process_image(img_path):
    """Reads an image, converts to RGB, and applies CLAHE."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, img_path
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_img = apply_clahe_preprocessing(img_rgb)
    return input_img, img_path


def process_folder(
    input_folder,
    output_folder,
    model,
    conf_threshold,
    img_size,
    batch_size,
    device,
    model_type="yolo",
):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"Directory {input_path} does not exist. Skipping.")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    # Find immediate subdirectories and the root folder
    immediate_subfolders = [d for d in input_path.iterdir() if d.is_dir()]
    targets = [input_path] + immediate_subfolders

    total_images_processed = 0

    for target_dir in targets:
        # If it's the root folder, only look for images directly inside it
        if target_dir == input_path:
            images = [
                f
                for f in target_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            csv_name = f"{input_path.name}_root.csv"
        else:
            # If it's a subfolder, gather ALL images within it, including any subsubfolders
            images = [
                f
                for f in target_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            csv_name = f"{target_dir.name}.csv"

        if not images:
            continue

        csv_path = output_path / csv_name
        print(
            f"Found {len(images)} images in '{target_dir.name}'. Saving predictions to {csv_name}..."
        )

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
            )

            with tqdm(total=len(images), desc=f"Processing {target_dir.name}") as pbar:
                for i in range(0, len(images), batch_size):
                    batch_img_paths = images[i : i + batch_size]
                    batch_input_imgs = []
                    valid_img_paths = []

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = list(executor.map(_process_image, batch_img_paths))

                    for input_img, img_path in results:
                        if input_img is None:
                            print(
                                f"Warning: Could not read image {img_path}. Skipping."
                            )
                        else:
                            batch_input_imgs.append(input_img)
                            valid_img_paths.append(img_path)

                    if not batch_input_imgs:
                        pbar.update(len(batch_img_paths))
                        continue

                    # Run batch inference
                    if model_type == "yolo":
                        batch_results = model.predict(
                            batch_input_imgs,
                            conf=conf_threshold,
                            imgsz=img_size,
                            verbose=False,
                            device=device,
                        )
                    elif model_type == "faster_rcnn":
                        batch_tensors = [
                            TF.to_tensor(img).to(device) for img in batch_input_imgs
                        ]
                        with torch.no_grad():
                            batch_results = model(batch_tensors)

                    for img_path, result in zip(valid_img_paths, batch_results):
                        if target_dir == input_path:
                            subfolder_name = "root"
                        else:
                            # Keep full relative path for the 'subfolder' column
                            subfolder_name = str(
                                img_path.parent.relative_to(input_path)
                            )

                        # Write all detections
                        if model_type == "yolo":
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                class_name = model.names[cls_id]
                                conf = float(box.conf[0])
                                x1, y1, x2, y2 = box.xyxy[0].tolist()

                                writer.writerow(
                                    [
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
                                )
                        elif model_type == "faster_rcnn":
                            faster_rcnn_names = {
                                1: "Other_Amphibian",
                                2: "Small_Mammal",
                                3: "Western_Leopard_Toad",
                            }
                            pred_boxes = result["boxes"].cpu().numpy()
                            pred_scores = result["scores"].cpu().numpy()
                            pred_labels = result["labels"].cpu().numpy()

                            for k in range(len(pred_scores)):
                                conf = float(pred_scores[k])
                                if conf >= conf_threshold:
                                    cls_id = int(pred_labels[k])
                                    class_name = faster_rcnn_names.get(
                                        cls_id, f"Unknown_{cls_id}"
                                    )
                                    x1, y1, x2, y2 = pred_boxes[k].tolist()

                                    writer.writerow(
                                        [
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
                                    )
                        total_images_processed += 1
                    pbar.update(len(batch_img_paths))
    return total_images_processed


def main():
    CONF_THRESHOLD = 0.01
    IMG_SIZE = 640
    BATCH_SIZE = 32
    DEVICE = 0

    folders = [
        "01.09.09.09.2025",
        "06.10-13.10.2025",
        "15.09-22.09.2025",
        "25.08-01.09.2025",
        "04.08.2025",
        "09.09.15.09.2025",
        "21-28.08.2025",
        "29.09.06.10.2025",
        "05.08.2025",
        "12.19.08.2025",
        "22.09-29.09.2025",
        "Cameras - AI Data",
        "Cameras-Master Data",
        "02.09.2024",
        "09.02.2024",
        "16.09.2024",
        "19.09.2024",
        "26.09.2024",
        "02.10.2024",
        "11.10.2024",
        "18.11.2024",
        "23.08.2024",
        "28.08.2024",
    ]

    models_config = [
        {
            "name": "faster_rcnn",
            "type": "faster_rcnn",
            "path": "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/faster_rcnn_finetune/weights/best_partial.pt",
            "output_prefix": "detect_faster_rcnn",
        },
        {
            "name": "rtdetr",
            "type": "yolo",
            "path": "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/rtdetr_finetune/subset_finetune/weights/best.pt",
            "output_prefix": "detect_rtdetr",
        },
        {
            "name": "yolo",
            "type": "yolo",
            "path": "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/yolo_finetune/subset_finetune/weights/best.pt",
            "output_prefix": "detect_yolo",
        },
    ]

    years = {
        "2023": "/srv/shared_leopard_toad/2023",
        "2024": "/srv/shared_leopard_toad/2024",
        "2025": "/srv/shared_leopard_toad/2025/Documents",
    }

    grand_total = 0

    for model_info in models_config:
        print(f"\n=========================================")
        print(f"LOADING MODEL: {model_info['name']}")
        print(f"=========================================")

        m_type = model_info["type"]
        m_path = model_info["path"]

        if m_type == "yolo":
            model = YOLO(m_path)
            eval_device = DEVICE
        elif m_type == "faster_rcnn":
            num_classes = 4  # Background + 3 classes
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
            state_dict = torch.load(m_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)

            eval_device = torch.device(
                f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu"
            )
            model.to(eval_device)
            model.eval()

        for year, base_input_dir in years.items():
            base_output_dir = f"/home/Joshua/Downloads/leopard_toad_identification/detection/results/{model_info['output_prefix']}/{year}"

            for folder in folders:
                in_dir = os.path.join(base_input_dir, folder)

                if not os.path.exists(in_dir):
                    continue

                out_dir = os.path.join(base_output_dir, folder)

                # Check if the output directory exists and already has contents (e.g. prediction CSVs)
                if (
                    os.path.exists(out_dir)
                    and os.path.isdir(out_dir)
                    and len(os.listdir(out_dir)) > 0
                ):
                    print(
                        f"\n--> Skipping folder (already processed): {year} / {folder} with model {model_info['name']}"
                    )
                    continue

                print(
                    f"\n--> Starting on folder: {year} / {folder} with model {model_info['name']}"
                )
                processed = process_folder(
                    in_dir,
                    out_dir,
                    model,
                    CONF_THRESHOLD,
                    IMG_SIZE,
                    BATCH_SIZE,
                    eval_device,
                    m_type,
                )
                grand_total += processed

        # Free memory of loaded model before proceeding to next model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nALL DONE! Processed {grand_total} total images across all models/years.")


if __name__ == "__main__":
    main()
