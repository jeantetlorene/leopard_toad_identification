import os
import cv2
import csv
from pathlib import Path
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from ultralytics import YOLO
from tqdm import tqdm
import concurrent.futures


def _process_image(img_path):
    try:
        img_bgr = cv2.imread(str(img_path))
        return img_bgr, img_path
    except Exception:
        return None, img_path


def get_camera_images(camera_name):
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = []

    YEARS = {
        "2023": "/srv/shared_leopard_toad/2023",
        "2024": "/srv/shared_leopard_toad/2024",
        "2025": "/srv/shared_leopard_toad/2025/Documents",
    }

    FOLDERS = [
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

    for year, base_dir in YEARS.items():
        for folder in FOLDERS:
            in_dir = Path(base_dir) / folder
            if not in_dir.exists():
                continue
            for img_path in in_dir.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    if f"/{camera_name}/" in str(img_path):
                        images.append(img_path)
    return images


def process_image_list(
    images,
    csv_path,
    model,
    conf_threshold,
    img_size,
    batch_size,
    device,
    model_type="yolo",
):
    if not images:
        return 0

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images. Saving predictions to {csv_path.name}...")

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

        total_images_processed = 0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
        with tqdm(total=len(images), desc=f"Processing {csv_path.name}") as pbar:
            for i in range(0, len(images), batch_size):
                batch_img_paths = images[i : i + batch_size]
                batch_input_imgs = []
                valid_img_paths = []

                results = list(executor.map(_process_image, batch_img_paths))

                for input_img, img_path in results:
                    if input_img is None:
                        pass
                    else:
                        batch_input_imgs.append(input_img)
                        valid_img_paths.append(img_path)

                if not batch_input_imgs:
                    pbar.update(len(batch_img_paths))
                    continue

                if model_type == "yolo":
                    batch_results = model.predict(
                        batch_input_imgs,
                        conf=conf_threshold,
                        imgsz=img_size,
                        verbose=False,
                        device=device,
                        half=True,
                        batch=len(batch_input_imgs),
                    )
                elif model_type == "faster_rcnn":
                    batch_tensors = []
                    for img in batch_input_imgs:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        batch_tensors.append(TF.to_tensor(img_rgb).to(device))
                    with torch.no_grad():
                        batch_results = model(batch_tensors)

                for img_path, result in zip(valid_img_paths, batch_results):
                    # For subfolder we use the parent folder name
                    subfolder_name = img_path.parent.name

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

                # Cleanup batch memory to prevent OOM
                del batch_input_imgs
                if model_type == "faster_rcnn":
                    del batch_tensors
                del batch_results

                total_images_processed += len(valid_img_paths)
                pbar.update(len(batch_img_paths))
        executor.shutdown(wait=False)
    return total_images_processed


def main():
    CONF_THRESHOLD = 0.1
    IMG_SIZE = 640
    DEVICE = 0

    base_dir = (
        "/home/Joshua/Downloads/leopard_toad_identification/detection/active learning"
    )

    models_config = [
        # Faster R-CNN
        {
            "name": "faster_rcnn_cycle_4_scratch_scratch",
            "type": "faster_rcnn",
            "path": f"{base_dir}/faster_rcnn/runs/cycle_4_scratch_scratch/weights/best.pt",
        },
        {
            "name": "faster_rcnn_cycle_4_pretrained_phase1",
            "type": "faster_rcnn",
            "path": f"{base_dir}/faster_rcnn/runs/cycle_4_pretrained_phase1/weights/best.pt",
        },
        {
            "name": "faster_rcnn_cycle_4_pretrained_phase2",
            "type": "faster_rcnn",
            "path": f"{base_dir}/faster_rcnn/runs/cycle_4_pretrained_phase2/weights/best.pt",
        },
        # YOLO
        {
            "name": "yolo_cycle_4_scratch_scratch",
            "type": "yolo",
            "path": f"{base_dir}/yolo/runs/cycle_4_scratch_scratch/weights/best.pt",
        },
        {
            "name": "yolo_cycle_4_pretrained_phase1",
            "type": "yolo",
            "path": f"{base_dir}/yolo/runs/cycle_4_pretrained_phase1/weights/best.pt",
        },
        {
            "name": "yolo_cycle_4_pretrained_phase2",
            "type": "yolo",
            "path": f"{base_dir}/yolo/runs/cycle_4_pretrained_phase2/weights/best.pt",
        },
        # RT-DETR
        {
            "name": "rtdetr_cycle_4_scratch_scratch",
            "type": "yolo",
            "path": f"{base_dir}/rtdetr/runs/cycle_4_scratch_scratch/weights/best.pt",
        },
        {
            "name": "rtdetr_cycle_4_pretrained_phase1",
            "type": "yolo",
            "path": f"{base_dir}/rtdetr/runs/cycle_4_pretrained_phase1/weights/best.pt",
        },
        {
            "name": "rtdetr_cycle_4_pretrained_phase2",
            "type": "yolo",
            "path": f"{base_dir}/rtdetr/runs/cycle_4_pretrained_phase2/weights/best.pt",
        },
    ]

    print("Scanning for val (4R) and test (5Z) images across all years...")
    val_images = get_camera_images("4R")
    test_images = get_camera_images("5Z")
    print(
        f"Found {len(val_images)} val images (4R) and {len(test_images)} test images (5Z)."
    )

    grand_total = 0

    for model_info in models_config:
        print(f"\n=========================================")
        print(f"LOADING MODEL: {model_info['name']}")
        print(f"=========================================")

        m_type = model_info["type"]
        m_path = model_info["path"]

        if not os.path.exists(m_path):
            print(f"Model path does not exist: {m_path}")
            continue

        if m_type == "yolo":
            model = YOLO(m_path)
            eval_device = DEVICE
            current_batch_size = 256
        elif m_type == "faster_rcnn":
            num_classes = 4  # Background + 3 classes
            model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
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
            current_batch_size = 32

        output_base_dir = f"/home/Joshua/Downloads/leopard_toad_identification/evaluation/{model_info['name']}"

        # Process Val (4R)
        processed = process_image_list(
            images=val_images,
            csv_path=f"{output_base_dir}/val.csv",
            model=model,
            conf_threshold=CONF_THRESHOLD,
            img_size=IMG_SIZE,
            batch_size=current_batch_size,
            device=eval_device,
            model_type=m_type,
        )
        grand_total += processed

        # Process Test (5Z)
        processed = process_image_list(
            images=test_images,
            csv_path=f"{output_base_dir}/test.csv",
            model=model,
            conf_threshold=CONF_THRESHOLD,
            img_size=IMG_SIZE,
            batch_size=current_batch_size,
            device=eval_device,
            model_type=m_type,
        )
        grand_total += processed

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nALL DONE! Processed {grand_total} total inferences.")


if __name__ == "__main__":
    main()
