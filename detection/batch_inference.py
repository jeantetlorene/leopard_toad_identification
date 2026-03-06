import os
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

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

def main():
    # --- CONFIGURATION VARIABLES ---
    MODEL_PATH = "/home/Joshua/Downloads/leopard_toad_identification/detection/runs/detect/Western_Leopard_Toad_Project/yolov8n_clahe_run2/weights/best.pt"
    INPUT_FOLDER = "/srv/shared_leopard_toad/2023/Cameras - AI Data"
    OUTPUT_FOLDER = "/home/Joshua/Downloads/leopard_toad_identification/detection/results/detect_2/2023"
    CONF_THRESHOLD = 0.25
    IMG_SIZE = 1280
    BATCH_SIZE = 16  # Adjust based on available RAM/VRAM
    DEVICE = 0       # Use 0 for GPU, or 'cpu' for CPU
    # -------------------------------

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    # Find immediate subdirectories and the root folder
    immediate_subfolders = [d for d in input_path.iterdir() if d.is_dir()]
    targets = [input_path] + immediate_subfolders
    
    total_images_processed = 0
    
    for target_dir in targets:
        # If it's the root folder, only look for images directly inside it
        if target_dir == input_path:
            images = [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
            csv_name = f"{input_path.name}_root.csv"
        else:
            # If it's a subfolder, gather ALL images within it, including any subsubfolders
            images = [f for f in target_dir.rglob('*') if f.is_file() and f.suffix.lower() in image_extensions]
            csv_name = f"{target_dir.name}.csv"
        
        if not images:
            continue
            
        csv_path = output_path / csv_name
        
        print(f"Found {len(images)} images in '{target_dir.name}'. Saving predictions to {csv_name}...")
        
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "image_name", "subfolder", "class_id", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"])
            
            with tqdm(total=len(images), desc=f"Processing {target_dir.name}") as pbar:
                for i in range(0, len(images), BATCH_SIZE):
                    batch_img_paths = images[i:i + BATCH_SIZE]
                    batch_input_imgs = []
                    valid_img_paths = []
                    
                    for img_path in batch_img_paths:
                        # Read image
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is None:
                            print(f"Warning: Could not read image {img_path}. Skipping.")
                            continue
                            
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Apply CLAHE preprocessing
                        input_img = apply_clahe_preprocessing(img_rgb)
                        batch_input_imgs.append(input_img)
                        valid_img_paths.append(img_path)
                        
                    if not batch_input_imgs:
                        pbar.update(len(batch_img_paths))
                        continue
                        
                    # Run batch inference
                    batch_results = model.predict(batch_input_imgs, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False, device=DEVICE)
                    
                    for img_path, result in zip(valid_img_paths, batch_results):
                        if target_dir == input_path:
                            subfolder_name = "root"
                        else:
                            # Keep full relative path for the 'subfolder' column so you know exactly where it came from
                            subfolder_name = str(img_path.parent.relative_to(input_path))

                        # Write all detections
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            class_name = model.names[cls_id]
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            writer.writerow([
                                str(img_path), 
                                img_path.name,
                                subfolder_name,
                                cls_id, 
                                class_name, 
                                f"{conf:.4f}", 
                                round(x1, 1), 
                                round(y1, 1), 
                                round(x2, 1), 
                                round(y2, 1)
                            ])
                        total_images_processed += 1
                    pbar.update(len(batch_img_paths))
                
    print(f"\nDone! Processed {total_images_processed} images total. Results saved in {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
