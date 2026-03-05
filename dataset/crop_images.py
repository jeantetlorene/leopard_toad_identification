"""
Crop images from a dataset based on YOLO labels.
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
IMAGES_DIR = "leopard_toad_identification/dataset/reid/images" 
LABELS_DIR = "leopard_toad_identification/dataset/labels"
OUTPUT_DIR = "leopard_toad_identification/dataset/dataset_reid_crops"

def yolo_to_pixels(yolo_coords, img_w, img_h):
    """
    Convert YOLO normalized coordinates to pixel coordinates.
    yolo_coords: [class_id, x_center, y_center, width, height]
    """
    # Parse values
    _, x_c, y_c, w, h = map(float, yolo_coords)
    
    # Calculate corners
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    
    # Clamp to image boundaries (prevent negative or out-of-bounds)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    
    return x1, y1, x2, y2

def generate_reid_dataset():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of images
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [f for f in os.listdir(IMAGES_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Found {len(image_files)} images. Starting cropping...")
    
    crops_created = 0
    
    for img_filename in tqdm(image_files):
        img_path = os.path.join(IMAGES_DIR, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)
        
        # Check if label exists
        if not os.path.exists(label_path):
            # Try checking if Label Studio messed with the filename (common issue)
            # If your labels have different names, we might need the "fuzzy matcher" again.
            # For now, assuming standard export structure.
            continue
            
        # Read Image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_h, img_w = img.shape[:2]
        
        # Read Labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Process each box in the file
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # Get coordinates
            x1, y1, x2, y2 = yolo_to_pixels(parts, img_w, img_h)
            
            # Crop
            crop = img[y1:y2, x1:x2]
            
            # Skip empty crops (e.g. if box was 0 size)
            if crop.size == 0: continue
            
            # Save Crop
            # Naming convention: originalName_cropIndex.jpg
            save_name = f"{os.path.splitext(img_filename)[0]}_crop{i}.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            cv2.imwrite(save_path, crop)
            crops_created += 1

    print("Processing Complete!")
    print(f"Total Toads Cropped: {crops_created}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_reid_dataset()