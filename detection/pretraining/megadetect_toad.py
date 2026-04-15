import argparse
import os
from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils

def get_all_images(folder_path):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image_paths.append(os.path.abspath(os.path.join(root, file)))
    return image_paths

def main():
    parser = argparse.ArgumentParser(description="Run MegaDetector on multiple folders")
    parser.add_argument("folders", nargs="+", help="One or more folders containing images")
    parser.add_argument("--output", default="/home/Joshua/Downloads/leopard_toad_identification/detection/pretraining/output1.json", help="Output JSON file path")
    
    args = parser.parse_args()
    
    all_image_file_names = []
    for folder in args.folders:
        folder_abs = os.path.abspath(folder)
        if os.path.isdir(folder_abs):
            print(f"Finding images in {folder_abs}...")
            images = get_all_images(folder_abs)
            all_image_file_names.extend(images)
            print(f"Found {len(images)} images in {folder_abs}.")
        else:
            print(f"Warning: {folder} is not a valid directory, skipping.")
            
    if not all_image_file_names:
        print("No images found in the provided folders.")
        return

    print(f"Running inference on a total of {len(all_image_file_names)} images...")
    
    # Run inference
    results = load_and_run_detector_batch('MDV5A', all_image_file_names)
    
    # Write results to file
    print(f"Saving results to {args.output}...")
    write_results_to_file(results, args.output, detector_file='MDV5A')
    print("Done!")

if __name__ == "__main__":
    main()