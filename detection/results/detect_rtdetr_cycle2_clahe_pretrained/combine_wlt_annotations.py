#!/usr/bin/env python3
import os
import pandas as pd
from PIL import Image


def extract_subfolder(path):
    path = str(path)
    # Year roots mapped in inference pipeline
    roots = {
        "2023": "/srv/shared_leopard_toad/2023",
        "2024": "/srv/shared_leopard_toad/2024",
        "2025": "/srv/shared_leopard_toad/2025/Documents",
    }

    for year, root in roots.items():
        if root in path:
            rel_to_root = os.path.relpath(path, root)
            parts = rel_to_root.split(os.sep)
            if len(parts) > 2:
                # The first part is the year target subfolder (e.g. "Cameras - AI Data", "02.09.2024")
                # Parts between that and the filename constitute the subfolder structure
                subfolder_parts = parts[1:-1]
                return "/".join(subfolder_parts)
            else:
                return "root"
    return os.path.basename(os.path.dirname(path))


def main():
    base_dir = "/home/Joshua/Downloads/leopard_toad_identification"
    results_dir = os.path.join(
        base_dir, "detection", "results", "detect_rtdetr_cycle2_clahe_pretrained"
    )
    eval_data_dir = os.path.join(base_dir, "detection", "evaluation", "data")

    preds_csv = os.path.join(results_dir, "wlt_predictions_gte_0.7_filtered.csv")
    output_csv = os.path.join(results_dir, "wlt_predictions.csv")
    evals_csv = os.path.join(
        results_dir, "wlt_predictions_gte_0.7_filtered_evaluations.csv"
    )
    mapping_csv = os.path.join(eval_data_dir, "image_mapping.csv")

    print("Loading predictions and evaluations...")
    df_preds = pd.read_csv(preds_csv)
    df_evals = pd.read_csv(evals_csv)

    # 2. Filter correct predictions based on evaluation
    print("Filtering correct predictions...")
    df_correct_evals = df_evals[df_evals["evaluation"] == "Correct"]
    correct_row_idxs = df_correct_evals["row_idx"].tolist()

    df_correct_preds = df_preds.iloc[correct_row_idxs].copy()
    print(
        f"Loaded {len(df_correct_preds)} correct predictions out of {len(df_preds)} total predictions."
    )

    # 3. Load image mapping to translate unique names to real full paths
    print("Loading image mapping...")
    df_map = pd.read_csv(mapping_csv)

    # Map unique name (without extension) -> row dictionary
    path_map = {}
    for idx, row in df_map.iterrows():
        uname = str(row["unique_name"])
        base_name = os.path.splitext(uname)[0]
        path_map[base_name] = {
            "original_path": row["original_path"],
            "split": row["split"],
            "unique_name": row["unique_name"],
        }

    # 4. Extract WLT annotations from val and test splits
    print("Extracting WLT annotations from test and val sets...")
    annotation_rows = []

    for split in ["val", "test"]:
        labels_dir = os.path.join(eval_data_dir, split, "labels")
        images_dir = os.path.join(eval_data_dir, split, "images")

        if not os.path.exists(labels_dir):
            print(f"Skipping {split} labels directory (not found).")
            continue

        for fname in os.listdir(labels_dir):
            if fname.endswith(".txt") and fname != "classes.txt":
                base_name = os.path.splitext(fname)[0]
                if base_name not in path_map:
                    print(
                        f"Warning: Label file {fname} not found in image_mapping.csv! Skipping."
                    )
                    continue

                map_info = path_map[base_name]
                orig_path = map_info["original_path"]
                unique_name = map_info["unique_name"]

                local_img_path = os.path.join(images_dir, unique_name)
                if not os.path.exists(local_img_path):
                    print(
                        f"Warning: Image file {unique_name} not found in {images_dir}! Skipping annotations."
                    )
                    continue

                # Get image size (width, height)
                try:
                    with Image.open(local_img_path) as img:
                        img_width, img_height = img.size
                except Exception as e:
                    print(f"Error opening image {local_img_path}: {e}. Skipping.")
                    continue

                # Parse annotations
                label_file = os.path.join(labels_dir, fname)
                with open(label_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        cls_id = int(parts[0])
                        # We only want WLT annotations (class index 2)
                        if cls_id == 2:
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])

                            # Convert from normalized coordinates to pixel coordinates
                            xmin = round((x_center - w / 2) * img_width, 1)
                            ymin = round((y_center - h / 2) * img_height, 1)
                            xmax = round((x_center + w / 2) * img_width, 1)
                            ymax = round((y_center + h / 2) * img_height, 1)

                            image_name = os.path.basename(orig_path)
                            subfolder = extract_subfolder(orig_path)

                            annotation_rows.append(
                                {
                                    "image_path": orig_path,
                                    "image_name": image_name,
                                    "subfolder": subfolder,
                                    "class_id": 2,
                                    "class_name": "Western_Leopard_Toad",
                                    "confidence": 1.0,  # Ground truth annotation has 100% confidence
                                    "xmin": xmin,
                                    "ymin": ymin,
                                    "xmax": xmax,
                                    "ymax": ymax,
                                }
                            )

    df_annotations = pd.DataFrame(annotation_rows)
    print(f"Extracted {len(df_annotations)} WLT annotations from val and test splits.")

    # 5. Combine correct predictions and WLT annotations
    print("Combining datasets...")
    df_combined = pd.concat([df_correct_preds, df_annotations], ignore_index=True)

    # 6. Save combined dataset to final output CSV
    print(f"Saving final combined CSV to {output_csv}...")
    df_combined.to_csv(output_csv, index=False)

    print("\n=======================================================")
    print("COMBINATION COMPLETED SUCCESSFULLY")
    print(f"  Correct predictions:    {len(df_correct_preds)}")
    print(f"  Test & Val annotations: {len(df_annotations)}")
    print(f"  Total combined rows:    {len(df_combined)}")
    print(f"  Output CSV saved:       {output_csv}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
