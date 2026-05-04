import os
import orjson
import pandas as pd
from tqdm import tqdm
from metrics import calculate_image_level_metrics, calculate_detection_metrics
from config import RESULTS_DIR, CONF_THRESHOLDS


def refresh_image_level_metrics():
    summary_rows = []
    folders = sorted(
        [
            f
            for f in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, f))
        ]
    )

    for model_folder in folders:
        folder_path = os.path.join(RESULTS_DIR, model_folder)
        filenames = sorted(
            [f for f in os.listdir(folder_path) if f.endswith("_raw.json")]
        )
        if not filenames:
            continue

        print(f"\n>>> Refreshing image-level metrics for {model_folder}...")

        for filename in tqdm(filenames, desc=f"Models in {model_folder}"):
            parts = filename.replace("_raw.json", "").split("_")
            cycle = int(parts[1])
            variant = parts[2]
            dataset = parts[3]

            folder_parts = model_folder.split("_")
            processing = folder_parts[-1]
            model_type = "_".join(folder_parts[:-1])

            with open(os.path.join(folder_path, filename), "rb") as f:
                results = orjson.loads(f.read())

            image_metrics = calculate_image_level_metrics(results, CONF_THRESHOLDS)
            mAP = calculate_detection_metrics(results)["mAP"]

            # Add mAP to metrics for CSV
            for m in image_metrics:
                m["mAP"] = mAP

            metrics_file = os.path.join(
                folder_path, filename.replace("_raw.json", "_metrics.csv")
            )
            metrics_df = pd.DataFrame(image_metrics)
            metrics_df.to_csv(metrics_file, index=False)

            # Summary entry (at 0.1 threshold)
            idx = (metrics_df["threshold"] - 0.1).abs().idxmin()
            summary_rows.append(
                {
                    "model": model_type,
                    "processing": processing,
                    "cycle": cycle,
                    "variant": variant,
                    "dataset": dataset,
                    "recall_0.1": metrics_df.loc[idx, "recall"],
                    "specificity_0.1": metrics_df.loc[idx, "specificity"],
                    "precision_0.1": metrics_df.loc[idx, "precision"],
                    "f1_0.1": metrics_df.loc[idx, "f1_score"],
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(RESULTS_DIR, "all_models_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAll models summary saved to {summary_path}")


if __name__ == "__main__":
    refresh_image_level_metrics()
