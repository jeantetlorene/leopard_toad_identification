import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import RESULTS_DIR, FILES_DIR, PLOTS_DIR, CONF_THRESHOLDS
from eval_utils.metrics import calculate_image_level_metrics
from eval_utils.data_utils import load_predictions_from_json


def main():
    all_binary_sweep = []
    plot_data = []

    # Find all model results folders
    folders = sorted(
        [
            f
            for f in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, f))
        ]
    )

    for model_folder in folders:
        folder_path = os.path.join(RESULTS_DIR, model_folder)
        # ONLY look for test_full_seq raw predictions (the test unlabeled pool)
        filenames = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if f.endswith("_test_full_seq_raw.json")
            ]
        )

        for filename in filenames:
            parts = filename.replace("_raw.json", "").split("_")
            cycle = int(parts[1])
            variant = parts[2]
            dataset = "test_full_seq"

            mf_parts = model_folder.split("_")
            processing = mf_parts[-1]
            model_type = "_".join(mf_parts[:-1])

            print(
                f"Processing {model_type} ({processing}) | Variant: {variant} | Cycle: {cycle}"
            )
            filepath = os.path.join(folder_path, filename)
            results = load_predictions_from_json(filepath, is_full_seq=True)

            if not results:
                print(f"Warning: No images from {filename} found. Skipping.")
                continue

            binary_gt = np.array([res["is_positive"] for res in results])
            binary_scores = np.array(
                [
                    max([p["conf"] for p in res["predictions"]] + [0.0])
                    for res in results
                ]
            )

            # Calculate Sweep
            binary_sweep_data = calculate_image_level_metrics(results, CONF_THRESHOLDS)
            for entry in binary_sweep_data:
                entry.update(
                    {
                        "model": model_type,
                        "processing": processing,
                        "cycle": cycle,
                        "variant": variant,
                        "dataset": dataset,
                    }
                )
                all_binary_sweep.append(entry)

            # Calculate ROC-AUC
            if len(np.unique(binary_gt)) > 1:
                binary_fpr, binary_tpr, _ = roc_curve(binary_gt, binary_scores)
                binary_auc = auc(binary_fpr, binary_tpr)
            else:
                binary_fpr, binary_tpr = None, None
                binary_auc = np.nan

            if binary_fpr is not None:
                plot_data.append(
                    {
                        "model": model_type,
                        "processing": processing,
                        "variant": variant,
                        "cycle": cycle,
                        "fpr": binary_fpr,
                        "tpr": binary_tpr,
                        "auc": binary_auc,
                    }
                )

    # Save CSV
    os.makedirs(FILES_DIR, exist_ok=True)
    df_sweep = pd.DataFrame(all_binary_sweep)
    if not df_sweep.empty:
        sweep_path = os.path.join(FILES_DIR, "binary_threshold_sweep_test_pool.csv")
        df_sweep.to_csv(sweep_path, index=False)
        print(f"\nSaved CSV to {sweep_path}")
    else:
        print("\nNo data to save for threshold sweep.")

    # Generate Plots
    if not plot_data:
        print("No plot data found.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    cycles = sorted(list(set(d["cycle"] for d in plot_data)))

    for c in cycles:
        plt.figure(figsize=(10, 8))
        subset = [d for d in plot_data if d["cycle"] == c]
        for d in subset:
            label = f"{d['model'].upper()} {d['variant'].capitalize()} ({d['processing']}): {d['auc']:.4f}"
            plt.plot(d["fpr"], d["tpr"], label=label, linewidth=2)

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"Image-Level ROC Curves - Test Unlabeled Pool (Cycle {c})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.gca().set_aspect("equal")

        plot_path = os.path.join(PLOTS_DIR, f"binary_roc_test_pool_cycle_{c}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
