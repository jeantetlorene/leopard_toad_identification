import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from config import RESULTS_DIR, CLASSES


def calculate_roc_auc_per_class():
    sweep_path = os.path.join(RESULTS_DIR, "per_class_threshold_sweep.csv")
    if not os.path.exists(sweep_path):
        print(f"Error: {sweep_path} not found.")
        return None

    df = pd.read_csv(sweep_path)

    auc_results = []

    # Group by model architecture and other variables
    groups = df.groupby(
        ["model", "processing", "cycle", "variant", "dataset", "class_id", "class_name"]
    )

    for name, group in groups:
        # Calculate FPR and sort by FPR to ensure monotonicity for AUC calculation
        group["fpr"] = 1 - group["specificity"]
        group = group.sort_values(["fpr", "recall"])

        tpr = group["recall"].values
        fpr = group["fpr"].values

        # Ensure it starts at 0,0 (infinitely high threshold)
        if fpr[0] > 0 or tpr[0] > 0:
            fpr = np.concatenate(([0.0], fpr))
            tpr = np.concatenate(([0.0], tpr))

        # Ensure it ends at 1,1 (infinitely low threshold)
        if fpr[-1] < 1.0 or tpr[-1] < 1.0:
            fpr = np.concatenate((fpr, [1.0]))
            tpr = np.concatenate((tpr, [1.0]))

        roc_auc = auc(fpr, tpr)

        auc_results.append(
            {
                "model": name[0],
                "processing": name[1],
                "cycle": name[2],
                "variant": name[3],
                "dataset": name[4],
                "class_id": name[5],
                "class_name": name[6],
                "roc_auc": roc_auc,
            }
        )

    auc_df = pd.DataFrame(auc_results)
    output_path = os.path.join(RESULTS_DIR, "per_class_roc_auc_summary.csv")
    auc_df.to_csv(output_path, index=False)
    print(f"Per-class ROC-AUC saved to {output_path}")
    return auc_df


def calculate_image_level_roc_auc():
    auc_results = []

    # Iterate through model folders
    for model_folder in os.listdir(RESULTS_DIR):
        folder_path = os.path.join(RESULTS_DIR, model_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith("_metrics.csv"):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)

                parts = filename.replace("_metrics.csv", "").split("_")
                cycle = int(parts[1])
                variant = parts[2]
                dataset = parts[3]

                folder_parts = model_folder.split("_")
                processing = folder_parts[-1]
                model_type = "_".join(folder_parts[:-1])

                # Sort by FPR
                df["fpr"] = 1 - df["specificity"]
                df = df.sort_values(["fpr", "recall"])

                tpr = df["recall"].values
                fpr = df["fpr"].values

                # Ensure it starts at 0,0
                if fpr[0] > 0 or tpr[0] > 0:
                    fpr = np.concatenate(([0.0], fpr))
                    tpr = np.concatenate(([0.0], tpr))

                # Ensure it ends at 1,1
                if fpr[-1] < 1.0 or tpr[-1] < 1.0:
                    fpr = np.concatenate((fpr, [1.0]))
                    tpr = np.concatenate((tpr, [1.0]))

                roc_auc = auc(fpr, tpr)

                auc_results.append(
                    {
                        "model": model_type,
                        "processing": processing,
                        "cycle": cycle,
                        "variant": variant,
                        "dataset": dataset,
                        "roc_auc": roc_auc,
                    }
                )

    auc_df = pd.DataFrame(auc_results)
    output_path = os.path.join(RESULTS_DIR, "image_level_roc_auc_summary.csv")
    auc_df.to_csv(output_path, index=False)
    print(f"Image-level ROC-AUC saved to {output_path}")
    return auc_df


def plot_best_roc_curves():
    sweep_path = os.path.join(RESULTS_DIR, "per_class_threshold_sweep.csv")
    if not os.path.exists(sweep_path):
        return

    df = pd.read_csv(sweep_path)

    # We want to compare models at Cycle 4 (last cycle)
    cycle_4_df = df[df["cycle"] == 4]

    for dataset in ["test", "val"]:
        for cls_id, cls_name in CLASSES.items():
            plot_df = cycle_4_df[
                (cycle_4_df["dataset"] == dataset) & (cycle_4_df["class_id"] == cls_id)
            ]
            if plot_df.empty:
                continue

            plt.figure(figsize=(10, 8))

            for (model, processing), group in plot_df.groupby(["model", "processing"]):
                # Calculate FPR and sort by FPR
                group["fpr"] = 1 - group["specificity"]
                group = group.sort_values(["fpr", "recall"])

                tpr = group["recall"].values
                fpr = group["fpr"].values

                # Ensure it starts at 0,0
                fpr = np.concatenate(([0.0], fpr))
                tpr = np.concatenate(([0.0], tpr))
                # Ensure it ends at 1,1
                fpr = np.concatenate((fpr, [1.0]))
                tpr = np.concatenate((tpr, [1.0]))

                roc_auc = auc(fpr, tpr)
                label = f"{model.upper()} ({processing}) - AUC: {roc_auc:.4f}"
                plt.plot(fpr, tpr, label=label, linewidth=2)

            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (1 - Specificity)")
            plt.ylabel("True Positive Rate (Recall)")
            plt.title(f"ROC Curves - {cls_name} (Cycle 4, {dataset.capitalize()} Set)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            plot_filename = f"roc_curves_{cls_name.lower()}_{dataset}_cycle4.png"
            plot_path = os.path.join(RESULTS_DIR, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"ROC curve plot saved to {plot_path}")


if __name__ == "__main__":
    print(">>> Calculating ROC-AUC for all models...")
    calculate_roc_auc_per_class()
    calculate_image_level_roc_auc()

    print("\n>>> Generating ROC Curve plots...")
    plot_best_roc_curves()

    print("\n>>> Done!")
