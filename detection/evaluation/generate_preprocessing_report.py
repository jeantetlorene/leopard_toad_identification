import pandas as pd
import os
import numpy as np

from config import FILES_DIR, PLOTS_DIR

UNIFIED_CSV = os.path.join(FILES_DIR, "unified_model_evaluation.csv")


def generate_report():
    if not os.path.exists(UNIFIED_CSV):
        print(
            "Error: Evaluation CSV not found. Please run evaluate_all_models.py first."
        )
        return

    # 1. Load Data
    df_unified = pd.read_csv(UNIFIED_CSV)

    # Filter for Cycle 0 and Test set
    df_unified = df_unified[
        (df_unified["cycle"] == 0) & (df_unified["dataset"] == "test")
    ]

    # 3. Get metrics
    df_final = df_unified[["model", "processing", "variant", "mAP", "mAR"]].copy()

    # 4. Generate Table (Markdown)
    report_path = os.path.join(FILES_DIR, "preprocessing_results.md")
    with open(report_path, "w") as f:
        f.write("# Results: Effect of Preprocessing (Cycle 0)\n\n")
        f.write(
            "This report summarizes the impact of CLAHE on Cycle 0 model performance.\n\n"
        )
        f.write("### Comparative Performance Table (Cycle 0, Test Set)\n")
        f.write("| Architecture | Variant | Processing | mAP | mean Average Recall |\n")
        f.write("|--------------|---------|------------|-----|---------------------|\n")

        # Sort for consistent display
        df_final = df_final.sort_values(["model", "variant", "processing"])

        for _, row in df_final.iterrows():
            f.write(
                f"| {row['model'].upper()} | {row['variant'].capitalize()} | {row['processing'].capitalize()} | {row['mAP']:.4f} | {row['mAR']:.4f} |\n"
            )

        f.write("\n### Precision-Recall Visualizations\n\n")
        for arch in df_final["model"].unique():
            f.write(f"#### {arch.upper()}\n")
            f.write(
                f"![{arch.upper()} PR Curve](../plots/pr_curve_{arch}_cycle0.png)\n\n"
            )

    print(f"Report saved to: {report_path}")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    generate_report()
