import os
import pandas as pd
import numpy as np


def main():
    print("Generating Comprehensive WLT-Specific Image-Level Evaluation Report...")

    # Paths
    base_dir = "/home/Joshua/Downloads/leopard_toad_identification/detection/evaluation"
    files_dir = os.path.join(base_dir, "results", "files")
    sweep_csv = os.path.join(files_dir, "wlt_binary_threshold_sweep_test_pool.csv")
    report_md = os.path.join(files_dir, "wlt_image_level_results.md")

    if not os.path.exists(sweep_csv):
        print(
            f"Error: Sweep CSV not found at {sweep_csv}. Please run the evaluation script first."
        )
        # Create a placeholder report explaining that the sweep needs to be run first
        os.makedirs(files_dir, exist_ok=True)
        with open(report_md, "w") as f:
            f.write(
                "# Results: WLT-Specific Image-Level Binary Classification & Labor Reduction\n\n"
            )
            f.write("> [!WARNING]\n")
            f.write(
                "> The threshold sweep data was not yet generated. Please execute the following command to populate this report:\n"
            )
            f.write("> ```bash\n")
            f.write(
                "> /home/Joshua/Downloads/leopard_toad_identification/.venv/bin/python3 pipelines/binary_eval_test_pool_wlt.py\n"
            )
            f.write("> ```\n\n")
        print(f"Created placeholder report at {report_md}")
        return

    df = pd.read_csv(sweep_csv)
    if df.empty:
        print("Sweep CSV is empty.")
        return

    # Columns of interest
    # model,processing,cycle,variant,dataset,threshold,tp,fp,tn,fn,recall,specificity,precision,f1_score,labor_reduction,auc

    # Compile the final report
    with open(report_md, "w") as f:
        f.write(
            "# Results: WLT-Specific Image-Level Binary Classification & Labor Reduction\n\n"
        )
        f.write(
            "This report documents the image-level binary filtering performance and manual annotation labor "
            "reduction achieved by custom models focusing strictly on the **Western Leopard Toad (WLT)** class "
            "on the unlabelled test pool (147,352 frames).\n\n"
        )

        f.write(
            "For each combination, we evaluate performance at three separate operating points:\n"
            "1.  **Optimal $F_1$-Score Operating Point**: Maximizes the geometric mean of Precision and Recall.\n"
            "2.  **High-Recall Safety Operating Point (Target $\\ge 95\\%$)**: Restricts the search space to "
            "configurations that guarantee at least $95\\%$ target recall, and maximizes specificity.\n"
            "3.  **Moderate High-Recall Operating Point (Target $\\ge 85\\%$)**: Restricts the search space to "
            "configurations that guarantee at least $85\\%$ target recall, and maximizes specificity.\n\n"
        )

        # Get unique cycles
        cycles = sorted(df["cycle"].unique())

        for cycle in cycles:
            f.write(f"## Active Learning Cycle {cycle}\n\n")
            f.write(
                "| Model | Process | Variant | Area Under ROC (AUC) | Metric Focus | Conf. Thresh | TP | FP | TN | FN | Recall | Specificity | Precision | F1-Score | Labor Saved |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

            df_cycle = df[df["cycle"] == cycle]

            # Group by model, processing, variant
            groups = df_cycle.groupby(["model", "processing", "variant"])

            for (model, processing, variant), df_group in sorted(
                groups, key=lambda x: (x[0][0], x[0][1], x[0][2])
            ):
                # Extract AUC (constant within group)
                auc_val = df_group["auc"].iloc[0]
                auc_str = f"{auc_val:.4f}" if pd.notna(auc_val) else "N/A"

                # 1. Max F1 Operating Point
                idx_max_f1 = df_group["f1_score"].idxmax()
                op_f1 = df_group.loc[idx_max_f1]

                # 2. High-Recall Target (>= 95%)
                df_95 = df_group[df_group["recall"] >= 0.95]
                op_95 = (
                    df_95.loc[df_95["threshold"].idxmax()] if not df_95.empty else None
                )

                # 3. High-Recall Target (>= 85%)
                df_85 = df_group[df_group["recall"] >= 0.85]
                op_85 = (
                    df_85.loc[df_85["threshold"].idxmax()] if not df_85.empty else None
                )

                # Display name
                model_name = model.upper()
                process_name = processing.upper()
                variant_name = variant.capitalize()

                # Write Max F1 row
                f.write(
                    f"| **{model_name}** | {process_name} | {variant_name} | **{auc_str}** | Max $F_1$ | {op_f1['threshold']:.2f} | {int(op_f1['tp'])} | {int(op_f1['fp'])} | {int(op_f1['tn'])} | {int(op_f1['fn'])} | {op_f1['recall'] * 100:.2f}% | {op_f1['specificity'] * 100:.2f}% | {op_f1['precision'] * 100:.2f}% | {op_f1['f1_score']:.4f} | **{op_f1['labor_reduction'] * 100:.2f}%** |\n"
                )

                # Write 95% Recall row
                if op_95 is not None:
                    f.write(
                        f"| | | | | Recall $\\ge 95\\%$ | {op_95['threshold']:.2f} | {int(op_95['tp'])} | {int(op_95['fp'])} | {int(op_95['tn'])} | {int(op_95['fn'])} | {op_95['recall'] * 100:.2f}% | {op_95['specificity'] * 100:.2f}% | {op_95['precision'] * 100:.2f}% | {op_95['f1_score']:.4f} | **{op_95['labor_reduction'] * 100:.2f}%** |\n"
                    )
                else:
                    max_recall_val = df_group["recall"].max()
                    f.write(
                        f"| | | | | Recall $\\ge 95\\%$ | N/A | - | - | - | - | N/A | N/A | N/A | N/A | **N/A (Max Rec: {max_recall_val * 100:.1f}%)** |\n"
                    )

                # Write 85% Recall row
                if op_85 is not None:
                    f.write(
                        f"| | | | | Recall $\\ge 85\\%$ | {op_85['threshold']:.2f} | {int(op_85['tp'])} | {int(op_85['fp'])} | {int(op_85['tn'])} | {int(op_85['fn'])} | {op_85['recall'] * 100:.2f}% | {op_85['specificity'] * 100:.2f}% | {op_85['precision'] * 100:.2f}% | {op_85['f1_score']:.4f} | **{op_85['labor_reduction'] * 100:.2f}%** |\n"
                    )
                else:
                    max_recall_val = df_group["recall"].max()
                    f.write(
                        f"| | | | | Recall $\\ge 85\\%$ | N/A | - | - | - | - | N/A | N/A | N/A | N/A | **N/A (Max Rec: {max_recall_val * 100:.1f}%)** |\n"
                    )

                f.write("| | | | | | | | | | | | | | | |\n")  # Divider row

            f.write(f"\n### WLT-Specific ROC Curve Visualization for Cycle {cycle}\n\n")
            f.write(
                f"![WLT-Specific ROC Curve (Cycle {cycle})](../plots/wlt_binary_roc_test_pool_cycle_{cycle}.png)\n\n"
            )
            f.write("---\n\n")

    print(f"Dual-table report successfully compiled and saved to: {report_md}")


if __name__ == "__main__":
    main()
