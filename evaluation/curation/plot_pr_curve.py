import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import os


def plot_pr_curve(csv_path):
    """
    Helper function to plot a Precision-Recall curve based on manual review.
    It automatically looks for the generated _evaluations.csv from the Gradio app.
    """
    eval_path = csv_path.replace(".csv", "_evaluations.csv")
    if not os.path.exists(eval_path):
        print(
            f"Error: Evaluations file not found at {eval_path}. Complete manual review in the Gradio app first."
        )
        return

    df = pd.read_csv(csv_path)
    eval_df = pd.read_csv(eval_path)

    # Merge the evaluations with the original predictions to get the confidence scores
    merged_df = pd.merge(df, eval_df, on="image_path", how="inner")

    if len(merged_df) == 0:
        print("No matching reviewed samples found.")
        return

    # Convert 'Correct'/'Incorrect' to boolean 1/0
    merged_df["is_correct"] = (merged_df["evaluation"] == "Correct").astype(int)

    y_true = merged_df["is_correct"]
    y_scores = merged_df["confidence"]

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".", label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Post-Curation)")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(os.path.dirname(csv_path), "pr_curve.png")
    plt.savefig(out_path)
    print(f"Saved PR curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Precision-Recall Curve")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the curation priority CSV to plot its PR curve (e.g. val_curation_priority.csv). Use this after completing manual review.",
    )

    args = parser.parse_args()
    plot_pr_curve(args.csv_path)


if __name__ == "__main__":
    main()
