import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_utils.config import FILES_DIR, PLOTS_DIR, CLASSES

CONF_THRESH = 0.5
MODELS_TO_PLOT = ["yolo", "rtdetr", "faster_rcnn"]
CM_CLASSES_RAW = list(CLASSES.values()) + ["Background"]
class_display_map = {
    "Western_Leopard_Toad": "WLT",
    "Other_Amphibian": "Others",
    "Small_Mammal": "Small Mammal",
}
CM_CLASSES = [class_display_map.get(c, c) for c in CM_CLASSES_RAW]
NUM_CLASSES = len(CM_CLASSES)


def plot_cm(cm, arch_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i == len(CLASSES) and j == len(CLASSES):
                continue  # Skip bg-bg
            c = cm[i, j]
            ax.text(
                j,
                i,
                str(c),
                va="center",
                ha="center",
                color="white" if c > np.max(cm) / 2 else "black",
            )

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CM_CLASSES, rotation=45, ha="left")
    ax.set_yticklabels(CM_CLASSES)

    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    # plt.title(f"Confusion Matrix: {arch_name.upper()} (Optimal Thresholds)", pad=20)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    png_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{arch_name}_cycle0.png")
    pdf_path = os.path.join(PLOTS_DIR, f"confusion_matrix_{arch_name}_cycle0.pdf")

    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)

    print(f"Saved CM plot to: {png_path} and {pdf_path}")
    plt.close()


def main():
    for arch in MODELS_TO_PLOT:
        cm_path = os.path.join(FILES_DIR, f"confusion_matrix_{arch}_cycle0.json")
        if not os.path.exists(cm_path):
            print(f"Warning: {cm_path} not found.")
            continue

        with open(cm_path, "r") as f:
            cm = np.array(json.load(f))

        plot_cm(cm, arch)


if __name__ == "__main__":
    main()
