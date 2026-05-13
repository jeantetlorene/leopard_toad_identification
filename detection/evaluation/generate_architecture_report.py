import os
import json
import torch
import numpy as np
import pandas as pd
import time

from config import MODEL_ROOTS, RESULTS_DIR, FILES_DIR, PLOTS_DIR, CLASSES, DEVICE
from models.faster_rcnn_wrapper import FasterRCNNWrapper
from models.ultralytics_wrapper import UltralyticsWrapper
from metrics import box_iou, calculate_map50_95

CYCLE = 0
VARIANT = "scratch"
PROCESSING = "clahe"
DATASET = "test"
CONF_THRESH = 0.5
IOU_THRESH = 0.5

MODELS_TO_EVALUATE = ["yolo", "rtdetr", "faster_rcnn"]

# Initialize classes + Background
CM_CLASSES = list(CLASSES.values()) + ["Background"]
NUM_CLASSES = len(CM_CLASSES)


def calculate_flops_params(wrapper, model_type):
    # Try to calculate params manually
    try:
        if model_type in ["yolo", "rtdetr"]:
            model = wrapper.model.model
        else:
            model = wrapper.model
        params = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        params = 0.0

    # Try to get FLOPs (using thop if available)
    flops = "N/A"
    try:
        from thop import profile

        dummy = torch.randn(1, 3, 640, 640).to(DEVICE)

        if model_type in ["yolo", "rtdetr"]:
            model_to_profile = wrapper.model.model
        else:
            model_to_profile = wrapper.model

        macs, _ = profile(model_to_profile, inputs=(dummy,), verbose=False)
        flops = f"{(macs * 2) / 1e9:.2f}"
    except ImportError:
        pass
    except Exception:
        pass

    return f"{params:.2f}", flops


def benchmark_inference(wrapper, model_type):
    print(f"Benchmarking inference speed for {model_type}...")
    dummy_images = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)]

    # Warmup
    for _ in range(5):
        wrapper.predict_batch(dummy_images)

    # Benchmark
    start_time = time.time()
    iters = 20
    for _ in range(iters):
        wrapper.predict_batch(dummy_images)

    ms_per_img = ((time.time() - start_time) / iters) * 1000
    fps = 1000 / ms_per_img
    return ms_per_img, fps


def generate_confusion_matrix(raw_results, arch_name):
    print(f"Generating Confusion Matrix for {arch_name}...")
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    bg_idx = len(CLASSES)

    for res in raw_results:
        gts = res["gt_boxes"]
        preds = [p for p in res["predictions"] if p["conf"] >= CONF_THRESH]

        # Track which predictions have been matched
        pred_matched = [False] * len(preds)

        for gt in gts:
            gt_cls = gt["cls"]
            best_iou = 0
            best_pred_idx = -1

            for i, p in enumerate(preds):
                if pred_matched[i]:
                    continue
                iou = box_iou(gt["bbox"], p["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i

            if best_iou >= IOU_THRESH:
                pred_cls = preds[best_pred_idx]["cls"]
                cm[gt_cls, pred_cls] += 1
                pred_matched[best_pred_idx] = True
            else:
                # False Negative: GT missed
                cm[gt_cls, bg_idx] += 1

        # False Positives: Predictions not matched to any GT
        for i, p in enumerate(preds):
            if not pred_matched[i]:
                cm[bg_idx, p["cls"]] += 1

    # True Negatives (Background -> Background) are not strictly counted in standard object detection CMs,
    # as they represent infinite empty space. We leave it as 0 or empty for clarity.

    # Save the CM data
    os.makedirs(FILES_DIR, exist_ok=True)
    cm_path = os.path.join(FILES_DIR, f"confusion_matrix_{arch_name}_cycle0.json")
    with open(cm_path, "w") as f:
        json.dump(cm.tolist(), f)

    print(f"Saved CM data to {cm_path}")
    return cm


def generate_report():
    print("Starting Architecture Benchmarking...")

    unified_csv = os.path.join(FILES_DIR, "unified_model_evaluation.csv")
    sweep_csv = os.path.join(FILES_DIR, "per_class_threshold_sweep.csv")

    if not os.path.exists(unified_csv) or not os.path.exists(sweep_csv):
        print("Required CSVs not found in files/. Please run evaluations first.")
        return

    df_unified = pd.read_csv(unified_csv)
    df_sweep = pd.read_csv(sweep_csv)

    results_table = []

    for m_type in MODELS_TO_EVALUATE:
        root_key = f"{m_type}_{PROCESSING}"
        root_dir = MODEL_ROOTS.get(root_key)

        runs_dir = os.path.join(root_dir, "runs")
        run_name = f"cycle_{CYCLE}_{VARIANT}_scratch"
        model_path = os.path.join(runs_dir, run_name, "weights", "best.pt")

        if not os.path.exists(model_path):
            print(f"Warning: Weights not found for {m_type} at {model_path}")
            continue

        # 1. Benchmark speed and params
        if m_type in ["yolo", "rtdetr"]:
            wrapper = UltralyticsWrapper(m_type, model_path, device=DEVICE)
        else:
            wrapper = FasterRCNNWrapper(model_path, device=DEVICE)

        ms_per_img, fps = benchmark_inference(wrapper, m_type)
        params, flops = calculate_flops_params(wrapper, m_type)

        # 2. Confusion Matrix
        raw_json_path = os.path.join(
            RESULTS_DIR, root_key, f"cycle_{CYCLE}_{VARIANT}_{DATASET}_raw.json"
        )
        map50_95 = "N/A"
        if os.path.exists(raw_json_path):
            with open(raw_json_path, "r") as f:
                raw_results = json.load(f)
            # 2. Confusion Matrix
            generate_confusion_matrix(raw_results, m_type)
            # Calculate mAP50-95
            map50_95 = calculate_map50_95(raw_results)
        else:
            print(f"Raw JSON not found for {m_type}: {raw_json_path}")

        # 3. Aggregate AP/AR metrics
        # mAP
        df_u = df_unified[
            (df_unified["model"] == m_type)
            & (df_unified["variant"] == VARIANT)
            & (df_unified["processing"] == PROCESSING)
            & (df_unified["cycle"] == CYCLE)
        ]
        map50 = df_u["mAP"].values[0] if not df_u.empty else "N/A"

        # AR
        df_s = df_sweep[
            (df_sweep["model"] == m_type)
            & (df_sweep["variant"] == VARIANT)
            & (df_sweep["processing"] == PROCESSING)
            & (df_sweep["cycle"] == CYCLE)
        ]
        if not df_s.empty:
            class_recalls = df_s.groupby("class_name")["recall"].max()
            ar = class_recalls.mean()
        else:
            ar = "N/A"

        results_table.append(
            {
                "Architecture": m_type.upper(),
                "mAP50": f"{map50:.4f}"
                if isinstance(map50, (float, np.floating))
                else map50,
                "mAP50-95": f"{map50_95:.4f}"
                if isinstance(map50_95, (float, np.floating))
                else map50_95,
                "AR": f"{ar:.4f}" if isinstance(ar, (float, np.floating)) else ar,
                "Params (M)": params,
                "GFLOPs": flops,
                "Inference (ms)": f"{ms_per_img:.1f}",
                "FPS": f"{fps:.1f}",
            }
        )

    # 4. Write Markdown Report
    os.makedirs(FILES_DIR, exist_ok=True)
    report_path = os.path.join(FILES_DIR, "architecture_results.md")
    with open(report_path, "w") as f:
        f.write("# Results: Effect of Architecture (Cycle 0)\n\n")
        f.write(
            "This report benchmarks the baseline network paradigms (YOLO, RT-DETR, Faster R-CNN) at Cycle 0, evaluating their computational efficiency and fundamental localization abilities before domain-specific transfer learning.\n\n"
        )

        f.write("### Comprehensive Architectural Benchmarking Table\n")
        f.write(
            "| Architecture | mAP50 | mAP50-95 | Average Recall | Params (M) | GFLOPs | Inference Speed (ms) | FPS |\n"
        )
        f.write(
            "|--------------|-------|----------|----------------|------------|--------|----------------------|-----|\n"
        )
        for row in results_table:
            f.write(
                f"| {row['Architecture']} | {row['mAP50']} | {row['mAP50-95']} | {row['AR']} | {row['Params (M)']} | {row['GFLOPs']} | {row['Inference (ms)']} | {row['FPS']} |\n"
            )

        f.write("\n### Architecture-Specific Confusion Matrices\n")
        f.write(
            "These matrices cross-reference predicted categories against actual ground-truth labels at a 0.5 confidence threshold, explicitly demonstrating inter-class confusion and background noise vulnerability.\n\n"
        )

        for row in results_table:
            arch = row["Architecture"].lower()
            f.write(f"#### {arch.upper()}\n")
            f.write(
                f"![{arch.upper()} Confusion Matrix](../plots/confusion_matrix_{arch}_cycle0.png)\n\n"
            )

    print(f"\nArchitecture evaluation complete! Report saved to {report_path}")


if __name__ == "__main__":
    generate_report()
