import os
import orjson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score
from metrics import calculate_detection_metrics, calculate_image_level_metrics
from config import RESULTS_DIR, CLASSES, DATASETS, CONF_THRESHOLDS

def get_model_info(folder_name):
    """Extract model type and processing from folder name."""
    parts = folder_name.split("_")
    processing = parts[-1]
    model_type = "_".join(parts[:-1])
    return model_type, processing

def get_eval_info(filename):
    """Extract cycle, variant, and dataset from filename."""
    parts = filename.replace("_raw.json", "").split("_")
    cycle = int(parts[1])
    variant = parts[2]
    dataset = parts[3]
    return cycle, variant, dataset

def run_evaluation_suite():
    all_metrics = []
    all_per_class_sweep = []
    all_binary_sweep = []
    
    # Get all model folders
    folders = sorted([f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))])
    
    for model_folder in folders:
        folder_path = os.path.join(RESULTS_DIR, model_folder)
        filenames = sorted([f for f in os.listdir(folder_path) if f.endswith("_raw.json")])
        if not filenames:
            continue
            
        model_type, processing = get_model_info(model_folder)
        print(f"\n>>> Evaluating {model_type} ({processing})...")
        
        for filename in tqdm(filenames, desc=f"Models in {model_folder}"):
            cycle, variant, dataset = get_eval_info(filename)
            
            with open(os.path.join(folder_path, filename), "rb") as f:
                results = orjson.loads(f.read())
            
            # 1. Detection-level Metrics (mAP)
            det_metrics = calculate_detection_metrics(results)
            mAP = det_metrics["mAP"]
            class_aps = det_metrics["class_aps"]
            
            # 2. Image-level Binary Metrics (Any Animal vs None)
            binary_gt = np.array([res["is_positive"] for res in results])
            binary_scores = np.array([max([p["conf"] for p in res["predictions"]] + [0.0]) for res in results])
            
            # Full Sweep for Binary
            binary_sweep_data = calculate_image_level_metrics(results, CONF_THRESHOLDS)
            for entry in binary_sweep_data:
                entry.update({"model": model_type, "processing": processing, "cycle": cycle, "variant": variant, "dataset": dataset})
                all_binary_sweep.append(entry)
            
            # ROC-AUC using all unique thresholds
            binary_fpr, binary_tpr, _ = roc_curve(binary_gt, binary_scores)
            binary_auc = auc(binary_fpr, binary_tpr)
            
            # Metrics at 0.1 for summary
            binary_preds_01 = binary_scores >= 0.1
            binary_recall_01 = recall_score(binary_gt, binary_preds_01, zero_division=0)
            binary_precision_01 = precision_score(binary_gt, binary_preds_01, zero_division=0)
            binary_f1_01 = f1_score(binary_gt, binary_preds_01, zero_division=0)
            
            # 3. Per-Class Image-level Metrics
            per_class_results = {}
            for cls_id, cls_name in CLASSES.items():
                cls_gt = np.array([any(gt["cls"] == cls_id for gt in res["gt_boxes"]) for res in results])
                cls_scores = np.array([max([p["conf"] for p in res["predictions"] if p["cls"] == cls_id] + [0.0]) for res in results])
                
                # Full Sweep for Per-Class
                for thresh in CONF_THRESHOLDS:
                    preds = cls_scores >= thresh
                    tp = np.sum(preds & cls_gt)
                    fp = np.sum(preds & ~cls_gt)
                    fn = np.sum(~preds & cls_gt)
                    tn = np.sum(~preds & ~cls_gt)
                    
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
                    
                    all_per_class_sweep.append({
                        "model": model_type, "processing": processing, "cycle": cycle, "variant": variant, "dataset": dataset,
                        "class_id": cls_id, "class_name": cls_name, "threshold": thresh,
                        "recall": recall, "specificity": spec, "precision": prec, "f1_score": f1,
                        "tp": tp, "fp": fp, "tn": tn, "fn": fn
                    })

                # ROC-AUC using all unique thresholds
                if len(np.unique(cls_gt)) > 1:
                    fpr, tpr, _ = roc_curve(cls_gt, cls_scores)
                    cls_auc = auc(fpr, tpr)
                else:
                    cls_auc = np.nan
                    fpr, tpr = None, None
                
                # Metrics at 0.1 for summary
                cls_preds_01 = cls_scores >= 0.1
                cls_recall_01 = recall_score(cls_gt, cls_preds_01, zero_division=0)
                cls_precision_01 = precision_score(cls_gt, cls_preds_01, zero_division=0)
                cls_f1_01 = f1_score(cls_gt, cls_preds_01, zero_division=0)
                
                per_class_results[cls_id] = {
                    "auc": cls_auc,
                    "fpr": fpr,
                    "tpr": tpr,
                    "recall_0.1": cls_recall_01,
                    "precision_0.1": cls_precision_01,
                    "f1_01": cls_f1_01,
                    "ap": class_aps.get(cls_id, 0.0)
                }
            
            # Record metrics
            row = {
                "model": model_type,
                "processing": processing,
                "cycle": cycle,
                "variant": variant,
                "dataset": dataset,
                "mAP": mAP,
                "binary_auc": binary_auc,
                "binary_recall_0.1": binary_recall_01,
                "binary_precision_0.1": binary_precision_01,
                "binary_f1_0.1": binary_f1_01
            }
            # Add per-class columns
            for cls_id, cls_name in CLASSES.items():
                res = per_class_results[cls_id]
                row[f"{cls_name}_auc"] = res["auc"]
                row[f"{cls_name}_ap"] = res["ap"]
                row[f"{cls_name}_recall_0.1"] = res["recall_0.1"]
                row[f"{cls_name}_precision_0.1"] = res["precision_0.1"]
                row[f"{cls_name}_f1_0.1"] = res["f1_01"]
                
            all_metrics.append(row)
            
            # Store FPR/TPR for plotting (only for Cycle 4)
            if cycle == 4:
                save_plot_data(model_type, processing, variant, dataset, per_class_results)

    # Save Summaries
    pd.DataFrame(all_metrics).to_csv(os.path.join(RESULTS_DIR, "unified_model_evaluation.csv"), index=False)
    pd.DataFrame(all_per_class_sweep).to_csv(os.path.join(RESULTS_DIR, "per_class_threshold_sweep.csv"), index=False)
    pd.DataFrame(all_binary_sweep).to_csv(os.path.join(RESULTS_DIR, "binary_threshold_sweep.csv"), index=False)
    
    print(f"\nEvaluation files saved to {RESULTS_DIR}")
    
    # Generate Plots
    print("\n>>> Generating ROC Plots...")
    generate_roc_plots()

# Global storage for plot data
plot_data = []

def save_plot_data(model, processing, variant, dataset, per_class_results):
    for cls_id, res in per_class_results.items():
        if res["fpr"] is not None:
            plot_data.append({
                "model": model,
                "processing": processing,
                "variant": variant,
                "dataset": dataset,
                "class_id": cls_id,
                "class_name": CLASSES[cls_id],
                "fpr": res["fpr"],
                "tpr": res["tpr"],
                "auc": res["auc"]
            })

def generate_roc_plots():
    df_plot = pd.DataFrame(plot_data)
    if df_plot.empty:
        return
        
    for dataset in ["test", "val"]:
        for cls_id, cls_name in CLASSES.items():
            subset = df_plot[(df_plot["dataset"] == dataset) & (df_plot["class_id"] == cls_id)]
            if subset.empty:
                continue
                
            plt.figure(figsize=(10, 8))
            
            for _, row in subset.iterrows():
                label = f"{row['model'].upper()} {row['variant'].capitalize()} ({row['processing']}) - AUC: {row['auc']:.4f}"
                plt.plot(row['fpr'], row['tpr'], label=label, linewidth=2)
                
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Chance')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (1 - Specificity)')
            plt.ylabel('True Positive Rate (Recall)')
            plt.title(f'ROC Curves - {cls_name} (Cycle 4, {dataset.capitalize()} Set)')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.gca().set_aspect('equal')
            
            plot_filename = f"final_roc_{cls_name.lower()}_{dataset}.png"
            plot_path = os.path.join(RESULTS_DIR, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC plot saved to {plot_path}")

if __name__ == "__main__":
    run_evaluation_suite()
