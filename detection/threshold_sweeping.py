#%%
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot aesthetics
sns.set_theme(style="white", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)

# Paths
BASE_DIR = "/home/Joshua/Downloads/leopard_toad_identification/detection/results"
VAL_CSV = os.path.join(BASE_DIR, "detect_2", "validation_dataset.csv")

MODELS = ["detect_faster_rcnn", "detect_rtdetr", "detect_yolo"]

# Helper to extract year
def extract_year(path):
    match = re.search(r'(202[345])', str(path))
    if match:
        return match.group(1)
    return "Unknown"

#%% [markdown]
# ### 1. Process Ground Truth

#%%
print("Loading Validation Dataset...")
val_df = pd.read_csv(VAL_CSV)
val_df['year'] = val_df['image_path'].apply(extract_year)

# Determine the unique classes we want to evaluate
classes = val_df['class_name'].unique().tolist()
valid_images = val_df['image_path'].unique().tolist()

print(f"Total Validation Images: {len(valid_images)}")
print(f"Classes Found: {classes}")

# Construct Ground Truth exactly at the image-level
gt_records = []
for img in valid_images:
    df_img = val_df[val_df['image_path'] == img]
    year = df_img['year'].iloc[0]
    
    img_gt = {'image_path': img, 'year': year}
    for cls in classes:
        # Image is positive for class if there's any 'Correct' evaluation row for that class
        is_positive = ((df_img['class_name'] == cls) & (df_img['evaluation'] == 'Correct')).any()
        img_gt[cls] = int(is_positive)
        
    gt_records.append(img_gt)

gt_df = pd.DataFrame(gt_records)
print("Ground truth formulated!")

#%% [markdown]
# ### 2. Load Model Predictions

#%%
print("\nLoading Model Predictions...")
pred_records = []

for model in MODELS:
    print(f"  -> Model: {model}")
    model_dir = os.path.join(BASE_DIR, model)
    csv_files = glob.glob(os.path.join(model_dir, "**", "*.csv"), recursive=True)
    
    # Batch load all CSVs for this model
    model_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Filter immediately to save memory (only validate against GT images)
            df = df[df['image_path'].isin(valid_images)]
            if not df.empty:
                model_dfs.append(df[['image_path', 'class_name', 'confidence']])
        except pd.errors.EmptyDataError:
            pass # Skip empty CSVs
    
    if len(model_dfs) > 0:
        model_all_preds = pd.concat(model_dfs, ignore_index=True)
        # Compute maximum confidence per class per image (Score at image level)
        img_level_preds = model_all_preds.groupby(['image_path', 'class_name'])['confidence'].max().reset_index()
        img_level_preds['model'] = model
        pred_records.append(img_level_preds)

all_preds_df = pd.concat(pred_records, ignore_index=True) if pred_records else pd.DataFrame()

# Map the generic model names to their proper training names
MODEL_MAPPING = {
    "detect_faster_rcnn": "Faster RCNN",
    "detect_rtdetr": "RT-DETR",
    "detect_yolo": "YOLO26"
}

if not all_preds_df.empty:
    all_preds_df['model'] = all_preds_df['model'].map(MODEL_MAPPING).fillna(all_preds_df['model'])

# Update MODELS list so the rest of the script loops over the mapped display names
MODELS = [MODEL_MAPPING.get(m, m) for m in MODELS]

#%% [markdown]
# ### 3. Combine GT and Predictions

#%%
print("\nCombining GT and predictions into evaluation master frame...")
# We need every combination of (model, image_path, class_name)
master_records = []

for model in MODELS:
    for cls in classes:
        for _, row in gt_df.iterrows():
            img = row['image_path']
            year = row['year']
            gt_val = row[cls]
            master_records.append({
                'model': model,
                'class_name': cls,
                'image_path': img,
                'year': year,
                'gt': gt_val
            })

master_df = pd.DataFrame(master_records)

# Merge the actual prediction scores onto the combinations
master_df = pd.merge(
    master_df,
    all_preds_df,
    on=['model', 'class_name', 'image_path'],
    how='left'
)

# If NaN, it means the model didn't predict that class for that image, so score is 0.0
master_df['confidence'] = master_df['confidence'].fillna(0.0)

print(f"Master evaluation frame shape: {master_df.shape}")

#%% [markdown]
# ### 4. Threshold Sweeping Calculation

#%%
print("\nSweeping Thresholds [0.01 -> 1.0]...")
thresholds = np.linspace(0.01, 1.0, 100)
metrics_list = []

years_to_evaluate = ['All Years'] + master_df['year'].unique().tolist()

for year in years_to_evaluate:
    for model in MODELS:
        for cls in classes:
            # Subset data for this specific categorical cut
            if year == 'All Years':
                df_cut = master_df[(master_df['model'] == model) & (master_df['class_name'] == cls)]
            else:
                df_cut = master_df[(master_df['model'] == model) & (master_df['class_name'] == cls) & (master_df['year'] == year)]
            
            if df_cut.empty:
                continue
                
            N = len(df_cut)
            y_true = df_cut['gt'].values
            y_scores = df_cut['confidence'].values
            
            # Fast vectorized evaluations for each threshold
            for t in thresholds:
                y_pred = (y_scores >= t).astype(int)
                
                TP = np.sum((y_pred == 1) & (y_true == 1))
                FP = np.sum((y_pred == 1) & (y_true == 0))
                FN = np.sum((y_pred == 0) & (y_true == 1))
                TN = np.sum((y_pred == 0) & (y_true == 0))
                
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                fp_per_1000 = (FP / N) * 1000 if N > 0 else 0
                flagged = TP + FP
                
                metrics_list.append({
                    'year': year,
                    'model': model,
                    'class_name': cls,
                    'threshold': t,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'fp_per_1000': fp_per_1000,
                    'images_flagged': flagged,
                    'total_images': N
                })

metrics_df = pd.DataFrame(metrics_list)
print("Threshold Sweeping Completed!")

#%% [markdown]
# ### 5. Extract Optimal Thresholds (95% Target Recall Constraint)

#%%
target_recall = 0.95
optimal_thresholds = []

for year in years_to_evaluate:
    for cls in classes:
        for model in MODELS:
            sub = metrics_df[(metrics_df['year'] == year) & (metrics_df['class_name'] == cls) & (metrics_df['model'] == model)]
            if sub.empty:
                continue
            
            # Find the highest threshold that maintains at least target_recall
            valid_cuts = sub[sub['recall'] >= target_recall]
            
            if not valid_cuts.empty:
                best_cut = valid_cuts.loc[valid_cuts['threshold'].idxmax()]
            else:
                # If no threshold provides 95% recall (even t=0.01), resort to the lowest bound
                best_cut = sub.loc[sub['threshold'].idxmin()]
                
            optimal_thresholds.append({
                'Year': year,
                'Class': cls,
                'Model': model,
                f'Threshold_at_{int(target_recall*100)}%_Recall': round(best_cut['threshold'], 2),
                'Recall': f"{best_cut['recall']:.3f}",
                'Precision': f"{best_cut['precision']:.3f}",
                'FP/1000': round(best_cut['fp_per_1000'], 1),
                'Images_Flagged': int(best_cut['images_flagged'])
            })

opt_df = pd.DataFrame(optimal_thresholds)

print(f"\n============= Recommended Operating Points ({int(target_recall*100)}% Recall constraint) =============")
# Print All Years separately for quick operational viewing
print("\n[ALL YEARS ANALYSIS]")
print(opt_df[opt_df['Year'] == 'All Years'].sort_values(['Class', 'Model']).to_string(index=False))

print("\n\n[BY YEAR ANALYSIS]")
print(opt_df[opt_df['Year'] != 'All Years'].sort_values(['Year', 'Class', 'Model']).to_string(index=False))
print("===================================================================================\n")

# Optionally write this out to a CSV
opt_csv_path = os.path.join(BASE_DIR, "optimal_thresholds_95_recall.csv")
opt_df.to_csv(opt_csv_path, index=False)
print(f"Optimal thresholds exported to: {opt_csv_path}")

#%% [markdown]
# ### 6. Visualize Evaluation Curves

#%%
# Helper to plot across years and classes cleanly
def plot_evaluation_curves(metrics, var_x, var_y, title_template, ylabel):
    # Iterate dynamically
    unique_years = metrics['year'].unique()
    unique_classes = metrics['class_name'].unique()
    
    for year in unique_years:
        for cls in unique_classes:
            data_cut = metrics[(metrics['year'] == year) & (metrics['class_name'] == cls)]
            if data_cut.empty: continue
            
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=data_cut, x=var_x, y=var_y, hue='model', linewidth=2.5)
            
            plt.title(f"{title_template} - {cls} ({year})", weight='bold', size=14)
            plt.ylabel(ylabel, size=12)
            plt.xlabel("Threshold" if var_x == 'threshold' else var_x.capitalize(), size=12)
            plt.legend(title='Model')
            plt.tight_layout()
            plt.show()

print("\n[PLOTTING] Recall vs Threshold...")
plot_evaluation_curves(metrics_df, 'threshold', 'recall', "Recall vs Threshold", "Image-Level Recall")

#%%
print("\n[PLOTTING] Precision vs Threshold...")
plot_evaluation_curves(metrics_df, 'threshold', 'precision', "Precision vs Threshold", "Image-Level Precision")

#%%
print("\n[PLOTTING] False Positives per 1000 Images vs Threshold...")
plot_evaluation_curves(metrics_df, 'threshold', 'fp_per_1000', "FP per 1000 vs Threshold", "False Positives per 1000 Images")

#%%
print("\n[PLOTTING] Review Burden (Images Flagged) vs Threshold...")
plot_evaluation_curves(metrics_df, 'threshold', 'images_flagged', "Review Burden vs Threshold", "Total Images to Review")

#%%
print("\n[PLOTTING] Precision-Recall (PR) Curve...")
# The PR curve is Precision vs Recall (x=Recall, y=Precision). 
for year in years_to_evaluate:
    for cls in classes:
        data_cut = metrics_df[(metrics_df['year'] == year) & (metrics_df['class_name'] == cls)]
        if data_cut.empty: continue
            
        plt.figure(figsize=(8, 6))
        # Sort by threshold descending to cleanly trace parametric curve without aggregation bands
        data_cut = data_cut.sort_values(by=['model', 'threshold'], ascending=[True, False])
        sns.lineplot(data=data_cut, x='recall', y='precision', hue='model', estimator=None, sort=False, linewidth=2.5)
            
        plt.title(f"Precision-Recall Curve - {cls} ({year})", weight='bold', size=14)
        plt.ylabel("Precision", size=12)
        plt.xlabel("Recall", size=12)
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()

#%% [markdown]
# ### 7. True Positive vs False Positive Confidence Distribution

#%%
# In this cell, we visualize the overlap between the confidence scores of GT positive vs GT negative.
# We render this specifically for 'All Years' to summarize the holistic separation properties of the models.

print("\n[PLOTTING] Confidence Distributions (TP vs FP) for All Years...")

for cls in classes:
    for model in MODELS:
        df_cut = master_df[(master_df['class_name'] == cls) & (master_df['model'] == model)]
        if df_cut.empty: continue
            
        # We only want to plot distributions for objects that the model actually predicted > 0.01 
        filtered_dist = df_cut[df_cut['confidence'] > 0.0]
        
        plt.figure(figsize=(8, 5))
        sns.kdeplot(
            data=filtered_dist, x='confidence', hue='gt', 
            fill=True, common_norm=False, palette={1: 'green', 0: 'red'}, alpha=0.5,
            warn_singular=False
        )
        plt.title(f"TP vs FP Density Overlay - {model} | {cls} (All Years)", weight='bold', size=14)
        plt.xlabel("Confidence", size=12)
        plt.ylabel("Density", size=12)
        
        # Add custom legend to prevent matplotlib legend overlap bugs
        handles = [plt.Rectangle((0,0),1,1, color='green', alpha=0.5), plt.Rectangle((0,0),1,1, color='red', alpha=0.5)]
        plt.legend(handles, ['True Positives', 'False Positives'])
        
        plt.tight_layout()
        plt.show()

# Now plot for all combined classes
print("\n[PLOTTING] Combined Classes Confidence Distributions (TP vs FP) for All Years...")
for model in MODELS:
    df_cut = master_df[master_df['model'] == model]
    if df_cut.empty: continue
        
    filtered_dist = df_cut[df_cut['confidence'] > 0.0]
    
    plt.figure(figsize=(8, 5))
    sns.kdeplot(
        data=filtered_dist, x='confidence', hue='gt', 
        fill=True, common_norm=False, palette={1: 'green', 0: 'red'}, alpha=0.5,
        warn_singular=False
    )
    plt.title(f"TP vs FP Density Overlay - {model} | All Classes Combined (All Years)", weight='bold', size=14)
    plt.xlabel("Confidence", size=12)
    plt.ylabel("Density", size=12)
    
    handles = [plt.Rectangle((0,0),1,1, color='green', alpha=0.5), plt.Rectangle((0,0),1,1, color='red', alpha=0.5)]
    plt.legend(handles, ['True Positives', 'False Positives'])
    
    plt.tight_layout()
    plt.show()

#%%
print("\nEvaluation Complete!")