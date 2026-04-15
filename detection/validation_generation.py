# %%
import os
import glob
import pandas as pd
import numpy as np

# Set the random seed for reproducibility
np.random.seed(42)

# Directory containing the YOLO predictions and evaluations
detect_2_dir = (
    "/home/Joshua/Downloads/leopard_toad_identification/detection/results/detect_2"
)

# Find all evaluation CSV files
eval_files = glob.glob(
    os.path.join(detect_2_dir, "**/*_evaluations.csv"), recursive=True
)

correct_dfs = []
incorrect_dfs = []

# %%
for eval_file in eval_files:
    # Original CSV file is the same name without "_evaluations"
    orig_file = eval_file.replace("_evaluations.csv", ".csv")

    if not os.path.exists(orig_file):
        continue

    try:
        # Load evaluations and original predictions
        df_eval = pd.read_csv(eval_file)
        df_orig = pd.read_csv(orig_file)

        # Merge on 'image_path' to assign the evaluation to all bounding boxes of that image
        df_merged = pd.merge(df_orig, df_eval, on="image_path", how="inner")

        # Filter correct and incorrect predictions
        correct_subset = df_merged[df_merged["evaluation"] == "Correct"]
        incorrect_subset = df_merged[df_merged["evaluation"] == "Incorrect"]

        if not correct_subset.empty:
            correct_dfs.append(correct_subset)
        if not incorrect_subset.empty:
            incorrect_dfs.append(incorrect_subset)

    except Exception as e:
        print(f"Error processing {eval_file}: {e}")

# %%
# Combine all data
all_correct_df = (
    pd.concat(correct_dfs, ignore_index=True) if correct_dfs else pd.DataFrame()
)
all_incorrect_df = (
    pd.concat(incorrect_dfs, ignore_index=True) if incorrect_dfs else pd.DataFrame()
)

print(f"Total correct predictions (bounding boxes): {len(all_correct_df)}")
print(f"Total incorrect predictions before sampling: {len(all_incorrect_df)}")

# Calculate 30% of the number of correct predictions to sample for the negative classes
target_incorrect_count = int(0.30 * len(all_correct_df))

if len(all_incorrect_df) > target_incorrect_count:
    # We sample at the image level to ensure all boxes for an image are kept together
    unique_incorrect_images = all_incorrect_df["image_path"].unique()
    np.random.shuffle(unique_incorrect_images)

    sampled_incorrect_list = []
    current_count = 0

    for img in unique_incorrect_images:
        img_df = all_incorrect_df[all_incorrect_df["image_path"] == img]
        sampled_incorrect_list.append(img_df)
        current_count += len(img_df)

        if current_count >= target_incorrect_count:
            break

    sampled_incorrect_df = pd.concat(sampled_incorrect_list, ignore_index=True)
else:
    sampled_incorrect_df = all_incorrect_df

# %%
print(f"Total incorrect predictions after sampling: {len(sampled_incorrect_df)}")

# Combine to create the final validation set
validation_set_df = pd.concat([all_correct_df, sampled_incorrect_df], ignore_index=True)

# Save to CSV
output_csv_path = os.path.join(detect_2_dir, "validation_dataset.csv")
validation_set_df.to_csv(output_csv_path, index=False)
print(f"\nSuccessfully saved validation dataset to: {output_csv_path}")
print(f"Final dataset contains {len(validation_set_df)} bounding boxes.")

# %%
# Perform statistics on the validation set
import re


# Helper function to extract the year from the image path
def extract_year(path):
    # Match years 2023, 2024, or 2025 in the file path
    match = re.search(r"(202[345])", str(path))
    if match:
        return match.group(1)
    return "Unknown"


# Add the 'year' column
validation_set_df["year"] = validation_set_df["image_path"].apply(extract_year)

print("\n" + "=" * 50)
print("STATISTICS OVERVIEW")
print("=" * 50)

# 1. Total number of bounding boxes present for each class
print("\n--- 1. Bounding Boxes per Class ---")
class_counts = validation_set_df["class_name"].value_counts()
print(class_counts.to_string())

print("\n--- Correct vs Incorrect per Class ---")
class_eval_counts = (
    validation_set_df.groupby(["class_name", "evaluation"]).size().unstack(fill_value=0)
)
print(class_eval_counts)

# %%
# 2. Number of unique images per year
print("\n--- 2. Unique Images per Year ---")
unique_images_df = validation_set_df.drop_duplicates(subset=["image_path"])
year_counts = unique_images_df["year"].value_counts().sort_index()
print(year_counts.to_string())

print("\n--- Evaluation Status of Unique Images per Year ---")
year_eval_counts = (
    unique_images_df.groupby(["year", "evaluation"]).size().unstack(fill_value=0)
)
print(year_eval_counts)

# 3. Average confidence scores
print("\n--- 3. Miscellaneous Stats: Average Confidence ---")
if "confidence" in validation_set_df.columns:
    conf_stats = validation_set_df.groupby("evaluation")["confidence"].agg(
        ["mean", "std", "min", "max"]
    )
    print(conf_stats)
else:
    print("'confidence' column not found.")

# %%
# 4. Correct predictions per class for each year
print("\n--- 4. Correct Predictions per Class by Year ---")
correct_only_df = validation_set_df[validation_set_df["evaluation"] == "Correct"]
if not correct_only_df.empty:
    correct_yearly_class_stats = (
        correct_only_df.groupby(["year", "class_name"]).size().unstack(fill_value=0)
    )
    print(correct_yearly_class_stats)
else:
    print("No correct predictions found.")
# %%
