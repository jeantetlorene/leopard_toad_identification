import pandas as pd
import os

csv_path = "/home/Joshua/Downloads/leopard_toad_identification/detection/results/detect_2/validation_dataset.csv"
df = pd.read_csv(csv_path)


# Extract camera from subfolder (assuming it's the first part before the slash)
def extract_camera(subfolder):
    if pd.isna(subfolder):
        return "Unknown"
    # Some subfolders might be like "5Z/100MEDIA" or "Cameras - AI Data/5Z/..."
    # The example lines show "5Z/100MEDIA", "4R/105MEDIA", etc.
    # But some lines (584+) show subfolder as "5Z" directly or have more complex paths.
    # Let's try splitting by slash and finding the one that matches our target camera codes.
    targets = {"3Z", "4R", "4Z", "5R", "5Z", "6R", "6Z"}
    parts = str(subfolder).split("/")
    for p in parts:
        if p.upper() in targets:
            return p.upper()
    return "Unknown"


df["camera"] = df["subfolder"].apply(extract_camera)

# Filter for correct predictions
df_correct = df[df["evaluation"] == "Correct"]

# Define target classes
target_classes = ["Western_Leopard_Toad", "Other_Amphibian", "Small_Mammal"]
target_cameras = ["3Z", "4R", "4Z", "5R", "5Z", "6R", "6Z"]

# 1. Stats per camera
# Initialize with zeros for all target cameras
camera_stats = pd.Series(0, index=target_cameras).sort_index()
actual_counts = (
    df_correct[df_correct["camera"].isin(target_cameras)].groupby("camera").size()
)
camera_stats.update(actual_counts)

# 2. Stats per camera and class
# Initialize a table with all target cameras and classes set to 0
class_stats = pd.DataFrame(0, index=target_cameras, columns=target_classes).sort_index()
actual_class_counts = (
    df_correct[df_correct["camera"].isin(target_cameras)]
    .groupby(["camera", "class_name"])
    .size()
    .unstack(fill_value=0)
)

# Merge actual counts into the template
for cam in actual_class_counts.index:
    for cls in actual_class_counts.columns:
        if cls in target_classes:
            class_stats.loc[cam, cls] = actual_class_counts.loc[cam, cls]

# Print results
print("### Correct Predictions per Camera:")
print(camera_stats.to_markdown())
print("\n### Correct Predictions per Camera and Class:")
print(class_stats.to_markdown())
