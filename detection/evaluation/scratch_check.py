import pandas as pd
import numpy as np

df = pd.read_csv("results/per_class_threshold_sweep.csv")
# Check for non-monotonicity in a group
groups = df.groupby(
    ["model", "processing", "cycle", "variant", "dataset", "class_id", "class_name"]
)
for name, group in groups:
    group = group.sort_values("threshold", ascending=False)
    fpr = 1 - group["specificity"].values
    if not np.all(np.diff(fpr) >= 0):
        print(f"Non-monotonic group: {name}")
        print(fpr)
        break
