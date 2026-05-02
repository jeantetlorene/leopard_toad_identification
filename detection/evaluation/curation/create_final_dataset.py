import pandas as pd
import argparse
import os


def process_split(consensus_csv, curation_csv, eval_csv, output_csv):
    if not os.path.exists(eval_csv):
        print(f"Skipping {consensus_csv}: No evaluations found at {eval_csv}")
        return

    print(f"Processing {os.path.basename(consensus_csv)}...")
    consensus_df = pd.read_csv(consensus_csv)
    curation_df = pd.read_csv(curation_csv)
    eval_df = pd.read_csv(eval_csv)

    # Find the cluster IDs whose representative was marked as Incorrect
    # Merge curation_df with eval_df on image_path
    merged = curation_df[curation_df["is_representative"] == True].merge(
        eval_df, on="image_path", how="left"
    )

    # Identify bad clusters
    bad_clusters = merged[merged["evaluation"] == "Incorrect"]["cluster_id"].unique()
    print(
        f"  Found {len(bad_clusters)} bad clusters out of {curation_df['cluster_id'].nunique()}"
    )

    # Find all predictions belonging to bad clusters
    bad_predictions = curation_df[curation_df["cluster_id"].isin(bad_clusters)]
    print(
        f"  Dropping {len(bad_predictions)} predictions associated with bad clusters."
    )

    # We match these bad predictions back to consensus_df to drop them.
    # Create a unique ID for each row based on path and coordinates to avoid floating point mismatch
    def make_id(df):
        return (
            df["image_path"]
            + "_"
            + df["xmin"].round(1).astype(str)
            + "_"
            + df["ymin"].round(1).astype(str)
            + "_"
            + df["xmax"].round(1).astype(str)
            + "_"
            + df["ymax"].round(1).astype(str)
        )

    consensus_df["_match_id"] = make_id(consensus_df)
    bad_predictions = bad_predictions.copy()
    bad_predictions["_match_id"] = make_id(bad_predictions)

    bad_ids = set(bad_predictions["_match_id"])

    # Filter out the bad IDs
    final_df = consensus_df[~consensus_df["_match_id"].isin(bad_ids)].drop(
        columns=["_match_id"]
    )

    print(f"  Original predictions: {len(consensus_df)}")
    print(f"  Final predictions: {len(final_df)}")
    print(f"  Total dropped: {len(consensus_df) - len(final_df)}")

    final_df.to_csv(output_csv, index=False)
    print(f"  Saved final dataset to {output_csv}\n")


if __name__ == "__main__":
    base_dir = "/home/Joshua/Downloads/leopard_toad_identification/evaluation/consensus_predictions"

    # Process Val
    process_split(
        os.path.join(base_dir, "val_consensus.csv"),
        os.path.join(base_dir, "val_curation_priority.csv"),
        os.path.join(base_dir, "val_curation_priority_evaluations.csv"),
        os.path.join(base_dir, "val_consensus_final.csv"),
    )

    # Process Test
    # Check if test evaluation exists first to avoid errors
    test_eval = os.path.join(base_dir, "test_curation_priority_evaluations.csv")
    if os.path.exists(test_eval):
        process_split(
            os.path.join(base_dir, "test_consensus.csv"),
            os.path.join(base_dir, "test_curation_priority.csv"),
            test_eval,
            os.path.join(base_dir, "test_consensus_final.csv"),
        )
