import os
import pandas as pd
import shutil
from tqdm import tqdm


def sort_toad_chips(csv_path, chips_dir):
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"CSV has {len(df)} rows")

    # Get all chip files
    chip_files = [f for f in os.listdir(chips_dir) if f.startswith("chips_img_id=")]
    print(f"Found {len(chip_files)} chip files")

    if len(df) != len(chip_files):
        print(
            f"WARNING: CSV rows ({len(df)}) and chip files ({len(chip_files)}) count mismatch!"
        )

    # Sort filenames numerically by ID
    def get_chip_id(filename):
        try:
            return int(filename.split("=")[1].split("_")[0])
        except (ValueError, IndexError):
            return float("inf")

    sorted_chip_files = sorted(chip_files, key=get_chip_id)

    # Move files into folders
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sorting chips"):
        if idx >= len(sorted_chip_files):
            print(f"Index {idx} exceeds number of chip files.")
            break

        toad_name = str(row["toad_name"])
        src_file = sorted_chip_files[idx]

        # Create destination directory
        dest_dir = os.path.join(chips_dir, toad_name)
        os.makedirs(dest_dir, exist_ok=True)

        # Move file
        src_path = os.path.join(chips_dir, src_file)
        dest_path = os.path.join(dest_dir, src_file)

        shutil.move(src_path, dest_path)

    print("Sorting complete.")


if __name__ == "__main__":
    CSV_PATH = "/home/Joshua/Downloads/leopard_toad_identification/identification/all_leopard_toad_chips/toad_id_inaturalist.csv"
    CHIPS_DIR = "/home/Joshua/Downloads/leopard_toad_identification/identification/all_leopard_toad_chips"

    sort_toad_chips(CSV_PATH, CHIPS_DIR)
