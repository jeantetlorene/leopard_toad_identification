import csv
import os
import shutil
import argparse
from pathlib import Path


def extract_metadata(csv_path):
    """
    Derives model_type, mode, and cycle from the CSV path/filename.
    Expected filename format: al_query_candidates_{mode}_cycle_{n}.csv
    Expected folder structure: .../{model_type}/al_query_candidates_...
    """
    csv_path = Path(csv_path)
    filename = csv_path.stem.lower()

    model_type = "unknown"
    for m in ["yolo", "rtdetr", "faster_rcnn"]:
        if m in str(csv_path).lower():
            model_type = m
            break

    mode = (
        "pretrained"
        if "pretrained" in filename
        else "scratch"
        if "scratch" in filename
        else "unknown"
    )

    # Find cycle number
    cycle = "cycle_unknown"
    parts = filename.split("_")
    for i, part in enumerate(parts):
        if part == "cycle" and i + 1 < len(parts):
            cycle = f"cycle_{parts[i + 1]}"
            break
        elif part.startswith("cycle"):
            cycle = part
            break

    return model_type, mode, cycle


def main():
    parser = argparse.ArgumentParser(
        description="Prepare images for annotation by copying them to a Downloads folder."
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the query candidates CSV file."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        stripped_path = Path(str(csv_path).lstrip("/"))
        if stripped_path.exists():
            csv_path = stripped_path

        else:
            print(f"Error: CSV file not found at {args.csv_path}")
            return

    model_type, mode, cycle = extract_metadata(csv_path)
    folder_name = f"{model_type}_{mode}_{cycle}"

    # Create folder in ~/Downloads
    dest_base = Path.home() / "Downloads"
    dest_dir = dest_base / folder_name

    print(f">> Identification: {model_type.upper()} | {mode.upper()} | {cycle.upper()}")
    print(f">> Target Directory: {dest_dir}")

    if dest_dir.exists():
        print(
            f"!! Warning: Directory {dest_dir} already exists. Images will be added/overwritten."
        )
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    missing_count = 0

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_path" not in reader.fieldnames:
            print(f"Error: CSV at {csv_path} is missing 'image_path' column.")
            return

        rows = list(reader)
        print(f">> Copying {len(rows)} images...")

        for row in rows:
            src_path_str = row["image_path"]
            src_path = Path(src_path_str)

            if not src_path.exists():
                print(f"   [MISSING] {src_path}")
                missing_count += 1
                continue

            # Use original filename and copy
            # Resolve naming conflicts
            dest_path = dest_dir / src_path.name
            if dest_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                print(f"   [CONFLICT] Renamed {src_path.name} -> {dest_path.name}")

            try:
                shutil.copy2(src_path, dest_path)
                success_count += 1
            except Exception as e:
                print(f"   [ERROR] Failed to copy {src_path.name}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Total Requested: {len(rows)}")
    print(f"  Successfully Copied: {success_count}")
    print(f"  Missing Images: {missing_count}")
    print(f"  Destination: {dest_dir}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
