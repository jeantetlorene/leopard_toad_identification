#!/usr/bin/env python3
"""
Bulk-download California Small Animals images + filtered annotations
for anything matching: toad, rat, or frog.

Examples:
  python download_california_small_animals.py

  python download_california_small_animals.py \
      --output-dir california_small_animals_subset

  python download_california_small_animals.py \
      --metadata-file /path/to/california-small-animals.json \
      --workers 16 \
      --download-missing-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import requests


DEFAULT_METADATA_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "california-small-animals/california_small_animals_with_sequences.zip"
)

DEFAULT_IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/california-small-animals"
)
DEFAULT_KEYWORDS = ["toad", "rat", "frog"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download California Small Animals images/annotations for toad, rat, or frog."
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=None,
        help="Path to local metadata JSON or JSON.ZIP file. If omitted, --metadata-url is used.",
    )
    parser.add_argument(
        "--metadata-url",
        type=str,
        default=DEFAULT_METADATA_URL,
        help="Remote metadata JSON.ZIP URL.",
    )
    parser.add_argument(
        "--image-base-url",
        type=str,
        default=DEFAULT_IMAGE_BASE_URL,
        help="Base URL for images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/Joshua/Downloads/leopard_toad_identification/dataset/california_small_animals_subset",
        help="Output directory.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Case-insensitive keywords to match in species/category names and file paths.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per image.",
    )
    parser.add_argument(
        "--download-missing-only",
        action="store_true",
        help="Skip image download if file already exists locally.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of matching images to download.",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def load_bytes_from_url(url: str, timeout: int) -> bytes:
    with requests.get(url, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        return r.content


def read_json_from_zip_bytes(blob: bytes) -> dict:
    with zipfile.ZipFile(BytesIO(blob)) as zf:
        json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
        if not json_names:
            raise ValueError("No JSON file found inside ZIP metadata.")
        with zf.open(json_names[0]) as f:
            return json.load(f)


def read_json_from_file(path: Path) -> dict:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".json.zip") or path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not json_names:
                raise ValueError(f"No JSON found inside ZIP: {path}")
            with zf.open(json_names[0]) as f:
                return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata(
    metadata_file: Optional[str], metadata_url: str, timeout: int
) -> dict:
    if metadata_file:
        return read_json_from_file(Path(metadata_file))
    blob = load_bytes_from_url(metadata_url, timeout=timeout)
    return read_json_from_zip_bytes(blob)


def normalize_keywords(keywords: Sequence[str]) -> List[str]:
    return [k.strip().lower() for k in keywords if k.strip()]


def image_text_blob(image_rec: dict) -> str:
    parts = [
        str(image_rec.get("id", "")),
        str(image_rec.get("file_name", "")),
        str(image_rec.get("location", "")),
    ]
    return " | ".join(parts).lower()


def build_category_maps(data: dict) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    for cat in data.get("categories", []):
        cid = cat.get("id")
        name = str(cat.get("name", "")).strip()
        if cid is not None:
            id_to_name[int(cid)] = name
        if name:
            name_to_id[name.lower()] = int(cid)
    return id_to_name, name_to_id


def matching_category_ids(
    id_to_name: Dict[int, str], keywords: Sequence[str]
) -> Set[int]:
    out: Set[int] = set()
    for cid, name in id_to_name.items():
        name_l = name.lower()
        if any(k in name_l for k in keywords):
            out.add(cid)
    return out


def filter_dataset(
    data: dict, keywords: Sequence[str]
) -> Tuple[List[dict], List[dict], Set[int]]:
    keywords = normalize_keywords(keywords)

    images: List[dict] = data.get("images", [])
    annotations: List[dict] = data.get("annotations", [])

    id_to_name, _ = build_category_maps(data)
    matched_category_ids = matching_category_ids(id_to_name, keywords)

    matched_image_ids: Set[int] = set()
    matched_images: List[dict] = []
    matched_annotations: List[dict] = []

    for im in images:
        blob = image_text_blob(im)
        if any(k in blob for k in keywords):
            iid = im.get("id")
            if iid is not None:
                matched_image_ids.add(iid)

    for ann in annotations:
        category_id = ann.get("category_id")
        image_id = ann.get("image_id")
        if category_id in matched_category_ids and image_id is not None:
            matched_image_ids.add(image_id)

    for im in images:
        iid = im.get("id")
        blob = image_text_blob(im)
        if (iid in matched_image_ids) or any(k in blob for k in keywords):
            matched_images.append(im)

    matched_image_ids_final = {im.get("id") for im in matched_images}
    for ann in annotations:
        if ann.get("image_id") in matched_image_ids_final:
            matched_annotations.append(ann)

    if not images and isinstance(data, list):
        flat_matches = []
        for rec in data:
            blob = image_text_blob(rec)
            if any(k in blob for k in keywords):
                flat_matches.append(rec)
        return flat_matches, [], set()

    return matched_images, matched_annotations, matched_category_ids


def make_filtered_output(
    original: dict,
    matched_images: List[dict],
    matched_annotations: List[dict],
    matched_category_ids: Set[int],
) -> dict:
    out = {}

    for key in ("info", "licenses"):
        if key in original:
            out[key] = original[key]

    if "categories" in original:
        if matched_category_ids:
            out["categories"] = [
                c for c in original["categories"] if c.get("id") in matched_category_ids
            ]
        else:
            out["categories"] = original["categories"]

    out["images"] = matched_images

    if "annotations" in original:
        out["annotations"] = matched_annotations

    return out


def safe_join_output(root: Path, relative_path: str) -> Path:
    return root.joinpath(Path(relative_path))


def download_one(
    image_rec: dict,
    output_root: Path,
    image_base_url: str,
    timeout: int,
    retries: int,
    skip_existing: bool,
) -> Tuple[str, bool, str]:
    rel_path = image_rec.get("file_name") or image_rec.get("id")
    if not rel_path:
        return ("<missing>", False, "missing file_name/id")

    rel_path = str(rel_path).lstrip("/")
    out_path = safe_join_output(output_root, rel_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_path.exists():
        return (rel_path, True, "exists")

    url = f"{image_base_url.rstrip('/')}/{rel_path}"

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return (rel_path, True, "downloaded")
        except Exception as e:
            last_err = f"attempt {attempt}/{retries}: {e}"
            time.sleep(min(2 * attempt, 5))

    return (rel_path, False, last_err)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Loading metadata...")
    data = load_metadata(args.metadata_file, args.metadata_url, args.timeout)

    if isinstance(data, list):
        matched_images, matched_annotations, matched_category_ids = filter_dataset(
            data, args.keywords
        )
        filtered = matched_images
    elif isinstance(data, dict):
        matched_images, matched_annotations, matched_category_ids = filter_dataset(
            data, args.keywords
        )
        filtered = make_filtered_output(
            data, matched_images, matched_annotations, matched_category_ids
        )
    else:
        raise ValueError("Unsupported metadata format. Expected a dict or list.")

    if args.limit is not None:
        matched_images = matched_images[: args.limit]
        if isinstance(filtered, dict):
            matched_ids = {im.get("id") for im in matched_images}
            filtered["images"] = matched_images
            if "annotations" in filtered:
                filtered["annotations"] = [
                    ann
                    for ann in filtered["annotations"]
                    if ann.get("image_id") in matched_ids
                ]
        elif isinstance(filtered, list):
            filtered = matched_images

    annotations_path = output_dir / "filtered_annotations.json"
    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2)

    log(f"Saved filtered annotations: {annotations_path}")
    log(f"Matched images: {len(matched_images)}")

    if not matched_images:
        log("No matching images found.")
        return 0

    log("Starting downloads...")
    ok = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(
                download_one,
                image_rec=im,
                output_root=output_dir,
                image_base_url=args.image_base_url,
                timeout=args.timeout,
                retries=args.retries,
                skip_existing=args.download_missing_only,
            )
            for im in matched_images
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            rel_path, success, status = fut.result()
            if success:
                ok += 1
            else:
                failed += 1

            if i % 50 == 0 or i == len(futures):
                log(f"[{i}/{len(futures)}] ok={ok} failed={failed}")

            if not success:
                log(f"FAILED: {rel_path} -> {status}")

    summary = {
        "matched_images": len(matched_images),
        "downloaded_ok": ok,
        "failed": failed,
        "keywords": args.keywords,
        "annotations_file": str(annotations_path),
        "output_dir": str(output_dir),
    }

    summary_path = output_dir / "download_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(f"Done. Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
