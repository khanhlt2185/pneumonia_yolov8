"""
Medical Image Preprocessing Pipeline for YOLOv8-Classification.
Specialized for Kaggle Chest X-Ray (Pneumonia) Dataset.

Key Objectives:
1. Re-split: Merge flawed original train/val splits and execute an 85/15 stratified
   split to resolve the 16-image validation set limitation.
2. Isolation: Strictly preserve the original test set (624 images) independently.
3. Enhancement: Apply Grayscale CLAHE (clipLimit=2.0, tileGridSize=(8,8)) to highlight
   subtle pulmonary opacities without amplifying background radiation noise.
4. Robustness: Filter corrupted/0-byte/non-image files, resilient I/O with pathlib and cv2.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    # Graceful fallback if tqdm is not installed
    def tqdm(iterable, desc="", **kwargs):
        if desc:
            print(f"[*] {desc}...")
        return iterable


# Global Reproducibility Seed & Processing Defaults
SEED: int = 42
DEFAULT_CLIP_LIMIT: float = 2.0
DEFAULT_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)
VALID_EXTENSIONS: set = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES: List[str] = ["NORMAL", "PNEUMONIA"]


def set_seed(seed: int = SEED) -> None:
    """Set global random seed for python and numpy to guarantee reproducible splits."""
    random.seed(seed)
    np.random.seed(seed)


def is_valid_image(file_path: Path) -> bool:
    """
    Validate whether a file is an authentic, readable, non-empty image.

    Checks:
    - Path existence and non-hidden/system file status (ignores .DS_Store, ._ files).
    - Valid image extension whitelist.
    - File size > 0 bytes.
    - Successful cv2 image decode.
    """
    try:
        # Ignore macOS / hidden metadata files
        if file_path.name.startswith(".") or file_path.name.startswith("._"):
            return False

        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            return False

        if not file_path.is_file() or file_path.stat().st_size == 0:
            return False

        # Read small probe to verify header integrity
        test_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if test_img is None or test_img.size == 0:
            return False

        return True
    except Exception:
        return False


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Medical Radiography Rationale:
    - Global histogram equalization often blows out soft tissue and increases high-frequency noise.
    - CLAHE operates on localized contextual tiles (tileGridSize, e.g. 8x8).
    - clipLimit=2.0 clips histogram bins to prevent excessive amplification of radiographic noise,
      making subtle infiltrates and consolidations sharper and distinguishable for CNN feature extraction.

    Args:
        image: Single-channel (grayscale) or multi-channel NumPy array.
        clip_limit: Threshold for contrast limiting. Default 2.0.
        tile_grid_size: Size of grid for histogram equalization. Default (8, 8).

    Returns:
        Enhanced single-channel (Grayscale) uint8 NumPy array.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    # Guarantee grayscale single-channel
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Ensure 8-bit depth
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced


def process_and_save_image(
    src_path: Path,
    dest_path: Path,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> bool:
    """
    Read an image from src_path in grayscale, apply CLAHE enhancement, and save to dest_path.

    Protected with try-except to ensure zero crash on edge-case I/O faults.
    """
    try:
        # Load directly in grayscale
        img_gray = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return False

        # Apply CLAHE enhancement
        enhanced_img = apply_clahe(img_gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        # Ensure parent destination directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write out enhanced image
        success = cv2.imwrite(str(dest_path), enhanced_img)
        return bool(success)
    except Exception as err:
        print(f"[!] Warning: Failed to process {src_path.name} -> {err}")
        return False


def collect_images_from_dirs(
    dir_paths: List[Path],
    class_name: str,
) -> Tuple[List[Path], int]:
    """
    Collect and sanitize all valid image paths from a list of directory paths for a given class.

    Returns:
        valid_paths: List of verified Path objects.
        corrupted_count: Count of skipped/corrupted/non-image files.
    """
    valid_paths: List[Path] = []
    corrupted_count: int = 0

    for d in dir_paths:
        class_dir = d / class_name
        if not class_dir.exists():
            continue

        for file_path in class_dir.iterdir():
            if is_valid_image(file_path):
                valid_paths.append(file_path)
            else:
                corrupted_count += 1

    return valid_paths, corrupted_count


def print_summary_table(
    original_stats: Dict[str, Dict[str, int]],
    final_stats: Dict[str, Dict[str, int]],
    skipped_count: int,
) -> None:
    """Print an informative, clinical-grade summary table of dataset distributions."""
    divider = "=" * 72
    print("\n" + divider)
    print("      CHEST X-RAY (PNEUMONIA) PREPROCESSING & SPLIT SUMMARY")
    print(divider)
    print(f"{'Split':<12} | {'Class':<12} | {'Original Count':<16} | {'Processed Count':<16}")
    print("-" * 72)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            orig = original_stats.get(split, {}).get(cls, 0)
            fin = final_stats.get(split, {}).get(cls, 0)
            print(f"{split:<12} | {cls:<12} | {orig:<16} | {fin:<16}")
        total_orig = sum(original_stats.get(split, {}).values())
        total_fin = sum(final_stats.get(split, {}).values())
        print(f"{'SUBTOTAL (' + split + ')':<26} | {total_orig:<16} | {total_fin:<16}")
        print("-" * 72)

    total_orig_all = sum(sum(v.values()) for v in original_stats.values())
    total_fin_all = sum(sum(v.values()) for v in final_stats.values())
    print(f"{'TOTAL IMAGES':<26} | {total_orig_all:<16} | {total_fin_all:<16}")
    print(f"{'CORRUPTED / SKIPPED FILES':<26} | {skipped_count:<16}")
    print(divider + "\n")


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.85,
    seed: int = SEED,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: Tuple[int, int] = DEFAULT_TILE_GRID_SIZE,
) -> None:
    """
    Executes the end-to-end dataset preparation pipeline:
    1. Reads and sanitizes raw input dataset (train, val, test).
    2. Isolates the original test set without leakage.
    3. Merges original train + val sets.
    4. Performs a stratified split into train_ratio (default 85%) and 1 - train_ratio (15%).
    5. Applies CLAHE enhancement to every image and writes to output_dir with identical structure.

    Args:
        input_dir: Root directory of raw chest_xray dataset containing train, val, test.
        output_dir: Destination directory for processed and enhanced YOLOv8 dataset.
        train_ratio: Ratio for training set (e.g., 0.85).
        seed: Random seed for reproducibility.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE grid size tuple.
    """
    set_seed(seed)
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")

    # Validate required subfolders
    for split in ["train", "val", "test"]:
        split_path = input_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Missing required subfolder: {split_path}")
        for cls in CLASSES:
            cls_path = split_path / cls
            if not cls_path.exists():
                raise FileNotFoundError(f"Missing required class subfolder: {cls_path}")

    print(f"[*] Input Directory  : {input_dir}")
    print(f"[*] Output Directory : {output_dir}")
    print(f"[*] Target Train/Val Split : {train_ratio * 100:.1f}% / {(1 - train_ratio) * 100:.1f}%")
    print(f"[*] CLAHE Configuration    : clipLimit={clip_limit}, tileGridSize={tile_grid_size}")
    print(f"[*] Random Seed            : {seed}\n")

    total_skipped = 0
    original_stats: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    # Step 1: Collect original test set (Strict Isolation)
    print("[1/4] Inspecting and sanitizing Test Set (Isolated)...")
    test_data: List[Tuple[Path, str]] = []
    for cls in CLASSES:
        paths, skipped = collect_images_from_dirs([input_dir / "test"], cls)
        total_skipped += skipped
        original_stats["test"][cls] = len(paths)
        for p in paths:
            test_data.append((p, cls))

    # Step 2: Merge original train + val sets
    print("[2/4] Merging and sanitizing Train + Val pools for Stratified Re-Split...")
    merged_train_val: List[Tuple[Path, str]] = []
    for cls in CLASSES:
        orig_train_paths, skipped_tr = collect_images_from_dirs([input_dir / "train"], cls)
        orig_val_paths, skipped_vl = collect_images_from_dirs([input_dir / "val"], cls)

        total_skipped += (skipped_tr + skipped_vl)
        original_stats["train"][cls] = len(orig_train_paths)
        original_stats["val"][cls] = len(orig_val_paths)

        for p in orig_train_paths + orig_val_paths:
            merged_train_val.append((p, cls))

    # Step 3: Stratified Re-split
    print("[3/4] Performing Stratified Split (85% Train / 15% Val)...")
    file_paths = [item[0] for item in merged_train_val]
    labels = [item[1] for item in merged_train_val]

    val_ratio = 1.0 - train_ratio
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths,
        labels,
        train_size=train_ratio,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    split_payloads = {
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
        "test": test_data,
    }

    # Step 4: Apply CLAHE and write to destination
    print("[4/4] Applying CLAHE enhancement and writing dataset...")
    final_stats: Dict[str, Dict[str, int]] = {
        "train": {cls: 0 for cls in CLASSES},
        "val": {cls: 0 for cls in CLASSES},
        "test": {cls: 0 for cls in CLASSES},
    }

    for split_name, items in split_payloads.items():
        desc = f"Processing {split_name.upper()} split"
        for src_path, cls_name in tqdm(items, desc=desc, unit="img"):
            dest_path = output_dir / split_name / cls_name / src_path.name
            success = process_and_save_image(
                src_path=src_path,
                dest_path=dest_path,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
            )
            if success:
                final_stats[split_name][cls_name] += 1
            else:
                total_skipped += 1

    # Print comprehensive diagnostic summary
    print_summary_table(original_stats, final_stats, total_skipped)
    print(f"[+] Dataset preprocessing completed successfully!")
    print(f"[+] Clean, CLAHE-enhanced dataset ready for YOLOv8 at: {output_dir}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical Preprocessing & Stratified Splitting for Chest X-Ray YOLOv8-cls"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="chest_xray",
        help="Path to raw chest_xray directory containing train, val, and test.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Path to output directory for balanced and CLAHE-enhanced dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.85,
        help="Ratio of merged train+val allocated to training (default: 0.85).",
    )
    parser.add_argument(
        "--clip_limit",
        type=float,
        default=DEFAULT_CLIP_LIMIT,
        help="CLAHE clip limit for contrast management (default: 2.0).",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=8,
        help="CLAHE tile grid size (default: 8 for (8, 8)).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Global random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        seed=args.seed,
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_size, args.tile_size),
    )
