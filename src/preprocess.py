"""
Pipeline Tiền Xử Lý Ảnh Y Tế Cho Mô Hình Phân Loại YOLOv8 (YOLOv8-Classification).
Chuyên biệt cho Bộ Dữ Liệu X-quang Ngực (Viêm Phổi) từ Kaggle.

Các Mục Tiêu Trọng Tâm:
1. Phân chia lại tập dữ liệu (Re-split): Gộp tập train và val gốc vốn bị lỗi (val chỉ có 16 ảnh),
   thực hiện phân chia tầng (stratified split) tỷ lệ 85% Train / 15% Validation để khắc phục
   triệt để vấn đề tập validation quá ít ảnh, đảm bảo tính đại diện thống kê.
2. Cô lập tập Test (Isolation): Bảo toàn nguyên vẹn và độc lập hoàn toàn tập test gốc (624 ảnh)
   để chống rò rỉ dữ liệu (data leakage).
3. Tăng cường ảnh y tế (CLAHE): Áp dụng cân bằng biểu đồ tần suất thích ứng giới hạn tương phản
   (CLAHE) trên ảnh mức xám (clipLimit=2.0, tileGridSize=(8,8)) nhằm làm rõ các tổn thương
   mờ đục phế trường (opacities) mà không gây khuếch đại nhiễu bức xạ nền.
4. Độ tin cậy & Ổn định: Tự động loại bỏ file hỏng, file 0-byte, file rác hệ điều hành,
   sử dụng pathlib và OpenCV với cơ chế bắt lỗi an toàn (try-except).
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
    # Cơ chế dự phòng khi môi trường chưa cài đặt thư viện tqdm
    def tqdm(iterable, desc="", **kwargs):
        if desc:
            print(f"[*] {desc}...")
        return iterable


# Cấu hình hạt giống ngẫu nhiên (Seed) & Các tham số tiền xử lý mặc định
SEED: int = 42
DEFAULT_CLIP_LIMIT: float = 2.0
DEFAULT_TILE_GRID_SIZE: Tuple[int, int] = (8, 8)
VALID_EXTENSIONS: set = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES: List[str] = ["NORMAL", "PNEUMONIA"]


def set_seed(seed: int = SEED) -> None:
    """Thiết lập hạt giống ngẫu nhiên toàn cục cho Python và NumPy nhằm đảm bảo khả năng tái lập kết quả."""
    random.seed(seed)
    np.random.seed(seed)


def is_valid_image(file_path: Path) -> bool:
    """
    Kiểm tra xem một file có phải là file ảnh hợp lệ, có thể đọc được và không bị rỗng hay không.

    Các bước kiểm tra:
    - Đường dẫn tồn tại và không phải file ẩn/file hệ thống (bỏ qua .DS_Store, ._* của macOS).
    - Phần mở rộng thuộc danh sách định dạng ảnh cho phép.
    - Kích thước file > 0 bytes.
    - OpenCV giải mã và đọc ảnh thành công.
    """
    try:
        # Bỏ qua các file ẩn và file metadata của macOS / hệ điều hành
        if file_path.name.startswith(".") or file_path.name.startswith("._"):
            return False

        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            return False

        if not file_path.is_file() or file_path.stat().st_size == 0:
            return False

        # Đọc thử một phần ảnh để xác thực tính toàn vẹn của phần tiêu đề (header)
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
    Áp dụng thuật toán cân bằng biểu đồ tần suất thích ứng giới hạn tương phản (CLAHE) cho ảnh.

    Cơ sở khoa học trong ảnh X-quang y tế:
    - Cân bằng biểu đồ tần suất toàn cục (Global Histogram Equalization) thường làm cháy sáng mô mềm
      và khuếch đại mạnh nhiễu tần số cao.
    - CLAHE chia ảnh thành các ô cục bộ theo lưới (tileGridSize, mặc định 8x8).
    - clipLimit=2.0 giới hạn độ dốc của hàm phân phối tích lũy (CDF) nhằm tránh phóng đại nhiễu bức xạ,
      từ đó làm nổi bật các vùng thâm nhiễm (infiltrates) và đông đặc (consolidations) phế trường,
      giúp mạng nơ-ron CNN trích xuất đặc trưng chính xác hơn.

    Tham số:
        image: Mảng NumPy chứa ảnh 1 kênh (grayscale) hoặc ảnh đa kênh.
        clip_limit: Ngưỡng giới hạn tương phản (mặc định 2.0).
        tile_grid_size: Kích thước lưới chia ô cục bộ (mặc định (8, 8)).

    Trả về:
        Mảng NumPy kiểu uint8 chứa ảnh đơn kênh (Grayscale) đã được tăng cường độ tương phản.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or None.")

    # Đảm bảo ảnh luôn ở dạng đơn kênh (Grayscale)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Đảm bảo độ sâu màu 8-bit (uint8)
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
    Đọc ảnh từ src_path ở dạng mức xám (grayscale), áp dụng tăng cường CLAHE và lưu vào dest_path.

    Được bọc trong khối try-except để đảm bảo script không bao giờ bị dừng đột ngột khi gặp lỗi I/O bất thường.
    """
    try:
        # Đọc trực tiếp dưới dạng ảnh mức xám
        img_gray = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return False

        # Áp dụng thuật toán tăng cường CLAHE
        enhanced_img = apply_clahe(img_gray, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        # Đảm bảo thư mục cha tại đích đến đã tồn tại
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Lưu ảnh đã được xử lý ra đĩa
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
    Thu thập và kiểm tra tính hợp lệ của tất cả các file ảnh từ danh sách thư mục cho một lớp cụ thể.

    Trả về:
        valid_paths: Danh sách đối tượng Path của các file ảnh hợp lệ.
        corrupted_count: Số lượng file bị hỏng / bỏ qua / không phải định dạng ảnh.
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
    """In bảng tổng kết thống kê chi tiết số lượng mẫu dữ liệu trước và sau tiền xử lý."""
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
    Thực hiện quy trình chuẩn bị và tiền xử lý toàn diện cho tập dữ liệu:
    1. Đọc và lọc sạch dữ liệu thô đầu vào (train, val, test).
    2. Cô lập hoàn toàn tập test gốc để tránh hiện tượng rò rỉ dữ liệu (data leakage).
    3. Hợp nhất hai tập train và val gốc thành một tập dữ liệu chung.
    4. Thực hiện chia tầng (stratified split) theo tỷ lệ train_ratio (mặc định 85%) và phần còn lại cho val (15%).
    5. Áp dụng tăng cường CLAHE cho từng ảnh và ghi ra output_dir với cấu trúc thư mục đồng nhất.

    Tham số:
        input_dir: Thư mục gốc chứa bộ dữ liệu thô chest_xray (gồm các thư mục con train, val, test).
        output_dir: Thư mục đích lưu trữ bộ dữ liệu đã cân bằng và tăng cường cho YOLOv8.
        train_ratio: Tỷ lệ phân bổ cho tập huấn luyện (mặc định 0.85 tương đương 85%).
        seed: Hạt giống ngẫu nhiên để tái lập kết quả phân chia.
        clip_limit: Ngưỡng giới hạn độ tương phản của CLAHE.
        tile_grid_size: Bộ giá trị kích thước lưới chia ô của CLAHE.
    """
    set_seed(seed)
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")

    # Kiểm tra sự tồn tại của các thư mục bắt buộc
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

    # Bước 1: Thu thập tập test gốc (Cô lập hoàn toàn, không trộn lẫn)
    print("[1/4] Inspecting and sanitizing Test Set (Isolated)...")
    test_data: List[Tuple[Path, str]] = []
    for cls in CLASSES:
        paths, skipped = collect_images_from_dirs([input_dir / "test"], cls)
        total_skipped += skipped
        original_stats["test"][cls] = len(paths)
        for p in paths:
            test_data.append((p, cls))

    # Bước 2: Gộp hai tập train và val gốc để chuẩn bị phân chia tầng lại
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

    # Bước 3: Phân chia tầng tập dữ liệu (85% Train / 15% Val)
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

    # Bước 4: Áp dụng thuật toán tăng cường CLAHE và lưu ảnh vào thư mục đích
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

    # In bảng thống kê chi tiết toàn bộ dữ liệu
    print_summary_table(original_stats, final_stats, total_skipped)
    print(f"[+] Dataset preprocessing completed successfully!")
    print(f"[+] Clean, CLAHE-enhanced dataset ready for YOLOv8 at: {output_dir}\n")


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh đầu vào."""
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
