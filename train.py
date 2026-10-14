"""
train.py - Huan luyen mo hinh YOLOv8 phan loai viem phoi tu anh X-quang.

Su dung YOLOv8-cls (Classification) de phan loai anh X-quang phoi
thanh 2 lop: NORMAL va PNEUMONIA.

Cach chay:
    python train.py
    python train.py --epochs 50 --model yolov8s-cls.pt --imgsz 224
"""

import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    # Thiet lap cac tham so dau vao cho qua trinh huan luyen
    parser = argparse.ArgumentParser(
        description="Huan luyen YOLOv8 Classification cho nhan dien viem phoi"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-cls.pt",
        help="Mo hinh YOLOv8-cls pretrained (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="chest_xray",
        help="Thu muc dataset (chua train/val/test)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="So epoch huan luyen")
    parser.add_argument(
        "--imgsz", type=int, default=224, help="Kich thuoc anh dau vao"
    )
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Thiet bi: rong (auto), 'cpu', '0' (GPU 0)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/classify",
        help="Thu muc luu ket qua",
    )
    parser.add_argument(
        "--name", type=str, default="pneumonia", help="Ten thu muc thi nghiem"
    )
    return parser.parse_args()


def prepare_dataset(data_path: str) -> str:
    """
    Kiem tra cau truc dataset va thong ke so luong anh tung tap.
    Yeu cau cau truc thu muc:
        dataset/
            train/ (NORMAL, PNEUMONIA)
            val/   (NORMAL, PNEUMONIA)
            test/  (NORMAL, PNEUMONIA)
    """
    data_path = Path(data_path)

    # Kiem tra cac thu muc bat buoc
    required_dirs = [
        data_path / "train" / "NORMAL",
        data_path / "train" / "PNEUMONIA",
        data_path / "val" / "NORMAL",
        data_path / "val" / "PNEUMONIA",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Khong tim thay thu muc: {d}")

    # Dem so luong anh trong cac tap
    train_normal = len(list((data_path / "train" / "NORMAL").glob("*.*")))
    train_pneumonia = len(list((data_path / "train" / "PNEUMONIA").glob("*.*")))
    val_normal = len(list((data_path / "val" / "NORMAL").glob("*.*")))
    val_pneumonia = len(list((data_path / "val" / "PNEUMONIA").glob("*.*")))

    print("=" * 60)
    print("THONG KE DATASET")
    print("=" * 60)
    print(f"  Train - NORMAL:    {train_normal:>5} anh")
    print(f"  Train - PNEUMONIA: {train_pneumonia:>5} anh")
    print(f"  Val   - NORMAL:    {val_normal:>5} anh")
    print(f"  Val   - PNEUMONIA: {val_pneumonia:>5} anh")

    if (data_path / "test").exists():
        test_normal = len(list((data_path / "test" / "NORMAL").glob("*.*")))
        test_pneumonia = len(list((data_path / "test" / "PNEUMONIA").glob("*.*")))
        print(f"  Test  - NORMAL:    {test_normal:>5} anh")
        print(f"  Test  - PNEUMONIA: {test_pneumonia:>5} anh")

    print(f"  Tong:              {train_normal + train_pneumonia + val_normal + val_pneumonia:>5} anh (train+val)")
    print("=" * 60)

    return str(data_path)


def train(args):
    """Thuc hien huan luyen mo hinh YOLOv8 Classification."""
    print("\n" + "=" * 60)
    print("BAT DAU HUAN LUYEN MO HINH YOLOV8 CLASSIFICATION")
    print("=" * 60)

    # Kiem tra dataset
    dataset_path = prepare_dataset(args.data)

    # Khoi tao mo hinh voi trong so pretrained
    print(f"\nDang tai mo hinh: {args.model}")
    model = YOLO(args.model)

    print(f"\nHuan luyen voi {args.epochs} epochs")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device: {args.device if args.device else 'auto'}")
    print()

    # Huan luyen voi cac thong so tang cuong du lieu va toi uu hoa
    results = model.train(
        data=dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Tang cuong du lieu (Data augmentation)
        hsv_h=0.0,      # X-quang la anh Grayscale, khong can map mau
        hsv_s=0.0,      # Khong can bo sung do bao hoa mau
        hsv_v=0.2,      # Thay doi do sang/toi 20% (mo phong cac loai may chup)
        degrees=10.0,   # Xoay nhe +-10 do
        translate=0.1,  # Dich chuyen nhe 10%
        scale=0.1,      # Giam scale xuong 10% de tranh cat mat goc phoi
        flipud=0.0,     # Tuyet doi khong lat doc
        fliplr=0.0,     # Tuyet doi khong lat ngang (Giu nguyen vi tri bong tim ben trai)
        # Toi uu hoa
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        verbose=True,
        plots=True,
    )

    # Danh gia tren tap test sau khi huan luyen
    print("\n" + "=" * 60)
    print("DANH GIA TREN TAP TEST")
    print("=" * 60)

    test_path = Path(args.data) / "test"
    if test_path.exists():
        # Tim file trong so tot nhat duoc luu
        best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
        if not best_model_path.exists():
            # Kiem tra thu muc long neu co
            nested_path = Path(args.project) / args.project / args.name / "weights" / "best.pt"
            if nested_path.exists():
                best_model_path = nested_path

        if best_model_path.exists():
            best_model = YOLO(str(best_model_path))
            test_results = best_model.val(
                data=dataset_path,
                split="test",
                imgsz=args.imgsz,
                batch=args.batch,
            )
            print(f"\nTest Accuracy (Top-1): {test_results.top1:.4f}")
            print(f"Test Accuracy (Top-5): {test_results.top5:.4f}")
        else:
            print("Khong tim thay best.pt, bo qua danh gia test.")
    else:
        print("Khong tim thay thu muc test, bo qua danh gia.")

    # Sao chep file trong so tot nhat ra thu muc goc
    best_src = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_src.exists():
        nested_path = Path(args.project) / args.project / args.name / "weights" / "best.pt"
        if nested_path.exists():
            best_src = nested_path

    best_dst = Path("best_pneumonia_model.pt")
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\nModel tot nhat da duoc luu tai: {best_dst.absolute()}")

    print("\n" + "=" * 60)
    print("HUAN LUYEN HOAN TAT")
    print(f"  Ket qua luu tai: {Path(args.project).absolute()}")
    print(f"  Model tot nhat:  {best_dst.absolute()}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
