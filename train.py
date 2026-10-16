"""
train.py - Huan luyen mo hinh YOLOv8 phan loai viem phoi tu anh X-quang.

Su dung YOLOv8-cls (Classification) de phan loai anh X-quang phoi
thanh 2 lop: NORMAL va PNEUMONIA.

Du lieu dau vao: Thu muc ./dataset/ da duoc tien xu ly bang CLAHE va chia tang
(stratified split 85/15) tu script src/preprocess.py.

Cach chay:
    python train.py
    python train.py --epochs 50 --model yolov8n-cls.pt --imgsz 224
    python train.py --model yolov8s-cls.pt --batch 8 --device 0
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# CAU HINH MAC DINH
# ============================================================
SEED = 42  # Hat giong ngau nhien co dinh de dam bao tai lap ket qua


def parse_args():
    """Thiet lap cac tham so dau vao cho qua trinh huan luyen."""
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
        default="dataset",
        help="Thu muc dataset da tien xu ly (chua train/val/test). Mac dinh: ./dataset/",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="So epoch huan luyen (mac dinh: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Kich thuoc anh dau vao (mac dinh: 224)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (mac dinh: 16, giam xuong 8 neu thieu VRAM)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="So epoch cho phep val_loss khong giam truoc khi dung som (Early Stopping)",
    )
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
        help="Thu muc goc luu ket qua huan luyen",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="pneumonia",
        help="Ten thu muc thi nghiem con ben trong --project",
    )
    return parser.parse_args()


def validate_dataset(data_path: Path) -> None:
    """
    Kiem tra cau truc thu muc dataset va in thong ke so luong anh.

    Cau truc bat buoc:
        dataset/
            train/ (NORMAL, PNEUMONIA)
            val/   (NORMAL, PNEUMONIA)
            test/  (NORMAL, PNEUMONIA)  [tuy chon nhung nen co]
    """
    # Kiem tra cac thu muc bat buoc cho viec huan luyen (train va val)
    required_dirs = [
        data_path / "train" / "NORMAL",
        data_path / "train" / "PNEUMONIA",
        data_path / "val" / "NORMAL",
        data_path / "val" / "PNEUMONIA",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc bat buoc: {d}\n"
                f"Hay chay 'python src/preprocess.py' truoc de tien xu ly du lieu."
            )

    # Dem va in thong ke so luong anh tung tap
    print("=" * 60)
    print("  THONG KE DU LIEU SAU TIEN XU LY (CLAHE + Stratified Split)")
    print("=" * 60)

    total_train = 0
    total_val = 0
    total_test = 0

    for cls in ["NORMAL", "PNEUMONIA"]:
        count = len(list((data_path / "train" / cls).glob("*.*")))
        total_train += count
        print(f"  Train - {cls:<12}: {count:>5} anh")

    for cls in ["NORMAL", "PNEUMONIA"]:
        count = len(list((data_path / "val" / cls).glob("*.*")))
        total_val += count
        print(f"  Val   - {cls:<12}: {count:>5} anh")

    if (data_path / "test").exists():
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = data_path / "test" / cls
            if cls_dir.exists():
                count = len(list(cls_dir.glob("*.*")))
                total_test += count
                print(f"  Test  - {cls:<12}: {count:>5} anh")

    print("-" * 60)
    print(f"  Tong Train : {total_train:>5} anh")
    print(f"  Tong Val   : {total_val:>5} anh")
    if total_test > 0:
        print(f"  Tong Test  : {total_test:>5} anh")
    print(f"  TONG CONG  : {total_train + total_val + total_test:>5} anh")
    print("=" * 60)


def train(args):
    """
    Thuc hien huan luyen mo hinh YOLOv8 Classification voi cac rang buoc y khoa.

    Cac rang buoc augmentation cho anh X-quang y te:
    -----------------------------------------------
    - fliplr=0.0 : KHONG lat ngang — bao toan vi tri bong tim (tim nam ben trai long nguc,
      lat ngang se lam dao nguoc cau truc giai phau, gay nham lan cho mo hinh).
    - flipud=0.0 : KHONG lat doc — phoi phai luon nam dung chieu trong luc,
      lat doc tao ra anh khong co trong thuc te lam sang.
    - hsv_h=0.0  : KHONG thay doi sac mau (hue) — anh X-quang chi co thang do xam (grayscale),
      khong ton tai thong tin mau sac.
    - hsv_s=0.0  : KHONG thay doi do bao hoa (saturation) — tuong tu, anh X-quang la grayscale.
    - hsv_v=0.2  : DUOC PHEP thay doi do sang +-20% — mo phong su khac biet cuong do tia X
      giua cac may chup khac nhau (hang san xuat, doi may, cai dat kVp/mAs khac nhau).
    - degrees=10.0 : DUOC PHEP xoay nhe +-10 do — mo phong truong hop benh nhan dung
      hoi nghieng khi chup (khong hoan toan thang dung truoc tam thu).
    """
    data_path = Path(args.data).resolve()

    print("\n" + "=" * 60)
    print("  BAT DAU HUAN LUYEN MO HINH YOLOV8 CLASSIFICATION")
    print("  Bai toan: Nhan dien Viem phoi (Pneumonia) tu anh X-quang")
    print("=" * 60)

    # Buoc 1: Kiem tra va thong ke dataset
    validate_dataset(data_path)

    # Buoc 2: Tai mo hinh pretrained
    print(f"\n[*] Dang tai mo hinh pretrained: {args.model}")
    model = YOLO(args.model)

    # Buoc 3: Hien thi toan bo cau hinh huan luyen
    print(f"\n{'─' * 60}")
    print(f"  CAU HINH HUAN LUYEN")
    print(f"{'─' * 60}")
    print(f"  Mo hinh       : {args.model}")
    print(f"  Dataset       : {data_path}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Image size    : {args.imgsz}x{args.imgsz}")
    print(f"  Patience (ES) : {args.patience} epochs")
    print(f"  Random Seed   : {SEED}")
    print(f"  Device        : {args.device if args.device else 'auto'}")
    print(f"{'─' * 60}")
    print(f"  RANG BUOC AUGMENTATION Y KHOA")
    print(f"{'─' * 60}")
    print(f"  fliplr  = 0.0  | KHONG lat ngang (bao toan vi tri bong tim)")
    print(f"  flipud  = 0.0  | KHONG lat doc (giu dung chieu trong luc)")
    print(f"  hsv_h   = 0.0  | KHONG doi sac mau (X-quang la grayscale)")
    print(f"  hsv_s   = 0.0  | KHONG doi do bao hoa (grayscale)")
    print(f"  hsv_v   = 0.2  | Cho phep dao dong do sang +-20%")
    print(f"  degrees = 10.0 | Cho phep xoay nhe +-10 do")
    print(f"{'─' * 60}\n")

    # Buoc 4: Thuc hien huan luyen
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=SEED,

        # ========================================================
        # EARLY STOPPING — Dung som neu khong co cai thien
        # ========================================================
        patience=args.patience,  # Dung huan luyen neu val_loss khong giam
                                 # sau 10 epoch lien tiep (chong overfitting)

        # ========================================================
        # RANG BUOC TANG CUONG DU LIEU CHO ANH Y TE (Medical Augmentation)
        # ========================================================

        # [KHONG LAT NGANG] Tim nam ben trai long nguc (situs solitus).
        # Lat ngang se dao nguoc toan bo cau truc giai phau phoi-tim,
        # khien mo hinh hoc sai vi tri cac ton thuong (VD: viem thuy giua phai
        # se bi hieu nham sang ben trai).
        fliplr=0.0,

        # [KHONG LAT DOC] Phoi phai luon nam dung chieu trong luc.
        # Lat doc tao ra hinh anh khong ton tai trong thuc te lam sang,
        # lam nhieu loat tin hieu hoc cua CNN.
        flipud=0.0,

        # [KHONG DOI SAC MAU] Anh X-quang la anh don kenh Grayscale,
        # khong chua thong tin mau sac (hue). Thay doi hue se khong co y nghia
        # va co the gay nhieu cho qua trinh trích xuat dac trung.
        hsv_h=0.0,

        # [KHONG DOI DO BAO HOA] Tuong tu nhu hsv_h, anh X-quang la grayscale
        # nen saturation (do bao hoa mau) luon bang 0 ve mat ky thuat.
        hsv_s=0.0,

        # [CHO PHEP DAO DONG DO SANG +-20%] Mo phong su khac biet cuong do
        # phoi nhiem (exposure) giua cac may chup X-quang khac nhau:
        # - May cu vs may moi co cuong do buc xa khac nhau
        # - Cai dat kVp/mAs thay doi giua cac lan chup
        # - BMI benh nhan khac nhau anh huong den do xuyen buc xa
        hsv_v=0.2,

        # [CHO PHEP XOAY NHE +-10 DO] Mo phong truong hop benh nhan
        # dung hoi nghieng/lech khi chup X-quang (khong hoan toan
        # thang dung truoc tam thu — pho bien trong thuc te lam sang).
        degrees=10.0,

        # Tang cuong hinh hoc bo sung (gia tri nho de giu nguyen context y khoa)
        translate=0.1,   # Dich chuyen nhe 10% — mo phong can chinh vi tri film
        scale=0.1,       # Phong to/thu nho nhe 10% — tranh cat mat goc phoi

        # ========================================================
        # TOI UU HOA (Optimizer & Learning Rate)
        # ========================================================
        optimizer="AdamW",       # AdamW: tot cho fine-tuning mo hinh pretrained
        lr0=0.001,               # Learning rate khoi tao
        lrf=0.01,                # He so giam lr cuoi cung (lr_final = lr0 * lrf)
        weight_decay=0.0005,     # L2 regularization chong overfitting
        warmup_epochs=3,         # Khoi dong nong 3 epoch dau (tang lr tu tu)

        # ========================================================
        # HIEN THI & LUU KET QUA
        # ========================================================
        verbose=True,    # Hien thi chi tiet trong console
        plots=True,      # Luu bieu do loss, accuracy, confusion matrix
    )

    # ============================================================
    # SAU HUAN LUYEN: Tim va sao chep trong so tot nhat (best.pt)
    # ============================================================
    print("\n" + "=" * 60)
    print("  HUAN LUYEN HOAN TAT — XU LY SAU HUAN LUYEN")
    print("=" * 60)

    # Tim duong dan best.pt tu ket qua huan luyen
    # Ultralytics luu tai: {project}/{name}/weights/best.pt
    best_src = Path(args.project) / args.name / "weights" / "best.pt"

    # Kiem tra truong hop thu muc bi long (bug cu cua phien ban Ultralytics truoc)
    if not best_src.exists():
        nested_path = (
            Path(args.project) / args.project / args.name / "weights" / "best.pt"
        )
        if nested_path.exists():
            best_src = nested_path
            print(f"[!] Phat hien thu muc long: {nested_path}")

    # Sao chep best.pt ra thu muc goc de tien su dung
    best_dst = Path("best_pneumonia_model.pt")
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"[+] Da sao chep trong so tot nhat:")
        print(f"    Nguon : {best_src.resolve()}")
        print(f"    Dich  : {best_dst.resolve()}")
    else:
        print("[!] Canh bao: Khong tim thay file best.pt sau huan luyen!")
        print(f"    Da kiem tra: {best_src}")

    # ============================================================
    # DANH GIA TREN TAP TEST (neu co)
    # ============================================================
    test_path = data_path / "test"
    if test_path.exists() and best_src.exists():
        print("\n" + "-" * 60)
        print("  DANH GIA NHANH TREN TAP TEST (624 anh)")
        print("-" * 60)
        best_model = YOLO(str(best_src))
        test_results = best_model.val(
            data=str(data_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
        )
        print(f"\n  Test Accuracy (Top-1): {test_results.top1:.4f}")
        print(f"  Test Accuracy (Top-5): {test_results.top5:.4f}")
    elif not test_path.exists():
        print("\n[*] Khong tim thay thu muc test — bo qua danh gia.")
    else:
        print("\n[*] Khong co best.pt — bo qua danh gia tren tap test.")

    # ============================================================
    # TONG KET CUOI CUNG
    # ============================================================
    print("\n" + "=" * 60)
    print("  TONG KET")
    print("=" * 60)
    print(f"  Ket qua huan luyen : {Path(args.project).resolve() / args.name}")
    print(f"  Trong so tot nhat  : {best_dst.resolve()}")
    print(f"  De danh gia chi tiet, chay: python evaluate.py --data {args.data}")
    print(f"  De chay web demo, chay   : python app.py")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
