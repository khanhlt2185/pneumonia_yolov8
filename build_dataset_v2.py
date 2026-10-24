"""
build_dataset_v2.py — Chia lại bộ dữ liệu chest_xray thành dataset_v2/.
CHỈ ĐỌC audit_files.csv, KHÔNG sửa dữ liệu gốc, KHÔNG ghi đè dataset/.

Cách chạy:
    python build_dataset_v2.py --dry-run          # xem số liệu, không copy
    python build_dataset_v2.py                     # tạo dataset_v2/
    python build_dataset_v2.py --force             # ghi đè nếu đã tồn tại

Phụ thuộc: scikit-learn (StratifiedGroupKFold), cv2 (CLAHE), stdlib.
Không tự cài thư viện — nếu thiếu sẽ báo lỗi rõ ràng.

Quyết định đã chốt (AGENTS.md):
- Giữ nguyên 624 ảnh test (không thay đổi).
- Loại khỏi train+val mọi bệnh nhân có mặt ở test:
    PNEUMONIA -> khóa nhóm person<ID>
    NORMAL    -> khóa nhóm IM-<số> (không phân biệt họ IM- hay NORMAL2-IM-)
- Loại ảnh trùng SHA-256 (giữ 1 ảnh, ưu tiên thứ tự tên file).
- Chia 80/20 theo nhóm bệnh nhân, phân tầng nhãn, seed cố định.
- Cộng 5 fold StratifiedGroupKFold để phục vụ cross-validation.
- CLAHE áp cho cả train/val và test (tái dùng apply_clahe() từ src/preprocess.py).
- Ghi vào thư mục tạm rồi rename — đảm bảo atomic.
- Bảo toàn tên file gốc.
"""

import argparse
import csv
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Kiểm tra phụ thuộc (không tự cài)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("[LỖI] Thiếu thư viện: numpy\n  Cài bằng: pip install numpy")

try:
    import cv2  # noqa: F401
except ImportError:
    sys.exit("[LỖI] Thiếu thư viện: opencv-python\n  Cài bằng: pip install opencv-python-headless")

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    sys.exit("[LỖI] Thiếu thư viện: scikit-learn\n  Cài bằng: pip install scikit-learn")

# Tái dùng đúng hàm CLAHE của v1 — không viết bản sao
try:
    from src.preprocess import apply_clahe
except ImportError:
    sys.exit(
        "[LỖI] Không import được src/preprocess.py\n"
        "  Chạy từ thư mục gốc dự án (có chứa src/)."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────────────────────────────────────
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
SEED = 42
N_FOLDS = 5
IMG_EXT = {".jpg", ".jpeg", ".png"}

# Số ảnh test cố định theo quyết định
EXPECTED_TEST = {"NORMAL": 234, "PNEUMONIA": 390}


# ──────────────────────────────────────────────────────────────────────────────
# Tiện ích
# ──────────────────────────────────────────────────────────────────────────────
def _sep(char="─", n=64):
    print(char * n, flush=True)


def _header(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")


def _step(n: int, total: int, msg: str):
    print(f"\n[{n}/{total}] {msg}", flush=True)


def _ok(msg: str):
    print(f"  OK  {msg}", flush=True)


def _warn(msg: str):
    print(f"  CANH BAO  {msg}", flush=True)


def _fail(msg: str):
    print(f"  THAT BAI: {msg}", flush=True)
    raise AssertionError(msg)


# ──────────────────────────────────────────────────────────────────────────────
# Đọc audit CSV
# ──────────────────────────────────────────────────────────────────────────────
def load_audit_csv(csv_path: Path) -> list:
    """Đọc audit_files.csv và trả về danh sách dict."""
    if not csv_path.exists():
        sys.exit(
            f"[LỖI] Không tìm thấy {csv_path}\n"
            "  Chạy: python audit_dataset.py --csv audit_files.csv"
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Bước 1: Tách test, lấy test_pids
# ──────────────────────────────────────────────────────────────────────────────
def extract_test_pids(rows: list) -> tuple:
    """Trả về (test_rows, devpool_rows, test_pids)."""
    test_rows = [r for r in rows if r["split"] == "test"]
    dev_rows  = [r for r in rows if r["split"] != "test"]
    test_pids = {r["patient_id"] for r in test_rows}
    return test_rows, dev_rows, test_pids


# ──────────────────────────────────────────────────────────────────────────────
# Bước 2: Loại bệnh nhân test khỏi train+val
# ──────────────────────────────────────────────────────────────────────────────
def remove_test_patients(dev_rows: list, test_pids: set) -> tuple:
    """Loại mọi bản ghi có patient_id in test_pids. Trả về (clean_rows, n_removed)."""
    clean   = [r for r in dev_rows if r["patient_id"] not in test_pids]
    removed = len(dev_rows) - len(clean)
    return clean, removed


# ──────────────────────────────────────────────────────────────────────────────
# Bước 3: Loại trùng SHA-256
# ──────────────────────────────────────────────────────────────────────────────
def dedup_sha256(rows: list) -> tuple:
    """
    Với mỗi nhóm SHA-256 >= 2 ảnh, giữ 1 ảnh (chọn lexicographically
    theo tên file — deterministic). Trả về (deduped_rows, n_dropped, n_dup_groups).
    """
    groups = defaultdict(list)
    for r in rows:
        groups[r["sha256"]].append(r)

    kept = []
    n_dropped = 0
    n_dup_groups = 0
    for sha, grp in groups.items():
        if len(grp) == 1:
            kept.append(grp[0])
        else:
            n_dup_groups += 1
            # Chọn ảnh có tên file nhỏ nhất (lexicographic) — reproducible
            chosen = min(grp, key=lambda r: r["file"])
            kept.append(chosen)
            n_dropped += len(grp) - 1

    return kept, n_dropped, n_dup_groups


# ──────────────────────────────────────────────────────────────────────────────
# Bước 4: Chia 80/20 bằng StratifiedGroupKFold
# ──────────────────────────────────────────────────────────────────────────────
def split_train_val(pool: list, fold_val: int = 0, n_splits: int = N_FOLDS, seed: int = SEED) -> tuple:
    """
    Dùng StratifiedGroupKFold để chia pool thành train/val.
    groups   = patient_id
    stratify = class (0=NORMAL, 1=PNEUMONIA)
    Fold `fold_val` -> val, còn lại -> train.
    Trả về (train_rows, val_rows, fold_assignments: idx->fold).
    """
    labels  = np.array([0 if r["class"] == "NORMAL" else 1 for r in pool])
    groups  = np.array([r["patient_id"] for r in pool])
    indices = np.arange(len(pool))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Gán fold cho từng mẫu
    fold_of = {}  # index -> fold id
    for fold_id, (_, val_idx) in enumerate(sgkf.split(indices, labels, groups)):
        for i in val_idx:
            fold_of[i] = fold_id

    train_rows = [pool[i] for i in range(len(pool)) if fold_of.get(i) != fold_val]
    val_rows   = [pool[i] for i in range(len(pool)) if fold_of.get(i) == fold_val]

    return train_rows, val_rows, fold_of


# ──────────────────────────────────────────────────────────────────────────────
# Bước 5: Xác minh / phân tích tỷ lệ lớp
# ──────────────────────────────────────────────────────────────────────────────
def class_ratio(rows: list) -> float:
    """Trả về tỷ lệ PNEUMONIA / tổng."""
    total = len(rows)
    if total == 0:
        return 0.0
    n_pneu = sum(1 for r in rows if r["class"] == "PNEUMONIA")
    return n_pneu / total


def count_by_class(rows: list) -> dict:
    cnt = defaultdict(int)
    for r in rows:
        cnt[r["class"]] += 1
    return dict(cnt)


# ──────────────────────────────────────────────────────────────────────────────
# Bước 6: Kiểm tra tự động trên kế hoạch (KT-1 đến KT-6)
# ──────────────────────────────────────────────────────────────────────────────
def run_plan_assertions(
    train_rows: list,
    val_rows: list,
    test_rows: list,
    orig_test_rows: list,
):
    """Chạy KT-1 đến KT-6 trên kế hoạch phân chia (trước khi copy)."""
    print("\n" + "─" * 64)
    print("  KIEM TRA KE HOACH (KT-1 den KT-6)")
    print("─" * 64)

    # --- Kiểm tra 1: Không bệnh nhân train ∩ val ---
    train_pids = {r["patient_id"] for r in train_rows}
    val_pids   = {r["patient_id"] for r in val_rows}
    test_pids  = {r["patient_id"] for r in test_rows}
    overlap_tv = train_pids & val_pids
    if overlap_tv:
        _fail(f"KT-1 — Benh nhan trung train^val: {len(overlap_tv)} ma, VD: {list(overlap_tv)[:3]}")
    else:
        _ok(f"KT-1 — Khong benh nhan nao vua o train vua o val (|train_pids|={len(train_pids)}, |val_pids|={len(val_pids)})")

    # --- Kiểm tra 2: (train ∪ val) ∩ test = ∅ ---
    overlap_devtest = (train_pids | val_pids) & test_pids
    if overlap_devtest:
        _fail(f"KT-2 — Benh nhan trung (train+val)^test: {len(overlap_devtest)} ma, VD: {list(overlap_devtest)[:3]}")
    else:
        _ok("KT-2 — Khong benh nhan nao vua o train/val vua o test")

    # --- Kiểm tra 3: SHA-256 không trùng giữa các tập ---
    sha_train = {r["sha256"] for r in train_rows}
    sha_val   = {r["sha256"] for r in val_rows}
    sha_test  = {r["sha256"] for r in test_rows}
    cross_tv   = sha_train & sha_val
    cross_ttte = sha_train & sha_test
    cross_vte  = sha_val   & sha_test
    if cross_tv or cross_ttte or cross_vte:
        msg = (
            f"train^val={len(cross_tv)}, "
            f"train^test={len(cross_ttte)}, "
            f"val^test={len(cross_vte)}"
        )
        _fail(f"KT-3 — SHA-256 trung giua tap: {msg}")
    else:
        _ok("KT-3 — Khong co SHA-256 nao xuat hien o hai tap khac nhau")

    # --- Kiểm tra 4a: Tập test đúng 234 NORMAL + 390 PNEUMONIA ---
    tc = count_by_class(test_rows)
    for cls, expected in EXPECTED_TEST.items():
        got = tc.get(cls, 0)
        if got != expected:
            _fail(f"KT-4a — Test {cls}: mong {expected}, thuc te {got}")
    _ok(f"KT-4a — Test dung {EXPECTED_TEST['NORMAL']} NORMAL + {EXPECTED_TEST['PNEUMONIA']} PNEUMONIA")

    # --- Kiểm tra 4b: Tên file test trong manifest = đúng 624 file test gốc ---
    orig_test_names = {r["file"] for r in orig_test_rows}
    new_test_names  = {r["file"] for r in test_rows}
    if orig_test_names != new_test_names:
        only_orig = orig_test_names - new_test_names
        only_new  = new_test_names - orig_test_names
        _fail(
            f"KT-4b — Ten file test khac voi tap goc: "
            f"chi-goc={len(only_orig)}, chi-moi={len(only_new)}"
        )
    _ok("KT-4b — Tap ten file test khop dung voi 624 file test goc")

    # --- Kiểm tra 5: Tỷ lệ lớp train/val lệch < 2 điểm phần trăm ---
    ratio_train = class_ratio(train_rows)
    ratio_val   = class_ratio(val_rows)
    diff_pp = abs(ratio_train - ratio_val) * 100
    if diff_pp >= 2.0:
        _fail(
            f"KT-5 — Ti le PNEUMONIA: train={ratio_train:.3f}, val={ratio_val:.3f}, "
            f"lech={diff_pp:.2f}pp >= 2pp"
        )
    _ok(
        f"KT-5 — Ti le PNEUMONIA: train={ratio_train*100:.1f}%, val={ratio_val*100:.1f}%, "
        f"lech={diff_pp:.2f}pp < 2pp"
    )

    # --- Kiểm tra 6: Cặp (class, file) không trùng trong toàn manifest ---
    all_rows = train_rows + val_rows + test_rows
    seen_pairs = set()
    dup_pairs = []
    for r in all_rows:
        pair = (r["class"], r["file"])
        if pair in seen_pairs:
            dup_pairs.append(pair)
        seen_pairs.add(pair)
    if dup_pairs:
        _fail(f"KT-6 — Cap (class, file) trung: {len(dup_pairs)} cap, VD: {dup_pairs[:3]}")
    _ok(f"KT-6 — Khong cap (class, file) nao trung trong manifest ({len(all_rows)} ban ghi)")


# ──────────────────────────────────────────────────────────────────────────────
# Bước 7 (hậu copy): Kiểm tra file thật trên thư mục tạm (KT-7 đến KT-9)
# ──────────────────────────────────────────────────────────────────────────────
def run_post_copy_assertions(all_rows: list, target_dir: Path):
    """
    Kiem tra tren thu muc tam SAU khi copy (buoc 7) nhung TRUOC khi rename thanh dataset_v2:
    - KT-7: Moi file trong manifest phai ton tai va doc duoc bang cv2.imread.
    - KT-8: Tong so file tren dia khop dung so dong manifest.
    - KT-9: Khong co bat ky file rac nao (.DS_Store, ._, __MACOSX).
    Raise AssertionError neu bat ky kiem tra nao that bai.
    """
    print("\n" + "─" * 64)
    print(f"  KIEM TRA FILE THUC TE TREN THU MUC TAM (KT-7 den KT-9)")
    print("─" * 64)

    # --- KT-7: Mọi file trong manifest tồn tại & đọc được bằng cv2.imread ---
    print("  KT-7 — Kiem tra moi file trong manifest ton tai va doc duoc bang cv2.imread...", flush=True)
    missing_files = []
    unreadable_files = []
    for r in all_rows:
        dest = target_dir / r["split"] / r["class"] / r["file"]
        if not dest.is_file():
            missing_files.append(str(dest))
        else:
            img = cv2.imread(str(dest))
            if img is None:
                unreadable_files.append(str(dest))

    if missing_files or unreadable_files:
        err_parts = []
        if missing_files:
            err_parts.append(f"{len(missing_files)} file khong ton tai (VD: {missing_files[:3]})")
        if unreadable_files:
            err_parts.append(f"{len(unreadable_files)} file khong doc duoc bang cv2.imread (VD: {unreadable_files[:3]})")
        _fail(f"KT-7 — " + "; ".join(err_parts))
    _ok(f"KT-7 — Tat ca {len(all_rows)} file trong manifest ton tai va doc duoc bang cv2.imread")

    # --- KT-8: Tổng file trong thư mục tạm khớp số dòng manifest ---
    print("  KT-8 — Kiem tra tong file tren dia khop manifest...", flush=True)
    actual_files = [
        p for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT
    ]
    if len(actual_files) != len(all_rows):
        _fail(
            f"KT-8 — Tong file tren dia ({len(actual_files)}) != "
            f"dong manifest ({len(all_rows)})"
        )
    _ok(f"KT-8 — Tong file tren dia = {len(actual_files)} = dong manifest")

    # --- KT-9: Không có file rác ---
    print("  KT-9 — Kiem tra file rac...", flush=True)
    junk = []
    for p in target_dir.rglob("*"):
        n = p.name
        if n.startswith("._") or n == ".DS_Store" or "__MACOSX" in str(p):
            junk.append(str(p))
    if junk:
        _fail(f"KT-9 — File rac tim thay: {len(junk)}, VD: {junk[:3]}")
    _ok("KT-9 — Khong co file rac (.DS_Store, ._, __MACOSX)")


# ──────────────────────────────────────────────────────────────────────────────
# Copy + CLAHE
# ──────────────────────────────────────────────────────────────────────────────
def copy_with_clahe(rows: list, split_name: str, root_dir: Path, dest_dir: Path) -> int:
    """
    Copy anh tu chest_xray/<orig_split>/<class>/<file> sang dest_dir/<split>/<class>/<file>,
    ap dung CLAHE (tai dung apply_clahe() tu src/preprocess.py).
    Tra ve so anh copy thanh cong.
    """
    ok_count = 0
    for r in rows:
        src = root_dir / r["orig_split"] / r["class"] / r["file"]
        dst = dest_dir / split_name / r["class"] / r["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            _warn(f"File nguon khong ton tai: {src}")
            continue

        img_gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            _warn(f"Khong doc duoc anh: {src}")
            continue

        enhanced = apply_clahe(img_gray)
        if not cv2.imwrite(str(dst), enhanced):
            _warn(f"Khong luu duoc anh: {dst}")
            continue

        ok_count += 1
    return ok_count


# ──────────────────────────────────────────────────────────────────────────────
# Xuất manifest CSV
# ──────────────────────────────────────────────────────────────────────────────
def write_manifest(
    train_rows: list,
    val_rows: list,
    test_rows: list,
    fold_of: dict,
    pool: list,
    manifest_path: Path,
):
    """Xuat manifest CSV voi cot: file, sha256, patient_id, class, split, fold, orig_split, clahe."""
    # Xây bản đồ (sha256, file) -> fold cho pool (train+val sau khi dedup)
    sha_file_to_fold = {}
    for i, r in enumerate(pool):
        sha_file_to_fold[(r["sha256"], r["file"])] = fold_of.get(i, -1)

    rows_out = []

    # Pool (train + val)
    for split_name, split_rows in [("train", train_rows), ("val", val_rows)]:
        for r in split_rows:
            fold_id = sha_file_to_fold.get((r["sha256"], r["file"]), -1)
            rows_out.append({
                "file":       r["file"],
                "sha256":     r["sha256"],
                "patient_id": r["patient_id"],
                "class":      r["class"],
                "split":      split_name,
                "fold":       fold_id,
                "orig_split": r["orig_split"],
                "clahe":      "true",
            })

    # Test
    for r in test_rows:
        rows_out.append({
            "file":       r["file"],
            "sha256":     r["sha256"],
            "patient_id": r["patient_id"],
            "class":      r["class"],
            "split":      "test",
            "fold":       -1,
            "orig_split": r["orig_split"],
            "clahe":      "true",
        })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "sha256", "patient_id", "class", "split", "fold", "orig_split", "clahe"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


# ──────────────────────────────────────────────────────────────────────────────
# In thống kê
# ──────────────────────────────────────────────────────────────────────────────
def print_stats(train_rows: list, val_rows: list, test_rows: list):
    _sep("─")
    print("  THONG KE PHAN CHIA")
    _sep("─")
    for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        cnt   = count_by_class(rows)
        total = len(rows)
        n_pneu = cnt.get("PNEUMONIA", 0)
        pct = n_pneu / total * 100 if total else 0
        print(
            f"  {split_name:<6}  NORMAL={cnt.get('NORMAL', 0):<5}  "
            f"PNEUMONIA={n_pneu:<5}  Tong={total:<5}  "
            f"(%PNEUMONIA={pct:.1f}%)"
        )
    # Số bệnh nhân
    train_pids = {r["patient_id"] for r in train_rows}
    val_pids   = {r["patient_id"] for r in val_rows}
    test_pids  = {r["patient_id"] for r in test_rows}
    print(f"\n  Benh nhan: train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")
    _sep("─")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(
        description="Chia lai bo du lieu X-quang phoi thanh dataset_v2/."
    )
    ap.add_argument("--csv",      default="audit_files.csv",         help="Duong dan audit_files.csv")
    ap.add_argument("--root",     default="chest_xray",              help="Thu muc chest_xray/ goc")
    ap.add_argument("--out",      default="dataset_v2",              help="Thu muc dich")
    ap.add_argument("--manifest", default="dataset_v2_manifest.csv", help="Manifest CSV dau ra")
    ap.add_argument("--fold-val", type=int, default=0,               help="Fold dung lam val (0-4)")
    ap.add_argument("--seed",     type=int, default=SEED,            help="Random seed")
    ap.add_argument("--dry-run",  action="store_true",               help="Chi in so lieu, khong copy")
    ap.add_argument("--force",    action="store_true",               help="Ghi de neu --out da ton tai")
    return ap.parse_args()


def main():
    args = parse_args()
    csv_path      = Path(args.csv)
    root_dir      = Path(args.root)
    out_dir       = Path(args.out)
    manifest_path = Path(args.manifest)

    _header("BUILD DATASET V2 — Chia lai du lieu X-quang phoi")
    print(f"  Che do   : {'DRY-RUN (khong copy)' if args.dry_run else 'THUC THI (copy anh)'}")
    print(f"  CSV audit: {csv_path}")
    print(f"  Root goc : {root_dir}")
    print(f"  Dau ra   : {out_dir}")
    print(f"  Manifest : {manifest_path}")
    print(f"  Fold val : {args.fold_val} / {N_FOLDS}")
    print(f"  Seed     : {args.seed}")

    # Kiểm tra root tồn tại
    if not root_dir.exists():
        sys.exit(f"[LOI] Khong tim thay thu muc nguon: {root_dir}")

    # Kiểm tra out đã tồn tại
    if out_dir.exists() and not args.dry_run:
        if not args.force:
            sys.exit(
                f"[LOI] Thu muc dich '{out_dir}' da ton tai.\n"
                "  Them --force de ghi de, hoac xoa thu muc truoc."
            )
        else:
            _warn(f"--force: se xoa va ghi de {out_dir}")

    # ────────────────────────────────────────────────────────────────────────
    TOTAL_STEPS = 7
    t0 = time.perf_counter()

    # BƯỚC 1 — Đọc CSV
    _step(1, TOTAL_STEPS, "Doc audit_files.csv")
    all_rows = load_audit_csv(csv_path)
    # Gán orig_split từ cột split gốc của audit
    for r in all_rows:
        r["orig_split"] = r["split"]
    print(f"  Doc {len(all_rows)} ban ghi tu {csv_path}")

    # BƯỚC 2 — Tách test, lấy test_pids
    _step(2, TOTAL_STEPS, "Tach tap test & xac dinh test_pids")
    test_rows, dev_rows, test_pids = extract_test_pids(all_rows)
    orig_test_rows = list(test_rows)  # bản sao để KT-4b
    tc = count_by_class(test_rows)
    print(f"  Test     : {tc.get('NORMAL', 0)} NORMAL + {tc.get('PNEUMONIA', 0)} PNEUMONIA = {len(test_rows)} anh")
    print(f"  test_pids: {len(test_pids)} ma benh nhan")
    print(f"  Dev pool : {len(dev_rows)} anh (train+val goc)")

    # BƯỚC 3 — Loại bệnh nhân test
    _step(3, TOTAL_STEPS, "Loai benh nhan test khoi train+val")
    clean_rows, n_removed = remove_test_patients(dev_rows, test_pids)
    nc = count_by_class(clean_rows)
    print(f"  Loai bo  : {n_removed} anh (benh nhan trung test)")
    print(f"  Con lai  : {len(clean_rows)} anh  (NORMAL={nc.get('NORMAL', 0)}, PNEUMONIA={nc.get('PNEUMONIA', 0)})")

    # BƯỚC 4 — Loại trùng SHA-256
    _step(4, TOTAL_STEPS, "Loai anh trung SHA-256")
    pool, n_dropped, n_dup_groups = dedup_sha256(clean_rows)
    pc = count_by_class(pool)
    print(f"  Nhom trung trong pool sach: {n_dup_groups} nhom")
    print(f"  Anh loai bo (du trung)    : {n_dropped} anh")
    print(f"  Pool sau dedup            : {len(pool)} anh  (NORMAL={pc.get('NORMAL', 0)}, PNEUMONIA={pc.get('PNEUMONIA', 0)})")

    # BƯỚC 5 — Chia 80/20
    _step(5, TOTAL_STEPS, f"Chia train/val bang StratifiedGroupKFold(n={N_FOLDS}, seed={args.seed})")
    train_rows, val_rows, fold_of = split_train_val(
        pool, fold_val=args.fold_val, n_splits=N_FOLDS, seed=args.seed
    )
    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"
    for r in test_rows:
        r["split"] = "test"
    print_stats(train_rows, val_rows, test_rows)

    # BƯỚC 6 — Kiểm tra tự động trên kế hoạch (KT-1 đến KT-6)
    _step(6, TOTAL_STEPS, "Kiem tra tu dong tren ke hoach (KT-1 den KT-6)")
    run_plan_assertions(
        train_rows, val_rows, test_rows, orig_test_rows,
    )

    if args.dry_run:
        elapsed = time.perf_counter() - t0
        print(f"\n{'=' * 64}")
        print(f"  DRY-RUN hoan tat ({elapsed:.1f}s) — khong co file nao duoc copy.")
        print(f"  KT-1 den KT-6 deu PASS tren ke hoach chia.")
        print(f"  (KT-7, KT-8, KT-9 se duoc kiem tra tren thu muc tam sau khi copy)")
        print(f"  Chay lai khong co --dry-run de tao {out_dir}/.")
        print(f"{'=' * 64}")
        return

    # BƯỚC 7 — Copy + CLAHE (ghi vào thư mục tạm, kiểm tra KT-7..9, rồi rename atomic)
    _step(7, TOTAL_STEPS, f"Copy + CLAHE vao thu muc tam, kiem tra va doi ten thanh {out_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="_dataset_v2_tmp_", dir=out_dir.parent))
    print(f"  Thu muc tam: {tmp_dir}")

    try:
        total_copied = 0
        for split_name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
            n = copy_with_clahe(rows, split_name, root_dir, tmp_dir)
            cnt = count_by_class(rows)
            print(
                f"  {split_name:<6}: {n}/{len(rows)} anh copy OK  "
                f"(NORMAL={cnt.get('NORMAL', 0)}, PNEUMONIA={cnt.get('PNEUMONIA', 0)})"
            )
            total_copied += n

        # KIỂM TRA FILE THẬT TRÊN THƯ MỤC TẠM (KT-7, KT-8, KT-9)
        # Chạy SAU bước copy nhưng TRƯỚC khi rename
        all_manifest_rows = train_rows + val_rows + test_rows
        run_post_copy_assertions(all_manifest_rows, tmp_dir)

        # Nếu out_dir đã tồn tại và có --force thì xóa thư mục cũ trước khi rename
        if out_dir.exists():
            if args.force:
                shutil.rmtree(out_dir)
                print(f"  Da xoa {out_dir} cu (--force)")
            else:
                raise FileExistsError(
                    f"Thu muc dich '{out_dir}' da ton tai. Dung --force de ghi de."
                )

        # Rename tmp -> out_dir (atomic trên cùng ổ đĩa)
        tmp_dir.rename(out_dir)
        print(f"  Rename {tmp_dir.name} -> {out_dir}")

        # Xuất manifest
        write_manifest(train_rows, val_rows, test_rows, fold_of, pool, manifest_path)
        total_manifest = len(all_manifest_rows)
        print(f"\n  OK  Manifest da ghi: {manifest_path} ({total_manifest} dong)")

    except Exception as exc:
        # Nếu kiểm tra thất bại hoặc có lỗi: xóa thư mục tạm, không tạo dataset_v2, và thoát với lỗi rõ ràng
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  Da xoa thu muc tam {tmp_dir} do kiem tra/copy that bai.")
        sys.exit(f"\n[LỖI] Qua trinh tao {out_dir} that bai: {exc}")

    elapsed = time.perf_counter() - t0
    _header(f"HOAN TAT ({elapsed:.1f}s) — {total_copied} anh da copy")
    print(f"  Dataset v2 : {out_dir.resolve()}")
    print(f"  Manifest   : {manifest_path.resolve()}")
    print(f"  De train   : python train.py --data {out_dir}")


if __name__ == "__main__":
    main()
