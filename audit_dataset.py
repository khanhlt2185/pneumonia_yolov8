"""Kiểm tra bộ dữ liệu Chest X-Ray (Pneumonia) - CHỈ ĐỌC, không sửa dữ liệu.

Cách chạy (tại thư mục dự án):
    python audit_dataset.py
    python audit_dataset.py --root chest_xray --dataset dataset --csv audit_files.csv
"""
import argparse
import csv
import hashlib
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def patient_id(name: str) -> str:
    m = re.match(r"^(person\d+)_", name)
    if m:
        return m.group(1)
    m = re.search(r"(IM-\d+)", name)
    if m:
        return m.group(1)
    return "UNPARSED:" + name


def family(name: str, cls: str) -> str:
    """Họ tên file: NORMAL có hai họ (IM-... và NORMAL2-IM-...), PNEUMONIA là person."""
    if cls == "NORMAL":
        return "NORMAL2-IM" if name.startswith("NORMAL2-") else "IM"
    return "person"


def list_images(base: Path):
    """Trả về [(split, class, path)] cho base/<split>/<class>/."""
    out = []
    for s in SPLITS:
        for c in CLASSES:
            d = base / s / c
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in IMG_EXT:
                    out.append((s, c, p))
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_all(items, label):
    total = len(items)
    print(f"  Đang băm {total} ảnh ({label})...", flush=True)
    done = 0
    result = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for h in ex.map(lambda t: sha256(t[2]), items):
            result.append(h)
            done += 1
            if done % 1000 == 0:
                print(f"    {done}/{total}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="chest_xray")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--csv", default=None, help="ghi danh sách file + mã băm + mã bệnh nhân")
    a = ap.parse_args()
    root, dsdir = Path(a.root), Path(a.dataset)

    items = list_images(root)
    if not items:
        sys.exit(f"Không thấy ảnh trong {root}\\<train|val|test>\\<NORMAL|PNEUMONIA>")

    print("== 1. Số ảnh thư mục chính ==")
    cnt = defaultdict(int)
    for s, c, _ in items:
        cnt[(s, c)] += 1
    for s in SPLITS:
        print("  {:<6} NORMAL={:<5} PNEUMONIA={}".format(s, cnt[(s, "NORMAL")], cnt[(s, "PNEUMONIA")]))
    print("  Tổng:", len(items))

    hashes = hash_all(items, "thư mục chính")
    uniq = set(hashes)
    print(f"\n== 2. Ảnh trùng hệt (SHA-256) ==\n  Nội dung khác nhau: {len(uniq)} / {len(items)}")
    groups = defaultdict(list)
    for (s, c, p), h in zip(items, hashes):
        groups[h].append((s, c, p))
    dups = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dups if len({x[0] for x in g}) > 1]
    print(f"  Nhóm trùng: {len(dups)}  (trong đó nằm ở nhiều tập khác nhau: {len(cross)})")
    for g in dups[:10]:
        print("   -", " | ".join(f"{x[0]}/{x[1]}/{x[2].name}" for x in g))

    nested = root / "chest_xray"
    print("\n== 3. Thư mục lồng chest_xray/chest_xray ==")
    if nested.is_dir():
        n_items = list_images(nested)
        n_hashes = hash_all(n_items, "bản lồng")
        miss = [i for i, h in enumerate(n_hashes) if h not in uniq]
        print(f"  Số ảnh bản lồng: {len(n_items)}; ảnh KHÔNG có ở thư mục chính: {len(miss)}")
        for i in miss[:10]:
            print("   -", n_items[i][2])
        others = [p for p in nested.rglob("*") if p.is_file() and p.suffix.lower() not in IMG_EXT]
        print(f"  File không phải ảnh trong bản lồng: {len(others)}")
        for p in others[:10]:
            print("   -", p)
    else:
        print("  Không có thư mục lồng.")

    print("\n== 4. Mã bệnh nhân ==")
    pids = [patient_id(p.name) for _, _, p in items]
    unparsed = [(p.name) for (_, _, p), pid in zip(items, pids) if pid.startswith("UNPARSED:")]
    print(f"  Ảnh không tách được mã: {len(unparsed)}", unparsed[:5])
    for c in CLASSES:
        sel = [pid for (s, cc, _), pid in zip(items, pids) if cc == c]
        per = defaultdict(int)
        for x in sel:
            per[x] += 1
        if per:
            vals = list(per.values())
            print(f"  {c}: {len(sel)} ảnh, {len(per)} bệnh nhân, "
                  f"TB {statistics.mean(vals):.2f} ảnh/người, nhiều nhất {max(vals)}")
    both = {pid for (s, c, _), pid in zip(items, pids) if c == "NORMAL"} & \
           {pid for (s, c, _), pid in zip(items, pids) if c == "PNEUMONIA"}
    print(f"  Mã xuất hiện ở cả hai lớp: {len(both)}")

    tv = {pid for (s, _, _), pid in zip(items, pids) if s in ("train", "val")}
    te = {pid for (s, _, _), pid in zip(items, pids) if s == "test"}
    ov = tv & te
    te_imgs = sum(1 for (s, _, _), pid in zip(items, pids) if s == "test" and pid in ov)
    print(f"  Bệnh nhân trùng giữa (train+val) và test: {len(ov)} / {len(te)} bệnh nhân test"
          f" ({te_imgs} ảnh test thuộc nhóm trùng)")
    tr = {pid for (s, _, _), pid in zip(items, pids) if s == "train"}
    va = {pid for (s, _, _), pid in zip(items, pids) if s == "val"}
    print(f"  Bệnh nhân trùng giữa train và val gốc: {len(tr & va)} / {len(va)}")

    print("\n== 4b. Rò rỉ theo từng lớp và chi phí nếu loại bệnh nhân trùng khỏi train/val ==")
    rec = [(s, c, p.name, pid, family(p.name, c)) for (s, c, p), pid in zip(items, pids)]
    dev_all = [r for r in rec if r[0] in ("train", "val")]
    lost_total = 0
    for c in CLASSES:
        dev = [r for r in rec if r[1] == c and r[0] in ("train", "val")]
        tst = [r for r in rec if r[1] == c and r[0] == "test"]
        dev_ids = {r[3] for r in dev}
        tst_ids = {r[3] for r in tst}
        ov_c = dev_ids & tst_ids
        lost = sum(1 for r in dev if r[3] in ov_c)
        lost_total += lost
        aff = sum(1 for r in tst if r[3] in ov_c)
        print(f"  {c}: bệnh nhân test {len(tst_ids)}, trùng train+val {len(ov_c)}; "
              f"ảnh test thuộc nhóm trùng {aff}/{len(tst)}; "
              f"ảnh train+val của các bệnh nhân này {lost}/{len(dev)}")
    print(f"  Tổng ảnh train+val sẽ mất nếu loại bệnh nhân trùng test: {lost_total}/{len(dev_all)}"
          f" ({100.0 * lost_total / max(1, len(dev_all)):.1f}%)")
    fam_ids = defaultdict(set)
    for r in rec:
        if r[1] == "NORMAL":
            fam_ids[r[4]].add(r[3])
    if len(fam_ids) > 1:
        fams = sorted(fam_ids)
        shared = set.intersection(*[fam_ids[f] for f in fams])
        print("  Họ tên NORMAL: " + ", ".join(f"{f}={len(fam_ids[f])} mã" for f in fams)
              + f"; mã số dùng chung giữa các họ: {len(shared)} (có thể là trùng số giả)")
        for f in fams:
            d = {r[3] for r in rec if r[1] == "NORMAL" and r[4] == f and r[0] in ("train", "val")}
            t = {r[3] for r in rec if r[1] == "NORMAL" and r[4] == f and r[0] == "test"}
            print(f"    họ {f}: trùng train+val/test = {len(d & t)} / {len(t)} mã test")

    print("\n== 5. Mức rò rỉ của chia v1 (dataset/) ==")
    ds = list_images(dsdir)
    if ds:
        dtr = {patient_id(p.name) for s, _, p in ds if s == "train"}
        dva_items = [(patient_id(p.name)) for s, _, p in ds if s == "val"]
        shared = sum(1 for x in dva_items if x in dtr)
        print(f"  Ảnh val v1 thuộc bệnh nhân đã có ở train v1: {shared} / {len(dva_items)}")
        print(f"  Bệnh nhân val v1 trùng train v1: {len(set(dva_items) & dtr)} / {len(set(dva_items))}")
        print("  (Chỉ có nghĩa nếu bước preprocess giữ nguyên tên file gốc.)")
    else:
        print(f"  Không thấy {dsdir}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["split", "class", "file", "sha256", "patient_id"])
            for (s, c, p), h, pid in zip(items, hashes, pids):
                w.writerow([s, c, p.name, h, pid])
        print(f"\nĐã ghi {a.csv}")


if __name__ == "__main__":
    main()