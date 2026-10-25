"""Thống kê đặc điểm ảnh gốc để tìm lệch phân bố giữa train/val/test - CHỈ ĐỌC.

Đọc ảnh gốc trong chest_xray/<orig_split>/<lớp>/<file> (trước CLAHE) theo dataset_v2_manifest.csv.
Trả lời hai câu hỏi:
  A. Chỉ dựa vào đặc điểm ảnh (kích thước, độ sáng, độ tương phản, viền tối, dung lượng file),
     có đoán được nhãn không? AUC cao nghĩa là có "đường tắt" không liên quan đến bệnh lý.
  B. Ảnh BÌNH THƯỜNG của test có khác ảnh bình thường của train/val về các đặc điểm đó không?
     (AUC phân biệt train+val với test; ~0.5 là không khác, cao là có lệch phân bố)

Cách chạy: python image_stats.py
"""
import argparse
import csv
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

FEATS = ["width", "height", "aspect", "mean", "std", "dark_frac", "file_kb"]


def group_of(name, cls):
    if cls == "NORMAL":
        return "NORMAL/NORMAL2-IM" if name.startswith("NORMAL2-") else "NORMAL/IM"
    return "PNEUMONIA"


def read_stats(job):
    path = job["path"]
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    return dict(width=w, height=h, aspect=w / h, mean=float(img.mean()), std=float(img.std()),
                dark_frac=float((img < 15).mean()), file_kb=os.path.getsize(path) / 1024.0)


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="dataset_v2_manifest.csv")
    ap.add_argument("--root", default="chest_xray")
    ap.add_argument("--out", default="evaluation_results/analysis")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    jobs = [dict(path=Path(a.root) / r["orig_split"] / r["class"] / r["file"], r=r) for r in rows]
    missing = [j["path"] for j in jobs if not j["path"].is_file()]
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} ảnh gốc, ví dụ: {missing[:3]}")
    print(f"Đang đọc {len(jobs)} ảnh gốc...", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (j, s) in enumerate(zip(jobs, ex.map(read_stats, jobs)), 1):
            if s is None:
                raise SystemExit(f"Không đọc được: {j['path']}")
            r = j["r"]
            s.update(split=r["split"], cls=r["class"], file=r["file"], pid=r["patient_id"],
                     group=group_of(r["file"], r["class"]))
            recs.append(s)
            if i % 1000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    out_csv = Path(a.out) / "image_stats.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "group", "file", "patient_id"] + FEATS)
        for s in recs:
            w.writerow([s["split"], s["cls"], s["group"], s["file"], s["pid"]] + [f"{s[k]:.4f}" for k in FEATS])
    print(f"Đã ghi {out_csv}")

    print("\n== Thống kê trung bình (độ lệch chuẩn) theo tập và nhóm ==")
    print(f"{'tập / nhóm':<32}{'n':>5}  " + "  ".join(f"{k:>16}" for k in FEATS))
    for split in ("train", "val", "test"):
        for g in ("NORMAL/IM", "NORMAL/NORMAL2-IM", "PNEUMONIA"):
            sel = [s for s in recs if s["split"] == split and s["group"] == g]
            if not sel:
                continue
            cells = []
            for k in FEATS:
                v = np.array([s[k] for s in sel])
                cells.append(f"{v.mean():8.2f} ({v.std():6.2f})")
            print(f"{split + ' / ' + g:<32}{len(sel):>5}  " + "  ".join(cells))

    X = np.array([[s[k] for k in FEATS] for s in recs])
    cls = np.array([1 if s["cls"] == "PNEUMONIA" else 0 for s in recs])
    split = np.array([s["split"] for s in recs])
    pid = np.array([s["pid"] for s in recs])
    rf = lambda: RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42,
                                        class_weight="balanced", n_jobs=-1)

    print("\n== A. Đoán NHÃN chỉ từ đặc điểm ảnh (huấn luyện trên train) ==")
    tr = split == "train"
    m = rf().fit(X[tr], cls[tr])
    for sp in ("val", "test"):
        idx = split == sp
        print(f"  AUC {sp}: {auc(cls[idx], m.predict_proba(X[idx])[:, 1]):.4f}   (0.5 = không đoán được, 1.0 = đoán hoàn hảo)")
    imp = sorted(zip(FEATS, m.feature_importances_), key=lambda t: -t[1])
    print("  Đặc điểm quan trọng nhất:", ", ".join(f"{k}={v:.2f}" for k, v in imp[:4]))

    print("\n== B. Ảnh BÌNH THƯỜNG: test có khác train+val không? (AUC chéo theo bệnh nhân) ==")
    for label, mask in (("tất cả ảnh bình thường", cls == 0),
                        ("chỉ họ NORMAL2-IM", np.array([s["group"] == "NORMAL/NORMAL2-IM" for s in recs])),
                        ("chỉ họ IM", np.array([s["group"] == "NORMAL/IM" for s in recs])),
                        ("ảnh VIÊM PHỔI (đối chứng)", cls == 1)):
        idx = np.where(mask)[0]
        dom = (split[idx] == "test").astype(int)
        if dom.sum() < 10 or (1 - dom).sum() < 10:
            continue
        oof = np.zeros(len(idx))
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for trn, tst in skf.split(X[idx], dom, groups=pid[idx]):
            oof[tst] = rf().fit(X[idx][trn], dom[trn]).predict_proba(X[idx][tst])[:, 1]
        print(f"  {label:<28} n={len(idx):<5} (test={int(dom.sum())})  AUC phân biệt test/train+val = {auc(dom, oof):.4f}")
    print("  (~0.5: không khác biệt đáng kể về các đặc điểm này; cao: có lệch phân bố)")


if __name__ == "__main__":
    main()