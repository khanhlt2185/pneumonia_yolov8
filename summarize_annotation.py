"""Tổng hợp bảng chú thích Grad-CAM sau khi điền (blind_labels.csv hoặc annotation.csv) - CHỈ ĐỌC.

Tự nhận các cột bắt đầu bằng 'focus_'. Mỗi dòng phải có đúng MỘT cột focus_* được đánh dấu (1, x);
để trống cả nhóm nếu không đánh giá được; đánh dấu nhiều cột bị coi là không hợp lệ.
Nhóm 'trường phổi' là cột có chữ 'lung'; 'lan tỏa' là cột 'diffuse'; các cột còn lại là 'ngoài phổi'.

Cách chạy: python summarize_annotation.py [đường_dẫn_csv]   (mặc định scratch/cam_visual/blind_labels.csv nếu có)
Đây là phân tích thăm dò: mỗi nhóm chỉ 25 ảnh, không kết luận chắc chắn từ p-value.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

GROUPS = ["FP", "TN", "TP"]
MARK = {"1", "x", "true", "yes", "y"}


def marked(v):
    return (v or "").strip().lower() in MARK


def main():
    default = "scratch/cam_visual/blind_labels.csv"
    path = sys.argv[1] if len(sys.argv) > 1 else (default if Path(default).exists() else "scratch/cam_visual/annotation.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    cats = [c for c in rows[0] if c.startswith("focus_")]
    if not cats or "group" not in rows[0] or "raw_peak" not in rows[0]:
        sys.exit("Thiếu cột focus_*, group hoặc raw_peak trong file.")
    lung = [c for c in cats if "lung" in c]
    diffuse = [c for c in cats if "diffuse" in c]
    non_lung = [c for c in cats if c not in lung + diffuse]
    print(f"Đọc {len(rows)} dòng từ {path}; các cột vị trí: {', '.join(c.replace('focus_', '') for c in cats)}\n")
    per = defaultdict(lambda: defaultdict(list))
    unlabeled = defaultdict(int)
    invalid = defaultdict(int)
    for r in rows:
        g = r["group"]
        ticks = [c for c in cats if marked(r[c])]
        if not ticks:
            unlabeled[g] += 1
        elif len(ticks) > 1:
            invalid[g] += 1
        else:
            per[g][ticks[0]].append(float(r["raw_peak"]))
    print(f"{'nhóm':<5}{'n':>4}{'trống':>7}{'sai':>5}   " + " ".join(f"{c.replace('focus_', '')[:15]:>16}" for c in cats))
    for g in GROUPS:
        n = sum(1 for r in rows if r["group"] == g)
        valid = sum(len(v) for v in per[g].values())
        cells = [f"{len(per[g][c]):>3} ({100 * len(per[g][c]) / max(1, valid):3.0f}%)" for c in cats]
        print(f"{g:<5}{n:>4}{unlabeled[g]:>7}{invalid[g]:>5}   " + " ".join(f"{x:>16}" for x in cells))
    print("\nraw_peak trung bình theo vị trí:")
    for g in GROUPS:
        parts = [f"{c.replace('focus_', '')}={np.mean(per[g][c]):.2f}(n={len(per[g][c])})" for c in cats if per[g][c]]
        print(f"  {g}: " + (" | ".join(parts) if parts else "(chưa có dữ liệu)"))

    def split(g):
        return sum(len(per[g][c]) for c in non_lung), sum(len(per[g][c]) for c in lung)

    print(f"\nTỷ lệ 'ngoài phổi' ({', '.join(c.replace('focus_', '') for c in non_lung)}) so với 'phổi', bỏ qua lan tỏa (Fisher chính xác):")
    for g1, g2 in (("FP", "TN"), ("FP", "TP"), ("TN", "TP")):
        a1, b1 = split(g1)
        a2, b2 = split(g2)
        if min(a1 + b1, a2 + b2) == 0:
            print(f"  {g1} so với {g2}: chưa đủ dữ liệu")
            continue
        _, p = fisher_exact([[a1, b1], [a2, b2]])
        print(f"  {g1}: {a1} ngoài phổi / {b1} phổi   |   {g2}: {a2} ngoài phổi / {b2} phổi   p={p:.3f}")
    print("\nNhắc lại: bản đồ 7x7 chuẩn hóa min-max theo từng ảnh; chỉ có ý nghĩa định tính, không phải định vị tổn thương.")


if __name__ == "__main__":
    main()