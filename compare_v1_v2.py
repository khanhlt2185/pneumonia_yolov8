"""So sánh ảnh giữa dataset (v1) và dataset_v2 - CHỈ ĐỌC.

Mọi ảnh có ở cả hai bộ đều xuất phát từ cùng ảnh gốc và cùng pipeline CLAHE,
nên phải giống hệt từng điểm ảnh (khớp theo lớp + tên file, không phân biệt tập).
Cách chạy: python compare_v1_v2.py [--v1 dataset] [--v2 dataset_v2]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")
EXT = {".jpg", ".jpeg", ".png"}

ap = argparse.ArgumentParser()
ap.add_argument("--v1", default="dataset")
ap.add_argument("--v2", default="dataset_v2")
a = ap.parse_args()
v1, v2 = Path(a.v1), Path(a.v2)

index = {}
for s in SPLITS:
    for c in CLASSES:
        for p in (v1 / s / c).glob("*"):
            if p.suffix.lower() in EXT:
                index[(c, p.stem)] = p
print(f"Đã lập chỉ mục {len(index)} ảnh của v1")

stat = defaultdict(lambda: defaultdict(int))
worst = 0.0
for s in SPLITS:
    for c in CLASSES:
        for f2 in sorted((v2 / s / c).glob("*")):
            if f2.suffix.lower() not in EXT:
                continue
            f1 = index.get((c, f2.stem))
            if f1 is None:
                stat[s]["không có ở v1"] += 1
                continue
            x, y = cv2.imread(str(f2)), cv2.imread(str(f1))
            if x is None or y is None or x.shape != y.shape:
                stat[s]["khác kích thước/không đọc được"] += 1
                print("Khác kích thước/không đọc được:", s, c, f2.name)
                continue
            d = float(np.abs(x.astype(int) - y.astype(int)).mean())
            worst = max(worst, d)
            if d == 0:
                stat[s]["giống hệt"] += 1
            else:
                stat[s]["khác điểm ảnh"] += 1
                print("Khác điểm ảnh:", s, c, f2.name, f"(TB {d:.3f})")

for s in SPLITS:
    print(f"  {s:<6}", dict(stat[s]))
bad = sum(v for s in SPLITS for k, v in stat[s].items() if k != "giống hệt" and k != "không có ở v1")
print(f"Sai khác trung bình lớn nhất: {worst:.4f}")
print("KẾT LUẬN:", "ĐẠT - mọi ảnh trùng giữa hai bộ đều giống hệt" if bad == 0 else f"KHÔNG ĐẠT - {bad} ảnh lệch, chưa được train")