"""Kiểm tra liên hệ giữa lỗi của mô hình và đặc điểm kỹ thuật của ảnh gốc - CHỈ ĐỌC.

Ghép evaluation_results/analysis/predictions.csv (từ analyze_models.py) với
evaluation_results/analysis/image_stats.csv (từ image_stats.py), rồi xem trên ảnh BÌNH THƯỜNG:
  - xác suất 'viêm phổi' của mô hình có tương quan với kích thước / dung lượng / mật độ byte của ảnh không
  - đặc hiệu thay đổi thế nào giữa nhóm ảnh có mật độ byte thấp, vừa, cao

Cách chạy: python shortcut_check.py
Chỉ là phân tích tương quan, không chứng minh nhân quả.
"""
import argparse
import csv
import warnings

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

FEATS = ["width", "height", "aspect", "mean", "std", "dark_frac", "file_kb", "kb_per_kpx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="evaluation_results/analysis/predictions.csv")
    ap.add_argument("--stats", default="evaluation_results/analysis/image_stats.csv")
    a = ap.parse_args()

    stats = {}
    for r in csv.DictReader(open(a.stats, encoding="utf-8")):
        d = {k: float(r[k]) for k in ("width", "height", "aspect", "mean", "std", "dark_frac", "file_kb")}
        d["kb_per_kpx"] = d["file_kb"] / (d["width"] * d["height"] / 1000.0)
        d["group"] = r["group"]
        stats[(r["split"], r["class"], r["file"])] = d

    rd = csv.DictReader(open(a.pred, encoding="utf-8"))
    models = [c for c in rd.fieldnames if c.startswith("p_pneumonia_")]
    rows = []
    for r in rd:
        s = stats.get((r["split"], r["class"], r["file"]))
        if s is None:
            raise SystemExit(f"Không ghép được: {r['split']}/{r['class']}/{r['file']}")
        rows.append((r, s))
    print(f"Đã ghép {len(rows)} ảnh có dự đoán với thống kê ảnh gốc.")

    for m in models:
        name = m.replace("p_pneumonia_", "")
        print("\n" + "=" * 72 + f"\nMÔ HÌNH: {name}\n" + "=" * 72)
        for split in ("val", "test"):
            sel = [(float(r[m]), s) for r, s in rows if r["split"] == split and r["class"] == "NORMAL"]
            if len(sel) < 20:
                continue
            p = np.array([x[0] for x in sel])
            print(f"\n[{split}] ảnh BÌNH THƯỜNG (n={len(sel)}), tương quan Spearman giữa P(viêm phổi) và đặc điểm ảnh:")
            for f in FEATS:
                v = np.array([x[1][f] for x in sel])
                rho = spearmanr(p, v).correlation
                print(f"    {f:<12} rho={rho:+.3f}")
            v = np.array([x[1]["kb_per_kpx"] for x in sel])
            cuts = np.quantile(v, [1 / 3, 2 / 3])
            print(f"  Đặc hiệu @0.5 theo mật độ byte (KB trên nghìn điểm ảnh), chia 3 phần bằng nhau:")
            for label, lo, hi in (("thấp", -np.inf, cuts[0]), ("vừa", cuts[0], cuts[1]), ("cao", cuts[1], np.inf)):
                idx = (v > lo) & (v <= hi) if lo != -np.inf else (v <= hi)
                if label == "cao":
                    idx = v > cuts[1]
                n = int(idx.sum())
                if n:
                    print(f"    {label:<5} n={n:<4} đặc hiệu={(p[idx] < 0.5).mean():.3f}  "
                          f"(mật độ TB={v[idx].mean():.3f}, P(viêm phổi) TB={p[idx].mean():.3f})")

    print("\nCách đọc: rho âm với kb_per_kpx nghĩa là ảnh nén mạnh/ít chi tiết bị gán 'viêm phổi' nhiều hơn.")
    print("Nếu val gần 0 nhưng test rõ rệt, lỗi trên test gắn với đặc điểm kỹ thuật của ảnh.")


if __name__ == "__main__":
    main()