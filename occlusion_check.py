"""Che từng phần ảnh để xem mô hình dựa vào vùng nào - CHỈ ĐỌC.

Với ảnh val/test của dataset_v2 (đã CLAHE), lấy hình vuông giữa ảnh (đúng vùng mô hình xét), chia 3 dải ngang bằng nhau
(trên / giữa / dưới) và thử:
  default      : đường dẫn file, Ultralytics tự xử lý (khớp evaluate.py)
  crop         : chỉ cắt hình vuông giữa (đối chứng của các bước che)
  mask_top/mid/bottom : che một dải bằng độ xám trung vị của ảnh
  only_top/mid/bottom : chỉ giữ một dải, che hai dải còn lại
Cách đọc: nếu che dải dưới làm tụt mạnh hơn che dải trên hay giữa, mô hình dựa nhiều vào vùng dưới ngực/bụng.
Nếu chỉ giữ dải dưới mà vẫn phân loại tốt (AUC cao), vùng dưới đủ để phân loại, đáng ngờ về đường tắt.
Ảnh bị che là ảnh lạ với mô hình nên sai lệch tuyệt đối khó diễn giải, hãy so các dải với nhau.
Chạy: python occlusion_check.py [--models v2a=weights/v2/best_v2a.pt] [--device 0]
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "crop": {0, 1, 2},
    "mask_top": {1, 2}, "mask_mid": {0, 2}, "mask_bottom": {0, 1},
    "only_top": {0}, "only_mid": {1}, "only_bottom": {2},
}


def square_crop(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def make_variant(sq, keep):
    s = sq.shape[0]
    cuts = [0, s // 3, 2 * s // 3, s]
    out = sq.copy()
    fill = int(np.median(sq))
    for b in range(3):
        if b not in keep:
            out[cuts[b]:cuts[b + 1]] = fill
    return out


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def metrics(y, p):
    pred = p >= 0.5
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    return (tp + tn) / len(y), tp / max(1, tp + fn), tn / max(1, tn + fp), auc(y, p), float(p[y == 1].mean()), float(p[y == 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["v2a=weights/v2/best_v2a.pt"])
    ap.add_argument("--manifest", default="dataset_v2_manifest.csv")
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--device", default="")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Thiếu ultralytics trong môi trường Python này.")
    rows = [r for r in csv.DictReader(open(ROOT / a.manifest, encoding="utf-8")) if r["split"] in ("val", "test")]
    paths = [ROOT / a.data / r["split"] / r["class"] / r["file"] for r in rows]
    for p in paths:
        if not p.is_file():
            sys.exit(f"Không thấy file: {p}")
    y = np.array([1 if r["class"] == "PNEUMONIA" else 0 for r in rows])
    split = np.array([r["split"] for r in rows])
    dev = a.device if a.device != "" else None
    models = {}
    for item in a.models:
        name, path = item.split("=", 1)
        m = YOLO(str(ROOT / path))
        models[name] = (m, [k for k, v in m.names.items() if str(v).upper() == "PNEUMONIA"][0])
    names = ["default"] + list(VARIANTS)
    P = {(n, v): np.zeros(len(rows)) for n in models for v in names}

    def run(m, idx, src):
        return [float(r.probs.data[idx]) for r in m.predict(src, imgsz=224, device=dev, verbose=False)]

    print(f"Đang dự đoán {len(rows)} ảnh x {len(names)} cách x {len(models)} mô hình...", flush=True)
    for s in range(0, len(rows), a.batch):
        chunk = paths[s:s + a.batch]
        sqs = [square_crop(cv2.imread(str(p))) for p in chunk]
        srcs = {"default": [str(p) for p in chunk]}
        for v, keep in VARIANTS.items():
            srcs[v] = [make_variant(q, keep) for q in sqs]
        for n, (m, idx) in models.items():
            for v, src in srcs.items():
                P[(n, v)][s:s + len(chunk)] = run(m, idx, src)
        if (s // a.batch) % 10 == 0:
            print(f"  {min(s + a.batch, len(rows))}/{len(rows)}", flush=True)

    for n in models:
        for sp in ("test", "val"):
            sel = split == sp
            print(f"\n=== {n} | tập {sp} (n={int(sel.sum())}) ===")
            print(f"{'cách':<14}{'acc':>7}{'nhạy':>7}{'đặc hiệu':>10}{'AUC':>8}{'P TB ảnh viêm phổi':>20}{'P TB ảnh bình thường':>22}")
            for v in names:
                acc, sens, spec, au, pp, pn = metrics(y[sel], P[(n, v)][sel])
                print(f"{v:<14}{acc:7.3f}{sens:7.3f}{spec:10.3f}{au:8.4f}{pp:20.3f}{pn:22.3f}")
    print("\nCách đọc: so mask_top, mask_mid, mask_bottom với crop (cùng diện tích bị che, mỗi dải 1/3).")
    print("Che dải nào làm AUC và độ nhạy tụt nhiều nhất thì mô hình dựa vào dải đó nhiều nhất. 'only_*' cho biết một dải đơn lẻ có đủ để phân loại không.")


if __name__ == "__main__":
    main()
