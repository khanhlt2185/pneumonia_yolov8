"""Thử nghiệm ở thời điểm dự đoán: kết quả có phụ thuộc cách đưa ảnh về hình vuông không? - CHỈ ĐỌC.

Với mỗi ảnh val/test của dataset_v2 (đã CLAHE) và mỗi mô hình, so sánh 3 cách đưa ảnh vào mô hình:
  default : đưa đường dẫn file, để Ultralytics tự xử lý (cách evaluate.py đang dùng)
  pad     : thêm viền đen cho ảnh thành hình vuông (giữ nguyên tỷ lệ, không cắt) rồi mới đưa vào
  stretch : kéo giãn thẳng về 224x224 bằng cv2.resize mặc định (không cắt, bóp méo tỷ lệ)
Chỉ là phép thử chẩn đoán: nếu kết quả đổi nhiều giữa các cách thì hình học/tỷ lệ ảnh đang ảnh hưởng mô hình.
KHÔNG chọn cách xử lý bằng cách nhìn điểm test; quyết định phải dựa trên lý do nguyên tắc rồi huấn luyện/đánh giá lại.

Cách chạy: python geometry_check.py --models v2a=weights/v2/best_v2a.pt v1=weights/baseline/v1_leaky.pt --device 0
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def to_pad(img):
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = img
    return out


def to_stretch(img):
    return cv2.resize(img, (224, 224))


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def summarize(y, p):
    pred = p >= 0.5
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    return (tp + tn) / len(y), tp / max(1, tp + fn), tn / max(1, tn + fp), auc(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default="dataset_v2_manifest.csv")
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--device", default="")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Thiếu ultralytics trong môi trường Python này.")

    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["split"] in ("val", "test")]
    paths = [Path(a.data) / r["split"] / r["class"] / r["file"] for r in rows]
    for p in paths:
        if not p.is_file():
            sys.exit(f"Không thấy file: {p}")
    y = np.array([1 if r["class"] == "PNEUMONIA" else 0 for r in rows])
    split = np.array([r["split"] for r in rows])
    models = {}
    for item in a.models:
        name, path = item.split("=", 1)
        m = YOLO(path)
        idx = [k for k, v in m.names.items() if str(v).upper() == "PNEUMONIA"][0]
        models[name] = (m, idx)
    P = {(n, v): np.zeros(len(rows)) for n in models for v in ("default", "pad", "stretch")}
    dev = a.device if a.device != "" else None

    def run(m, idx, src):
        return [float(r.probs.data[idx]) for r in m.predict(src, imgsz=224, device=dev, verbose=False)]

    print(f"Đang dự đoán {len(rows)} ảnh x 3 cách x {len(models)} mô hình...", flush=True)
    for s in range(0, len(rows), a.batch):
        chunk = paths[s:s + a.batch]
        imgs = [cv2.imread(str(p)) for p in chunk]
        srcs = {"default": [str(p) for p in chunk], "pad": [to_pad(i) for i in imgs],
                "stretch": [to_stretch(i) for i in imgs]}
        for n, (m, idx) in models.items():
            for v, src in srcs.items():
                P[(n, v)][s:s + len(chunk)] = run(m, idx, src)
        if (s // a.batch) % 10 == 0:
            print(f"  {min(s + a.batch, len(rows))}/{len(rows)}", flush=True)

    aspect = np.array([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape for p in paths])
    aspect = aspect[:, 1] / aspect[:, 0]
    print(f"\n{'mô hình / cách':<18}{'tập':<6}{'acc':>7}{'nhạy':>7}{'đặc hiệu':>10}{'AUC':>8}")
    for (n, v), p in P.items():
        for sp in ("val", "test"):
            sel = split == sp
            acc, sens, spec, au = summarize(y[sel], p[sel])
            print(f"{n + ' / ' + v:<18}{sp:<6}{acc:7.3f}{sens:7.3f}{spec:10.3f}{au:8.4f}")
    print("\nTương quan Spearman giữa P(viêm phổi) và tỷ lệ khung (rộng/cao) trên ảnh BÌNH THƯỜNG của test:")
    nm = (split == "test") & (y == 0)
    for (n, v), p in P.items():
        print(f"  {n + ' / ' + v:<18} rho={spearmanr(p[nm], aspect[nm]).correlation:+.3f}")
    print("\nNếu 'pad' hay 'stretch' khác hẳn 'default' (đặc hiệu, AUC test, rho): hình học ảnh đang ảnh hưởng mô hình.")
    print("Nếu hầu như không đổi: nguyên nhân không nằm ở việc cắt/bóp ảnh về hình vuông.")


if __name__ == "__main__":
    main()