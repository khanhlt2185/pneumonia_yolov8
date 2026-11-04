"""Tính hai hằng số cho giao diện Grad-CAM từ tập VAL (không dùng test) - CHỈ ĐỌC (ghi một file kết quả).

Với mỗi ảnh val (ảnh gốc qua đúng đường backend), lấy raw_peak của bản đồ lớp PNEUMONIA và xác suất viêm phổi, rồi chia nhóm theo ngưỡng --thr:
  TP: viêm phổi thật, p >= thr | FN: viêm phổi thật, p < thr | FP: bình thường, p >= thr | TN: bình thường, p < thr
Đề xuất:
  WEAK_PEAK = phân vị 10 của raw_peak ở nhóm TP  (yếu hơn 90% ca viêm phổi val được phát hiện đúng thì coi là 'bằng chứng yếu', không vẽ bản đồ)
  REF_PEAK  = phân vị 90 của raw_peak ở nhóm TP  (độ đậm tuyệt đối = min(1, raw_peak / REF_PEAK))
Cách chạy (từ bất kỳ thư mục nào):  python val_raw_peak.py --thr 0.85
Kết quả cũng được ghi vào evaluation_results/analysis/val_raw_peak.txt
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/v2/best_v2a.pt")
    ap.add_argument("--manifest", default="dataset_v2_manifest.csv")
    ap.add_argument("--thr", type=float, default=0.85)
    ap.add_argument("--out", default="evaluation_results/analysis/val_raw_peak.txt")
    a = ap.parse_args()

    from backend.app.core.config import settings
    settings.MODEL_PATH = (ROOT / a.model).resolve()
    from backend.app.services.predictor import PredictorService
    from backend.app.services.cam_service import CamService, compute_cam_raw

    pred = PredictorService.get_instance()
    pred.initialize()
    cam = CamService.get_instance()
    cam.initialize()

    rows = [r for r in csv.DictReader(open(ROOT / a.manifest, encoding="utf-8-sig")) if r["split"] == "val"]
    print(f"Tập val: {len(rows)} ảnh; ngưỡng p(viêm phổi) >= {a.thr}", flush=True)
    groups = {"TP": [], "FN": [], "FP": [], "TN": []}
    for i, r in enumerate(rows, 1):
        path = ROOT / "chest_xray" / r["orig_split"] / r["class"] / r["file"]
        if not path.is_file():
            sys.exit(f"Không thấy ảnh gốc: {path}")
        data = path.read_bytes()
        p = float(pred.predict(data)["probabilities"]["PNEUMONIA"])
        rgb = pred.preprocess_image(data)
        tensor = cam._get_input_tensor(rgb)
        _, cam_raw, _, _ = compute_cam_raw(cam._cam_model, cam._cam_layer, tensor, 1)
        peak = float(cam_raw.max())
        if r["class"] == "PNEUMONIA":
            groups["TP" if p >= a.thr else "FN"].append(peak)
        else:
            groups["FP" if p >= a.thr else "TN"].append(peak)
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    lines = [f"raw_peak của bản đồ PNEUMONIA trên tập VAL, ngưỡng p >= {a.thr}, mô hình {a.model}",
             f"{'nhóm':<6}{'n':>5}{'p10':>9}{'p50':>9}{'p90':>9}"]
    for g in ("TP", "FN", "FP", "TN"):
        v = groups[g]
        lines.append(f"{g:<6}{len(v):>5}{pct(v, 10):>9.3f}{pct(v, 50):>9.3f}{pct(v, 90):>9.3f}")
    weak, ref = pct(groups["TP"], 10), pct(groups["TP"], 90)
    below = {g: (float(np.mean(np.array(groups[g]) < weak)) if groups[g] else float("nan")) for g in ("TP", "FP", "TN")}
    lines += ["",
              f"WEAK_PEAK = {weak:.3f}   (phân vị 10 của TP)",
              f"REF_PEAK  = {ref:.3f}   (phân vị 90 của TP)",
              f"Tỷ lệ nằm dưới WEAK_PEAK: TP {100 * below['TP']:.0f}% (theo định nghĩa ~10%), FP {100 * below['FP']:.0f}%, TN {100 * below['TN']:.0f}%",
              "Chỉ dùng hai hằng số này trong giao diện; không chỉnh theo test."]
    text = "\n".join(lines)
    print("\n" + text)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes((text + "\n").encode("utf-8"))
    print(f"\nĐã ghi {out}")


if __name__ == "__main__":
    main()
