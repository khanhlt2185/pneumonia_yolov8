"""Đỉnh của bản đồ Grad-CAM 7x7 nằm ở đâu? Có phải hiệu ứng viền của mạng tích chập không? - CHỈ ĐỌC.

Với toàn bộ 624 ảnh test (chia nhóm FP/TN/TP theo evaluation_results/analysis/predictions.csv), tìm ô có giá trị lớn nhất
trong bản đồ 7x7 (lớp PNEUMONIA) và thống kê theo hàng/cột. Đối chứng:
  - ảnh nhiễu ngẫu nhiên và ảnh dốc sáng (không có giải phẫu) qua mô hình đã học
  - ảnh test thật qua mô hình có backbone bị khởi tạo ngẫu nhiên (--rand-baseline N, mặc định 150)
Nếu đối chứng cũng dồn đỉnh vào viền như ảnh thật thì phần lớn xu hướng dồn về viền là hiệu ứng của kiến trúc, không phải do ảnh.
Trong 49 ô, có 24 ô thuộc vòng ngoài (49%); nếu đỉnh phân bố đều thì khoảng 49% đỉnh nằm ở vòng ngoài.
Hàng 0 = phần trên của vùng mô hình xét, hàng 6 = phần dưới (với ảnh ngang là sát mép dưới ảnh).

Cách chạy (từ thư mục dự án hoặc bất cứ đâu): python cam_peak_location.py
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
N = 7


def summarize(name, peaks, maps, skipped):
    n = len(peaks)
    if n == 0:
        print(f"{name:<34} (không có bản đồ nào dùng được, bỏ {skipped})")
        return
    r = np.array([p[0] for p in peaks])
    c = np.array([p[1] for p in peaks])
    ring = ((r == 0) | (r == N - 1) | (c == 0) | (c == N - 1)).mean()
    print(f"{name:<34}{n:>5}{skipped:>6}{100 * (r == 0).mean():>9.0f}%{100 * (r == N - 1).mean():>9.0f}%"
          f"{100 * (c == 0).mean():>9.0f}%{100 * (c == N - 1).mean():>9.0f}%{100 * ring:>10.0f}%")
    return np.mean(maps, axis=0), np.bincount(r, minlength=N) / n


def collect(get_map, items, use_pre=False):
    """get_map trả (cam_pre, cam_raw). use_pre=True: lấy đỉnh trên bản đồ trước ReLU (cho mô hình ngẫu nhiên, vì sau ReLU thường toàn 0)."""
    peaks, maps, skipped = [], [], 0
    for it in items:
        pre, raw = get_map(it)
        m = pre if use_pre else raw
        if (not use_pre and float(m.max()) <= 0.0) or float(m.max()) == float(m.min()):
            skipped += 1
            continue
        peaks.append(np.unravel_index(int(np.argmax(m)), m.shape))
        lo, hi = float(m.min()), float(m.max())
        maps.append((m - lo) / (hi - lo))
    return peaks, maps, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/v2/best_v2a.pt")
    ap.add_argument("--pred", default="evaluation_results/analysis/predictions.csv")
    ap.add_argument("--col", default="p_pneumonia_v2a")
    ap.add_argument("--noise", type=int, default=100)
    ap.add_argument("--rand-baseline", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from backend.app.core.config import settings
    settings.MODEL_PATH = (ROOT / a.model).resolve()
    from backend.app.services.predictor import PredictorService
    from backend.app.services.cam_service import CamService, compute_cam_raw

    pred = PredictorService.get_instance()
    pred.initialize()
    cam = CamService.get_instance()
    cam.initialize()

    rows = [r for r in csv.DictReader(open(ROOT / a.pred, encoding="utf-8-sig")) if r["split"] == "test"]
    groups = {"FP": [], "TN": [], "TP": [], "FN": []}
    for r in rows:
        p = float(r[a.col])
        key = ("TP" if p >= 0.5 else "FN") if r["class"] == "PNEUMONIA" else ("FP" if p >= 0.5 else "TN")
        groups[key].append(ROOT / "chest_xray" / "test" / r["class"] / r["file"])
    for k, v in groups.items():
        for p in v:
            if not p.is_file():
                sys.exit(f"Không thấy ảnh gốc: {p}")
    print("Số ảnh test theo nhóm:", {k: len(v) for k, v in groups.items()})

    def real_map(path, model=None, layer=None):
        rgb = pred.preprocess_image(Path(path).read_bytes())
        tensor = cam._get_input_tensor(rgb)
        cam_pre, cam_raw, _, _ = compute_cam_raw(model or cam._cam_model, layer or cam._cam_layer, tensor, 1)
        return cam_pre, cam_raw

    def synth_map(img):
        ok, buf = cv2.imencode(".png", img)
        rgb = pred.preprocess_image(buf.tobytes())
        tensor = cam._get_input_tensor(rgb)
        cam_pre, cam_raw, _, _ = compute_cam_raw(cam._cam_model, cam._cam_layer, tensor, 1)
        return cam_pre, cam_raw

    header = f"{'nhóm':<34}{'n':>5}{'bỏ':>6}{'hàng 0':>10}{'hàng 6':>10}{'cột 0':>10}{'cột 6':>10}{'vòng ngoài':>11}"
    print("\nTỷ lệ đỉnh nằm ở hàng/cột ngoài cùng (kỳ vọng nếu phân bố đều: vòng ngoài ≈ 49%, mỗi hàng/cột ngoài cùng ≈ 14%)")
    print(header)
    mean_maps = {}
    rowdist = {}
    for g in ("FP", "TN", "TP"):
        res = summarize(f"{g} (ảnh thật, mô hình đã học)", *collect(real_map, groups[g]))
        if res:
            mean_maps[g] = res
            rowdist[f"{g} (ảnh thật, mô hình đã học)"] = res[1]

    rng = np.random.default_rng(a.seed)
    noise_imgs = []
    for _ in range(a.noise):
        h, w = int(rng.integers(1000, 1800)), int(rng.integers(1200, 2000))
        noise_imgs.append(rng.integers(0, 256, (h, w), dtype=np.uint8))
    res = summarize("Đối chứng: nhiễu (mô hình đã học)", *collect(synth_map, noise_imgs))
    if res:
        rowdist["Đối chứng: nhiễu (mô hình đã học)"] = res[1]
    grad_imgs = []
    for _ in range(a.noise):
        h, w = int(rng.integers(1000, 1800)), int(rng.integers(1200, 2000))
        top, bot = int(rng.integers(20, 90)), int(rng.integers(160, 235))
        g = np.tile(np.linspace(top, bot, h)[:, None], (1, w))
        grad_imgs.append((g + rng.normal(0, 3, g.shape)).clip(0, 255).astype(np.uint8))
    res = summarize("Đối chứng: dốc sáng (mô hình đã học)", *collect(synth_map, grad_imgs))
    if res:
        rowdist["Đối chứng: dốc sáng (mô hình đã học)"] = res[1]

    if a.rand_baseline > 0:
        try:
            import copy
            import torch
        except ImportError:
            print("Bỏ qua đối chứng backbone ngẫu nhiên: thiếu torch")
        else:
            torch.manual_seed(a.seed)
            rmodel = copy.deepcopy(cam._cam_model)
            for idx in range(9):
                for sub in rmodel.get_submodule(f"model.{idx}").modules():
                    if hasattr(sub, "reset_parameters") and callable(sub.reset_parameters):
                        sub.reset_parameters()
            rlayer = rmodel.get_submodule(getattr(settings, "CAM_LAYER", "model.9.conv"))
            allp = [p for v in (groups["FP"], groups["TN"], groups["TP"]) for p in v]
            random.Random(a.seed).shuffle(allp)
            res = summarize("Đối chứng: ảnh thật, backbone ngẫu nhiên",
                            *collect(lambda p: real_map(p, rmodel, rlayer), allp[:a.rand_baseline], use_pre=True))
            if res:
                rowdist["Đối chứng: ảnh thật, backbone ngẫu nhiên"] = res[1]

    print("\nPhân bố đỉnh theo hàng (% ảnh; hàng 0 = trên cùng, hàng 6 = dưới cùng; nếu đều: mỗi hàng 14%, hai hàng cuối 29%):")
    print(f"{'nhóm':<46}" + "".join(f"{i:>6}" for i in range(N)) + f"{'hàng 5-6':>10}")
    for name, rd in rowdist.items():
        print(f"{name:<46}" + "".join(f"{100 * v:>5.0f}%" for v in rd) + f"{100 * (rd[N - 2] + rd[N - 1]):>9.0f}%")
    print("\nBản đồ trung bình đã chuẩn hóa (x100), hàng 0 ở trên, hàng 6 ở dưới:")
    for g, (mm, rowhist) in mean_maps.items():
        print(f"  {g}:")
        for i in range(N):
            print("    " + " ".join(f"{int(round(100 * v)):>3}" for v in mm[i]) + f"   | đỉnh ở hàng {i}: {100 * rowhist[i]:3.0f}%")
    print("\nCách đọc: so tỷ lệ 'vòng ngoài' và 'hàng 6' của ảnh thật với các đối chứng.")
    print("  Giống đối chứng: xu hướng dồn về viền chủ yếu do kiến trúc/đệm biên, không do nội dung ảnh.")
    print("  Cao hơn hẳn đối chứng: nội dung ảnh (vùng cơ hoành, bụng) đóng góp; vẫn chưa chứng minh mô hình hiểu bệnh lý.")


if __name__ == "__main__":
    main()
