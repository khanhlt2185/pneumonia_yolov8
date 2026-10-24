"""Phân tích sâu các mô hình trên val/test của dataset_v2 - CHỈ ĐỌC (trừ file kết quả).

Cho mỗi mô hình in ra:
  1. Độ chính xác test ở ngưỡng 0.5 (phải khớp với evaluate.py để chắc script đúng)
  2. AUC trên val và test (AUC không phụ thuộc ngưỡng), khoảng tin cậy bootstrap theo bệnh nhân
  3. Ngưỡng chọn trên VAL (Youden và độ nhạy mục tiêu) rồi áp dụng lên test
  4. Kết quả theo nhóm con của test (họ tên NORMAL, loại viêm phổi vi khuẩn/virus)
  5. Kết quả theo nhóm bệnh nhân mà mô hình v1 đã/chưa thấy lúc huấn luyện (cần --v1-data)

Cách chạy (tại thư mục dự án):
  python analyze_models.py --models v2a=weights/v2/best_v2a.pt v1=weights/baseline/v1_leaky.pt --device 0
Lưu ý: ngưỡng phải được chọn bằng val; không được chọn ngưỡng nhìn vào test.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}


def patient_id(name):
    m = re.match(r"^(person\d+)_", name)
    if m:
        return m.group(1)
    m = re.search(r"(IM-\d+)", name)
    return m.group(1) if m else "UNPARSED:" + name


def group_of(name, cls):
    if cls == "NORMAL":
        return "NORMAL/NORMAL2-IM" if name.startswith("NORMAL2-") else "NORMAL/IM"
    if "_bacteria_" in name:
        return "PNEUMONIA/vi khuẩn"
    if "_virus_" in name:
        return "PNEUMONIA/virus"
    return "PNEUMONIA/khác"


def predict(model, paths, device, bs=64):
    idx = [k for k, v in model.names.items() if str(v).upper() == "PNEUMONIA"][0]
    out = []
    for i in range(0, len(paths), bs):
        res = model.predict([str(p) for p in paths[i:i + bs]], imgsz=224,
                            device=device if device != "" else None, verbose=False)
        out += [float(r.probs.data[idx]) for r in res]
    return np.array(out)


def counts(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    return tp, fn, tn, fp


def metrics(y, p, thr=0.5):
    tp, fn, tn, fp = counts(y, p, thr)
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(1e-12, prec + sens)
    return dict(acc=(tp + tn) / max(1, len(y)), sens=sens, spec=spec, prec=prec, f1=f1,
                tp=tp, fn=fn, tn=tn, fp=fp)


def auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def boot_ci(y, p, pids, fn, B, rng):
    groups = defaultdict(list)
    for i, g in enumerate(pids):
        groups[g].append(i)
    keys = list(groups)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[k]] for k in pick])
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def pick_thresholds(y, p, target_sens):
    cand = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999, 0.9995, 0.9999]]))
    best, best_j = 0.5, -1.0
    thr_sens = None
    for t in cand:
        m = metrics(y, p, t)
        j = m["sens"] + m["spec"] - 1
        if j > best_j + 1e-12:
            best, best_j = float(t), j
        if m["sens"] >= target_sens:
            thr_sens = float(t)
    return best, thr_sens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tên=đường_dẫn.pt")
    ap.add_argument("--manifest", default="dataset_v2_manifest.csv")
    ap.add_argument("--data", default="dataset_v2")
    ap.add_argument("--v1-data", default="dataset", help="thư mục train v1 để xác định nhóm bệnh nhân đã thấy")
    ap.add_argument("--device", default="")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--target-sens", type=float, default=0.98)
    ap.add_argument("--out", default="evaluation_results/analysis")
    a = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Thiếu ultralytics trong môi trường Python này.")

    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["split"] in ("val", "test")]
    data = Path(a.data)
    rec = []
    for r in rows:
        path = data / r["split"] / r["class"] / r["file"]
        if not path.is_file():
            sys.exit(f"Không thấy file: {path}")
        rec.append(dict(path=path, split=r["split"], cls=r["class"], file=r["file"],
                        pid=r["patient_id"], y=1 if r["class"] == "PNEUMONIA" else 0,
                        group=group_of(r["file"], r["class"])))

    seen = None
    tr = Path(a.v1_data) / "train"
    if tr.is_dir():
        seen = {patient_id(p.name) for p in tr.rglob("*") if p.suffix.lower() in IMG_EXT}
        for r in rec:
            r["seen_v1"] = r["pid"] in seen
        print(f"Nhóm 'v1 đã thấy' dựa trên {len(seen)} mã bệnh nhân trong {tr} (giả định dataset/ hiện tại = dữ liệu v1 đã train)")

    val = [i for i, r in enumerate(rec) if r["split"] == "val"]
    test = [i for i, r in enumerate(rec) if r["split"] == "test"]
    y = np.array([r["y"] for r in rec])
    pids = np.array([r["pid"] for r in rec])
    rng = np.random.default_rng(42)
    probs = {}

    for item in a.models:
        name, path = item.split("=", 1)
        print(f"\nĐang dự đoán {len(rec)} ảnh bằng {name} ({path})...", flush=True)
        probs[name] = predict(YOLO(path), [r["path"] for r in rec], a.device)

    for name, p in probs.items():
        print("\n" + "=" * 70 + f"\nMÔ HÌNH: {name}\n" + "=" * 70)
        pv, pt, yv, yt = p[val], p[test], y[val], y[test]
        m = metrics(yt, pt, 0.5)
        print(f"[1] Test @0.5: acc={m['acc']:.4f}  nhạy={m['sens']:.4f}  đặc hiệu={m['spec']:.4f}  "
              f"precision={m['prec']:.4f}  F1={m['f1']:.4f}  (TP={m['tp']} FN={m['fn']} TN={m['tn']} FP={m['fp']})")
        print("    -> acc phải khớp với evaluate.py; nếu lệch là script hoặc cách tiền xử lý khác")
        lo, hi = boot_ci(yt, pt, pids[test], auc, a.boot, rng)
        print(f"[2] AUC val={auc(yv, pv):.4f} | AUC test={auc(yt, pt):.4f} (95% CI bệnh nhân {lo:.4f}-{hi:.4f})")
        t_j, t_s = pick_thresholds(yv, pv, a.target_sens)
        for label, t in (("Youden trên val", t_j), (f"nhạy>={a.target_sens} trên val", t_s)):
            if t is None:
                print(f"[3] {label}: không có ngưỡng nào đạt")
                continue
            mv, mt = metrics(yv, pv, t), metrics(yt, pt, t)
            slo, shi = boot_ci(yt, pt, pids[test], lambda yy, pp: metrics(yy, pp, t)["spec"], a.boot, rng)
            print(f"[3] {label}: ngưỡng={t:.4f} | val nhạy={mv['sens']:.3f} đặc hiệu={mv['spec']:.3f} | "
                  f"TEST nhạy={mt['sens']:.3f} đặc hiệu={mt['spec']:.3f} (CI {slo:.3f}-{shi:.3f}) acc={mt['acc']:.3f}")
        print("[4] Theo nhóm con của test @0.5 (đúng/tổng):")
        by = defaultdict(list)
        for i in test:
            by[rec[i]["group"]].append(i)
        for g in sorted(by):
            idx = by[g]
            ok = sum(1 for i in idx if (p[i] >= 0.5) == bool(y[i]))
            print(f"    {g:<22} {ok}/{len(idx)} = {ok / len(idx):.3f}")
        if seen is not None:
            print("[5] Theo nhóm bệnh nhân v1 đã/chưa thấy (test @0.5):")
            for flag, label in ((True, "v1 ĐÃ thấy"), (False, "v1 CHƯA thấy")):
                idx = [i for i in test if rec[i]["seen_v1"] == flag]
                if not idx:
                    continue
                yy, pp = y[idx], p[idx]
                mm = metrics(yy, pp, 0.5)
                print(f"    {label:<14} n={len(idx):<4} acc={mm['acc']:.3f}  nhạy={mm['sens']:.3f} "
                      f"({int(yy.sum())} ảnh viêm phổi)  đặc hiệu={mm['spec']:.3f} ({int((yy == 0).sum())} ảnh bình thường)")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    outp = Path(a.out) / "predictions.csv"
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "file", "patient_id", "group"] + [f"p_pneumonia_{n}" for n in probs])
        for i, r in enumerate(rec):
            w.writerow([r["split"], r["cls"], r["file"], r["pid"], r["group"]] + [f"{probs[n][i]:.6f}" for n in probs])
    print(f"\nĐã ghi {outp}")


if __name__ == "__main__":
    main()