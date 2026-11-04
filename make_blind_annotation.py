"""Tạo công cụ chú thích MÙ cho ảnh ghép Grad-CAM (trang HTML chạy trên máy bạn, không cần mạng).

Đọc FP_montage.png, TN_montage.png, TP_montage.png và annotation.csv trong thư mục cam_visual,
cắt từng ô, xóa nhãn chữ (P và PK) ở góc trên bên trái, xáo trộn 75 ô và ẩn nhóm.
Mở file HTML tạo ra, bấm phím 1-6 để chọn vị trí vùng nóng nhất (0 = không rõ), phím mũi tên để đi lại.
Tiến độ tự lưu trong trình duyệt. Cuối cùng bấm 'Xuất CSV' để tải blind_labels.csv.

Cách chạy: python make_blind_annotation.py [--dir scratch/cam_visual]
Giả định: ô thứ k (hàng trước, cột sau) của ảnh ghép nhóm G ứng với dòng thứ k của nhóm G trong annotation.csv.
"""
import argparse
import base64
import csv
import json
import random
import sys
from pathlib import Path

import cv2

CATS = [
    ("focus_lung_field", "Trường phổi"),
    ("focus_diaphragm_abdomen", "Cơ hoành / dưới cơ hoành / bụng"),
    ("focus_mediastinum_heart", "Trung thất / bóng tim"),
    ("focus_shoulder_neck_apex", "Vai / cổ / đỉnh phổi"),
    ("focus_marker_device_edge", "Chữ, dây, thiết bị hoặc viền ngoài ảnh"),
    ("focus_diffuse", "Lan tỏa, không có đỉnh rõ"),
]
GROUPS = ["FP", "TN", "TP"]

HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Chú thích mù Grad-CAM</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#1b1b1b;color:#eee}
#wrap{display:flex;gap:24px;padding:16px;align-items:flex-start;flex-wrap:wrap}
#img{width:640px;height:640px;image-rendering:auto;background:#000;border:1px solid #555}
#side{max-width:460px}
.btn{display:block;width:100%;text-align:left;margin:4px 0;padding:8px 10px;border:1px solid #666;background:#2b2b2b;color:#eee;cursor:pointer;border-radius:4px}
.btn.sel{background:#2d6a4f;border-color:#52b788}
#bar{height:8px;background:#333;margin:8px 0}#fill{height:8px;background:#52b788;width:0}
small{color:#aaa}
</style></head><body><div id="wrap"><div><img id="img"><div id="bar"><div id="fill"></div></div></div>
<div id="side"><h3 id="prog"></h3>
<p><small>Chọn vị trí của VÙNG NÓNG NHẤT (đỏ/vàng). Khung vàng là vùng mô hình xét. Mỗi ô chỉ chọn một mục.</small></p>
<div id="btns"></div>
<button class="btn" onclick="setLabel('0')">0 — Không rõ / bỏ qua</button>
<p><button class="btn" onclick="prev()">← Ô trước</button><button class="btn" onclick="next()">Ô sau →</button></p>
<p><button class="btn" onclick="exportCsv()">Xuất CSV (blind_labels.csv)</button></p>
<p><small id="stat"></small></p></div></div>
<script>
const CATS = __CATS__;
const ITEMS = __ITEMS__;
const KEY = "cam_blind_v1_" + ITEMS.length;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }
let i = 0;
for (let k = 0; k < ITEMS.length; k++) { if (!(k in labels)) { i = k; break; } if (k === ITEMS.length - 1) i = k; }
function save(){ try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }
function render(){
  document.getElementById('img').src = ITEMS[i].img;
  document.getElementById('prog').textContent = "Ô " + (i+1) + " / " + ITEMS.length;
  const box = document.getElementById('btns'); box.innerHTML = "";
  CATS.forEach((c, idx) => {
    const b = document.createElement('button'); b.className = 'btn' + (labels[i] === String(idx+1) ? ' sel' : '');
    b.textContent = (idx+1) + " — " + c[1]; b.onclick = () => setLabel(String(idx+1)); box.appendChild(b);
  });
  const done = Object.keys(labels).length;
  document.getElementById('fill').style.width = (100*done/ITEMS.length) + "%";
  document.getElementById('stat').textContent = "Đã chú thích: " + done + "/" + ITEMS.length + (labels[i] !== undefined ? " | ô này: " + labels[i] : "");
}
function setLabel(v){ labels[i] = v; save(); if (i < ITEMS.length - 1) i++; render(); }
function next(){ if (i < ITEMS.length - 1) i++; render(); }
function prev(){ if (i > 0) i--; render(); }
document.addEventListener('keydown', e => {
  if (e.key >= '0' && e.key <= String(CATS.length)) setLabel(e.key);
  else if (e.key === 'ArrowRight') next(); else if (e.key === 'ArrowLeft') prev();
});
function esc(v){ v = String(v === undefined || v === null ? "" : v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function exportCsv(){
  const rows = ITEMS.map((it, k) => ({ it, k })).sort((a, b) => a.it.order - b.it.order);
  const base = Object.keys(rows[0].it.row).filter(c => !c.startsWith('focus_'));
  const header = base.concat(CATS.map(c => c[0]));
  const lines = [header.map(esc).join(",")];
  rows.forEach(({ it, k }) => {
    const rec = base.map(c => esc(it.row[c]));
    CATS.forEach((c, idx) => rec.push(labels[k] === String(idx+1) ? "1" : ""));
    lines.push(rec.join(","));
  });
  const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = "blind_labels.csv"; a.click();
}
render();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="scratch/cam_visual")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--mask-w", type=int, default=176)
    ap.add_argument("--mask-h", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    d = Path(a.dir)
    csv_path = Path(a.csv) if a.csv else d / "annotation.csv"
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    by = {g: [r for r in rows if r["group"] == g] for g in GROUPS}
    n = a.grid * a.grid
    for g in GROUPS:
        if len(by[g]) != n:
            sys.exit(f"Nhóm {g} có {len(by[g])} dòng trong {csv_path}, cần {n}")
    items = []
    order = 0
    for g in GROUPS:
        img = cv2.imread(str(d / f"{g}_montage.png"))
        if img is None:
            sys.exit(f"Không đọc được {d / (g + '_montage.png')}")
        h, w = img.shape[:2]
        ch, cw = h // a.grid, w // a.grid
        for k in range(n):
            r, c = divmod(k, a.grid)
            cell = img[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy()
            cell[:a.mask_h, :a.mask_w] = 0
            ok, buf = cv2.imencode(".png", cell)
            uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
            items.append({"img": uri, "row": by[g][k], "order": order})
            order += 1
    random.Random(a.seed).shuffle(items)
    out_dir = d / "blind"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__CATS__", json.dumps(CATS, ensure_ascii=False)).replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    out = out_dir / "annotate.html"
    out.write_text(html, encoding="utf-8")
    print(f"Đã tạo {out} ({out.stat().st_size / 1e6:.1f} MB), {len(items)} ô đã xáo trộn.")
    print("Kiểm tra giả định thứ tự: 3 ô đầu của mỗi nhóm trong annotation.csv (so với nhãn PK ở ảnh ghép gốc):")
    for g in GROUPS:
        for k in range(3):
            r = by[g][k]
            print(f"  {g} ô #{k + 1:02d}: file={r.get('file')} raw_peak={float(r['raw_peak']):.2f} p_backend={float(r['p_pneu_backend']):.2f}")


if __name__ == "__main__":
    main()