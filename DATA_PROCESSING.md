# Ghi chép xử lý dữ liệu trước huấn luyện: chia lại bộ Chest X-Ray Pneumonia

*Tổng hợp từ quá trình làm việc với Claude, tháng 10/2026. Mọi con số trong bảng là số đã in ra từ terminal của bạn. Những chỗ là suy luận hoặc chưa có số được đánh dấu rõ là giả định hoặc chưa đo.*

## 1. Tóm tắt ngắn

- Bộ dữ liệu gốc (Kaggle, Paul Timothy Mooney) có nhiều ảnh của cùng một bệnh nhân. Cách chia cũ (v1) chia theo từng ảnh nên các tập train/val/test bị rò rỉ bệnh nhân.
- Đã đo: **554/785 ảnh val của v1 (70,6%)** thuộc bệnh nhân đã có ở train v1; **264/404 bệnh nhân test (65%)** có ảnh trong train+val của bộ gốc (riêng viêm phổi: 170/202 bệnh nhân, 84%).
- Giải pháp: giữ nguyên 624 ảnh test, loại khỏi train/val mọi bệnh nhân có mặt ở test, bỏ ảnh trùng hệt, rồi chia train/val theo nhóm bệnh nhân (80/20). Kết quả là `dataset_v2`, đã kiểm chứng độc lập bằng hai script riêng.
- Chưa có: kết quả huấn luyện v2a và đánh giá trên test (xem mục 12).

## 2. Nguồn dữ liệu và cấu trúc gốc

| Tập | NORMAL | PNEUMONIA | Tổng |
| --- | --- | --- | --- |
| train | 1.341 | 3.875 | 5.216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |
| **Tổng** | **1.583** | **4.273** | **5.856** |

- Dữ liệu chỉ gồm trẻ em 1-5 tuổi từ một trung tâm (theo tài liệu dự án).
- Thư mục tải về có thêm `chest_xray/chest_xray` (bản lồng) và `__MACOSX`.
- Tên file mang mã bệnh nhân: viêm phổi dạng `person93_bacteria_454.jpeg`; bình thường dạng `IM-0036-0001.jpeg` hoặc `NORMAL2-IM-0036-0001.jpeg`.

## 3. Những gì đã kiểm tra và kết quả (script `audit_dataset.py`)

| Kiểm tra | Kết quả |
| --- | --- |
| Bản lồng `chest_xray/chest_xray` | Bản sao y hệt: 0 ảnh khác thư mục chính. File không phải ảnh chỉ là `.DS_Store` (đó cũng là lý do bản lồng nhiều hơn 4 file lúc đếm) |
| Ảnh trùng hệt (SHA-256) | 5.856 ảnh, 5.824 nội dung khác nhau: 30 nhóm trùng, 32 ảnh dư. Tất cả nằm trong train, là các ảnh liền số của cùng một bệnh nhân. Không nhóm nào nằm ở nhiều tập |
| Mã bệnh nhân | 0 ảnh không tách được mã. NORMAL: 1.583 ảnh, 1.116 bệnh nhân (TB 1,42 ảnh/người, nhiều nhất 6). PNEUMONIA: 4.273 ảnh, 1.674 bệnh nhân (TB 2,55, nhiều nhất 31). Không mã nào xuất hiện ở cả hai lớp |
| Bệnh nhân test | 404 (202 mỗi lớp) |
| **Trùng train+val với test** | **264/404 bệnh nhân; 416/624 ảnh test thuộc bệnh nhân trùng** |
| Trùng theo lớp | PNEUMONIA: 170/202 bệnh nhân, 320/390 ảnh test, 457 ảnh train+val. NORMAL (gộp họ tên): 94 bệnh nhân, 96/234 ảnh test, 94 ảnh train+val |
| Hai họ tên NORMAL | `IM-` = 65 mã test, `NORMAL2-IM-` = 160 mã test; trong từng họ riêng không có trùng nào (0/65, 0/160). Hai họ dùng chung 328 mã số, nên 94 trường hợp trùng của NORMAL có thể là trùng số giả (chưa chứng minh) |
| Trùng train với val gốc | 0/15 (val gốc chỉ 16 ảnh, không dùng được để chọn mô hình) |
| Chi phí loại bệnh nhân trùng test | 551/5.232 ảnh train+val (10,5%) |

## 4. Bộ v1 (`dataset/`) và mô hình v1

- Tạo bằng `src/preprocess.py` (tham số mặc định): gộp train+val gốc rồi chia 85/15 phân tầng theo từng ảnh, seed 42, kèm CLAHE (clipLimit 2.0, lưới 8x8). Không resize.
- Số ảnh: train 4.447 (1.147/3.300), val 785 (202/583), test 624 (234/390).
- **Rò rỉ train-val của v1:** 554/785 ảnh val (70,6%) thuộc bệnh nhân đã có ở train; 443/662 bệnh nhân val (66,9%) cũng có ở train.
- Mô hình v1 (`best_pneumonia_model.pt`): ghi lúc 01:10:51 ngày 17/10/2026, 2.963.585 byte, SHA-256 `299C9C984C4271052A95149C19821A35B12C980A6A11AC109F451A39DC267860`. Đã sao lưu bản giống hệt (cùng mã băm) tại `weights/baseline/v1_leaky.pt`, kèm `weights/baseline/evaluation_results_v1`.
- Cấu hình huấn luyện v1 (từ `args.yaml`, thư mục `runs/classify/runs/classify/pneumonia`, ghi lúc 01:00:18 ngày 17/10): `yolov8n-cls.pt`, 50 epoch, patience 10, batch 16, imgsz 224, AdamW (lr0 0,001, lrf 0,01), seed 42, deterministic, pretrained, dropout 0, fliplr 0, flipud 0, hsv\_h 0, hsv\_s 0, hsv\_v 0,2, degrees 10, translate 0,1, scale 0,1, shear 0, mosaic 1,0, mixup 0, cutmix 0, **auto\_augment randaugment, erasing 0,4** (hai giá trị này là mặc định của Ultralytics, `train.py` không tắt).
- Dữ liệu v1 đã train nằm ở `<thư mục Desktop của máy cũ>/doAn/dataset`, **thư mục này không còn**. Vì vậy chưa chứng minh được `dataset/` hiện tại giống hệt bản v1 đã dùng (giả định: cùng pipeline, cùng seed nên rất có thể giống).

## 5. Quyết định thiết kế và lý do

1. **Đơn vị độc lập là bệnh nhân, không phải ảnh.** Ảnh cùng người rất giống nhau; để chúng ở hai tập sẽ thổi phồng điểm số.
2. **Giữ nguyên 624 ảnh test** để vẫn so sánh được với tài liệu và với v1, nhưng **loại khỏi train/val mọi bệnh nhân có mặt ở test** (mất 551 ảnh).
3. **Khóa nhóm:** PNEUMONIA = `person<ID>`; NORMAL = `IM-<số>` không phân biệt họ `IM-` hay `NORMAL2-IM-`. Lựa chọn thận trọng: thà mất 94 ảnh còn hơn chấp nhận rủi ro rò rỉ.
4. **Bỏ ảnh trùng hệt** (giữ một ảnh mỗi nhóm), làm sau khi loại bệnh nhân trùng test, rồi mới chia.
5. **Chia 80/20 theo nhóm, phân tầng theo nhãn** bằng `StratifiedGroupKFold(n=5, shuffle, seed 42)`; fold 0 làm val, lưu cả 5 fold trong manifest để có thể cross-validation.
6. **Giữ CLAHE y hệt v1** (tái dùng `apply_clahe()`), giữ nguyên tên file, áp cho cả test.
7. **Chỉ đổi dữ liệu, giữ nguyên siêu tham số** khi train v2a, để chênh lệch kết quả quy được cho dữ liệu.
8. Chọn ngưỡng quyết định và checkpoint bằng val; tập test chỉ đánh giá một lần ở cuối.

## 6. Quy trình đã thực hiện

1. Tạo nhánh git `feature/retrain`; sao lưu mô hình và kết quả v1.
2. Di chuyển (không xóa) `chest_xray/chest_xray` và `__MACOSX` ra `<thư mục ngoài dự án>/_raw_backup/`.
3. Chạy `audit_dataset.py` (SHA-256, mã bệnh nhân, rò rỉ theo lớp, chi phí loại, mức rò rỉ của v1); xuất `audit_files.csv`.
4. Thêm quy tắc chia dữ liệu vào `AGENTS.md` (sau khi sửa `.gitignore` đang chặn file này).
5. Agent lập kế hoạch (`resplit_plan.md`) và viết `build_dataset_v2.py`; chạy `--dry-run`, rồi chạy thật (220,7 giây, 5.283 ảnh).
6. Kiểm chứng độc lập `dataset_v2` bằng `audit_dataset.py` và `compare_v1_v2.py`.
7. Sửa `train.py` (thêm `--export`, `--seed`, `--skip-test-eval`) để huấn luyện không ghi đè mô hình đang dùng.

## 7. Kết quả `dataset_v2`

| Bước | Số liệu |
| --- | --- |
| Pool train+val gốc | 5.232 ảnh |
| Loại bệnh nhân trùng test | -551, còn 4.681 (NORMAL 1.255, PNEUMONIA 3.426) |
| Bỏ ảnh trùng hệt | 20 nhóm trong pool sạch, bỏ 22 ảnh, còn 4.659 (NORMAL 1.254, PNEUMONIA 3.405) |
| **train** | 3.754 ảnh (1.000/2.754), 1.909 bệnh nhân, 73,4% viêm phổi |
| **val** | 905 ảnh (254/651), 477 bệnh nhân, 71,9% viêm phổi |
| **test** | 624 ảnh (234/390), 404 bệnh nhân, 62,5% viêm phổi |
| Manifest | `dataset_v2_manifest.csv`, 5.283 dòng; cột gồm file, sha256, patient\_id, class, split, fold, orig\_split |

Tổng 5.856 - 573 = 5.283 (551 ảnh bệnh nhân trùng test cộng 22 ảnh trùng hệt).

## 8. Kiểm chứng độc lập

**Script chia tự kiểm tra (KT-1 đến KT-9):** không bệnh nhân nào ở hai tập; không SHA-256 nào ở hai tập khác nhau; test đúng 234/390 và khớp từng tên file với 624 file gốc; lệch tỷ lệ viêm phổi train-val 1,43 điểm phần trăm (dưới ngưỡng 2); không cặp (lớp, tên file) trùng; mọi file tồn tại và đọc được bằng cv2.imread; tổng file khớp manifest; không file rác. Lỗi thứ tự đã được sửa: KT-7 đến KT-9 ban đầu chạy trước khi copy nên luôn thất bại (không tạo gì), nay chạy trên thư mục tạm sau khi copy và trước khi đổi tên.

**`audit_dataset.py` chạy lại trên `dataset_v2`:** test 234/390; bệnh nhân trùng (train+val) với test **0/404**; train với val **0/477**; ảnh val thuộc bệnh nhân có ở train **0/905** (v1: 554/785); 0 ảnh không tách được mã. Còn 6 nhóm ảnh trùng hệt, **cả 6 nằm trong tập test** (giữ nguyên vì test không được sửa), không nhóm nào nằm ở nhiều tập.

**`compare_v1_v2.py`:** 3.754 (train) + 905 (val) + 624 (test) = 5.283 ảnh giống hệt bản v1 từng điểm ảnh, sai khác lớn nhất 0,0000. Nghĩa là tiền xử lý v2 đồng nhất với v1 và mô hình mới dùng được với backend hiện tại.

## 9. So sánh ba bộ dữ liệu

|  | `chest_xray` (gốc) | `dataset/` (v1) | `dataset_v2` |
| --- | --- | --- | --- |
| Train | 5.216 | 4.447 | 3.754 |
| Val | 16 | 785 | 905 |
| Test | 624 | 624 | 624 (giữ nguyên) |
| Cách chia | Chia sẵn của Kaggle | Gộp, chia 85/15 theo ảnh | Theo nhóm bệnh nhân, 80/20 |
| Rò rỉ train-val | Val quá nhỏ | 70,6% ảnh val | 0 |
| Rò rỉ train-test | 65% bệnh nhân test | Như gốc | 0 |
| Tiền xử lý | Ảnh thô | CLAHE | CLAHE giống hệt v1 |
| Truy vết | Không | Không | Manifest + seed |

Tỷ lệ NORMAL: train v1 25,8%, val v1 25,7%; train v2 26,6%, val v2 28,1%; test 37,5%. Lệch phân bố giữa train và test vẫn còn ở cả hai bộ.

## 10. File đã tạo, sửa, di chuyển, xóa

- **Tạo:** `audit_dataset.py`, `audit_files.csv`, `audit_v2_files.csv`, `build_dataset_v2.py`, `compare_v1_v2.py`, `dataset_v2/`, `dataset_v2_manifest.csv`, `resplit_plan.md`, `AGENTS.md`, `weights/baseline/` (v1\_leaky.pt và evaluation\_results\_v1), `PROJECT_OVERVIEW.md`, `SPEC.md`.
- **Sửa:** `.gitignore` (bỏ dòng chặn AGENTS.md; `dataset_v2/` và `_raw_backup/` đã được bỏ qua), `README.md`, `requirements.txt` (thêm torchcam), `train.py`.
- **Di chuyển:** `chest_xray/chest_xray` và `__MACOSX` sang `<thư mục ngoài dự án>/_raw_backup/`.
- **Xóa:** chỉ các file `*.cache` cũ.
- **Không đụng tới:** `chest_xray/` gốc, `dataset/` (v1), `best_pneumonia_model.pt`.
- Git (nhánh `feature/retrain`): các commit cho .gitignore, AGENTS.md và script kiểm tra, sửa thứ tự kiểm tra, và manifest. Thay đổi `train.py` đã được sửa và kiểm tra, cần xác nhận đã commit. Kho chỉ nặng khoảng 2,7 MiB, không chứa ảnh dữ liệu.

## 11. Môi trường

Python 3.12, torch 2.6.0+cu124, ultralytics 8.4.175, opencv-python 4.12.0.88, torchcam 0.5.0 (cho Grad-CAM, chưa dùng), GPU NVIDIA GeForce RTX 3050 Laptop.

## 12. Việc còn lại (chưa có số liệu)

- Train v2a: `python train.py --model yolov8n-cls.pt --data dataset_v2 --epochs 50 --patience 10 --imgsz 224 --batch 16 --device 0 --name v2a_group_split --export weights/v2/best_v2a.pt --skip-test-eval`.
- So sánh `args.yaml` của v2a với v1 (chỉ nên khác data, name, save\_dir, device).
- Đánh giá v2a và v1 trên cùng test bằng `evaluate.py`; tách test thành nhóm bệnh nhân v1 đã thấy và chưa thấy (khoảng 416 và 208 ảnh theo `dataset/` hiện tại).
- Chọn ngưỡng trên val; báo cáo recall, specificity, precision, F1, AUC, ma trận nhầm lẫn, khoảng tin cậy bootstrap theo bệnh nhân.
- Sau cùng mới thay mô hình vào backend và làm lại Grad-CAM (lớp mục tiêu đề xuất `model.9.conv`, phải kiểm chứng lại trên mô hình mới).
- **Kết quả huấn luyện v2a: chưa có. Hãy điền ở đây:** \_\_\_\_

## 13. Giới hạn và giả định (nên nêu trong báo cáo)

- Mã bệnh nhân suy ra từ tên file, chưa được xác nhận bằng nguồn khác. Hai họ tên NORMAL được gộp theo số như giả định thận trọng.
- Không còn bản `dataset` mà v1 đã train, nên rò rỉ của v1 đo trên `dataset/` hiện tại (cùng pipeline, cùng seed).
- 6 nhóm ảnh trùng hệt nằm trong tập test; 6 ảnh dư trên 624 là tác động nhỏ.
- Lệch phân bố lớp giữa train (khoảng 73% viêm phổi) và test (62,5%) vẫn còn, nên ngưỡng quyết định cần chọn trên val.
- Dữ liệu chỉ gồm trẻ 1-5 tuổi từ một trung tâm; chưa thể kết luận cho người lớn hay nơi khác.
- `RandAugment` và `erasing` vẫn ở mặc định Ultralytics, chưa tắt (cần ghi nhận; nếu tắt thì coi là thí nghiệm riêng, ví dụ v2b).
- v2 mất khoảng 15,6% dữ liệu train so với v1, nên so sánh v1 và v2 trộn lẫn hai tác động: bỏ rò rỉ và có ít dữ liệu hơn.
- Điểm val của v1 và v2 không so sánh trực tiếp được; chỉ so sánh trên test chung.

## 14. Gợi ý cấu trúc phần báo cáo về xử lý dữ liệu

1. Mô tả bộ dữ liệu và cách chia ban đầu.
2. Vấn đề rò rỉ bệnh nhân: phương pháp phát hiện (mã bệnh nhân từ tên file, SHA-256) và số liệu ở mục 3-4.
3. Phương pháp chia lại (mục 5) kèm sơ đồ các bước và bảng số liệu mục 7.
4. Kiểm chứng (mục 8), nêu rõ có hai lớp kiểm tra: tự động trong script và độc lập bên ngoài.
5. So sánh v1 và v2 (mục 9, và kết quả huấn luyện khi đã có).
6. Hạn chế (mục 13).

**Nên tránh khẳng định:** v2 chắc chắn khái quát tốt hơn; mô hình dùng được trên lâm sàng; v1 chắc chắn bị rò rỉ theo bản dữ liệu gốc (chỉ nói theo bản tái tạo). Chỉ khẳng định điều đã đo.

## 15. Lệnh chính (tham khảo)

- Kiểm tra: `python audit_dataset.py --root dataset_v2 --dataset dataset_v2 --csv audit_v2_files.csv` và `python compare_v1_v2.py`.
- Dựng lại v2: `python build_dataset_v2.py` (có `--dry-run`; `--force` chỉ khi cần ghi đè, vì sẽ xóa `dataset_v2` hiện có sau khi bản mới được kiểm tra).
- Đánh giá: `python evaluate.py --model weights/v2/best_v2a.pt --data dataset_v2 --output evaluation_results/v2a` và tương tự cho `weights/baseline/v1_leaky.pt`.
## 16. Kết quả huấn luyện và đánh giá (cập nhật)

- Huấn luyện v2a: cấu hình giống hệt v1 (chỉ khác data, device, name, save_dir theo so sánh args.yaml). Dừng sớm ở epoch 19 (tốt nhất epoch 9), 0,188 giờ, top-1 val 0,982.
- Test 624 ảnh @0,5: v1 acc 0,7885 (nhạy 1,000; đặc hiệu 0,4359; 132 ảnh bình thường báo nhầm); v2a acc 0,8253 (nhạy 1,000; đặc hiệu 0,5342; 109 báo nhầm). Số đo v1 giống hệt trên dataset và dataset_v2.
- AUC test: v2a 0,9674 (0,9512-0,9798); v1 0,9653 (0,9472-0,9782). Không khác biệt có ý nghĩa. AUC val v2a 0,9981.
- Ngưỡng chọn trên val (nhạy >= 0,98): v2a ngưỡng 0,85, test nhạy 0,995, đặc hiệu 0,637 (0,563-0,699).
- Lỗi dồn vào ảnh bình thường họ NORMAL2-IM (165/234 ảnh bình thường của test): đặc hiệu 0,461 (IM: 0,710). Viêm phổi 390/390 đúng (242 vi khuẩn, 148 virus).
- Kết luận thận trọng: rò rỉ trong cách chia v1 là có thật (số liệu mục 3-4) nhưng chưa phát hiện tác động lên hiệu năng trên test chính thức. Khoảng cách val-test (AUC 0,998 xuống 0,967) giống nhau ở cả hai mô hình, nên nhiều khả năng do lệch phân bố của tập test, chưa kiểm chứng.
- Ghi chú: tài liệu cũ ghi accuracy 85-90%; số đo thật của v1 là 78,85%.
## 17. Phát hiện về đặc điểm kỹ thuật của ảnh (image_stats.py)

- Ảnh gốc bình thường trong train/val: trung bình khoảng 1680x1380 điểm ảnh, file 585-616 KB (khoảng 0,25 KB/nghìn điểm ảnh). Ảnh viêm phổi: khoảng 1200x820, file 83 KB (0,085 KB/nghìn điểm ảnh). Ước thô từ giá trị trung bình.
- Chỉ từ đặc điểm file (kích thước, tỷ lệ khung, độ sáng, tương phản, viền tối, dung lượng) một mô hình rừng ngẫu nhiên đoán nhãn với AUC 0,9991 trên val và 0,8907 trên test.
- Ảnh bình thường của test khác ảnh bình thường của train+val: AUC phân biệt 0,9975 (ảnh viêm phổi đối chứng: 0,6418). Test: họ IM 288 KB (0,10 KB/nghìn điểm ảnh), họ NORMAL2-IM 174 KB (0,076), gần mật độ byte của ảnh viêm phổi, không gần của ảnh bình thường lúc train.
- Cách đọc thận trọng: nhãn bị lẫn với đặc điểm thu nhận/lưu ảnh nên val 98,2% không chứng minh hiểu biết bệnh lý; lỗi trên test (đặc hiệu thấp, dồn ở họ NORMAL2-IM) nhất quán với một đường tắt về chất lượng ảnh, nhưng đây mới là tương quan. Chưa biết mô hình có dùng đường tắt này hay không (ảnh đưa vào mô hình đã qua CLAHE và thu nhỏ 224).
- Việc tiếp theo: shortcut_check.py (tương quan P(viêm phổi) với mật độ byte trên ảnh bình thường), Grad-CAM, thí nghiệm biến đổi chất lượng ảnh ngẫu nhiên độc lập với nhãn.
### 17.2 Phép thử hình học ảnh khi suy luận (geometry_check.py)
- Backend (predictor.py) không resize: giải mã ảnh, áp CLAHE (2.0, 8x8) đúng logic src/preprocess.py rồi đưa mảng ảnh vào model.predict; tức dùng bộ biến đổi mặc định của Ultralytics như evaluate.py.
- Ba cách đưa ảnh vào (default, pad thành hình vuông, stretch về 224x224) trên test: v2a acc 0,825 / 0,854 / 0,788, độ nhạy 1,000 / 0,797 / 1,000, đặc hiệu 0,534 / 0,949 / 0,436, AUC 0,9674 / 0,9581 / 0,9724. v1: acc 0,788 / 0,808 / 0,785, độ nhạy 1,000 / 0,713 / 1,000, đặc hiệu 0,436 / 0,966 / 0,427, AUC 0,9653 / 0,9643 / 0,9818.
- Pad tạo ảnh có viền đen mà mô hình chưa gặp lúc huấn luyện nên làm độ nhạy giảm mạnh (khoảng 20-29 điểm phần trăm); không thể quy riêng cho tỷ lệ khung. AUC gần như không đổi. Điều chắc chắn: xác suất đầu ra và ngưỡng quyết định rất nhạy với cách đưa ảnh vào. Chọn cách xử lý theo val: giữ default (khớp evaluate.py và backend).

### 17.3 Hình học ảnh trong Ultralytics (đọc từ mã nguồn bản 8.4.175)
- Suy luận/validation: Resize cạnh ngắn về 224 giữ tỷ lệ, rồi CenterCrop 224, mô hình chỉ thấy hình vuông ở giữa (cạnh bằng chiều ngắn của ảnh).
- Huấn luyện: RandomResizedCrop (cắt ngẫu nhiên, kéo giãn vào hình vuông). Hình học huấn luyện và suy luận khác nhau; ảnh tỷ lệ khung lớn bị cắt hai bên nhiều hơn khi suy luận (ảnh tỷ lệ 1,5 mất khoảng 33% chiều ngang, tỷ lệ 1,2 mất khoảng 17%).
- Hệ quả: Grad-CAM chỉ phủ hình vuông trung tâm; cần trả crop_box.

## 18. Kiểm chứng Grad-CAM (v2a)
- Nhất quán: xác suất đường CAM khớp model.predict (độ lệch lớn nhất 2,2e-6); khớp TorchCAM (20/20 ảnh Spearman > 0,95); ngẫu nhiên hóa backbone làm tương quan trước ReLU giảm (trung vị abs(rho) 0,08); ngẫu nhiên hóa độc lập từng khối: trung vị abs(rho) 0,08-0,43. Chỉ chứng minh bản đồ phụ thuộc trọng số đã học.
- Chú thích mù 75 ảnh (một người chú thích): vùng nóng nhất ở cơ hoành/bụng 61%, lan tỏa 20%, vai/cổ/đỉnh 14%, tim 4%, trường phổi 1%; FP và TN giống nhau về vị trí, khác nhau về độ mạnh (raw_peak 0,38 và 0,13 ở cùng hạng mục).
- Vị trí đỉnh (bản đồ 7x7): 68-71% ở hai hàng dưới cùng với ảnh thật, nhưng 79% với ảnh nhiễu và 97% với ảnh dốc sáng; backbone ngẫu nhiên chỉ 6%. Xu hướng dồn xuống dưới do huấn luyện tạo ra và không phụ thuộc nội dung ảnh.
- Che từng dải 1/3 (AUC test/val): default 0,9674/0,9981; che trên 0,9595/0,9963; che giữa 0,9286/0,9918; che dưới 0,9447/0,9922. Chỉ giữ một dải: trên 0,8342/0,9293, giữa 0,9153/0,9746, dưới 0,8493/0,9557. Thông tin phân tán khắp ảnh; che dải dưới (nơi bản đồ nhiệt dồn) không phải tác động lớn nhất.
- Kết luận thận trọng: Grad-CAM của mô hình này chỉ cho biết vùng mô hình chú ý, không định vị tổn thương và không phản ánh trung thực vùng mô hình dựa vào. Bản đồ 7x7 chuẩn hóa min-max từng ảnh làm ảnh bằng chứng yếu vẫn hiện điểm đỏ; giao diện phải ẩn mặc định, dùng độ đậm tuyệt đối và nêu rõ giới hạn.

## 19. Quyết định chốt về mô hình và ngưỡng

- Mô hình mặc định của backend là v2a (`weights/v2/best_v2a.pt`), đặt qua biến `MODEL_PATH` (giá trị mặc định trong `backend/app/core/config.py` và `backend/.env.example`). Mô hình v1 (`best_pneumonia_model.pt`) và bản sao `weights/baseline/v1_leaky.pt` chỉ giữ làm mốc so sánh. Thư mục `weights/` và các file `.pt` không nằm trong git: sau khi tải repo phải tự đặt mô hình vào.
- Ngưỡng nhãn giữ ở 0,5: ở ngưỡng này v2a bắt đủ 390/390 ca viêm phổi trên test (độ đặc hiệu 0,534). Giao diện thêm mốc cảnh báo «Mức tin cậy thấp» khi xác suất viêm phổi dưới 0,85 (quy tắc nhạy >= 0,98 trên val). Ở ngưỡng 0,85 test cho nhạy 0,995 và đặc hiệu 0,637 nhưng bỏ sót khoảng 2 ca, nên 0,85 không được dùng làm ngưỡng nhãn. Ca âm tính sát ngưỡng chưa có cảnh báo riêng.
- Hiển thị Grad-CAM trên giao diện: ẩn mặc định, chỉ có nút khi kết quả là PNEUMONIA; vẽ theo `crop_box`; độ đậm = thanh trượt x min(1, raw_peak / 0,945); không vẽ khi `is_empty` hoặc raw_peak < 0,402 (hai hằng số tính từ val ở ngưỡng 0,85 cho mô hình v2a); luôn kèm cảnh báo độ trung thực thấp. Theo định nghĩa, khoảng 10% ca viêm phổi phát hiện đúng ở val bị coi là bằng chứng yếu và không có bản đồ.
- Banner khuyến cáo nêu phạm vi huấn luyện (trẻ 1-5 tuổi, một cơ sở); ảnh ngoài phạm vi không đáng tin (ví dụ ảnh người lớn tải từ mạng cho xác suất sát 50%).
- Chưa làm: huấn luyện thêm seed; tắt RandAugment và erasing; cảnh báo cho ca âm tính sát ngưỡng; kiểm chứng ngoài miền.
