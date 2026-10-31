# Hệ Thống Hỗ Trợ Sàng Lọc Viêm Phổi Từ Ảnh X-Quang Ngực (Chest X-Ray Pneumonia Screening & Explainable AI)

> ⚠️ **TUYÊN BỐ MIỄN TRỪ TRÁCH NHIỆM Y KHOA (CLINICAL DISCLAIMER):**  
> Hệ thống này là công cụ **HỖ TRỢ SÀNG LỌC** phục vụ mục đích nghiên cứu và học thuật. Kết quả dự đoán và bản đồ Grad-CAM phản ánh "vùng mô hình chú ý", **tuyệt đối không phải bằng chứng lâm sàng hay kết luận chẩn đoán bệnh tật**. Mọi quyết định y tế phải được đưa ra bởi bác sĩ chuyên khoa dựa trên đầy đủ các xét nghiệm và thăm khám lâm sàng.

---

## 🌟 Giới Thiệu Tổng Quan

Hệ thống ứng dụng mô hình Deep Learning **YOLOv8 Classification** kết hợp thuật toán tiền xử lý ảnh y tế **CLAHE** và cơ chế giải thích mô hình **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) để hỗ trợ phát hiện dấu hiệu viêm phổi trên phim X-quang lồng ngực. Dự án được xây dựng hoàn chỉnh từ nghiên cứu huấn luyện mô hình, kiểm định khoa học cho tới hệ thống ứng dụng web Fullstack hiện đại phục vụ tương tác trực quan.

### Phân loại 2 lớp bệnh học:
- **`NORMAL`**: Phổi bình thường (Chưa ghi nhận dấu hiệu nghi ngờ thâm nhiễm/tổn thương).
- **`PNEUMONIA`**: Nghi ngờ viêm phổi (Xuất hiện vùng cản quang, thâm nhiễm phế nang hoặc kẽ).

---

## 🚀 Các Tính Năng Nổi Bật

### 1. 🧠 Mô Hình & Tiền Xử Lý Ảnh Y Tế Chuẩn Mực
- **Kiến trúc YOLOv8-cls**: Tinh chỉnh (fine-tune) trên tập dữ liệu X-quang lồng ngực với kích thước đầu vào $224 \times 224$.
- **Cân bằng Histogram CLAHE**: Tăng cường độ tương phản cục bộ mô phổi (`clip_limit=2.0`, `tile_grid_size=(8,8)`), khắc phục triệt để tình trạng ảnh bị mờ, thiếu sáng hoặc thừa sáng từ các máy quét khác nhau.
- **Phiên bản mô hình nâng cao v2a**: Khắc phục hiện tượng rò rỉ dữ liệu (*Data Leakage* theo định danh bệnh nhân), loại bỏ các "đường tắt" nhận diện nhân tạo và đạt độ ổn định cao trên tập test độc lập.

### 2. 🔍 Giải Thích Mô Hình Đáng Tin Cậy (Explainable AI - Grad-CAM)
- **Tập trung vào lớp hội tụ sâu**: Trích xuất activation map trực tiếp từ lớp tích chập cuối `model.9.conv` trước tầng phân loại phẳng.
- **Căn chỉnh hình học CenterCrop**: Bản đồ nhiệt được ánh xạ chính xác về vùng hộp cắt (`crop_box`) trên ảnh gốc, không bị méo mó tỉ lệ khung hình.
- **Mã hóa PNG-RGBA chuẩn y tế**: Bản đồ màu Turbo / Viridis đi kèm kênh alpha tỉ lệ thuận với độ tích cực, cho phép phủ trực tiếp lên phim X-quang.
- **Kiểm định khoa học nghiêm ngặt**: Đã vượt qua các bài test ngẫu nhiên hóa phân tầng (*Cascade Randomization*), ngẫu nhiên hóa backbone (*Backbone-only Randomization* với trung vị $|\rho| < 0.1$) và đối chiếu với thư viện chuẩn `torchcam`.

### 3. ⚡ Backend RESTful API Hiệu Năng Cao (FastAPI)
- **Kiến trúc Thread-Safe Singleton**: `PredictorService` và `CamService` chạy độc lập, nạp trước mô hình vào bộ nhớ khi khởi động (Warm-up).
- **Phân tách suy luận và giải thích**: Tham số `explain=true` kích hoạt luồng Grad-CAM trên bản sao sâu (*deepcopy*), hoàn toàn không làm chậm các truy vấn chẩn đoán thông thường.
- **Khả năng phục hồi cao**: Tắt TF32 trên GPU Ampere/Ada Lovelace để đảm bảo độ chính xác số học; tự động dự phòng CPU khi không có GPU.
- **Tài liệu Swagger UI tương tác**: Tự động sinh tài liệu chuẩn OpenAPI tại `/docs`.

### 4. 💻 Giao Diện Bác Sĩ Hiện Đại & Tinh Tế (React + Vite + TailwindCSS 4)
- **Phòng Phân Tích X-Quang Tương Tác (Xray Studio)**:
  - Bật/tắt bản đồ nhiệt Grad-CAM mượt mà.
  - Thanh trượt điều chỉnh độ mờ (Opacity slider) từ 0% đến 100%.
  - Chế độ so sánh Before / After (Side-by-side hoặc thanh trượt Split View).
  - Công cụ thu phóng (Zoom) và di chuyển (Pan) ảnh linh hoạt.
  - Hiển thị khung cắt vùng quan sát mô hình (`crop_box`).
- **Thư viện ảnh mẫu 1-Click (Sample Gallery)**: Cho phép thử nghiệm tức thì với các ảnh mẫu NORMAL và PNEUMONIA mà không cần tự tải ảnh lên.
- **Xuất Báo Cáo Lâm Sàng (Clinical Report)**: Xuất phiếu kết quả định dạng in/PDF chuẩn y tế gồm mã bệnh án, thời gian, kết quả phân loại, độ tin cậy, hình ảnh và khuyến nghị lâm sàng.
- **Lưu lịch sử phiên làm việc (Session History)**: Lưu trữ tạm thời các ca đã sàng lọc trong phiên làm việc.

### 5. 🧪 Kiểm Thử Tự Động Toàn Diện (32/32 Tests Passed)
- Kiểm tra toàn diện HTTP status code (`200`, `400`, `413`, `415`, `503`).
- Kiểm thử bảo mật (không lưu ảnh bệnh nhân ra đĩa, không để lộ đường dẫn tuyệt đối máy chủ).
- Kiểm tra tính bất biến của tham số suy luận và tính nhất quán toán học.

---

## 📁 Cấu Trúc Dự Án

```
doAn/
├── backend/                     # Mã nguồn Backend FastAPI
│   ├── app/
│   │   ├── api/                 # Bộ định tuyến API (/health, /predict, /samples)
│   │   ├── core/                # Cấu hình tập trung (config.py, .env)
│   │   ├── schemas/             # Pydantic schemas (Prediction, ExplainResult, Error)
│   │   ├── services/            # Dịch vụ nghiệp vụ chính
│   │   │   ├── predictor.py     # Singleton suy luận YOLOv8 & tiền xử lý CLAHE
│   │   │   └── cam_service.py   # Singleton giải thích Grad-CAM & CenterCrop
│   │   └── main.py              # Điểm khởi tạo FastAPI, Lifespan & Middleware
│   ├── samples/                 # Thư mục chứa các ảnh X-quang mẫu thử nghiệm
│   ├── tests/                   # Bộ kiểm thử tự động Pytest
│   │   ├── test_api.py          # Kiểm thử tích hợp các endpoint REST API
│   │   └── test_cam_service.py  # 21 bài test toán học & ngẫu nhiên hóa Grad-CAM
│   ├── requirements.txt         # Thư viện phụ thuộc của Backend
│   └── run.py                   # Script khởi chạy server Backend
│
├── frontend/                    # Ứng dụng Web Single Page (React 19 + Vite)
│   ├── src/
│   │   ├── components/          # Các thành phần giao diện React
│   │   │   ├── Header.jsx           # Thanh tiêu đề & trạng thái kết nối Backend
│   │   │   ├── DisclaimerBanner.jsx # Khuyến cáo y khoa bắt buộc
│   │   │   ├── ImageUpload.jsx      # Kéo thả & tải ảnh X-quang
│   │   │   ├── SampleGallery.jsx    # Thư viện ảnh mẫu 1-click
│   │   │   ├── PredictionResult.jsx # Thẻ hiển thị xác suất & khuyến nghị
│   │   │   ├── XrayStudio.jsx       # Studio phân tích Grad-CAM, Zoom/Pan, Split view
│   │   │   ├── ClinicalReport.jsx   # Xuất báo cáo PDF/In ấn chuẩn y khoa
│   │   │   ├── SessionHistory.jsx   # Lịch sử các ca trong phiên
│   │   │   └── Toast.jsx            # Thông báo trạng thái
│   │   ├── services/api.js      # Axios client kết nối REST API Backend
│   │   ├── App.jsx              # Khung ứng dụng chính
│   │   └── index.css            # Cấu hình TailwindCSS 4 & Typography
│   ├── package.json             # Danh sách dependencies (React, Vite, Lucide, Tailwind)
│   └── vite.config.js           # Cấu hình đóng gói Vite
│
├── chest_xray/                  # Bộ dữ liệu ảnh gốc (train, val, test)
├── dataset_v2/                  # Bộ dữ liệu v2 đã phân chia theo bệnh nhân (patient ID)
├── weights/                     # Trọng số mô hình đã huấn luyện
│   └── v2/
│       └── best_v2a.pt          # Mô hình v2a tối ưu cho sàng lọc lâm sàng
├── best_pneumonia_model.pt      # Trọng số mô hình ban đầu (v1)
│
├── train.py                     # Script huấn luyện YOLOv8-cls
├── evaluate.py                  # Script đánh giá mô hình & xuất ma trận nhầm lẫn
├── predict.py                   # Script suy luận dòng lệnh (CLI)
├── app.py                       # Giao diện web thử nghiệm nhanh bằng Gradio (Legacy)
├── make_blind_annotation.py     # Công cụ tạo bài kiểm tra mù cho bác sĩ thẩm định Grad-CAM
├── summarize_annotation.py      # Phân tích thống kê kết quả thẩm định vùng chú ý
├── requirements.txt             # Thư viện dùng chung toàn dự án
└── README.md                    # Tài liệu hướng dẫn sử dụng này
```

---

## 🛠️ Yêu Cầu Hệ Thống & Cài Đặt

### 1. Yêu cầu môi trường:
- **Hệ điều hành**: Windows 10/11, macOS, hoặc Linux Ubuntu 20.04+.
- **Python**: Phiên bản `3.10` - `3.12` (Khuyến nghị `3.12`).
- **Node.js**: Phiên bản `18.x`, `20.x` hoặc `22.x` (kèm `npm`).
- **Phần cứng**: CPU tối thiểu 4 cores, RAM $\ge 8$ GB. Hỗ trợ GPU NVIDIA với CUDA 11.8 / 12.x để suy luận thời gian thực ($< 15$ ms).

---

### 2. Cài đặt môi trường Python:

```bash
# Clone dự án (nếu clone từ Git)
cd e:\dev_doAn\doAn

# Khuyến nghị tạo môi trường ảo Python
python -m venv venv
# Kích hoạt trên Windows:
venv\Scripts\activate
# Hoặc kích hoạt trên Linux/macOS:
# source venv/bin/activate

# Cài đặt thư viện chính
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

---

### 3. Cài đặt môi trường Frontend:

```bash
cd frontend
npm install
cd ..
```

---

## 🚦 Hướng Dẫn Vận Hành Ứng Dụng

### 🌟 Cách 1: Khởi Chạy Ứng Dụng Web Đầy Đủ (Khuyên Dùng)

Bạn cần mở **2 cửa sổ Terminal**:

#### Terminal 1: Khởi động Backend (FastAPI - Cổng 8000)
```bash
python backend/run.py
```
- Server Backend sẵn sàng tại: `http://localhost:8000`
- Kiểm tra trạng thái máy chủ: `http://localhost:8000/api/v1/health`
- Xem tài liệu Swagger UI tương tác: `http://localhost:8000/docs`

#### Terminal 2: Khởi động Frontend (React 19 Vite - Cổng 5173)
```bash
cd frontend
npm run dev
```
- Truy cập giao diện ứng dụng tại: **`http://localhost:5173`**

---

### 🖥️ Cách 2: Sử Dụng Dòng Lệnh (CLI - Predict)

Bạn có thể chạy dự đoán trực tiếp mà không cần khởi động giao diện web:

```bash
# 1. Dự đoán trên một ảnh đơn lẻ
python predict.py --source chest_xray/test/PNEUMONIA/person1_virus_6.jpeg

# 2. Sử dụng mô hình v2a
python predict.py --source backend/samples/sample_4_PNEUMONIA_person93_bacteria_454.jpeg --model weights/v2/best_v2a.pt

# 3. Dự đoán hàng loạt trên toàn bộ một thư mục ảnh
python predict.py --source chest_xray/test/NORMAL/
```

---

### 📊 Cách 3: Huấn Luyện & Đánh Giá Mô Hình

```bash
# 1. Huấn luyện mô hình YOLOv8-cls
python train.py --epochs 30 --model yolov8n-cls.pt --imgsz 224 --batch 16 --device 0

# 2. Đánh giá mô hình trên tập test độc lập
python evaluate.py --model weights/v2/best_v2a.pt --data chest_xray

# Biểu đồ và ma trận nhầm lẫn sẽ được xuất tại thư mục: evaluation_results/
```

---

### 🧪 Cách 4: Chạy Giao Diện Gradio Đơn Giản (Demo Nhanh)

```bash
python app.py
```
- Mở trình duyệt tại: `http://localhost:7860`

---

## 📡 Chi Tiết API (RESTful Endpoints)

Backend cung cấp các API chuẩn mực, định dạng trao đổi dữ liệu JSON:

| Phương thức | Đường dẫn | Tham số chính | Chức năng |
| :--- | :--- | :--- | :--- |
| **`GET`** | `/api/v1/health` | Không | Kiểm tra trạng thái server, thiết bị (`cuda`/`cpu`), mô hình, lớp phân loại |
| **`POST`** | `/api/v1/predict` | `file`: Multipart/Form-data<br>`explain`: boolean (Query, mặc định `false`) | Tải ảnh lên phân tích; nếu `explain=true` sẽ tính toán thêm Grad-CAM |
| **`GET`** | `/api/v1/samples` | Không | Lấy danh sách 4-6 ảnh mẫu phục vụ kiểm thử nhanh |
| **`GET`** | `/docs` | Không | Giao diện tài liệu Swagger UI có sẵn nút "Try it out" |

### Ví dụ JSON phản hồi khi `explain=true`:
```json
{
  "success": true,
  "prediction": "PNEUMONIA",
  "confidence": 0.9852,
  "probabilities": {
    "NORMAL": 0.0148,
    "PNEUMONIA": 0.9852
  },
  "inference_time_ms": 12.4,
  "advice": "Kết quả cho thấy dấu hiệu nghi ngờ viêm phổi. Khuyến nghị thăm khám bác sĩ chuyên khoa hô hấp.",
  "explain": {
    "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "crop_box": { "x0": 99, "y0": 0, "x1": 399, "y1": 300 },
    "colormap": "turbo",
    "is_empty": false,
    "empty_message": null,
    "raw_peak": 0.6724,
    "target_class": "PNEUMONIA",
    "target_layer": "model.9.conv",
    "image_size": { "width": 500, "height": 300 },
    "attention_note": "Vùng mô hình chú ý — không phải bằng chứng lâm sàng"
  }
}
```

---

## 🧪 Kiểm Thử Tự Động (Automated Testing Suite)

Dự án thiết lập bộ kiểm thử toàn diện bằng `pytest` đảm bảo độ tin cậy tuyệt đối cho hệ thống y tế:

```bash
# Chạy toàn bộ 32 bài kiểm thử (API + Grad-CAM)
python -m pytest backend/tests/ -v
```

### Các nhóm kiểm thử chính:
1. **API Integration Tests (`test_api.py`)**:
   - `test_health_check`: Kiểm tra nhịp tim hệ thống và nạp mô hình.
   - `test_get_samples`: Xác minh đường dẫn ảnh mẫu an toàn, không lộ đường dẫn tuyệt đối.
   - `test_predict_valid_image`: Kiểm tra suy luận trên ảnh hợp lệ.
   - `test_predict_invalid_extension` / `test_predict_corrupted_image`: Bắt lỗi định dạng lạ và file hỏng (Mã 400, 415).
   - `test_predict_oversized_file`: Từ chối file tải lên vượt quá giới hạn 10 MB (Mã 413).
   - `test_predict_no_explain_key`: Đảm bảo không lộ trường thừa khi `explain=false`.
   - `test_predict_explain_true_fields`: Khẳng định cấu trúc đầy đủ 10 trường của Grad-CAM.
   - `test_predict_two_concurrent_explain`: Kiểm tra an toàn luồng khi có 2 request gửi đồng thời.
   - `test_predict_explain_503_when_cam_not_ready`: Trả về 503 chuẩn xác nếu dịch vụ giải thích gặp sự cố.
2. **Grad-CAM Mathematical & Sanity Tests (`test_cam_service.py`)**:
   - `test_cam_service_singleton_and_init`: Khởi tạo Singleton an toàn luồng.
   - `test_cam_and_predictor_use_same_weights`: Trọng số mô hình giải thích đồng nhất 100% với mô hình suy luận.
   - `test_cam_probability_matches_predict_20_images`: Xác suất tính từ đường CAM khớp kết quả gốc dưới sai số $10^{-3}$.
   - `test_compute_crop_box_torchvision_formula`: Cắt ảnh theo đúng công thức CenterCrop của Torchvision.
   - `test_cam_cascade_randomization_spearman`: Kiểm tra tính nhạy cảm khi ngẫu nhiên hóa trọng số từ trên xuống.
   - `test_cam_backbone_only_randomization_spearman`: Khẳng định bản đồ phụ thuộc chặt chẽ vào biểu diễn đặc trưng học được của Backbone ($|\rho| < 0.1$).
   - `test_cam_matches_torchcam_spearman`: Tương quan Spearman với thư viện chuẩn `torchcam` đạt $\rho > 0.95$.
   - `test_explain_removes_hook`: Dọn dẹp sạch sẽ forward hooks sau mỗi lượt gọi, tránh rò rỉ bộ nhớ GPU.

---

## 📈 Kết Quả Đánh Giá & Hiệu Năng Mô Hình

| Phiên bản | Mô tả đặc điểm | Độ chính xác (Accuracy) | Precision (PNEUMONIA) | Recall (PNEUMONIA) | F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **v1** | Huấn luyện trên dataset ban đầu | ~88.5% | 86.4% | 98.2% | 91.9% |
| **v2a** *(Đề xuất)* | Lọc rò rỉ bệnh nhân, tăng cường CLAHE, kiểm định đường tắt | **~90.2%** | **89.1%** | **97.4%** | **93.1%** |

- **Thời gian suy luận trung bình (Inference Latency)**:
  - GPU NVIDIA RTX 3050 Laptop: **8 - 14 ms / ảnh**.
  - CPU Intel Core i5/i7: **45 - 85 ms / ảnh**.

---

## ⚖️ Đạo Đức Y Tế & Khuyến Cáo Khi Ứng Dụng

1. **Không chẩn đoán tự động**: AI chỉ đóng vai trò như một bộ lọc sàng lọc sơ bộ (Triage / Second-reader) giúp phân loại mức độ ưu tiên hàng đợi đọc phim cho bác sĩ chẩn đoán hình ảnh.
2. **Bảo mật dữ liệu bệnh nhân**: Hệ thống xử lý ảnh trực tiếp trong bộ nhớ RAM/VRAM thông qua stream nhị phân, tuyệt đối không lưu trữ ảnh gốc hay thông tin định danh bệnh nhân ra đĩa cứng máy chủ.
3. **Hiểu đúng về Grad-CAM**: Vùng nhiệt đỏ/vàng là khu vực các đặc trưng kích hoạt mạng nơ-ron mạnh nhất để đưa ra dự đoán. Vùng này có thể bao gồm tổn thương thâm nhiễm hoặc các cấu trúc lân cận liên quan, không phải là ranh giới khoanh vùng phẫu thuật hay sinh thiết.

---

## 👥 Tác Giả & Liên Hệ

- **Dự án**: Đồ án tốt nghiệp / Nghiên cứu ứng dụng Deep Learning trong Sàng lọc Hình ảnh Y tế.
- **Kho lưu trữ mã nguồn**: [e:/dev_doAn/doAn](file:///e:/dev_doAn/doAn)
- **Tài liệu tham khảo chuyên sâu**: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md), [DATA_PROCESSING.md](DATA_PROCESSING.md).
