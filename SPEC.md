\# SPEC: Hệ thống hỗ trợ sàng lọc viêm phổi từ ảnh X-quang ngực (đồ án tốt nghiệp)

Đây là công cụ HỖ TRỢ SÀNG LỌC phục vụ nghiên cứu, không phải công cụ chẩn đoán.

\## 1. Hiện trạng

\- Thư mục gốc: E:\\dev\_doAn\\doAn

\- Trọng số: best\_pneumonia\_model.pt ở thư mục gốc (YOLOv8-cls, 2 lớp NORMAL/PNEUMONIA).

\- Môi trường: Windows; Python 3.12.5; Node v22.20.0, npm 10.9.3. Phiên bản thư viện nêu sau đây chỉ để tham khảo, phải xác minh bằng `pip list`: fastapi 0.143.0, uvicorn 0.54.0, ultralytics 8.4.175, torch 2.6.0+cu124.

\- Nếu PowerShell lỗi script policy khi chạy npm thì dùng: cmd.exe /c "npm ..."

\- Trên Windows dùng curl.exe (không dùng alias curl của PowerShell); dùng pathlib cho đường dẫn.

\- KHÔNG sửa/xóa: train.py, evaluate.py, dữ liệu, file .pt. Chỉ tạo file trong backend/ và frontend/ (và file cấu hình gốc như .gitignore, README.md).

\- Tiền xử lý: suy luận phải dùng đúng pipeline lúc huấn luyện/đánh giá (đọc train.py, evaluate.py để lấy imgsz và mọi bước như CLAHE, resize, chuyển kênh). Không tự thêm hay bớt bước nào. Tên lớp lấy từ model.names, không hard-code.

\## 2. Backend (FastAPI, cổng 8000)

Cấu trúc: backend/app/{main.py, api/endpoints.py, core/config.py, services/predictor.py, schemas/prediction.py}, backend/{requirements.txt, run.py, .env.example}.

\- Config: PORT, MODEL\_PATH (tính từ thư mục gốc dự án, không phụ thuộc thư mục chạy lệnh, ghi đè được bằng .env), CORS\_ORIGINS (http://localhost:5173, http://127.0.0.1:5173), MAX\_UPLOAD\_MB (mặc định 10).

\- Model: nạp một lần bằng lifespan (singleton); tự chọn device (cuda nếu có, ngược lại cpu); warm-up 1 lần khi khởi động; không chặn event loop (dùng def hoặc run\_in\_threadpool).

\- GET /api/v1/health: trạng thái server, model, device, tên lớp.

\- POST /api/v1/predict: nhận UploadFile (jpg/jpeg/png), trả:

&#x20; {"success": true, "prediction": "PNEUMONIA", "confidence": 0.985, "probabilities": {"NORMAL": 0.015, "PNEUMONIA": 0.985}, "inference\_time\_ms": 3.4, "advice": "..."}

&#x20; inference\_time\_ms đo bằng time.perf\_counter quanh bước suy luận (thời gian trên máy chủ).

&#x20; advice là câu cố định, không sinh tự do:

&#x20;   PNEUMONIA: "Ảnh có dấu hiệu nghi ngờ viêm phổi theo mô hình. Cần được bác sĩ chuyên khoa xem xét."

&#x20;   NORMAL: "Mô hình chưa ghi nhận dấu hiệu nghi ngờ. Kết quả này không loại trừ bệnh; nếu có triệu chứng, hãy đi khám."

\- GET /api/v1/samples: trả id, nhãn gốc, URL tương đối của 4-6 ảnh mẫu (ít nhất 2 NORMAL, 2 PNEUMONIA), copy ngẫu nhiên từ tập TEST vào backend/samples/, phục vụ qua StaticFiles. Không trả đường dẫn tuyệt đối trên máy.

\- Upload và lỗi: không tin content-type, phải giải mã bằng OpenCV/Pillow để xác thực là ảnh; giới hạn dung lượng; lỗi trả mã 400/413/415/500 kèm {"success": false, "error": "..."}; không lộ stack trace. Không lưu ảnh ra đĩa, không log nội dung ảnh.

\## 3. Frontend (cổng 5173)

Vite + React (JavaScript) + Tailwind CSS v4 (plugin @tailwindcss/vite) + axios + lucide-react. baseURL lấy từ VITE\_API\_URL (mặc định http://localhost:8000/api/v1).

\- Giao diện y tế sạch, tone blue/slate/emerald.

\- Kéo-thả hoặc chọn file, xem trước ảnh; kiểm tra định dạng và dung lượng phía client; thu hồi URL.revokeObjectURL khi đổi ảnh hoặc unmount.

\- 4-6 ảnh mẫu: bấm vào thì tải blob rồi gọi lại đúng endpoint /predict.

\- Bảng "Kết quả phân loại": card nổi bật, thanh tiến trình "Xác suất dự đoán", thời gian xử lý, lời khuyến nghị (từ advice).

\- Trạng thái: spinner/skeleton khi tải, toast lỗi khi mất kết nối backend, nút "Phân tích ảnh khác" để đặt lại.

\- PNEUMONIA -> "Nghi ngờ viêm phổi" (đỏ); NORMAL -> "Chưa ghi nhận dấu hiệu nghi ngờ" (xanh, kèm dòng nhắc không loại trừ bệnh).

\- Không dùng các từ: "chẩn đoán", "phát hiện bệnh", "bình thường/khỏe mạnh", "an tâm", "độ tin cậy", "khám lại".

\- Luôn hiển thị cố định: "Công cụ hỗ trợ sàng lọc phục vụ nghiên cứu, không thay thế chẩn đoán của bác sĩ. Mô hình được huấn luyện trên ảnh X-quang ngực bệnh nhi (1-5 tuổi) từ một cơ sở; chưa được kiểm chứng lâm sàng."

\## 4. Quy tắc làm việc

\- Chỉ làm đúng bước được giao; không làm sang bước khác.

\- Code sạch, modular, comment tiếng Việt, dễ trình bày trong báo cáo.

\- Không đổi hợp đồng JSON của API khi chưa hỏi.

\- Gặp điều chưa rõ thì hỏi trước, không tự đoán.

\- Cuối mỗi bước: liệt kê (1) các giả định đã đặt ra, (2) những gì chưa kiểm tra được, (3) lệnh để tôi tự kiểm tra.

