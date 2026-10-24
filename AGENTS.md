# AGENTS.md

## Bối cảnh
Công cụ SÀNG LỌC viêm phổi từ X-quang ngực (YOLOv8-cls + FastAPI + React 19 / Vite / Tailwind v4).
Đây là công cụ hỗ trợ, KHÔNG phải công cụ chẩn đoán. Giao diện bằng tiếng Việt.

## Quy tắc y đức (không được vi phạm)
- Dùng "Nghi ngờ viêm phổi" / "Không phát hiện dấu hiệu nghi ngờ". KHÔNG dùng "Chẩn đoán", "Bạn bị bệnh".
- Luôn hiển thị banner cảnh báo: kết quả chỉ mang tính hỗ trợ, không thay thế bác sĩ.
- Không lưu ảnh bệnh nhân ra đĩa hay ghi vào log; xử lý trong bộ nhớ.
- Khi thiết kế lại giao diện, các quy tắc trên vẫn phải được giữ nguyên.

## Giao diện
- Phong cách lâm sàng: tối giản, tương phản cao, đáng tin cậy. Không dùng hiệu ứng hào nhoáng (glassmorphism, shiny button, marquee...).
- Hỗ trợ Light/Dark nếu đã thống nhất trong thiết kế; vùng xem ảnh X-quang ưu tiên nền trung tính.
- Đạt WCAG AA. Không truyền đạt kết quả chỉ bằng màu sắc (kèm chữ hoặc biểu tượng).
- Tailwind v4: cấu hình CSS-first bằng `@theme`, không dùng `tailwind.config.js`.
- Mọi chữ hiển thị cho người dùng bằng tiếng Việt, đủ dấu.

## Mô hình / ML
- Không thêm augmentation lật ngang/dọc hay đổi màu.
- CLAHE khi suy luận phải giống hệt lúc huấn luyện (clipLimit=2.0, tile 8x8).
- Không dùng tập test (624 ảnh) để huấn luyện hay chỉnh tham số.
- Tên lớp đọc từ `model.names`, không hard-code.

## Cách làm việc
- Việc lớn: nêu kế hoạch ngắn trước khi sửa. Chỉ sửa file liên quan, không refactor lan man.
- Sau khi sửa phải chạy test/lint/build và báo kết quả thật. Không xóa test để cho pass.
- Không thêm thư viện mới khi chưa hỏi. Không hard-code khóa API; dùng biến môi trường.
- Không chỉnh sửa trọng số mô hình (`*.pt`) hay thư mục `dataset/`, `weights/`, `runs/` nếu không được yêu cầu rõ.
- Cổng mặc định: backend 8000, frontend 5173.
## Grad-CAM
- Gọi là "vùng mô hình chú ý", KHÔNG gọi là "vị trí bệnh", "ổ viêm" hay "vùng tổn thương".
- Kèm ghi chú trên giao diện: bản đồ chỉ giải thích mô hình, không phải bằng chứng lâm sàng.
- Tiền xử lý (CLAHE...) phải dùng lại đúng hàm của luồng suy luận hiện có, không viết bản sao.
- Tính CAM ngoài torch.inference_mode(); chạy qua run_in_threadpool.
- Không lưu ảnh gốc hay heatmap ra đĩa; trả về dạng base64 hoặc stream.
- Nếu bản đồ trống (toàn 0 hoặc hằng), báo "không có vùng nổi bật", không bịa diễn giải.
- Test: kích thước bản đồ khớp ảnh, giá trị trong [0,1], có test cho trường hợp bản đồ trống.
## Chia dữ liệu và huấn luyện lại
- Chia train/val theo NHÓM BỆNH NHÂN (person<ID> cho PNEUMONIA, IM-<số> cho NORMAL, không phân biệt họ IM- hay NORMAL2-IM-), không chia theo từng ảnh.
- Tập test chính thức 624 ảnh giữ nguyên. Mọi bệnh nhân có mặt ở test phải bị loại khỏi train/val (hiện có khoảng 65% bệnh nhân test trùng với train+val).
- Loại ảnh trùng hệt (SHA-256) trước khi chia.
- Tập test chỉ dùng đánh giá một lần ở cuối; chọn mô hình và ngưỡng bằng val.
- Chỉ đọc các file ảnh (.jpg/.jpeg/.png); bỏ qua .DS_Store, ._* và thư mục __MACOSX.
- Lưu manifest CSV (file, SHA-256, mã bệnh nhân, nhãn, tập, fold) và ghi seed; không ghi đè dataset/ cũ, tạo dataset_v2/.
- Không ghi đè mô hình cũ; lưu mô hình mới kèm phiên bản.
- Sau khi train lại: chạy evaluate.py, pytest, kiểm chứng lại Grad-CAM, rồi mới cập nhật tài liệu.