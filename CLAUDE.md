@AGENTS.md

## Claude Code

- Nhánh làm việc hiện tại: `feature/retrain`. Trước khi sửa, chạy `git branch --show-current`; không commit trực tiếp lên `main`.
- Kiểm tra sau mỗi thay đổi backend: `python -m pytest backend/tests -q`, và báo kết quả thật (số test xanh, đỏ).
- Không bao giờ sửa, xóa hay ghi đè: `chest_xray/`, `dataset/`, `dataset_v2/`, `weights/`, `runs/`, `best_pneumonia_model.pt`. Mô hình mới lưu ra `weights/<phiên_bản>/` bằng `train.py --export`.
- Không nới test, ngưỡng hay assertion để cho xanh; không dùng skip hoặc xfail. Nếu cho rằng test sai thì dừng và hỏi.
- Việc lớn (nhiều file, đổi hợp đồng API, đổi tiền xử lý ảnh): dùng plan mode, nêu kế hoạch rồi chờ xác nhận trước khi sửa.
- Số liệu đưa vào tài liệu phải lấy từ kết quả đo thật (`evaluate.py`, `analyze_models.py`), không chép từ tài liệu cũ.
- Tiến độ và kết quả phân tích: xem `DATA_PROCESSING.md`. Khi tài liệu và mã nguồn mâu thuẫn, tin mã nguồn và kết quả chạy.
- Commit: tieng Viet khong dau, dang '<loai>: <mo ta ngan>' (xem CONTRIBUTING.md); khong tu push.
