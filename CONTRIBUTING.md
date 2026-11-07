# Quy ước làm việc và cập nhật dự án

Tài liệu ngắn để việc cập nhật dự án luôn đơn giản và nhất quán.

## Quy ước commit

Viết bằng **tiếng Việt không dấu**, một dòng, theo dạng:

```
<loai>: <mo ta ngan>
```

| Loại | Dùng khi | Ví dụ |
|---|---|---|
| `them` | Thêm tính năng, file hoặc phân tích mới | `them: nut xem vung mo hinh chu y tren giao dien` |
| `sua` | Sửa lỗi | `sua: khung nhin lech khi anh doc` |
| `cap nhat` | Cải tiến hoặc đổi hành vi có sẵn | `cap nhat: dat nguong tin cay thap duoi 0.85` |
| `kiem thu` | Thêm hoặc sửa test | `kiem thu: them test cho phan hoi 503` |
| `tai lieu` | Chỉ sửa tài liệu | `tai lieu: cap nhat ket qua danh gia v2a` |
| `don dep` | Dọn dẹp, không đổi chức năng | `don dep: bo script mot lan` |

Quy tắc: tối đa khoảng 70 ký tự, viết thường, không dấu chấm cuối, nói *làm gì* chứ không nói "sửa linh tinh".

## Quy trình cập nhật (4 bước)

```bash
git status --short                          # xem thay doi
python -m pytest backend/tests -q           # test phai xanh (33 passed)
git add -A                                  # .gitignore da chan du lieu, mo hinh, .env
git commit -m "them: mo ta ngan"
git push
```

Trước khi `git add -A`, kiểm tra `git status --short` không có file `.env`, `.pt`, hay thư mục dữ liệu (chúng đã bị `.gitignore` chặn; nếu thấy, dừng lại).

## Việc không được làm

- Không commit dữ liệu (`chest_xray/`, `dataset*/`), mô hình (`weights/`, `*.pt`), `backend/.env`, thông tin cá nhân.
- Không nới test hay ngưỡng để cho xanh.
- Không ghi đè `best_pneumonia_model.pt` (mô hình v1 lịch sử); mô hình mới lưu ra `weights/<phien_ban>/` bằng `train.py --export`.
- Số liệu trong tài liệu phải lấy từ kết quả đo thật, không chép từ tài liệu cũ.

## Phát hành

Khi hoàn tất một mốc, đánh dấu phiên bản:

```bash
git tag -a v1.0 -m "phien ban ..."
git push origin v1.0
```
