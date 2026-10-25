"""
backend/app/schemas/prediction.py
Định nghĩa Pydantic models cho các yêu cầu và phản hồi API.
Tuân thủ nghiêm ngặt hợp đồng JSON trong SPEC.md.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Phản hồi kiểm tra trạng thái sức khỏe hệ thống (GET /api/v1/health)."""

    status: str = Field(..., description="Trạng thái server ('healthy')")
    model_loaded: bool = Field(..., description="Mô hình đã được nạp hay chưa")
    device: str = Field(..., description="Thiết bị tính toán đang dùng ('cuda' hoặc 'cpu')")
    classes: List[str] = Field(..., description="Danh sách tên lớp phân loại từ model.names")
    use_clahe: bool = Field(..., description="Cấu hình tiền xử lý CLAHE đang bật hay tắt")


class PredictionSuccessResponse(BaseModel):
    """
    Phản hồi khi suy luận thành công (POST /api/v1/predict).
    Đúng định dạng quy định trong SPEC.md:
    {"success": true, "prediction": "PNEUMONIA", "confidence": 0.985,
     "probabilities": {"NORMAL": 0.015, "PNEUMONIA": 0.985},
     "inference_time_ms": 3.4, "advice": "..."}
    """

    success: bool = True
    prediction: str = Field(..., description="Lớp dự đoán có xác suất cao nhất (NORMAL hoặc PNEUMONIA)")
    confidence: float = Field(..., description="Độ tin cậy của lớp dự đoán (từ 0.0 đến 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Chi tiết phân phối xác suất của từng lớp")
    inference_time_ms: float = Field(..., description="Thời gian suy luận thuần trên máy chủ (ms)")
    advice: str = Field(..., description="Lời khuyến nghị y tế cố định theo lớp dự đoán")


class ErrorResponse(BaseModel):
    """
    Phản hồi khi có lỗi (mã 400, 413, 415, 500).
    Đúng định dạng SPEC.md: {"success": false, "error": "..."}
    """

    success: bool = False
    error: str = Field(..., description="Thông báo lỗi mô tả nguyên nhân")


class SampleItem(BaseModel):
    """Thông tin một ảnh mẫu."""

    id: str = Field(..., description="Mã định danh ảnh mẫu")
    original_label: str = Field(..., description="Nhãn gốc của ảnh (NORMAL hoặc PNEUMONIA)")
    label: str = Field(..., description="Bí danh của nhãn gốc để thuận tiện cho client")
    url: str = Field(..., description="Đường dẫn URL tương đối phục vụ qua StaticFiles")


class SampleListResponse(BaseModel):
    """Danh sách ảnh mẫu (GET /api/v1/samples)."""

    success: bool = True
    samples: List[SampleItem] = Field(..., description="Danh sách 4-6 ảnh mẫu")


class CropBox(BaseModel):
    """Tọa độ hộp cắt trung tâm tương ứng với CenterCrop trên ảnh gốc."""

    x0: int = Field(..., description="Tọa độ x góc trên bên trái")
    y0: int = Field(..., description="Tọa độ y góc trên bên trái")
    x1: int = Field(..., description="Tọa độ x góc dưới bên phải")
    y1: int = Field(..., description="Tọa độ y góc dưới bên phải")


class ImageSize(BaseModel):
    """Kích thước chiều rộng và chiều cao ảnh gốc."""

    width: int = Field(..., description="Chiều rộng ảnh (pixel)")
    height: int = Field(..., description="Chiều cao ảnh (pixel)")


class ExplainResult(BaseModel):
    """
    Kết quả giải thích Grad-CAM phục vụ nghiên cứu và hỗ trợ sàng lọc.
    Tuân thủ nghiêm ngặt nguyên tắc đạo đức y tế:
    - Không khẳng định vị trí tổn thương lâm sàng.
    - Chú thích rõ ràng vùng chú ý của mô hình.
    """

    heatmap_base64: str = Field(..., description="Chuỗi Base64 ảnh heatmap PNG-RGBA (không có tiền tố data:)")
    crop_box: CropBox = Field(..., description="Hộp cắt tương ứng vùng mô hình quan sát")
    colormap: str = Field(..., description="Bảng màu hiển thị (turbo hoặc viridis)")
    is_empty: bool = Field(..., description="Cờ đánh dấu bản đồ nhiệt không có tín hiệu nổi bật")
    empty_message: Optional[str] = Field(None, description="Thông điệp cảnh báo nếu bản đồ rỗng")
    raw_peak: float = Field(..., description="Giá trị cực đại của activation map trước khi chuẩn hóa")
    target_class: str = Field(..., description="Tên lớp được giải thích")
    target_layer: str = Field(..., description="Tên lớp mạng được hook activation")
    image_size: ImageSize = Field(..., description="Kích thước gốc của ảnh đầu vào")
    attention_note: str = Field(
        "Vùng mô hình chú ý — không phải bằng chứng lâm sàng",
        description="Lưu ý y tế bắt buộc",
    )

