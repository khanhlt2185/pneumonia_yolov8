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
