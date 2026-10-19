"""
backend/app/api/endpoints.py
Định nghĩa các router API:
- GET  /api/v1/health
- POST /api/v1/predict
- GET  /api/v1/samples
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.app.core.config import settings
from backend.app.schemas.prediction import (
    ErrorResponse,
    HealthResponse,
    PredictionSuccessResponse,
    SampleItem,
    SampleListResponse,
)
from backend.app.services.predictor import ImageValidationError, PredictorService

router = APIRouter(prefix="/api/v1")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái hệ thống và mô hình",
    description="Trả về trạng thái hoạt động của server, mô hình, thiết bị tính toán và các lớp phân loại.",
)
async def health_check() -> HealthResponse:
    """Endpoint kiểm tra sức khỏe của server."""
    predictor = PredictorService.get_instance()
    return HealthResponse(
        status="healthy" if predictor.is_ready else "starting",
        model_loaded=predictor.is_ready,
        device=predictor.device,
        classes=predictor.class_names_list,
        use_clahe=settings.USE_CLAHE,
    )


@router.post(
    "/predict",
    response_model=PredictionSuccessResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Dữ liệu yêu cầu không hợp lệ"},
        413: {"model": ErrorResponse, "description": "Dung lượng file vượt quá giới hạn"},
        415: {"model": ErrorResponse, "description": "Định dạng file không được hỗ trợ"},
        500: {"model": ErrorResponse, "description": "Lỗi nội bộ server"},
    },
    summary="Dự đoán phân loại viêm phổi từ ảnh X-quang",
    description="Nhận file ảnh X-quang (jpg/jpeg/png), thực hiện tiền xử lý CLAHE và suy luận mô hình.",
)
async def predict_image(
    file: UploadFile = File(..., description="File ảnh X-quang ngực cần phân tích"),
) -> Any:
    """Nhận file ảnh tải lên, giải mã, chạy tiền xử lý CLAHE và suy luận mô hình."""
    # 1. Kiểm tra phần mở rộng định dạng file cơ bản
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.VALID_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content={
                "success": False,
                "error": f"Định dạng file '{suffix}' không được hỗ trợ. Chỉ chấp nhận JPG, JPEG, PNG.",
            },
        )

    # 2. Đọc nội dung file ảnh vào bộ nhớ (không lưu ra đĩa)
    try:
        contents = await file.read()
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "error": "Không thể đọc dữ liệu từ file tải lên."},
        )

    # 3. Kiểm tra giới hạn dung lượng file
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            content={
                "success": False,
                "error": f"Dung lượng file ({len(contents) / (1024 * 1024):.2f} MB) vượt quá giới hạn tối đa {settings.MAX_UPLOAD_MB} MB.",
            },
        )

    # 4. Thực thi suy luận qua predictor service (chạy trong threadpool để không chặn event loop)
    predictor = PredictorService.get_instance()
    try:
        result = await run_in_threadpool(predictor.predict, contents)
        return result
    except ImageValidationError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"success": False, "error": e.message},
        )
    except Exception as e:
        # Không lộ chi tiết traceback nhạy cảm ra ngoài client
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Đã xảy ra lỗi trong quá trình xử lý ảnh trên máy chủ.",
            },
        )


@router.get(
    "/samples",
    response_model=SampleListResponse,
    summary="Lấy danh sách ảnh mẫu thử nghiệm",
    description="Trả về danh sách 4-6 ảnh mẫu được lưu trữ trong thư mục backend/samples.",
)
async def get_sample_images() -> SampleListResponse:
    """
    Trả về danh sách ảnh mẫu phục vụ qua StaticFiles.
    Không để lộ đường dẫn tuyệt đối trên máy chủ.
    """
    samples_dir = settings.SAMPLES_DIR
    sample_items: List[SampleItem] = []

    if samples_dir.exists():
        files = sorted(
            [
                f
                for f in samples_dir.iterdir()
                if f.is_file() and f.suffix.lower() in settings.VALID_EXTENSIONS
            ]
        )
        for idx, f in enumerate(files, start=1):
            # Xác định nhãn dựa trên tên file
            fname_upper = f.name.upper()
            if "NORMAL" in fname_upper:
                label = "NORMAL"
            elif "PNEUMONIA" in fname_upper:
                label = "PNEUMONIA"
            else:
                label = "UNKNOWN"

            sample_items.append(
                SampleItem(
                    id=f"sample_{idx}",
                    original_label=label,
                    label=label,
                    url=f"/static/samples/{f.name}",
                )
            )

    return SampleListResponse(success=True, samples=sample_items)
