"""
backend/app/main.py
Điểm vào chính của ứng dụng FastAPI.
Cấu hình vòng đời ứng dụng (lifespan), CORS middleware, StaticFiles và xử lý lỗi tập trung.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.endpoints import router as api_router
from backend.app.core.config import settings
from backend.app.services.predictor import PredictorService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời khởi động và tắt của ứng dụng FastAPI.
    Nạp mô hình YOLOv8-cls đúng 1 lần khi server khởi động (Singleton).
    """
    print("[*] Đang khởi động hệ thống Backend và nạp mô hình...")
    predictor = PredictorService.get_instance()
    predictor.initialize()
    yield
    print("[*] Đang tắt hệ thống Backend...")


app = FastAPI(
    title="Hệ thống hỗ trợ sàng lọc viêm phổi từ ảnh X-quang ngực",
    description="Công cụ HỖ TRỢ SÀNG LỌC phục vụ nghiên cứu, không phải công cụ chẩn đoán.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 1. Cấu hình CORS middleware cho Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Cấu hình phục vụ ảnh tĩnh (StaticFiles) cho ảnh mẫu
settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/samples",
    StaticFiles(directory=str(settings.SAMPLES_DIR)),
    name="samples",
)

# 3. Đăng ký các router API
app.include_router(api_router)


# 4. Trình xử lý lỗi toàn cục để không để lộ stack trace nhạy cảm
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Bắt mọi ngoại lệ không mong muốn và trả về định dạng chuẩn."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Đã xảy ra lỗi nội bộ trên máy chủ. Vui lòng thử lại sau.",
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    """Trang chủ chuyển hướng / gợi ý tài liệu."""
    return {
        "message": "Chest X-Ray Pneumonia Screening API is running",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
