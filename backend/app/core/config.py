"""
backend/app/core/config.py
Cấu hình tập trung cho hệ thống Backend FastAPI.
Đọc các biến môi trường từ file .env và tính toán đường dẫn tương đối từ thư mục gốc.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Xác định thư mục gốc dự án (thư mục chứa file best_pneumonia_model.pt và SPEC.md)
# backend/app/core/config.py -> parents[3] là thư mục doAn
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]

# Đường dẫn nạp file .env từ backend/.env hoặc .env ở gốc
ENV_PATH_BACKEND = CURRENT_FILE.parents[2] / ".env"
ENV_PATH_ROOT = PROJECT_ROOT / ".env"

if ENV_PATH_BACKEND.exists():
    load_dotenv(dotenv_path=ENV_PATH_BACKEND)
elif ENV_PATH_ROOT.exists():
    load_dotenv(dotenv_path=ENV_PATH_ROOT)
else:
    load_dotenv()


class Settings:
    """Cấu hình các tham số hệ thống."""

    # Cổng chạy FastAPI server
    PORT: int = int(os.getenv("PORT", "8000"))

    # Đường dẫn mô hình YOLOv8-cls
    # Tính từ thư mục gốc dự án, không phụ thuộc vào thư mục chạy lệnh
    _raw_model_path: str = os.getenv("MODEL_PATH", "best_pneumonia_model.pt")
    MODEL_PATH: Path = (
        Path(_raw_model_path)
        if Path(_raw_model_path).is_absolute()
        else (PROJECT_ROOT / _raw_model_path).resolve()
    )

    # Nguồn gốc cho phép CORS (Frontend)
    _raw_cors: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    CORS_ORIGINS: List[str] = [
        origin.strip() for origin in _raw_cors.split(",") if origin.strip()
    ]

    # Giới hạn kích thước upload ảnh (Megabytes và bytes)
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

    # Bật/tắt CLAHE khi suy luận (mặc định: True)
    USE_CLAHE: bool = os.getenv("USE_CLAHE", "true").lower() in ("true", "1", "yes")

    # Đường dẫn thư mục ảnh mẫu (Static files)
    SAMPLES_DIR: Path = (CURRENT_FILE.parents[2] / "samples").resolve()

    # Kích thước ảnh đầu vào YOLOv8 mặc định
    IMGSZ: int = 224

    # Định dạng ảnh hợp lệ
    VALID_EXTENSIONS: set = {".jpg", ".jpeg", ".png"}


settings = Settings()
