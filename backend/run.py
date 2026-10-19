"""
backend/run.py
Script chạy ứng dụng backend FastAPI trên cổng cấu hình (mặc định 8000).

Cách chạy:
    python backend/run.py
"""

import sys
from pathlib import Path

# Đảm bảo mã hóa UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import 'backend.app' an toàn
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from backend.app.core.config import settings

if __name__ == "__main__":
    print(f"[*] Khởi động server tại: http://0.0.0.0:{settings.PORT}")
    print(f"[*] API Health check : http://localhost:{settings.PORT}/api/v1/health")
    print(f"[*] API Swagger docs : http://localhost:{settings.PORT}/docs")
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=False,
    )
