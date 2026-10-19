"""
backend/tests/test_api.py
Bộ kiểm thử tự động (Unit / Integration Tests) cho các API của Backend.
Sử dụng pytest và FastAPI TestClient (httpx) với context manager để kích hoạt lifespan.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app


@pytest.fixture(scope="session")
def client():
    """Tạo client kiểm thử với vòng đời lifespan được kích hoạt đầy đủ."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Kiểm tra endpoint GET /api/v1/health."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["device"] in ("cuda", "cpu")
    assert "NORMAL" in data["classes"]
    assert "PNEUMONIA" in data["classes"]
    assert data["use_clahe"] is True


def test_get_samples(client):
    """Kiểm tra endpoint GET /api/v1/samples."""
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["samples"]) >= 4

    labels = [s["original_label"] for s in data["samples"]]
    assert labels.count("NORMAL") >= 2
    assert labels.count("PNEUMONIA") >= 2

    # Đảm bảo URL trả về là tương đối, không phải đường dẫn tuyệt đối
    for s in data["samples"]:
        assert s["url"].startswith("/static/samples/")
        assert not s["url"].startswith("E:") and not s["url"].startswith("C:")


def test_predict_valid_image(client):
    """Kiểm tra suy luận với ảnh mẫu hợp lệ."""
    sample_files = list(settings.SAMPLES_DIR.glob("*.jpeg"))
    assert len(sample_files) > 0, "Không tìm thấy ảnh mẫu để kiểm thử."

    sample_img_path = sample_files[0]
    with open(sample_img_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/v1/predict",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= data["confidence"] <= 1.0
    assert "NORMAL" in data["probabilities"]
    assert "PNEUMONIA" in data["probabilities"]
    assert data["inference_time_ms"] > 0
    assert len(data["advice"]) > 0


def test_predict_invalid_extension(client):
    """Kiểm tra tải lên file không đúng định dạng cho phép (.txt)."""
    fake_txt = io.BytesIO(b"Hello world not an image")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 415
    data = response.json()
    assert data["success"] is False
    assert "error" in data


def test_predict_corrupted_image(client):
    """Kiểm tra tải lên file có đuôi .jpg nhưng nội dung bị hỏng."""
    corrupt_data = io.BytesIO(b"This is just corrupt binary data")
    response = client.post(
        "/api/v1/predict",
        files={"file": ("corrupt.jpg", corrupt_data, "image/jpeg")},
    )
    assert response.status_code in (400, 415)
    data = response.json()
    assert data["success"] is False
    assert "error" in data


def test_predict_oversized_file(client):
    """Kiểm tra tải lên file vượt quá MAX_UPLOAD_MB."""
    # Tạo dữ liệu giả lập 11MB (> 10MB)
    oversized_data = b"0" * (11 * 1024 * 1024)
    response = client.post(
        "/api/v1/predict",
        files={"file": ("oversized.jpg", io.BytesIO(oversized_data), "image/jpeg")},
    )
    assert response.status_code == 413
    data = response.json()
    assert data["success"] is False
    assert "vượt quá giới hạn" in data["error"]
