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


def test_predict_no_explain_key(client):
    """Kiểm tra khi không có tham số explain hoặc explain=false thì không có khóa explain."""
    sample_files = list(settings.SAMPLES_DIR.glob("*.jpeg"))
    sample_img_path = sample_files[0]
    with open(sample_img_path, "rb") as f:
        file_bytes = f.read()

    expected_keys = {
        "success",
        "prediction",
        "confidence",
        "probabilities",
        "inference_time_ms",
        "advice",
    }

    # TH1: không truyền tham số explain
    res1 = client.post(
        "/api/v1/predict",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert set(data1.keys()) == expected_keys
    assert "explain" not in data1

    # TH2: explain=false
    res2 = client.post(
        "/api/v1/predict?explain=false",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert set(data2.keys()) == expected_keys
    assert "explain" not in data2


def test_predict_explain_true_fields(client):
    """Kiểm tra khi explain=true thì trả về đủ 10 trường trong explain và các khóa cũ giữ nguyên."""
    sample_files = list(settings.SAMPLES_DIR.glob("*.jpeg"))
    sample_img_path = sample_files[0]
    with open(sample_img_path, "rb") as f:
        file_bytes = f.read()

    # Gọi với explain=false
    res_no_exp = client.post(
        "/api/v1/predict?explain=false",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    data_no_exp = res_no_exp.json()

    # Gọi với explain=true
    res_exp = client.post(
        "/api/v1/predict?explain=true",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    assert res_exp.status_code == 200
    data_exp = res_exp.json()

    # Các khóa cũ phải khớp y hệt
    for key in ("success", "prediction", "confidence", "probabilities", "advice"):
        assert data_exp[key] == data_no_exp[key]

    # Kiểm tra cấu trúc trường explain
    assert "explain" in data_exp
    exp = data_exp["explain"]

    # 1. heatmap_base64
    assert isinstance(exp.get("heatmap_base64"), str)
    assert not exp["heatmap_base64"].startswith("data:")

    # 2. crop_box
    assert isinstance(exp.get("crop_box"), dict)
    for coord in ("x0", "y0", "x1", "y1"):
        assert isinstance(exp["crop_box"].get(coord), int)

    # 3. colormap
    assert exp.get("colormap") in ("turbo", "viridis")

    # 4. is_empty & 5. empty_message
    assert isinstance(exp.get("is_empty"), bool)
    assert exp.get("empty_message") is None or isinstance(exp.get("empty_message"), str)

    # 6. raw_peak
    assert isinstance(exp.get("raw_peak"), float)

    # 7. target_class & 8. target_layer
    assert exp.get("target_class") == data_exp["prediction"]
    assert isinstance(exp.get("target_layer"), str)

    # 9. image_size
    assert isinstance(exp.get("image_size"), dict)
    assert "width" in exp["image_size"] and "height" in exp["image_size"]
    assert isinstance(exp["image_size"]["width"], int)
    assert isinstance(exp["image_size"]["height"], int)

    # 10. attention_note
    assert exp.get("attention_note") == "Vùng mô hình chú ý — không phải bằng chứng lâm sàng"


def test_predict_two_concurrent_explain(client):
    """Hai request song song với hai ảnh khác nhau đều thành công và trả heatmap khác nhau."""
    from concurrent.futures import ThreadPoolExecutor

    sample_files = list(settings.SAMPLES_DIR.glob("*.jpeg"))
    assert len(sample_files) >= 2, "Cần ít nhất 2 ảnh mẫu để kiểm thử đồng thời."

    with open(sample_files[0], "rb") as f0:
        b0 = f0.read()
    with open(sample_files[1], "rb") as f1:
        b1 = f1.read()

    def send_req(filename, file_bytes):
        return client.post(
            "/api/v1/predict?explain=true",
            files={"file": (filename, file_bytes, "image/jpeg")},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_req0 = executor.submit(send_req, sample_files[0].name, b0)
        f_req1 = executor.submit(send_req, sample_files[1].name, b1)

        r0 = f_req0.result()
        r1 = f_req1.result()

    assert r0.status_code == 200
    assert r1.status_code == 200

    d0 = r0.json()
    d1 = r1.json()

    assert "explain" in d0
    assert "explain" in d1
    assert d0["explain"]["heatmap_base64"] != d1["explain"]["heatmap_base64"]


def test_predict_explain_503_when_cam_not_ready(client, monkeypatch):
    """Khi CamService chưa sẵn sàng, explain=true trả 503, còn explain=false vẫn 200."""
    from backend.app.services.cam_service import CamService

    cam_svc = CamService.get_instance()
    monkeypatch.setattr(cam_svc, "is_ready", False)

    sample_files = list(settings.SAMPLES_DIR.glob("*.jpeg"))
    sample_img_path = sample_files[0]
    with open(sample_img_path, "rb") as f:
        file_bytes = f.read()

    # explain=true phải trả 503
    res_true = client.post(
        "/api/v1/predict?explain=true",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    assert res_true.status_code == 503
    d_true = res_true.json()
    assert d_true["success"] is False
    assert "error" in d_true

    # explain=false vẫn hoạt động bình thường (200)
    res_false = client.post(
        "/api/v1/predict?explain=false",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )
    assert res_false.status_code == 200
    assert res_false.json()["success"] is True


def test_predict_unchanged_after_explain(client):
    """Suy luận bình thường không bị thay đổi sau khi gọi explain=true."""
    sample_files = list(settings.SAMPLES_DIR.glob("*.jpeg"))
    sample_img_path = sample_files[0]
    with open(sample_img_path, "rb") as f:
        file_bytes = f.read()

    # 1. Gọi trước
    r_before = client.post(
        "/api/v1/predict?explain=false",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    ).json()

    # 2. Gọi explain
    _ = client.post(
        "/api/v1/predict?explain=true",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    )

    # 3. Gọi sau
    r_after = client.post(
        "/api/v1/predict?explain=false",
        files={"file": (sample_img_path.name, file_bytes, "image/jpeg")},
    ).json()

    assert r_before["prediction"] == r_after["prediction"]
    assert r_before["confidence"] == r_after["confidence"]
    assert r_before["probabilities"] == r_after["probabilities"]

