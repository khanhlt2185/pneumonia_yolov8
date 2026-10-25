"""Bảo đảm mô hình mặc định của backend là v2a ở cả cấu hình và file mẫu (chống lệch giữa mã và tài liệu)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_config_default_and_env_example_use_v2a():
    cfg = (ROOT / "backend" / "app" / "core" / "config.py").read_text(encoding="utf-8")
    env = (ROOT / "backend" / ".env.example").read_text(encoding="utf-8-sig")
    assert 'os.getenv("MODEL_PATH", "weights/v2/best_v2a.pt")' in cfg
    assert "MODEL_PATH=weights/v2/best_v2a.pt" in env
