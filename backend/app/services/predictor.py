"""
backend/app/services/predictor.py
Service quản lý mô hình YOLOv8-cls và thực hiện tiền xử lý, suy luận.
Áp dụng cơ chế Singleton, tự chọn device, warm-up và pipeline tiền xử lý CLAHE chuẩn xác.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.app.core.config import settings


class ImageValidationError(Exception):
    """Ngoại lệ khi ảnh không hợp lệ hoặc không thể giải mã."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PredictorService:
    """Singleton service quản lý vòng đời mô hình và thực hiện suy luận."""

    _instance: Optional["PredictorService"] = None

    def __init__(self) -> None:
        self.model: Optional[YOLO] = None
        self.device: str = "cpu"
        self.classes: Dict[int, str] = {}
        self.class_names_list: List[str] = []
        self.is_ready: bool = False

    @classmethod
    def get_instance(cls) -> "PredictorService":
        """Lấy đối tượng singleton của PredictorService."""
        if cls._instance is None:
            cls._instance = PredictorService()
        return cls._instance

    def initialize(self) -> None:
        """
        Nạp mô hình YOLOv8, xác định thiết bị (cuda/cpu) và chạy warm-up 1 lần.
        Được gọi trong lifespan của FastAPI app.
        """
        model_path = settings.MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file trọng số mô hình tại: {model_path}. "
                f"Vui lòng kiểm tra cấu hình MODEL_PATH trong .env."
            )

        # Tự động chọn thiết bị tính toán
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Nạp mô hình (YOLOv8-cls)
        self.model = YOLO(str(model_path))

        # Lấy tên các lớp trực tiếp từ mô hình, không hard-code
        self.classes = dict(self.model.names)
        self.class_names_list = [self.classes[i] for i in sorted(self.classes.keys())]

        # Khởi động nóng (warm-up) 1 lần với ảnh rỗng kích thước imgsz
        try:
            dummy_img = np.zeros(
                (settings.IMGSZ, settings.IMGSZ, 3), dtype=np.uint8
            )
            self.model.predict(
                source=dummy_img,
                imgsz=settings.IMGSZ,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            # Nếu warm-up thất bại, ghi log cảnh báo nhưng không làm sập server
            print(f"[!] Warm-up warning: {e}")

        self.is_ready = True
        print(
            f"[+] PredictorService đã sẵn sàng: Device={self.device}, "
            f"Classes={self.classes}, Model={model_path.name}"
        )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Xác thực và tiền xử lý ảnh từ mảng byte.

        Quy trình chuẩn y tế:
        1. Giải mã bằng OpenCV (không tin Content-Type của client).
        2. Nếu bật USE_CLAHE: chuyển mức xám, chuẩn hóa uint8, áp dụng CLAHE
           (clipLimit=2.0, tileGridSize=(8, 8)) theo ĐÚNG logic src/preprocess.py.
        3. Chuyển sang 3 kênh (RGB) để đưa vào YOLOv8-cls.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ImageValidationError("File rỗng, không có dữ liệu.", status_code=400)

        # Chuyển đổi bytes thành mảng numpy để OpenCV giải mã
        np_arr = np.frombuffer(image_bytes, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if decoded is None or decoded.size == 0:
            raise ImageValidationError(
                "Không thể giải mã dữ liệu ảnh. Định dạng không hợp lệ hoặc file bị hỏng.",
                status_code=415,
            )

        # Pipeline tiền xử lý theo cấu hình USE_CLAHE
        if settings.USE_CLAHE:
            # 1. Đảm bảo ảnh luôn ở dạng đơn kênh (Grayscale)
            if len(decoded.shape) == 3:
                # Nếu ảnh có kênh Alpha (4 kênh PNG)
                if decoded.shape[2] == 4:
                    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            else:
                gray = decoded

            # 2. Đảm bảo độ sâu màu 8-bit (uint8)
            if gray.dtype != np.uint8:
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(
                    np.uint8
                )

            # 3. Áp dụng thuật toán cân bằng biểu đồ tần suất thích ứng giới hạn tương phản (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)

            # 4. Chuyển sang 3 kênh (RGB) theo yêu cầu kiến trúc của YOLOv8-cls
            processed = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
        else:
            # Nếu tắt CLAHE: chuyển trực tiếp sang RGB
            if len(decoded.shape) == 2:
                processed = cv2.cvtColor(decoded, cv2.COLOR_GRAY2RGB)
            elif decoded.shape[2] == 4:
                bgr = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
                processed = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                processed = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        return processed

    def get_advice(self, prediction_label: str) -> str:
        """Lời khuyến nghị y tế cố định theo SPEC.md (không sinh tự do)."""
        if prediction_label == "PNEUMONIA":
            return (
                "Ảnh có dấu hiệu nghi ngờ viêm phổi theo mô hình. "
                "Cần được bác sĩ chuyên khoa xem xét."
            )
        elif prediction_label == "NORMAL":
            return (
                "Mô hình chưa ghi nhận dấu hiệu nghi ngờ. "
                "Kết quả này không loại trừ bệnh; nếu có triệu chứng, hãy đi khám."
            )
        else:
            return "Kết quả cần được bác sĩ chuyên khoa xem xét."

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Thực hiện suy luận phân loại cho một ảnh X-quang.

        Đo thời gian inference_time_ms chính xác quanh bước suy luận mô hình.
        Trả về kết quả chuẩn theo hợp đồng JSON của SPEC.md.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError("Mô hình chưa được nạp vào bộ nhớ.")

        # Bước tiền xử lý ảnh
        processed_img = self.preprocess_image(image_bytes)

        # Đo thời gian suy luận (inference time) bằng time.perf_counter quanh bước predict
        t_start = time.perf_counter()
        results = self.model.predict(
            source=processed_img,
            imgsz=settings.IMGSZ,
            device=self.device,
            verbose=False,
        )
        t_end = time.perf_counter()
        inference_time_ms = round((t_end - t_start) * 1000.0, 2)

        result = results[0]
        probs = result.probs

        # Lấy lớp có xác suất cao nhất
        top1_idx = int(probs.top1)
        pred_label = self.classes.get(top1_idx, f"Class_{top1_idx}")
        confidence = float(probs.top1conf.item())

        # Chi tiết xác suất từng lớp theo tên lớp (không hard-code)
        probabilities: Dict[str, float] = {}
        for idx, cls_name in self.classes.items():
            prob_val = float(probs.data[idx].item())
            probabilities[cls_name] = round(prob_val, 4)

        advice = self.get_advice(pred_label)

        return {
            "success": True,
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "inference_time_ms": inference_time_ms,
            "advice": advice,
        }
