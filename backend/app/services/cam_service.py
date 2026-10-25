"""
backend/app/services/cam_service.py
Dịch vụ giải thích mô hình bằng phương pháp Grad-CAM (Gradient-weighted Class Activation Mapping).
Tuân thủ nghiêm ngặt các nguyên tắc:
- Bản sao sâu (deepcopy) riêng của mô hình, không làm thay đổi mô hình suy luận dùng chung.
- Sử dụng torch.autograd.grad trên logit đầu ra, ngoài inference_mode.
- Thread-safe thông qua threading.Lock.
- Không lưu ảnh hay heatmap ra đĩa; mã hóa Base64 PNG-RGBA trực tiếp trong bộ nhớ.
"""

import base64
import copy
import threading
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from ultralytics import YOLO

from backend.app.core.config import settings
from backend.app.services.predictor import PredictorService

COLORMAP_NAME: str = "turbo" if hasattr(cv2, "COLORMAP_TURBO") else "viridis"
COLORMAP_CODE: int = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_VIRIDIS)


def configure_process_precision() -> None:
    """
    Cấu hình cờ độ chính xác toán học toàn tiến trình (process-wide flags) của PyTorch trên GPU:
    Tắt TensorFloat32 (TF32) cho cuDNN và CUDA matmul để ép tính toán ma trận thực hiện ở chuẩn float32
    đầy đủ (24-bit mantissa thay vì 10-bit mantissa của TF32).

    Lý do:
    Cờ này ảnh hưởng đến toàn bộ tiến trình Python hiện tại. Nếu bật TF32 (mặc định trên GPU Ampere+),
    sai lệch làm tròn giữa phép tính thủ công và engine dự báo có thể lên đến ~2.3e-3, vượt ngưỡng sai số 1e-3.
    Khi tắt TF32, độ lệch tối đa giảm về ~4.8e-5.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def compute_cam_raw(
    cam_model: nn.Module,
    layer: nn.Module,
    tensor: torch.Tensor,
    class_idx: int,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Tính toán Grad-CAM thô ở cấp module:
    1. Clone tensor và bật requires_grad_(True).
    2. Đăng ký forward hook lưu activation trong đồ thị (không detach).
    3. Forward qua mô hình trong torch.enable_grad().
    4. Trích xuất logit từ out[1] nếu đầu ra là tuple/list.
    5. Tính đạo hàm autograd.grad(logit, activation) và trọng số GAP.
    6. Tính cam_pre (trước ReLU) và cam_raw (sau ReLU).
    7. Gỡ hook trong khối finally.
    
    Trả về:
        (cam_pre, cam_raw, activation.detach(), logits.detach())
    """
    tensor_input = tensor.detach().clone().requires_grad_(True)
    activation: Optional[torch.Tensor] = None

    def hook_fn(module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor) -> None:
        nonlocal activation
        activation = output_tensor

    hook_handle = layer.register_forward_hook(hook_fn)

    try:
        with torch.enable_grad():
            out = cam_model(tensor_input)
            logits = out[1] if isinstance(out, (tuple, list)) else out
            logit = logits[0, class_idx]

            grads = torch.autograd.grad(logit, activation, retain_graph=False, create_graph=False)[0]
            # Global Average Pooling qua chiều không gian (H, W)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            # Tổng có trọng số của các bản đồ đặc trưng
            cam_pre_tensor = (weights * activation).sum(dim=1)[0]
            cam_pre = cam_pre_tensor.detach().cpu().numpy().astype(np.float32)
            cam_raw = np.maximum(cam_pre, 0.0).astype(np.float32)

            return cam_pre, cam_raw, activation.detach(), logits.detach()
    finally:
        hook_handle.remove()


class CamService:
    """
    Dịch vụ giải thích Grad-CAM áp dụng cơ chế Singleton,
    quản lý vòng đời bản sao mô hình và xử lý các yêu cầu explain.
    """

    _instance: Optional["CamService"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.device: str = "cpu"
        self.is_ready: bool = False
        self._cam_model: Optional[nn.Module] = None
        self._cam_layer: Optional[nn.Module] = None
        self._transforms: Optional[Callable] = None
        self._cam_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "CamService":
        """Lấy đối tượng singleton của CamService đảm bảo an toàn luồng (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = CamService()
        return cls._instance

    def initialize(self) -> None:
        """
        Nạp mô hình YOLO bản sao riêng, trích xuất chuỗi transforms chuẩn xác từ predictor,
        sao chép yolo.model sang _cam_model (đối tượng ClassificationModel thuần túy),
        xác định lớp mục tiêu từ cấu hình và đặt chế độ eval().
        """
        model_path = settings.MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file trọng số mô hình tại: {model_path}")

        # Tắt TF32 toàn tiến trình để đảm bảo độ chính xác float32 đầy đủ
        configure_process_precision()

        # Đồng bộ thiết bị tính toán với PredictorService
        predictor = PredictorService.get_instance()
        self.device = predictor.device if predictor.is_ready else ("cuda" if torch.cuda.is_available() else "cpu")

        # Nạp bản sao YOLO riêng biệt
        cam_yolo = YOLO(str(model_path))

        # Chạy warm-up 1 lần với ảnh rỗng để Ultralytics khởi tạo predictor.transforms
        dummy_img = np.zeros((settings.IMGSZ, settings.IMGSZ, 3), dtype=np.uint8)
        cam_yolo.predict(
            source=dummy_img,
            imgsz=settings.IMGSZ,
            device=self.device,
            verbose=False,
        )

        # Trích xuất transforms chuẩn xác từ predictor
        self._transforms = getattr(cam_yolo.model, "transforms", None) or getattr(cam_yolo.predictor, "transforms", None)

        # Bản sao sâu yolo.model thuần túy (không bọc thêm lớp ngoài)
        self._cam_model = copy.deepcopy(cam_yolo.model)
        self._cam_model.eval()
        self._cam_model.to(self.device)

        # Không thay đổi requires_grad của tham số (giữ nguyên False)

        # Lấy lớp mục tiêu cho Grad-CAM
        cam_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")
        self._cam_layer = self._cam_model.get_submodule(cam_layer_name)

        self.is_ready = True
        print(f"[+] CamService đã sẵn sàng: Device={self.device}, Layer={cam_layer_name}")

    def _get_input_tensor(self, rgb_arr: np.ndarray) -> torch.Tensor:
        """
        Chuẩn bị tensor đầu vào cho mô hình theo đúng pipeline ClassificationPredictor.preprocess:
        1. cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)
        2. PIL.Image.fromarray(...)
        3. transforms(...)
        4. unsqueeze(0), chuyển sang device và float32.
        Trả về tensor không yêu cầu gradient (requires_grad=False).
        """
        rgb_converted = cv2.cvtColor(rgb_arr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_converted)
        tensor = self._transforms(pil_img).unsqueeze(0).to(self.device).float()
        return tensor

    @staticmethod
    def _normalize(cam_raw: np.ndarray) -> np.ndarray:
        """Chuẩn hóa Min-Max giữ nguyên hình dạng thô; nếu là mảng hằng thì trả về toàn 0."""
        c_min = float(cam_raw.min())
        c_max = float(cam_raw.max())
        if abs(c_max - c_min) < 1e-12:
            return np.zeros_like(cam_raw, dtype=np.float32)
        norm = (cam_raw - c_min) / (c_max - c_min)
        return norm.astype(np.float32)

    @staticmethod
    def _resize_cam(cam_norm: np.ndarray, size: int = 224) -> np.ndarray:
        """Phóng to bản đồ nhiệt lên kích thước mong muốn bằng nội suy tuyến tính, giới hạn trong [0, 1]."""
        resized = cv2.resize(cam_norm, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _detect_empty(cam_prerelu: np.ndarray, cam_raw: np.ndarray) -> bool:
        """
        Phát hiện bản đồ trống:
        - Toàn 0 sau ReLU.
        - Hoặc peak < 1e-4 * max(|cam_prerelu|.max(), 1e-12).
        Trả về kiểu bool Python chuẩn.
        """
        if bool((cam_raw == 0.0).all()):
            return True
        peak = float(cam_raw.max())
        max_prerelu = float(np.max(np.abs(cam_prerelu)))
        threshold = 1e-4 * max(max_prerelu, 1e-12)
        return bool(peak < threshold)

    @staticmethod
    def _encode_heatmap(cam_norm: np.ndarray) -> str:
        """
        Mã hóa bản đồ nhiệt sang Base64 PNG RGBA:
        - Kênh màu: áp dụng COLORMAP_CODE trên OpenCV BGR gốc (không đổi sang RGB).
        - Kênh alpha: độ trong suốt tỉ lệ thuận với giá trị cam_norm.
        Trả về chuỗi base64 thuần túy.
        """
        idx = np.round(cam_norm * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(idx, COLORMAP_CODE)
        alpha = idx[:, :, np.newaxis]
        bgra = np.concatenate([bgr, alpha], axis=-1)

        success, buf = cv2.imencode(".png", bgra)
        if not success:
            raise RuntimeError("Không thể mã hóa heatmap sang định dạng PNG.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _compute_crop_box(H: int, W: int, imgsz: int = 224) -> Dict[str, int]:
        """
        Tính toán crop_box trên ảnh gốc theo đúng công thức CenterCrop của torchvision:
        - Cạnh dài mới = int(imgsz * dài / ngắn)
        - offset = round((dài_mới - imgsz) / 2)
        - Tọa độ gốc = round(offset * dài / dài_mới)
        - Kích thước cắt = cạnh ngắn gốc
        """
        if W > H:
            long_side = W
            short_side = H
            long_new = int(imgsz * long_side / short_side)
            offset = round((long_new - imgsz) / 2)
            x0 = round(offset * long_side / long_new)
            x1 = x0 + short_side
            y0 = 0
            y1 = short_side
        elif H > W:
            long_side = H
            short_side = W
            long_new = int(imgsz * long_side / short_side)
            offset = round((long_new - imgsz) / 2)
            y0 = round(offset * long_side / long_new)
            y1 = y0 + short_side
            x0 = 0
            x1 = short_side
        else:
            x0 = 0
            y0 = 0
            x1 = W
            y1 = H

        return {
            "x0": int(x0),
            "y0": int(y0),
            "x1": int(x1),
            "y1": int(y1),
        }

    def explain(self, image_bytes: bytes, class_idx: int) -> Dict[str, Any]:
        """
        Thực hiện giải thích Grad-CAM cho một ảnh X-quang:
        Đảm bảo an toàn luồng với threading.Lock, trả về dict đầy đủ theo hợp đồng API.
        """
        if not self.is_ready or self._cam_model is None or self._cam_layer is None:
            raise RuntimeError("CamService chưa được khởi tạo hoàn tất.")

        with self._cam_lock:
            predictor = PredictorService.get_instance()
            rgb_arr = predictor.preprocess_image(image_bytes)
            orig_h, orig_w = rgb_arr.shape[:2]

            tensor = self._get_input_tensor(rgb_arr)
            cam_pre, cam_raw, _, _ = compute_cam_raw(
                self._cam_model,
                self._cam_layer,
                tensor,
                class_idx=class_idx,
            )

            is_empty = self._detect_empty(cam_pre, cam_raw)
            cam_norm = self._normalize(cam_raw)
            cam_resized = self._resize_cam(cam_norm, size=settings.IMGSZ)
            heatmap_b64 = self._encode_heatmap(cam_resized)
            crop_box = self._compute_crop_box(orig_h, orig_w, imgsz=settings.IMGSZ)

            target_class_name = predictor.classes.get(class_idx, f"Class_{class_idx}") if predictor.classes else (
                predictor.class_names_list[class_idx] if class_idx < len(predictor.class_names_list) else str(class_idx)
            )
            target_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")

            return {
                "heatmap_base64": str(heatmap_b64),
                "crop_box": crop_box,
                "colormap": COLORMAP_NAME,
                "is_empty": bool(is_empty),
                "empty_message": "Không có vùng nổi bật" if is_empty else None,
                "raw_peak": float(cam_raw.max()),
                "target_class": str(target_class_name),
                "target_layer": str(target_layer_name),
                "image_size": {
                    "width": int(orig_w),
                    "height": int(orig_h),
                },
                "attention_note": "Vùng mô hình chú ý — không phải bằng chứng lâm sàng",
            }
