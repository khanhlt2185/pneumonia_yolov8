"""
backend/tests/test_cam_service.py
Bộ kiểm thử đơn vị và tích hợp cho CamService và hàm compute_cam_raw.
Tuân thủ nghiêm ngặt nguyên tắc Test-Driven Development (TDD):
- Không dùng skip, xfail, try/except nuốt lỗi quanh assertion.
- Dùng pytest.MonkeyPatch phạm vi module đặt settings.MODEL_PATH về weights/v2/best_v2a.pt và dọn dẹp singleton.
- Các hàm thuần là staticmethod của CamService: _normalize, _resize_cam, _detect_empty (trả bool), _encode_heatmap, _compute_crop_box.
- Lõi tính CAM là compute_cam_raw(cam_model, layer, tensor, class_idx) trả (cam_prerelu, cam_raw, activation, logits).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import base64
import copy
import random
import threading

import cv2
import numpy as np
from PIL import Image
import pytest
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torchcam.methods import GradCAM
from ultralytics import YOLO

from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.services.predictor import PredictorService
from backend.app.services.cam_service import CamService, compute_cam_raw


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_module_environment():
    """
    Fixture phạm vi module:
    1. Đặt settings.MODEL_PATH trỏ về weights/v2/best_v2a.pt và CAM_LAYER bằng pytest.MonkeyPatch.
    2. Reset singleton trước và sau phiên test để không ảnh hưởng đến các file test khác.
    """
    mp = pytest.MonkeyPatch()
    v2a_path = (PROJECT_ROOT / "weights" / "v2" / "best_v2a.pt").resolve()
    assert v2a_path.exists(), f"Không tìm thấy file mô hình tại {v2a_path}"

    mp.setattr(settings, "MODEL_PATH", v2a_path)
    mp.setattr(settings, "CAM_LAYER", "model.9.conv", raising=False)

    # Reset singletons trước khi khởi tạo
    PredictorService._instance = None
    if hasattr(CamService, "_instance"):
        CamService._instance = None

    yield

    # Teardown: dọn dẹp singleton và hoàn tác monkeypatch
    PredictorService._instance = None
    if hasattr(CamService, "_instance"):
        CamService._instance = None
    mp.undo()


@pytest.fixture(scope="module")
def predictor_instance(setup_module_environment):
    """Khởi tạo PredictorService với mô hình v2a."""
    predictor = PredictorService.get_instance()
    if not predictor.is_ready:
        predictor.initialize()
    return predictor


@pytest.fixture(scope="module")
def cam_service_instance(setup_module_environment, predictor_instance):
    """Lấy và khởi tạo CamService singleton."""
    cam_svc = CamService.get_instance()
    if not cam_svc.is_ready:
        cam_svc.initialize()
    return cam_svc


def get_20_test_image_paths():
    """Lấy mẫu cố định 20 ảnh thô từ chest_xray/test (10 NORMAL, 10 PNEUMONIA, seed 42)."""
    normal_dir = PROJECT_ROOT / "chest_xray" / "test" / "NORMAL"
    pneumonia_dir = PROJECT_ROOT / "chest_xray" / "test" / "PNEUMONIA"
    valid_exts = {".jpg", ".jpeg", ".png"}

    normal_files = sorted([f for f in normal_dir.iterdir() if f.suffix.lower() in valid_exts])
    pneumonia_files = sorted([f for f in pneumonia_dir.iterdir() if f.suffix.lower() in valid_exts])

    rng = random.Random(42)
    selected_normal = rng.sample(normal_files, 10)
    selected_pneumonia = rng.sample(pneumonia_files, 10)
    return selected_normal + selected_pneumonia


def create_smooth_gradient_image(height: int, width: int) -> np.ndarray:
    """Tạo ảnh mảng numpy 3 kênh với gradient mượt để kiểm tra hình học biến đổi."""
    y = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    ch0 = (x + y) / 2.0
    ch1 = np.abs(x - y)
    ch2 = 255.0 - ch0
    img = np.stack([ch0, ch1, ch2], axis=-1).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Nhóm 1: Khởi động, kiến trúc lớp và kiểm tra đồng bộ trọng số
# ---------------------------------------------------------------------------

def test_cam_service_singleton_and_init(cam_service_instance, predictor_instance):
    """Kiểm tra CamService tuân thủ singleton, nạp mô hình riêng và cùng thiết bị."""
    svc1 = CamService.get_instance()
    svc2 = CamService.get_instance()
    assert svc1 is svc2
    assert svc1.is_ready is True
    assert svc1.device == predictor_instance.device
    assert svc1._cam_model is not None


def test_cam_and_predictor_use_same_weights(cam_service_instance, predictor_instance):
    """Mô hình của CamService và PredictorService phải dùng cùng checkpoint và trọng số giống hệt."""
    assert Path(settings.MODEL_PATH).name == "best_v2a.pt"

    pred_state = predictor_instance.model.model.state_dict()
    cam_state = cam_service_instance._cam_model.state_dict()

    assert set(pred_state.keys()) == set(cam_state.keys())
    for k in pred_state.keys():
        assert torch.equal(pred_state[k].cpu(), cam_state[k].cpu()), f"Trọng số tensor {k} không đồng nhất giữa hai mô hình"


def test_cam_layer_accessible_and_4d_output(cam_service_instance):
    """Kiểm tra lớp model.9.conv truy cập được qua get_submodule và cho activation 4 chiều."""
    cam_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")
    layer = cam_service_instance._cam_model.get_submodule(cam_layer_name)
    assert layer is not None
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=cam_service_instance.device)
    _, _, activation, _ = compute_cam_raw(
        cam_service_instance._cam_model,
        layer,
        dummy,
        class_idx=0
    )
    assert activation.ndim == 4
    assert activation.shape[0] == 1
    assert activation.shape[2] == 7
    assert activation.shape[3] == 7


# ---------------------------------------------------------------------------
# Nhóm 2: Xác suất tính từ đường CAM khớp model.predict trong sai số 1e-3
# ---------------------------------------------------------------------------

def test_cam_probability_matches_predict_20_images(cam_service_instance, predictor_instance):
    """Xác suất softmax(logits, dim=-1).squeeze() đường CAM phải khớp model.predict trong sai số 1e-3 trên 20 ảnh."""
    test_paths = get_20_test_image_paths()
    assert len(test_paths) == 20

    max_diffs = []
    for p in test_paths:
        with open(p, "rb") as f:
            img_bytes = f.read()

        # Tiền xử lý thống nhất qua đúng hàm preprocess_image của predictor
        rgb_arr = predictor_instance.preprocess_image(img_bytes)

        # 1. Đường suy luận chuẩn của Predictor
        predict_res = predictor_instance.model.predict(
            source=rgb_arr,
            imgsz=settings.IMGSZ,
            device=predictor_instance.device,
            verbose=False,
        )[0]
        predict_probs = predict_res.probs.data.cpu().numpy()

        # 2. Đường tensor CAM
        tensor = cam_service_instance._get_input_tensor(rgb_arr)
        _, _, _, logits = compute_cam_raw(
            cam_service_instance._cam_model,
            cam_service_instance._cam_layer,
            tensor,
            class_idx=0,
        )
        assert logits.ndim == 2 and logits.shape[0] == 1 and logits.shape[1] == 2
        cam_probs = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()

        diff = np.max(np.abs(predict_probs - cam_probs))
        max_diffs.append(diff)
        assert diff <= 1e-3, f"Ảnh {p.name} có chênh lệch xác suất {diff:.6e} > 1e-3"

    print(f"\n[Test Prob Match] Max diff trên 20 ảnh: {max(max_diffs):.6e}")


# ---------------------------------------------------------------------------
# Nhóm 3: Bản đồ và Heatmap PNG RGBA
# ---------------------------------------------------------------------------

def test_cam_output_shape_and_range(cam_service_instance, predictor_instance):
    """
    Kiểm tra hình dạng và khoảng giá trị:
    - _normalize(cam_raw) có hình dạng (7, 7), float32, trong [0, 1].
    - _resize_cam(cam_norm, size=224) có hình dạng (224, 224), float32, trong [0, 1].
    """
    test_paths = get_20_test_image_paths()
    with open(test_paths[0], "rb") as f:
        img_bytes = f.read()
    rgb_arr = predictor_instance.preprocess_image(img_bytes)
    tensor = cam_service_instance._get_input_tensor(rgb_arr)

    _, cam_raw, _, _ = compute_cam_raw(
        cam_service_instance._cam_model,
        cam_service_instance._cam_layer,
        tensor,
        class_idx=1,
    )
    assert cam_raw.shape == (7, 7)

    # 1. _normalize giữ nguyên hình dạng thô (7, 7)
    cam_norm = CamService._normalize(cam_raw)
    assert cam_norm.shape == (7, 7)
    assert cam_norm.dtype == np.float32
    assert 0.0 <= cam_norm.min()
    assert cam_norm.max() <= 1.0

    # 2. _resize_cam phóng lên hình dạng (224, 224)
    cam_resized = CamService._resize_cam(cam_norm, size=224)
    assert cam_resized.shape == (224, 224)
    assert cam_resized.dtype == np.float32
    assert 0.0 <= cam_resized.min()
    assert cam_resized.max() <= 1.0


def test_encode_heatmap_png_rgba_exact_colormap_and_alpha():
    """Kiểm tra heatmap PNG RGBA giải mã khớp bảng màu Turbo (BGR) và alpha theo cam_norm."""
    # Dựng cam_norm dốc mượt từ 0 đến 1 biết trước
    cam_norm = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape((224, 224))

    b64_str = CamService._encode_heatmap(cam_norm)
    png_bytes = base64.b64decode(b64_str)
    decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    assert decoded is not None
    assert decoded.shape == (224, 224, 4)
    assert decoded.dtype == np.uint8

    # Chỉ mục màu dùng chung: np.round(cam_norm * 255).astype(np.uint8) cho cả applyColorMap và alpha
    color_indices = np.round(cam_norm * 255.0).astype(np.uint8)

    # So sánh 3 kênh màu với applyColorMap(COLORMAP_TURBO) trên OpenCV BGR gốc
    expected_bgr = cv2.applyColorMap(color_indices, cv2.COLORMAP_TURBO)
    diff_bgr = np.abs(decoded[:, :, :3].astype(np.int32) - expected_bgr.astype(np.int32))
    assert np.max(diff_bgr) <= 1, "Màu RGB/BGR sai lệch quá 1 mức so với COLORMAP_TURBO"

    # So sánh kênh alpha với color_indices
    diff_alpha = np.abs(decoded[:, :, 3].astype(np.int32) - color_indices.astype(np.int32))
    assert np.max(diff_alpha) <= 1, "Kênh Alpha sai lệch quá 1 mức so với color_indices"


def test_normalize_cam_range_and_peak():
    """Kiểm tra hàm _normalize đưa dữ liệu về [0, 1] và giữ nguyên hình dạng."""
    raw = np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32)
    norm = CamService._normalize(raw)
    assert norm.shape == (2, 2)
    assert norm.min() == 0.0
    assert norm.max() == 1.0
    assert norm[0, 0] == 0.0
    assert norm[1, 1] == 1.0


def test_normalize_constant_map():
    """Mảng hằng cho kết quả toàn 0, float32, không NaN."""
    const_arr = np.full((7, 7), 5.0, dtype=np.float32)
    norm = CamService._normalize(const_arr)
    assert norm.shape == (7, 7)
    assert norm.dtype == np.float32
    assert not np.isnan(norm).any()
    assert (norm == 0.0).all()


# ---------------------------------------------------------------------------
# Nhóm 4: Phát hiện bản đồ trống
# ---------------------------------------------------------------------------

def test_detect_empty_static_method():
    """
    Kiểm tra _detect_empty:
    - Bắt buộc trả về kiểu bool chuẩn của Python (is True / is False).
    - Phát hiện đúng khi CAM toàn 0 hoặc peak quá nhỏ so với max abs trước ReLU (< 1e-4).
    """
    # Toàn 0
    zeros = np.zeros((7, 7), dtype=np.float32)
    prerelu_zeros = np.zeros((7, 7), dtype=np.float32)
    res1 = CamService._detect_empty(prerelu_zeros, zeros)
    assert isinstance(res1, bool)
    assert res1 is True

    # Peak quá nhỏ so với max abs trước ReLU (< 1e-4)
    prerelu_huge = np.array([[-100.0, -50.0], [-200.0, 0.00001]], dtype=np.float32)
    postrelu_tiny = np.maximum(prerelu_huge, 0.0)
    res2 = CamService._detect_empty(prerelu_huge, postrelu_tiny)
    assert isinstance(res2, bool)
    assert res2 is True

    # Bản đồ bình thường
    prerelu_normal = np.array([[-2.0, 3.0], [5.0, 1.0]], dtype=np.float32)
    postrelu_normal = np.maximum(prerelu_normal, 0.0)
    res3 = CamService._detect_empty(prerelu_normal, postrelu_normal)
    assert isinstance(res3, bool)
    assert res3 is False


def test_cam_empty_map_zeroed_classifier(cam_service_instance):
    """Zero hóa trọng số và bias của model.9.linear trên bản deepcopy phải cho is_empty is True."""
    model_copy = copy.deepcopy(cam_service_instance._cam_model)
    linear_layer = model_copy.get_submodule("model.9.linear")

    with torch.no_grad():
        linear_layer.weight.zero_()
        if linear_layer.bias is not None:
            linear_layer.bias.zero_()

    dummy = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=cam_service_instance.device)
    cam_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")
    layer = model_copy.get_submodule(cam_layer_name)

    cam_raw_prerelu, cam_raw, _, _ = compute_cam_raw(
        model_copy,
        layer,
        dummy,
        class_idx=0,
    )
    is_empty = CamService._detect_empty(cam_raw_prerelu, cam_raw)
    assert isinstance(is_empty, bool)
    assert is_empty is True, "Bản đồ không được nhận diện là trống khi zero hóa lớp phân loại"


# ---------------------------------------------------------------------------
# Nhóm 5: Hình học ảnh và crop_box độc lập
# ---------------------------------------------------------------------------

def test_compute_crop_box_torchvision_formula():
    """
    Kiểm tra _compute_crop_box tính đúng công thức torchvision với các tỷ lệ khung hình:
    - Ảnh ngang 300x500 (H=300, W=500): dài mới = int(224*500/300) = 373, offset = round((373-224)/2) = 74,
      x0 = round(74*500/373) = 99, x1 = 399, y0 = 0, y1 = 300.
    - Ảnh dọc 500x300 (H=500, W=300): y0 = 99, y1 = 399, x0 = 0, x1 = 300.
    - Ảnh vuông 400x400: (0, 0, 400, 400).
    """
    # 1. Ảnh vuông
    box_sq = CamService._compute_crop_box(400, 400, imgsz=224)
    assert isinstance(box_sq, dict)
    assert (box_sq["x0"], box_sq["y0"], box_sq["x1"], box_sq["y1"]) == (0, 0, 400, 400)

    # 2. Ảnh ngang 300x500 (H=300, W=500)
    box_land = CamService._compute_crop_box(300, 500, imgsz=224)
    assert isinstance(box_land, dict)
    assert box_land["x0"] == 99
    assert box_land["x1"] == 399
    assert box_land["y0"] == 0
    assert box_land["y1"] == 300

    # 3. Ảnh dọc 500x300 (H=500, W=300)
    box_port = CamService._compute_crop_box(500, 300, imgsz=224)
    assert isinstance(box_port, dict)
    assert box_port["x0"] == 0
    assert box_port["x1"] == 300
    assert box_port["y0"] == 99
    assert box_port["y1"] == 399


def test_crop_box_matches_torchvision_transforms_pixel_diff(cam_service_instance):
    """Cắt ảnh theo crop_box rồi resize 224 bằng bilinear phải khớp tensor của transforms trong 3/255."""
    test_dims = [(300, 500), (500, 300), (400, 400)]

    for H, W in test_dims:
        img_arr = create_smooth_gradient_image(H, W)
        pil_img = Image.fromarray(img_arr)

        # 1. Cắt theo crop_box độc lập và resize PIL bilinear
        box = CamService._compute_crop_box(H, W, imgsz=224)
        cropped = pil_img.crop((box["x0"], box["y0"], box["x1"], box["y1"]))
        manual_resized = cropped.resize((224, 224), resample=Image.Resampling.BILINEAR)
        manual_arr = np.array(manual_resized, dtype=np.float32) / 255.0

        # 2. Tensor tạo ra từ transforms của predictor
        tfm = cam_service_instance._transforms
        tfm_tensor = tfm(pil_img)  # Tensor (3, 224, 224)
        tfm_arr = tfm_tensor.permute(1, 2, 0).cpu().numpy()

        mean_diff = np.mean(np.abs(manual_arr - tfm_arr))
        assert mean_diff <= (3.0 / 255.0), f"Dim ({H},{W}) sai khác {mean_diff:.5f} > 3/255"


# ---------------------------------------------------------------------------
# Nhóm 6: Kiểm chứng ngẫu nhiên hóa (Cascade Randomization)
# ---------------------------------------------------------------------------

def test_cam_cascade_randomization_spearman(cam_service_instance, predictor_instance):
    """
    Kiểm chứng ngẫu nhiên hóa cascade từ linear -> 9.conv -> backbone 8 xuống 0:
    - Đặt torch.manual_seed(42) trước cascade.
    - Gọi reset_parameters() cho mọi submodule có phương thức đó (không dùng ndim==1 -> zeros_).
    - Đảm bảo BatchNorm gamma không bao giờ bằng 0.
    - In Spearman correlation ở từng bước; yêu cầu bước toàn bộ |rho| < 0.3 ở ít nhất 18/20 ảnh.
    - Số NaN không vượt quá 5/20.
    """
    torch.manual_seed(42)
    test_paths = get_20_test_image_paths()
    tensors = []
    for p in test_paths:
        with open(p, "rb") as f:
            b = f.read()
        rgb = predictor_instance.preprocess_image(b)
        tensors.append(cam_service_instance._get_input_tensor(rgb))

    # Tính bản đồ CAM gốc trên mô hình chưa ngẫu nhiên hóa
    orig_cams = []
    layer_orig = cam_service_instance._cam_layer
    for t in tensors:
        _, cam_raw, _, _ = compute_cam_raw(
            cam_service_instance._cam_model,
            layer_orig,
            t,
            class_idx=1,
        )
        orig_cams.append(cam_raw.copy())

    # Deepcopy để thực hiện cascade randomization
    model_cascade = copy.deepcopy(cam_service_instance._cam_model)
    cam_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")
    layer_cascade = model_cascade.get_submodule(cam_layer_name)

    cascade_steps = [
        ("linear", model_cascade.get_submodule("model.9.linear")),
        ("9.conv", model_cascade.get_submodule("model.9.conv")),
    ]
    # Thêm các khối backbone từ 8 xuống 0
    for idx in range(8, -1, -1):
        cascade_steps.append((f"model.{idx}", model_cascade.get_submodule(f"model.{idx}")))

    print("\n--- BẮT ĐẦU CASCADE RANDOMIZATION ---")
    for step_name, module in cascade_steps:
        # Gọi reset_parameters() cho mọi submodule có phương thức đó
        for submod in module.modules():
            if hasattr(submod, "reset_parameters") and callable(submod.reset_parameters):
                submod.reset_parameters()
            # Đảm bảo gamma của BatchNorm không bao giờ bằng 0
            if isinstance(submod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if submod.weight is not None and (submod.weight == 0).any():
                    with torch.no_grad():
                        nn.init.ones_(submod.weight)

        # Tính CAM tại bước hiện tại
        rhos = []
        for orig, t in zip(orig_cams, tensors):
            _, rand_cam, _, _ = compute_cam_raw(
                model_cascade,
                layer_cascade,
                t,
                class_idx=1,
            )
            val, _ = spearmanr(orig.flatten(), rand_cam.flatten())
            rhos.append(val)

        valid_rhos = [r for r in rhos if not np.isnan(r)]
        mean_rho = np.mean(valid_rhos) if valid_rhos else float("nan")
        print(f"Bước {step_name:>10}: mean Spearman rho = {mean_rho:+.4f} (NaN: {len(rhos)-len(valid_rhos)})")

    # Đánh giá bước toàn bộ (sau khi đã ngẫu nhiên hóa tất cả)
    nan_count = 0
    pass_count = 0
    for r in rhos:
        if np.isnan(r):
            nan_count += 1
            abs_rho = 0.0  # NaN tính |rho| = 0
        else:
            abs_rho = abs(r)

        if abs_rho < 0.3:
            pass_count += 1

    print(f"Kết quả ngẫu nhiên hóa toàn bộ: {pass_count}/20 ảnh có |rho| < 0.3, số NaN: {nan_count}")
    assert nan_count <= 5, f"Số lượng NaN ({nan_count}) vượt quá ngưỡng cho phép 5/20"
    assert pass_count >= 18, f"Chỉ có {pass_count}/20 ảnh có |rho| < 0.3 (yêu cầu >= 18)"


# ---------------------------------------------------------------------------
# Nhóm 7: Đối chiếu với torchcam
# ---------------------------------------------------------------------------

def test_cam_matches_torchcam_spearman(cam_service_instance, predictor_instance):
    """
    So sánh bản đồ tự tính với torchcam.methods.GradCAM trên deepcopy (SiLU.inplace=False):
    - Tính trước cả 20 bản đồ tự tính bằng compute_cam_raw ngoài khối GradCAM.
    - Mở khối with GradCAM và chỉ tính bản torchcam.
    - tensor được detach().clone().requires_grad_(True) trước khi forward trong torch.enable_grad().
    - Yêu cầu Spearman rho > 0.95 ở ít nhất 18/20 ảnh.
    """
    model_copy = copy.deepcopy(cam_service_instance._cam_model)
    # Tắt inplace cho toàn bộ SiLU để tránh lỗi in-place gradient
    for mod in model_copy.modules():
        if isinstance(mod, nn.SiLU):
            mod.inplace = False

    class _Wrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            out = self.net(x)
            return out[1] if isinstance(out, (tuple, list)) else out

    wrapped = _Wrapper(model_copy).eval()
    test_paths = get_20_test_image_paths()

    # 1. TÍNH TRƯỚC cả 20 bản đồ tự tính bằng compute_cam_raw ngoài khối GradCAM
    custom_cams = []
    tensors = []
    layer_submod = model_copy.get_submodule(getattr(settings, "CAM_LAYER", "model.9.conv"))
    for p in test_paths:
        with open(p, "rb") as f:
            b = f.read()
        rgb = predictor_instance.preprocess_image(b)
        tensor = cam_service_instance._get_input_tensor(rgb)
        tensors.append(tensor)
        _, custom_cam, _, _ = compute_cam_raw(
            model_copy,
            layer_submod,
            tensor,
            class_idx=1,
        )
        custom_cams.append(custom_cam)

    # 2. Mở khối with GradCAM và chỉ tính bản torchcam
    pass_count = 0
    with torch.enable_grad():
        with GradCAM(wrapped, target_layer="net.model.9.conv") as cam_extractor:
            for custom_cam, tensor in zip(custom_cams, tensors):
                tensor_torchcam = tensor.detach().clone().requires_grad_(True)
                scores = wrapped(tensor_torchcam)
                torchcam_map = cam_extractor(class_idx=1, scores=scores)[0].squeeze().detach().cpu().numpy()

                # So trực tiếp trên feature map (7, 7)
                if custom_cam.shape != torchcam_map.shape:
                    custom_cam_7x7 = cv2.resize(custom_cam, (torchcam_map.shape[1], torchcam_map.shape[0]))
                else:
                    custom_cam_7x7 = custom_cam

                rho, _ = spearmanr(custom_cam_7x7.flatten(), torchcam_map.flatten())
                if not np.isnan(rho) and rho > 0.95:
                    pass_count += 1

    print(f"\n[Test vs torchcam] Số ảnh đạt rho > 0.95: {pass_count}/20")
    assert pass_count >= 18, f"Chỉ có {pass_count}/20 ảnh khớp torchcam với rho > 0.95"


# ---------------------------------------------------------------------------
# Nhóm 8: An toàn đa luồng và Bất biến của Predictor dùng chung
# ---------------------------------------------------------------------------

def test_cam_concurrent_requests_two_different_images(cam_service_instance):
    """Hai luồng gọi CAM đồng thời trên 2 ảnh khác nhau, cả 2 đều thành công và trả heatmap khác nhau."""
    test_paths = get_20_test_image_paths()
    with open(test_paths[0], "rb") as f:
        bytes1 = f.read()
    with open(test_paths[1], "rb") as f:
        bytes2 = f.read()

    results = {}
    errors = {}

    def worker(key, img_bytes, class_idx):
        try:
            results[key] = cam_service_instance.explain(img_bytes, class_idx)
        except Exception as e:
            errors[key] = e

    t1 = threading.Thread(target=worker, args=("req1", bytes1, 0))
    t2 = threading.Thread(target=worker, args=("req2", bytes2, 1))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors, f"Lỗi trong quá trình chạy đa luồng: {errors}"
    assert "heatmap_base64" in results["req1"]
    assert "heatmap_base64" in results["req2"]
    assert results["req1"]["heatmap_base64"] != results["req2"]["heatmap_base64"]


def test_predictor_unchanged_after_cam_call(cam_service_instance, predictor_instance):
    """Mô hình dự đoán dùng chung không bị đổi cờ requires_grad và cho xác suất đồng nhất sau khi gọi CAM."""
    test_paths = get_20_test_image_paths()
    with open(test_paths[0], "rb") as f:
        sample_bytes = f.read()

    # Lưu trạng thái cờ requires_grad và xác suất trước khi gọi CAM
    flags_before = [p.requires_grad for p in predictor_instance.model.model.parameters()]
    assert all(not f for f in flags_before), "Mô hình dùng chung có cờ requires_grad không phải False"
    pred_before = predictor_instance.predict(sample_bytes)

    # Thực thi giải thích qua CAM
    cam_service_instance.explain(sample_bytes, class_idx=1)

    # Kiểm tra sau khi gọi CAM
    flags_after = [p.requires_grad for p in predictor_instance.model.model.parameters()]
    assert flags_before == flags_after, "Cờ requires_grad của mô hình dùng chung bị thay đổi sau CAM"

    pred_after = predictor_instance.predict(sample_bytes)
    for c in predictor_instance.class_names_list:
        diff = abs(pred_before["probabilities"][c] - pred_after["probabilities"][c])
        assert diff <= 1e-6, f"Xác suất lớp {c} bị lệch {diff} sau khi gọi CAM"


def test_explain_returns_all_fields(cam_service_instance, predictor_instance):
    """
    Gọi explain trên ảnh thô, khẳng định trả về dict với đủ các khóa:
    - heatmap_base64: str base64 thuần, giải mã được thành PNG (224, 224, 4), không có tiền tố data:
    - crop_box: dict 4 số nguyên x0, y0, x1, y1
    - colormap == "turbo"
    - is_empty: bool
    - empty_message: None hoặc str
    - raw_peak: float
    - target_class: str, đúng tên lớp của class_idx
    - target_layer: str
    - image_size: dict có width và height bằng kích thước ảnh gốc
    - attention_note: str chứa cụm "vùng mô hình chú ý"
    """
    test_paths = get_20_test_image_paths()
    with open(test_paths[0], "rb") as f:
        img_bytes = f.read()

    # Kích thước ảnh gốc
    orig_img = Image.open(test_paths[0])
    orig_w, orig_h = orig_img.size

    class_idx = 1
    expected_class_name = predictor_instance.class_names_list[class_idx]

    result = cam_service_instance.explain(img_bytes, class_idx=class_idx)
    assert isinstance(result, dict)

    # heatmap_base64
    assert "heatmap_base64" in result
    assert isinstance(result["heatmap_base64"], str)
    assert not result["heatmap_base64"].startswith("data:")
    decoded_png = cv2.imdecode(np.frombuffer(base64.b64decode(result["heatmap_base64"]), np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded_png is not None
    assert decoded_png.shape == (224, 224, 4)

    # crop_box
    assert "crop_box" in result
    assert isinstance(result["crop_box"], dict)
    for k in ("x0", "y0", "x1", "y1"):
        assert k in result["crop_box"]
        assert type(result["crop_box"][k]) is int

    # colormap
    assert result.get("colormap") == "turbo"

    # is_empty & empty_message
    assert type(result.get("is_empty")) is bool
    assert result.get("empty_message") is None or isinstance(result.get("empty_message"), str)

    # raw_peak
    assert type(result.get("raw_peak")) is float

    # target_class & target_layer
    assert result.get("target_class") == expected_class_name
    assert isinstance(result.get("target_layer"), str)

    # image_size
    assert "image_size" in result
    assert isinstance(result["image_size"], dict)
    assert result["image_size"].get("width") == orig_w
    assert result["image_size"].get("height") == orig_h

    # attention_note
    assert "attention_note" in result
    assert isinstance(result["attention_note"], str)
    assert result["attention_note"] == "Vùng mô hình chú ý — không phải bằng chứng lâm sàng"


def test_explain_crop_box_inside_image(cam_service_instance):
    """crop_box của explain bằng _compute_crop_box(H, W) và nằm trọn trong ảnh."""
    test_paths = get_20_test_image_paths()
    with open(test_paths[0], "rb") as f:
        img_bytes = f.read()

    orig_img = Image.open(test_paths[0])
    orig_w, orig_h = orig_img.size

    result = cam_service_instance.explain(img_bytes, class_idx=0)
    box = result["crop_box"]
    expected_box = CamService._compute_crop_box(orig_h, orig_w, imgsz=224)

    assert box == expected_box
    assert 0 <= box["x0"] < box["x1"] <= orig_w
    assert 0 <= box["y0"] < box["y1"] <= orig_h


def test_explain_removes_hook(cam_service_instance):
    """Sau khi gọi explain, hook forward phải được gỡ bỏ hoàn toàn (len == 0)."""
    test_paths = get_20_test_image_paths()
    with open(test_paths[0], "rb") as f:
        img_bytes = f.read()

    assert len(cam_service_instance._cam_layer._forward_hooks) == 0
    _ = cam_service_instance.explain(img_bytes, class_idx=0)
    assert len(cam_service_instance._cam_layer._forward_hooks) == 0


def test_cam_backbone_only_randomization_spearman(cam_service_instance):
    """
    đây kiểm tra bản đồ có phụ thuộc đặc trưng backbone đã học hay không, ngưỡng 0.5 là phán đoán.
    So sánh bản đồ TRƯỚC ReLU (cam_pre) giữa bản gốc và bản ngẫu nhiên hóa backbone (model.0 -> model.8).
    Số NaN phải <= 2/20 và trung vị abs(rho) < 0.5.
    """
    image_paths = get_20_test_image_paths()
    tensors = []
    orig_pre_cams = []

    cam_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")
    layer = cam_service_instance._cam_layer

    for path in image_paths:
        with open(path, "rb") as f:
            b = f.read()
        rgb = PredictorService.get_instance().preprocess_image(b)
        t = cam_service_instance._get_input_tensor(rgb)
        tensors.append(t)

        cam_pre, _, _, _ = compute_cam_raw(
            cam_service_instance._cam_model,
            layer,
            t,
            class_idx=1,
        )
        orig_pre_cams.append(cam_pre.copy())

    # Deepcopy và ngẫu nhiên hóa chỉ backbone model.0 -> model.8
    model_bb = copy.deepcopy(cam_service_instance._cam_model)
    layer_bb = model_bb.get_submodule(cam_layer_name)

    torch.manual_seed(42)
    for idx in range(9):
        mod = model_bb.get_submodule(f"model.{idx}")
        for submod in mod.modules():
            if hasattr(submod, "reset_parameters") and callable(submod.reset_parameters):
                submod.reset_parameters()
            if isinstance(submod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if submod.weight is not None and (submod.weight == 0).any():
                    with torch.no_grad():
                        nn.init.ones_(submod.weight)

    rhos = []
    print("\n--- BẢNG SPEARMAN BACKBONE-ONLY RANDOMIZATION (cam_pre, 20 ẢNH) ---")
    for i, (path, orig_pre, t) in enumerate(zip(image_paths, orig_pre_cams, tensors)):
        rand_pre, _, _, _ = compute_cam_raw(
            model_bb,
            layer_bb,
            t,
            class_idx=1,
        )
        val, _ = spearmanr(orig_pre.flatten(), rand_pre.flatten())
        rhos.append(val)
        print(f"Ảnh {i+1:02d} ({path.name}): Spearman rho = {val:+.4f}")

    nan_count = sum(1 for r in rhos if np.isnan(r))
    abs_rhos = [abs(r) for r in rhos if not np.isnan(r)]
    median_abs_rho = float(np.median(abs_rhos)) if abs_rhos else 0.0
    print(f"--> Số NaN: {nan_count}/20 | Trung vị |rho| (backbone-only): {median_abs_rho:.4f}")

    assert nan_count <= 2, f"Số NaN {nan_count} vượt quá 2/20"
    assert median_abs_rho < 0.5, f"Trung vị |rho| {median_abs_rho:.4f} không nhỏ hơn 0.5"


def test_cam_independent_block_randomization_report(cam_service_instance):
    """
    Báo cáo ngẫu nhiên hóa từng khối độc lập (independent block randomization).
    Với mỗi khối trong [model.0, ..., model.8, model.9.linear], deepcopy mô hình gốc,
    chỉ ngẫu nhiên hóa khối đó (seed 42), tính trung vị abs(rho) giữa cam_pre gốc và ngẫu nhiên.
    Khẳng định duy nhất: số NaN <= 2/20 ở mọi khối.
    """
    image_paths = get_20_test_image_paths()
    tensors = []
    orig_pre_cams = []

    cam_layer_name = getattr(settings, "CAM_LAYER", "model.9.conv")
    layer = cam_service_instance._cam_layer

    for path in image_paths:
        with open(path, "rb") as f:
            b = f.read()
        rgb = PredictorService.get_instance().preprocess_image(b)
        t = cam_service_instance._get_input_tensor(rgb)
        tensors.append(t)

        cam_pre, _, _, _ = compute_cam_raw(
            cam_service_instance._cam_model,
            layer,
            t,
            class_idx=1,
        )
        orig_pre_cams.append(cam_pre.copy())

    blocks_to_test = [f"model.{i}" for i in range(9)] + ["model.9.linear"]

    print("\n--- BẢNG BÁO CÁO NGẪU NHIÊN HÓA TỪNG KHỐI ĐỘC LẬP (INDEPENDENT BLOCK) ---")
    for block_name in blocks_to_test:
        model_block = copy.deepcopy(cam_service_instance._cam_model)
        layer_block = model_block.get_submodule(cam_layer_name)

        torch.manual_seed(42)
        target_mod = model_block.get_submodule(block_name)
        for submod in target_mod.modules():
            if hasattr(submod, "reset_parameters") and callable(submod.reset_parameters):
                submod.reset_parameters()
            if isinstance(submod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if submod.weight is not None and (submod.weight == 0).any():
                    with torch.no_grad():
                        nn.init.ones_(submod.weight)

        rhos = []
        for orig_pre, t in zip(orig_pre_cams, tensors):
            rand_pre, _, _, _ = compute_cam_raw(
                model_block,
                layer_block,
                t,
                class_idx=1,
            )
            val, _ = spearmanr(orig_pre.flatten(), rand_pre.flatten())
            rhos.append(val)

        nan_count = sum(1 for r in rhos if np.isnan(r))
        valid_abs = [abs(r) for r in rhos if not np.isnan(r)]
        median_abs = float(np.median(valid_abs)) if valid_abs else float("nan")

        print(f"Khối {block_name:>15}: median |rho| = {median_abs:.4f} (NaN: {nan_count}/20)")
        assert nan_count <= 2, f"Khối {block_name} có số NaN {nan_count} vượt quá 2/20"


