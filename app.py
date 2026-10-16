"""
app.py - Ung dung web demo nhan dien viem phoi tu anh X-quang bang Gradio.

Chuc nang:
- Upload anh X-quang phoi
- Nhan ket qua phan loai: NORMAL hoac PNEUMONIA
- Hien thi do tin cay (confidence)
- Cung cap anh mau thu nghiem truc tiep

Cach chay:
    python app.py
    python app.py --model best_pneumonia_model.pt --port 7860
"""

import argparse
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Demo nhan dien viem phoi")
    parser.add_argument(
        "--model",
        type=str,
        default="best_pneumonia_model.pt",
        help="Duong dan toi model",
    )
    parser.add_argument("--port", type=int, default=7860, help="Port chay server")
    parser.add_argument(
        "--share", action="store_true", help="Tao public link qua Gradio"
    )
    return parser.parse_args()


# Bien toan cuc luu tru mo hinh da tai
model = None


def load_model(model_path: str):
    """Tai mo hinh YOLOv8 vao bo nho."""
    global model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Khong tim thay model tai: {model_path}\n"
            f"Hay chay 'python train.py' truoc de huan luyen model."
        )
    model = YOLO(model_path)
    print(f"Da tai model: {model_path}")
    print(f"Cac lop: {model.names}")
    return model


def predict(image):
    """
    Du doan ket qua phan loai cho anh X-quang.

    Args:
        image: PIL Image hoac numpy array

    Returns:
        tuple: (annotated_image, result_html, confidence_dict)
    """
    if image is None:
        return (
            None,
            create_placeholder_html(),
            {},
        )

    if model is None:
        return (
            None,
            create_error_html("Model chưa được tải vào bộ nhớ. Vui lòng khởi động lại ứng dụng."),
            {},
        )

    # Du doan voi mo hinh
    results = model.predict(source=image, imgsz=224, verbose=False)
    result = results[0]

    # Lay nhan va do tin cay
    probs = result.probs
    pred_class_idx = probs.top1
    pred_class_name = model.names[pred_class_idx]
    pred_conf = probs.top1conf.item()

    # Thong ke xac suat cho tung lop
    confidence_dict = {}
    for idx, class_name in model.names.items():
        conf = probs.data[idx].item()
        confidence_dict[class_name] = round(conf, 4)

    # Tao anh chu thich hien thi ket qua
    if isinstance(image, np.ndarray):
        annotated = Image.fromarray(image)
    else:
        annotated = image.copy()

    annotated = annotated.convert("RGB")
    display_size = (500, 500)
    annotated = annotated.resize(display_size, Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(annotated)

    # Thiet lap thong bao va vien theo ket qua
    if pred_class_name == "PNEUMONIA":
        color = (220, 53, 69)  # Mau do canh bao
        status_vi = "VIÊM PHỔI"
        status_en = "PNEUMONIA"
        advice = "Phát hiện dấu hiệu nghi ngờ viêm phổi. Cần tham khảo thêm ý kiến bác sĩ chuyên khoa."
    else:
        color = (25, 135, 84)  # Mau xanh an toan
        status_vi = "BÌNH THƯỜNG"
        status_en = "NORMAL"
        advice = "Không phát hiện dấu hiệu bất thường về viêm phổi trên ảnh chụp."

    # Ve vien mau quanh anh
    border_width = 6
    for i in range(border_width):
        draw.rectangle(
            [i, i, display_size[0] - 1 - i, display_size[1] - 1 - i],
            outline=color,
        )

    # Ve nhan ket qua tren anh
    label = f"{pred_class_name} ({pred_conf:.1%})"
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Banner tren anh
    banner_h = text_h + 20
    draw.rectangle(
        [border_width, border_width, display_size[0] - border_width - 1, banner_h + border_width],
        fill=color,
    )
    text_x = (display_size[0] - text_w) // 2
    draw.text(
        (text_x, border_width + 8),
        label,
        fill="white",
        font=font,
    )

    # Tao HTML ket qua chi tiet
    result_html = create_result_html(
        pred_class_name, pred_conf, status_vi, status_en, advice, confidence_dict
    )

    return annotated, result_html, confidence_dict


def create_placeholder_html():
    """Tao HTML placeholder khi chua co anh."""
    return """
<div style="
    text-align: center;
    padding: 36px 20px;
    border: 2px dashed #3D5A99;
    border-radius: 16px;
    background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
    color: #3D5A99;
    font-family: 'Segoe UI', sans-serif;
">
    <h3 style="margin: 0 0 8px 0; color: #1a3a7a;">Chờ ảnh X-quang</h3>
    <p style="margin: 0; opacity: 0.7; font-size: 14px;">
        Vui lòng tải lên ảnh X-quang phổi hoặc chọn ảnh mẫu bên dưới để bắt đầu phân tích.
    </p>
</div>
"""


def create_error_html(message):
    """Tao HTML thong bao loi."""
    return f"""
<div style="
    text-align: center;
    padding: 28px 20px;
    border: 2px solid #dc3545;
    border-radius: 16px;
    background: linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%);
    color: #721c24;
    font-family: 'Segoe UI', sans-serif;
">
    <h3 style="margin: 0 0 8px 0;">Lỗi hệ thống</h3>
    <p style="margin: 0; font-size: 14px;">{message}</p>
</div>
"""


def create_result_html(pred_class, pred_conf, status_vi, status_en, advice, confidence_dict):
    """Tao HTML hien thi ket qua phan tich chi tiet."""
    # Mau sac theo ket qua
    if pred_class == "PNEUMONIA":
        gradient = "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"
        bg_card = "linear-gradient(135deg, #fff5f5 0%, #ffe8e8 100%)"
        border_color = "#dc3545"
        badge_bg = "#dc3545"
        bar_color = "#dc3545"
        text_color = "#721c24"
    else:
        gradient = "linear-gradient(135deg, #198754 0%, #157347 100%)"
        bg_card = "linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%)"
        border_color = "#198754"
        badge_bg = "#198754"
        bar_color = "#198754"
        text_color = "#0f5132"

    # Tinh phan tram confidence
    conf_pct = pred_conf * 100
    normal_pct = confidence_dict.get("NORMAL", 0) * 100
    pneumonia_pct = confidence_dict.get("PNEUMONIA", 0) * 100

    html = f"""
<div style="font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 100%;">

    <!-- Header ket qua -->
    <div style="
        background: {gradient};
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    ">
        <div>
            <div style="font-size: 12px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;">
                Kết quả chẩn đoán
            </div>
            <div style="font-size: 24px; font-weight: 700; margin-top: 4px;">
                {status_vi}
            </div>
            <div style="font-size: 14px; opacity: 0.85; margin-top: 2px;">
                {status_en} &mdash; Độ tin cậy: {conf_pct:.1f}%
            </div>
        </div>
    </div>

    <!-- Thanh confidence -->
    <div style="
        background: white;
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    ">
        <div style="font-size: 14px; font-weight: 600; color: #1a3a7a; margin-bottom: 14px;">
            Phân bố xác suất
        </div>

        <!-- NORMAL bar -->
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 13px; font-weight: 600; color: #198754;">
                    NORMAL (Bình thường)
                </span>
                <span style="font-size: 13px; font-weight: 700; color: #198754;">
                    {normal_pct:.1f}%
                </span>
            </div>
            <div style="
                width: 100%; height: 10px;
                background: #e9ecef; border-radius: 5px;
                overflow: hidden;
            ">
                <div style="
                    width: {normal_pct:.1f}%; height: 100%;
                    background: linear-gradient(90deg, #198754, #20c997);
                    border-radius: 5px;
                    transition: width 0.6s ease;
                "></div>
            </div>
        </div>

        <!-- PNEUMONIA bar -->
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 13px; font-weight: 600; color: #dc3545;">
                    PNEUMONIA (Viêm phổi)
                </span>
                <span style="font-size: 13px; font-weight: 700; color: #dc3545;">
                    {pneumonia_pct:.1f}%
                </span>
            </div>
            <div style="
                width: 100%; height: 10px;
                background: #e9ecef; border-radius: 5px;
                overflow: hidden;
            ">
                <div style="
                    width: {pneumonia_pct:.1f}%; height: 100%;
                    background: linear-gradient(90deg, #dc3545, #ff6b6b);
                    border-radius: 5px;
                    transition: width 0.6s ease;
                "></div>
            </div>
        </div>
    </div>

    <!-- Loi khuyen -->
    <div style="
        background: {bg_card};
        border-left: 4px solid {border_color};
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    ">
        <div style="font-size: 13px; font-weight: 600; color: {text_color}; margin-bottom: 4px;">
            Nhận xét
        </div>
        <div style="font-size: 13px; color: #333; line-height: 1.5;">
            {advice}
        </div>
    </div>

    <!-- Disclaimer -->
    <div style="
        text-align: center;
        font-size: 11px;
        color: #6c757d;
        padding: 8px 0;
        border-top: 1px solid #e9ecef;
        font-style: italic;
    ">
        Lưu ý: Hệ thống mang tính chất hỗ trợ tham khảo kỹ thuật, không thay thế chẩn đoán y tế chính thức.
    </div>
</div>
"""
    return html


def get_sample_images():
    """Lay danh sach mot so anh mau tu tap test, can doi giua NORMAL va PNEUMONIA."""
    samples = []
    test_dir = Path("dataset/test")

    if test_dir.exists():
        # Lay 5 anh NORMAL
        normal_dir = test_dir / "NORMAL"
        if normal_dir.exists():
            normal_imgs = sorted(normal_dir.glob("*.jpeg"))[:5]
            samples.extend([str(p) for p in normal_imgs])

        # Lay 5 anh PNEUMONIA
        pneumonia_dir = test_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            pneumonia_imgs = sorted(pneumonia_dir.glob("*.jpeg"))[:5]
            samples.extend([str(p) for p in pneumonia_imgs])

    return samples if samples else None


def create_app():
    """Khoi tao giao dien Gradio voi theme Cobalt Blue chuyen nghiep."""
    sample_images = get_sample_images()

    # === CUSTOM CSS — Cobalt Blue Theme ===
    custom_css = """
    /* === Import Google Font === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* === Global Styles === */
    * {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        background: linear-gradient(180deg, #0a1628 0%, #0f2044 30%, #162d5a 60%, #1a3570 100%) !important;
        min-height: 100vh;
    }

    /* === Header Banner === */
    .header-banner {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 30%, #1976d2 60%, #2196f3 100%);
        border-radius: 20px;
        padding: 36px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .header-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .header-banner h1 {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        margin: 0 0 8px 0 !important;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .header-banner p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 15px !important;
        margin: 0 !important;
        line-height: 1.5;
    }
    .header-banner .badge-row {
        display: flex;
        gap: 8px;
        margin-top: 14px;
        flex-wrap: wrap;
    }
    .header-banner .badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.2);
        letter-spacing: 0.3px;
    }

    /* === Card / Panel Styling === */
    .panel-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px);
        border-radius: 16px !important;
        border: 1px solid rgba(13, 71, 161, 0.1) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08) !important;
        overflow: hidden;
    }

    /* === Gradio Component Overrides === */
    .gr-group {
        border-radius: 16px !important;
        border: 1px solid rgba(13, 71, 161, 0.15) !important;
        background: rgba(255, 255, 255, 0.97) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06) !important;
    }

    /* Primary Button — Cobalt Blue */
    .gr-button-primary, button.primary {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 28px !important;
        text-transform: none !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 16px rgba(13, 71, 161, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    .gr-button-primary:hover, button.primary:hover {
        background: linear-gradient(135deg, #0b3d91 0%, #1256a8 50%, #1565c0 100%) !important;
        box-shadow: 0 6px 24px rgba(13, 71, 161, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    .gr-button-primary:active, button.primary:active {
        transform: translateY(0) !important;
    }

    /* Secondary Button */
    .gr-button-secondary, button.secondary {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%) !important;
        border: 1px solid #90caf9 !important;
        border-radius: 12px !important;
        color: #0d47a1 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    .gr-button-secondary:hover, button.secondary:hover {
        background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%) !important;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.2) !important;
    }

    /* === Tab Styling === */
    .tab-nav button {
        background: transparent !important;
        color: rgba(255,255,255,0.6) !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
    }
    .tab-nav button.selected {
        color: white !important;
        border-bottom-color: #42a5f5 !important;
        background: rgba(255,255,255,0.08) !important;
    }
    .tab-nav button:hover {
        color: rgba(255,255,255,0.9) !important;
        background: rgba(255,255,255,0.05) !important;
    }

    /* === Image Component === */
    .image-container, .gr-image {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* === Label Component === */
    .gr-label {
        border-radius: 12px !important;
    }

    /* === Section Headers === */
    .section-header {
        background: linear-gradient(135deg, rgba(13, 71, 161, 0.08) 0%, rgba(21, 101, 192, 0.04) 100%);
        border-left: 4px solid #1565c0;
        border-radius: 0 12px 12px 0;
        padding: 12px 18px;
        margin: 8px 0 16px 0;
    }
    .section-header h3 {
        color: #0d47a1 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .section-header p {
        color: #546e7a !important;
        font-size: 13px !important;
        margin: 4px 0 0 0 !important;
    }

    /* === Info Cards (bottom) === */
    .info-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        color: white;
    }
    .info-card h4 {
        color: #90caf9 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        margin: 0 0 10px 0 !important;
    }
    .info-card p, .info-card li {
        color: rgba(255,255,255,0.75) !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
    }

    /* === Examples Gallery === */
    .gr-examples {
        border-radius: 12px !important;
    }
    .gr-examples .gr-sample-textbox {
        border-radius: 8px !important;
    }

    /* === Footer === */
    .footer-bar {
        text-align: center;
        padding: 20px;
        color: rgba(255,255,255,0.4);
        font-size: 12px;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 24px;
    }

    /* === Markdown inside dark bg === */
    .dark-text h3, .dark-text h4 {
        color: #90caf9 !important;
    }
    .dark-text p, .dark-text li, .dark-text span {
        color: rgba(255,255,255,0.75) !important;
    }

    /* === Scrollbar === */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: #1565c0; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #0d47a1; }
    """

    # === Gradio Soft Theme ===
    cobalt_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        button_primary_background_fill="linear-gradient(135deg, #0d47a1, #1976d2)",
        button_primary_background_fill_hover="linear-gradient(135deg, #0b3d91, #1565c0)",
        button_primary_text_color="white",
        button_primary_border_color="transparent",
        block_title_text_color="#0d47a1",
        block_label_text_color="#1565c0",
        input_border_color="#bbdefb",
        input_border_color_focus="#1976d2",
    )

    with gr.Blocks(
        title="Hệ thống nhận diện viêm phổi - YOLOv8",
    ) as app:
        # === HEADER ===
        gr.HTML("""
        <div class="header-banner">
            <div>
                <h1>Hệ Thống Nhận Diện Viêm Phổi từ Ảnh X-Quang</h1>
                <p>Ứng dụng trí tuệ nhân tạo phân loại ảnh X-quang phổi sử dụng kiến trúc YOLOv8 Classification</p>
                <div class="badge-row">
                    <span class="badge">YOLOv8</span>
                    <span class="badge">2 lớp: NORMAL / PNEUMONIA</span>
                    <span class="badge">224×224px</span>
                    <span class="badge">Thời gian thực</span>
                </div>
            </div>
        </div>
        """)

        # === MAIN CONTENT ===
        with gr.Row(equal_height=True):
            # -- Cot trai: Upload & nut phan tich --
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="section-header">
                    <h3>Tải ảnh X-quang</h3>
                    <p>Chọn hoặc kéo thả ảnh X-quang phổi vào khung bên dưới</p>
                </div>
                """)
                input_image = gr.Image(
                    type="pil",
                    label="Ảnh X-quang phổi",
                    height=400,
                    sources=["upload", "clipboard"],
                    show_label=True,
                    elem_classes=["panel-card"],
                )
                predict_btn = gr.Button(
                    "Phân Tích Ảnh X-Quang",
                    variant="primary",
                    size="lg",
                    elem_classes=["predict-btn"],
                )
                clear_btn = gr.Button(
                    "Xóa & Nhập ảnh mới",
                    variant="secondary",
                    size="sm",
                )

            # -- Cot phai: Ket qua --
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="section-header">
                    <h3>Kết quả phân tích</h3>
                    <p>Kết quả chẩn đoán và phân bố xác suất</p>
                </div>
                """)
                output_image = gr.Image(
                    type="pil",
                    label="Ảnh kết quả",
                    height=400,
                    interactive=False,
                    show_label=True,
                    elem_classes=["panel-card"],
                )
                output_label = gr.Label(
                    label="Xác suất từng lớp",
                    num_top_classes=2,
                    elem_classes=["panel-card"],
                )

        # === KET QUA CHI TIET ===
        result_html = gr.HTML(
            value=create_placeholder_html(),
            label="Chi tiết kết quả",
        )

        # === GAn SU KIEN ===
        predict_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_image, result_html, output_label],
        )

        input_image.change(
            fn=predict,
            inputs=[input_image],
            outputs=[output_image, result_html, output_label],
        )

        def clear_all():
            return None, None, create_placeholder_html(), {}

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[input_image, output_image, result_html, output_label],
        )

        # === ANH MAU DE KIEM THU ===
        if sample_images:
            gr.HTML("""
            <div style="margin-top: 24px;">
                <div class="section-header">
                    <h3>Ảnh mẫu kiểm thử</h3>
                    <p>Chọn một ảnh mẫu từ tập test để thử nghiệm nhanh — bao gồm cả ảnh NORMAL và PNEUMONIA</p>
                </div>
            </div>
            """)
            gr.Examples(
                examples=sample_images,
                inputs=input_image,
                outputs=[output_image, result_html, output_label],
                fn=predict,
                cache_examples=False,
                examples_per_page=10,
                label="Chọn ảnh mẫu (5 NORMAL + 5 PNEUMONIA)",
            )

        # === HUONG DAN & THONG TIN ===
        gr.HTML("""
        <div style="margin-top: 24px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
            <div class="info-card">
                <h4>Hướng dẫn sử dụng</h4>
                <ol style="padding-left: 18px; margin: 0;">
                    <li>Tải ảnh X-quang phổi lên hoặc chọn ảnh mẫu</li>
                    <li>Nhấn nút <b>"Phân Tích Ảnh X-Quang"</b></li>
                    <li>Xem kết quả phân loại và mức độ tin cậy</li>
                    <li>Tham khảo nhận xét và lời khuyên</li>
                </ol>
            </div>
            <div class="info-card">
                <h4>Về mô hình AI</h4>
                <p style="margin: 0;">
                    Mô hình sử dụng kiến trúc <b>YOLOv8 Classification</b> (Ultralytics),
                    được huấn luyện trên bộ dữ liệu X-quang phổi với 2 lớp phân loại:
                    <b>NORMAL</b> (bình thường) và <b>PNEUMONIA</b> (viêm phổi).
                    Kích thước ảnh đầu vào: 224×224 pixel.
                </p>
            </div>
            <div class="info-card">
                <h4>Lưu ý quan trọng</h4>
                <p style="margin: 0;">
                    Kết quả từ hệ thống AI chỉ mang tính chất <b>hỗ trợ tham khảo kỹ thuật</b>.
                    Không thay thế chẩn đoán y tế chính thức từ bác sĩ chuyên khoa.
                    Vui lòng liên hệ cơ sở y tế nếu cần tư vấn chuyên sâu.
                </p>
            </div>
        </div>
        """)

        # === FOOTER ===
        gr.HTML("""
        <div class="footer-bar">
            <p style="margin: 0;">
                Hệ thống nhận diện viêm phổi &mdash; Powered by <b>YOLOv8</b> &amp; <b>Gradio</b>
                &nbsp;|&nbsp; Đồ án nghiên cứu khoa học
            </p>
        </div>
        """)

    return app, cobalt_theme, custom_css


if __name__ == "__main__":
    args = parse_args()

    # Tai model
    load_model(args.model)

    # Khoi chay ung dung
    app, theme, css = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=theme,
        css=css,
        ssr_mode=False,
    )
