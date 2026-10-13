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
        tuple: (annotated_image, result_text, confidence_dict)
    """
    if image is None:
        return None, "Vui long upload mot anh X-quang phoi de phan tich.", {}

    if model is None:
        return None, "Model chua duoc tai vao bo nho.", {}

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

    # Thiet lap thong bao va vien vien theo ket qua
    if pred_class_name == "PNEUMONIA":
        color = (220, 53, 69)  # Mau do
        status = "VIEM PHOI (PNEUMONIA)"
        advice = "Phat hien dau hieu nghi ngo viem phoi. Can tham khao them y kien bac si chuyen khoa."
    else:
        color = (40, 167, 69)  # Mau xanh
        status = "BINH THUONG (NORMAL)"
        advice = "Khong phat hien dau hieu bat thuong ve viem phoi tren anh chup."

    border_width = 6
    for i in range(border_width):
        draw.rectangle(
            [i, i, display_size[0] - 1 - i, display_size[1] - 1 - i],
            outline=color,
        )

    label = f"{pred_class_name} ({pred_conf:.1%})"
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    draw.rectangle(
        [border_width, border_width, text_w + border_width + 16, text_h + border_width + 12],
        fill=color,
    )
    draw.text(
        (border_width + 8, border_width + 4),
        label,
        fill="white",
        font=font,
    )

    result_text = f"""
### KET QUA CHAN DOAN: {status}

- Phan loai: **{pred_class_name}**
- Do tin cay: **{pred_conf:.1%}**

**Loi khuyen:**
{advice}

---
*Luu y: He thong mang tinh chat ho tro tham khao ky thuat, khong thay the chan doan y te chinh thuc.*
"""

    return annotated, result_text, confidence_dict


def get_sample_images():
    """Lay danh sach mot so anh mau tu tap test."""
    samples = []
    test_dir = Path("chest_xray/test")

    if test_dir.exists():
        normal_dir = test_dir / "NORMAL"
        if normal_dir.exists():
            normal_imgs = sorted(normal_dir.glob("*.jpeg"))[:3]
            samples.extend([str(p) for p in normal_imgs])

        pneumonia_dir = test_dir / "PNEUMONIA"
        if pneumonia_dir.exists():
            pneumonia_imgs = sorted(pneumonia_dir.glob("*.jpeg"))[:3]
            samples.extend([str(p) for p in pneumonia_imgs])

    return samples if samples else None


def create_app():
    """Khoi tao giao dien Gradio."""
    sample_images = get_sample_images()

    custom_css = """
    .gradio-container {
        max-width: 1100px !important;
        margin: auto !important;
    }
    """

    with gr.Blocks(title="He thong nhan dien viem phoi - YOLOv8") as app:
        gr.Markdown(
            """
            # He Thong Nhan Dien Viem Phoi tu Anh X-Quang
            Mo hinh phan loai anh su dung kien truc YOLOv8 Classification (NORMAL / PNEUMONIA).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    type="pil",
                    label="Upload anh X-quang phoi",
                    height=380,
                )
                predict_btn = gr.Button(
                    "Phan tich anh",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    type="pil",
                    label="Ket qua phan tich",
                    height=380,
                )
                output_label = gr.Label(
                    label="Xac suat tung lop",
                    num_top_classes=2,
                )

        result_text = gr.Markdown(label="Chi tiet ket qua")

        # Gan su kien nhan nut va thay doi anh
        predict_btn.click(
            fn=predict,
            inputs=[input_image],
            outputs=[output_image, result_text, output_label],
        )

        input_image.change(
            fn=predict,
            inputs=[input_image],
            outputs=[output_image, result_text, output_label],
        )

        # Anh mau de kiem thu nhanh
        if sample_images:
            gr.Markdown("### Anh mau kiem thu")
            gr.Examples(
                examples=sample_images,
                inputs=input_image,
                outputs=[output_image, result_text, output_label],
                fn=predict,
                cache_examples=False,
            )

        gr.Markdown(
            """
            ---
            ### Huong dan su dung
            1. Chon anh X-quang tu may hoac chon mot trong cac anh mau ben tren.
            2. Nhan nut 'Phan tich anh' de nhan ket qua phan loai va muc do tin cay.
            """
        )

    return app


if __name__ == "__main__":
    args = parse_args()

    # Tai model
    load_model(args.model)

    # Khoi chay ung dung
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
