"""
predict.py - Du doan viem phoi tren anh don le hoac ca thu muc anh.

Cach chay:
    python predict.py --source path/to/image.jpg
    python predict.py --source path/to/folder/
    python predict.py --source chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Du doan viem phoi tu anh X-quang")
    parser.add_argument(
        "--model",
        type=str,
        default="best_pneumonia_model.pt",
        help="Duong dan model",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Anh hoac thu muc anh can du doan",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Kich thuoc anh")
    parser.add_argument(
        "--conf", type=float, default=0.5, help="Nguong confidence toi thieu"
    )
    parser.add_argument(
        "--save", action="store_true", help="Luu anh ket qua"
    )
    parser.add_argument(
        "--output", type=str, default="predictions", help="Thu muc luu ket qua"
    )
    return parser.parse_args()


def predict_single(model, image_path, imgsz=224):
    """Du doan phan loai cho mot anh don le."""
    results = model.predict(source=image_path, imgsz=imgsz, verbose=False)
    result = results[0]

    pred_class_idx = result.probs.top1
    pred_class_name = model.names[pred_class_idx]
    pred_conf = result.probs.top1conf.item()

    # Lay chi tiet xac suat cua tung lop
    all_probs = {}
    for idx, name in model.names.items():
        all_probs[name] = result.probs.data[idx].item()

    return pred_class_name, pred_conf, all_probs


def main():
    args = parse_args()

    # Tai mo hinh
    model = YOLO(args.model)
    print(f"Model loaded: {args.model}")
    print(f"Classes: {model.names}\n")

    source = Path(args.source)

    if source.is_file():
        # Du doan tren mot file anh
        pred_class, conf, all_probs = predict_single(model, str(source), args.imgsz)

        print(f"Anh: {source.name}")
        print(f"Ket qua: {pred_class} (Confidence: {conf:.1%})")
        for cls, prob in all_probs.items():
            print(f"  - {cls}: {prob:.4f}")

    elif source.is_dir():
        # Du doan tren toan bo anh trong thu muc
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = [
            f
            for f in sorted(source.rglob("*"))
            if f.suffix.lower() in image_extensions
        ]

        print(f"Tim thay {len(images)} anh trong {source}\n")

        correct = 0
        total = 0
        results_list = []

        for img_path in images:
            pred_class, conf, all_probs = predict_single(
                model, str(img_path), args.imgsz
            )

            # Xac dinh nhan thuc te dua theo ten thu muc cha neu co
            true_label = img_path.parent.name
            is_correct = true_label == pred_class

            if true_label in model.names.values():
                total += 1
                if is_correct:
                    correct += 1
                status = "Dung" if is_correct else "Sai"
            else:
                status = "Khong ro"

            print(
                f"  [{status:<7}] {img_path.name:<45} {pred_class:<12} "
                f"conf={conf:.1%}"
            )

            results_list.append(
                {
                    "path": str(img_path),
                    "prediction": pred_class,
                    "confidence": conf,
                    "true_label": true_label,
                    "correct": is_correct,
                }
            )

        if total > 0:
            accuracy = correct / total
            print(f"\n{'='*60}")
            print(f"Do chinh xac: {correct}/{total} = {accuracy:.1%}")
            print(f"{'='*60}")

    else:
        print(f"Khong tim thay duong dan: {source}")


if __name__ == "__main__":
    main()
