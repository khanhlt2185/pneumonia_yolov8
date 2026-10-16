"""
evaluate.py - Danh gia chi tiet mo hinh YOLOv8 tren tap test.

Script thuc hien:
- Tinh toan Confusion Matrix (Ma tran nham lan)
- Tinh toan Classification Report (Precision, Recall, F1-Score)
- Luu anh truc quan hoa ket qua du doan mau va phan phoi do tin cay

Cach chay:
    python evaluate.py
    python evaluate.py --model best_pneumonia_model.pt --data dataset
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Danh gia mo hinh YOLOv8-cls")
    parser.add_argument(
        "--model",
        type=str,
        default="best_pneumonia_model.pt",
        help="Duong dan toi file model .pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset",
        help="Duong dan toi thu muc dataset",
    )
    parser.add_argument("--imgsz", type=int, default=224, help="Kich thuoc anh")
    parser.add_argument(
        "--output", type=str, default="evaluation_results", help="Thu muc luu ket qua"
    )
    return parser.parse_args()


def evaluate(args):
    """Danh gia chi tiet mo hinh tren tap kiem thu."""
    print("=" * 60)
    print("DANH GIA CHI TIET MO HINH")
    print("=" * 60)

    # Tao thu muc chua ket qua danh gia
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tai mo hinh da huan luyen
    model = YOLO(args.model)
    class_names = model.names

    print(f"\nDa tai model: {args.model}")
    print(f"Danh sach lop: {class_names}")

    test_dir = Path(args.data) / "test"
    if not test_dir.exists():
        print("Khong tim thay thu muc test")
        return

    # Thu thap toan bo anh test va nhan thuc te
    all_images = []
    true_labels = []
    pred_labels = []
    pred_confs = []

    for class_name in sorted(os.listdir(test_dir)):
        class_dir = test_dir / class_name
        if not class_dir.is_dir() or class_name.startswith("__"):
            continue
        for img_file in sorted(class_dir.glob("*.*")):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                all_images.append(str(img_file))
                true_labels.append(class_name)

    print(f"\nDang du doan tren {len(all_images)} anh test...")

    # Du doan theo lo tren tap test
    results = model.predict(
        source=all_images,
        imgsz=args.imgsz,
        verbose=False,
    )

    for r in results:
        pred_class_idx = r.probs.top1
        pred_class_name = class_names[pred_class_idx]
        pred_conf = r.probs.top1conf.item()
        pred_labels.append(pred_class_name)
        pred_confs.append(pred_conf)

    # Tinh toan cac chi so danh gia
    unique_classes = sorted(set(true_labels) | set(pred_labels))

    # Tinh ma tran nham lan (Confusion matrix)
    n_classes = len(unique_classes)
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[class_to_idx[t]][class_to_idx[p]] += 1

    # Tinh Precision, Recall, F1-Score cho tung lop
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 60)

    total_correct = 0
    total_samples = 0

    for i, cls in enumerate(unique_classes):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n_classes)) - tp
        fn = sum(cm[i][j] for j in range(n_classes)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = sum(cm[i])

        total_correct += tp
        total_samples += support

        print(f"{cls:<15} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print("-" * 60)
    print(f"{'Accuracy':<15} {'':>10} {'':>10} {accuracy:>10.4f} {total_samples:>10}")
    print("=" * 60)

    # Ve va luu bieu do Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_classes,
        yticklabels=unique_classes,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Confusion Matrix - Pneumonia Detection", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion Matrix da luu tai: {cm_path}")

    # Ve va luu mot so anh du doan mau
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Mot so ket qua du doan mau", fontsize=16)

    indices = np.random.choice(len(all_images), min(12, len(all_images)), replace=False)

    for idx, ax in zip(indices, axes.flat):
        img = Image.open(all_images[idx]).convert("RGB")
        ax.imshow(img, cmap="gray")

        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        conf = pred_confs[idx]

        color = "green" if true_label == pred_label else "red"
        status = "Dung" if true_label == pred_label else "Sai"

        ax.set_title(
            f"[{status}] True: {true_label}\nPred: {pred_label} ({conf:.1%})",
            fontsize=10,
            color=color,
        )
        ax.axis("off")

    plt.tight_layout()
    samples_path = output_dir / "sample_predictions.png"
    plt.savefig(samples_path, dpi=150)
    plt.close()
    print(f"Sample predictions da luu tai: {samples_path}")

    # Ve bieu do phan phoi do tin cay (Confidence distribution)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    correct_confs = [c for t, p, c in zip(true_labels, pred_labels, pred_confs) if t == p]
    wrong_confs = [c for t, p, c in zip(true_labels, pred_labels, pred_confs) if t != p]

    axes[0].hist(correct_confs, bins=30, color="green", alpha=0.7, edgecolor="black")
    axes[0].set_title("Do tin cay - Du doan dung", fontsize=12)
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("So luong")

    if wrong_confs:
        axes[1].hist(wrong_confs, bins=30, color="red", alpha=0.7, edgecolor="black")
    axes[1].set_title("Do tin cay - Du doan sai", fontsize=12)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("So luong")

    plt.tight_layout()
    conf_path = output_dir / "confidence_distribution.png"
    plt.savefig(conf_path, dpi=150)
    plt.close()
    print(f"Confidence distribution da luu tai: {conf_path}")

    print(f"\nTat ca ket qua danh gia da luu trong: {output_dir.absolute()}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
