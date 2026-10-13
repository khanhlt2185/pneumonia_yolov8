# He Thong Nhan Dien Viem Phoi tu Anh X-Quang

He thong su dung mo hinh **YOLOv8 Classification** de phan loai anh X-quang phoi thanh 2 lop:
- **NORMAL**: Phoi binh thuong
- **PNEUMONIA**: Viem phoi

## Cau truc du an

```
doAn/
├── chest_xray/              # Thu muc dataset anh X-quang
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── train.py                 # Script huan luyen mo hinh
├── evaluate.py              # Script danh gia chi tiet tren tap test
├── predict.py               # Script du doan dong lenh (CLI)
├── app.py                   # Demo giao dien web bang Gradio
├── requirements.txt         # Danh sach thu vien can thiet
├── best_pneumonia_model.pt  # File trong so mo hinh sau khi huan luyen
└── README.md                # Tai lieu huong dan
```

## Huong dan su dung

### 1. Cai dat thu vien

```bash
pip install -r requirements.txt
```

### 2. Huan luyen mo hinh

```bash
# Huan luyen co ban tren GPU
python train.py --epochs 30 --model yolov8n-cls.pt --imgsz 224 --batch 32 --device 0

# Huan luyen tren CPU neu khong co GPU
python train.py --epochs 30 --model yolov8n-cls.pt --device cpu
```

### 3. Danh gia mo hinh tren tap test

```bash
python evaluate.py --model best_pneumonia_model.pt --data chest_xray
```

Ket qua se duoc luu tai thu muc `evaluation_results/`:
- `confusion_matrix.png`: Ma tran nham lan
- `sample_predictions.png`: Mot so anh du doan mau
- `confidence_distribution.png`: Bieu do phan phoi do tin cay

### 4. Du doan anh qua CLI

```bash
# Du doan tren mot anh cu the
python predict.py --source chest_xray/test/PNEUMONIA/person1_virus_6.jpeg

# Du doan tren mot thu muc anh
python predict.py --source chest_xray/test/
```

### 5. Chay giao dien web demo

```bash
python app.py
```

Mo trinh duyet tai: `http://localhost:7860`

## Thong tin ky thuat

- Framework: Ultralytics YOLOv8
- Tac vu: Image Classification
- Kich thuoc dau vao: 224x224
- Cac lop: NORMAL, PNEUMONIA
- Optimizer: AdamW
- Loss / Metrics: Top-1 Accuracy, Precision, Recall, F1-Score
