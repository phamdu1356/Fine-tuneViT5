# Fine-tuneViT5

Fine-tune **VietAI/vit5-base** cho bài toán tóm tắt văn bản tiếng Việt (abstractive summarisation).

## Tổng quan

| Hạng mục | Thông tin |
|---|---|
| **Model** | `VietAI/vit5-base` (T5 encoder-decoder) |
| **Task** | `article` → `abstract` (Vietnamese news) |
| **Dataset** | VietNews v2 — 143,811 mẫu (train/val/test: 65/15/20) |
| **Metrics** | ROUGE-1, ROUGE-2, ROUGE-L |

## Cài đặt môi trường

### Windows (local / smoke test)
```bat
# CPU (smoke test)
setup_env.bat

# CUDA (nếu có GPU driver)
setup_env.bat cuda

# Kích hoạt
venv\Scripts\activate
```

### Linux / Cloud GPU
```bash
# Auto-detect CUDA
bash setup_env.sh

# Chỉ định CUDA version
bash setup_env.sh cu128   # CUDA 12.8
bash setup_env.sh cu121   # CUDA 12.1
bash setup_env.sh cu118   # CUDA 11.8
bash setup_env.sh cpu     # CPU only

# Kích hoạt
source venv/bin/activate
```

## Cách chạy

### Smoke test (CPU, ~2 phút)
```bash
python train.py --config configs/train_smoke.yaml --smoke
```

### Full training (cloud GPU)
```bash
python train.py --config configs/train_v1.yaml
```

### Resume sau khi bị gián đoạn
```bash
python train.py --config configs/train_v1.yaml --resume auto
# hoặc
python train.py --config configs/train_v1.yaml --resume outputs/checkpoints/checkpoint-1000
```

### Multi-GPU với accelerate
```bash
accelerate config   # cấu hình lần đầu
accelerate launch train.py --config configs/train_v1.yaml
```

### Override tham số
```bash
python train.py --config configs/train_v1.yaml \
    --override training.learning_rate=1e-4 \
    --override training.num_train_epochs=3 \
    --override training.per_device_train_batch_size=4
```

### Evaluation
```bash
# Baseline (model gốc, trước fine-tune)
python evaluate.py --model VietAI/vit5-base \
    --config configs/eval_baseline.yaml \
    --split validation

# Fine-tuned model
python evaluate.py --model outputs/checkpoints/best \
    --config configs/eval_baseline.yaml \
    --split test \
    --output_dir outputs/final_eval
```

### Đóng gói model
```bash
python package_model.py \
    --checkpoint outputs/checkpoints \
    --config configs/train_v1.yaml
```

## Cấu trúc thư mục

```
Fine-tuneViT5/
├── train.py                        ← Script training chính
├── evaluate.py                     ← Script evaluation (baseline + fine-tuned)
├── package_model.py                ← Đóng gói model tốt nhất
├── requirements.txt                ← Dependencies
├── setup_env.bat / setup_env.sh    ← Tạo venv
├── configs/
│   ├── train_v1.yaml               ← Config training đầy đủ (cloud)
│   ├── train_smoke.yaml            ← Config smoke test (CPU local)
│   └── eval_baseline.yaml          ← Config evaluation (locked bởi Hải Anh)
├── data/processed/v2/
│   └── vietnews_v2.parquet         ← Dataset đã lock (Git LFS)
├── outputs/                        ← Git-ignored
│   ├── checkpoints/                ← Checkpoints training
│   └── checkpoints/best/           ← Best model (bàn giao G5)
└── reports/
    └── hardware_and_training.md    ← Báo cáo training (Giang điền)
```

## Cấu hình quan trọng

Chỉnh sửa `configs/train_v1.yaml` để thay đổi:
- `model.name_or_path` — model ID hoặc local path
- `data.max_source_length` — độ dài cắt input (512 mặc định)
- `training.learning_rate` — learning rate
- `training.num_train_epochs` — số epochs
- `training.fp16` / `training.bf16` — precision
- `training.per_device_train_batch_size` + `gradient_accumulation_steps`

## Ràng buộc quan trọng

- **Không dùng test split** để chọn hyperparameter hoặc checkpoint
- **Không thay đổi** `configs/eval_baseline.yaml` sau khi Hải Anh lock (G3)
- **Không commit** `outputs/`, `venv/`, model weights, `.env` vào Git
- **Dataset v2 đã lock** — không sửa `data/processed/v2/`
- **Seed = 42** cho mọi random operation