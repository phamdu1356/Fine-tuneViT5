# reports/hardware_and_training.md
# Template — điền vào sau khi chạy training xong (Giang)
# ──────────────────────────────────────────────────────────

# Báo cáo Phần cứng và Training — Fine-tuneViT5

**Người phụ trách:** Giang
**Ngày hoàn thành:** _(điền sau)_
**Config sử dụng:** `configs/train_v1.yaml`
**Dataset:** VietNews v2 — SHA256: `8064243b0e8fc6dbb841fe459d7440f87a12b7b8515f45f932c47112a8451681`

---

## 1. Phần cứng

| Thành phần | Thông tin |
|---|---|
| **GPU** | _(ví dụ: NVIDIA A100 40GB)_ |
| **VRAM** | _(GB)_ |
| **GPU count** | _(số lượng GPU)_ |
| **CPU** | _(ví dụ: Intel Xeon, 32 cores)_ |
| **RAM** | _(GB)_ |
| **Dung lượng đĩa** | _(GB trống)_ |

---

## 2. Môi trường phần mềm

| Thư viện | Phiên bản |
|---|---|
| Python | _(ví dụ: 3.11.8)_ |
| PyTorch | _(ví dụ: 2.2.0+cu121)_ |
| CUDA | _(ví dụ: 12.1)_ |
| transformers | _(ví dụ: 5.13.1)_ |
| datasets | _(ví dụ: 2.18.0)_ |
| accelerate | _(ví dụ: 0.30.0)_ |
| sentencepiece | _(ví dụ: 0.2.0)_ |
| rouge_score | _(ví dụ: 0.1.2)_ |

---

## 3. Cấu hình Training

| Tham số | Giá trị |
|---|---|
| Model | `VietAI/vit5-base` |
| Revision | _(điền hash commit)_ |
| Precision | _(fp16 / bf16 / fp32)_ |
| Per-device batch size | _(ví dụ: 8)_ |
| Gradient accumulation | _(ví dụ: 4)_ |
| Effective batch size | _(tính: batch × accum × n_gpu)_ |
| Learning rate | _(ví dụ: 5e-4)_ |
| LR scheduler | _(ví dụ: linear)_ |
| Warmup ratio | _(ví dụ: 0.05)_ |
| Epochs | _(ví dụ: 5)_ |
| Max source length | _(ví dụ: 512)_ |
| Max target length | _(ví dụ: 128)_ |
| Seed | 42 |
| Num beams (eval) | _(ví dụ: 4)_ |

---

## 4. Smoke Test

| Hạng mục | Kết quả |
|---|---|
| Forward pass | _(OK / FAIL)_ |
| Backward pass | _(OK / FAIL)_ |
| Loss hữu hạn (không NaN) | _(OK / FAIL)_ |
| Validation step | _(OK / FAIL)_ |
| Save checkpoint | _(OK / FAIL)_ |
| Load checkpoint | _(OK / FAIL)_ |
| Inference từ checkpoint | _(OK / FAIL)_ |
| **Kết luận smoke test** | _(PASS / FAIL)_ |

**Lệnh smoke test đã dùng:**
```bash
python train.py --config configs/train_smoke.yaml --smoke
```

---

## 5. Kết quả Training

### Loss curve (điền sau)

| Epoch | Train Loss | Val Loss | Val ROUGE-1 | Val ROUGE-2 | Val ROUGE-L |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

### Checkpoint tốt nhất

| Hạng mục | Giá trị |
|---|---|
| Checkpoint | _(ví dụ: checkpoint-epoch-3)_ |
| Tiêu chí chọn | Validation ROUGE-L cao nhất |
| Val ROUGE-L | _(ví dụ: 28.5)_ |
| Val ROUGE-1 | |
| Val ROUGE-2 | |

---

## 6. Tài nguyên sử dụng

| Hạng mục | Giá trị |
|---|---|
| VRAM peak | _(GB)_ |
| RAM peak | _(GB)_ |
| Thời gian train | _(ví dụ: 4h 32m)_ |
| Throughput | _(samples/s)_ |

---

## 7. Lệnh đã dùng

```bash
# Tạo môi trường
bash setup_env.sh                  # hoặc setup_env.bat trên Windows

# Smoke test (xác nhận pipeline)
python train.py --config configs/train_smoke.yaml --smoke

# Full training
python train.py --config configs/train_v1.yaml

# Resume (nếu bị gián đoạn)
python train.py --config configs/train_v1.yaml --resume auto

# Đóng gói model tốt nhất
python package_model.py --checkpoint outputs/checkpoints --config configs/train_v1.yaml
```

---

## 8. Vấn đề gặp phải và giải pháp

_(Ghi lại các lỗi, warnings, và cách xử lý)_

| Vấn đề | Giải pháp |
|---|---|
| _(ví dụ: OOM với batch_size=16)_ | _(giảm xuống 8, tăng gradient_accumulation)_ |

---

## 9. Bàn giao cho Khải (G5)

- [ ] Model, tokenizer, generation config tại: `outputs/checkpoints/best/`
- [ ] `training_info.json` chứa seed, dataset hash, config, metrics
- [ ] `HANDOFF.md` hướng dẫn sử dụng
- [ ] Config eval: `configs/eval_baseline.yaml` (khóa bởi Hải Anh)
- [ ] Lệnh eval:
  ```bash
  python evaluate.py --model outputs/checkpoints/best \
      --config configs/eval_baseline.yaml \
      --split test \
      --output_dir outputs/final_eval
  ```
