#!/usr/bin/env python3
"""
train.py — Fine-tune VietAI/vit5-base cho bài toán tóm tắt văn bản tiếng Việt.

Dataset: data/processed/v2/vietnews_v2.parquet  (VietNews v2, 143,811 mẫu)
Model:   VietAI/vit5-base  (T5 encoder-decoder)
Task:    Sequence-to-sequence summarisation (article → abstract)

Cách dùng
─────────
  # Full training trên cloud GPU:
  python train.py --config configs/train_v1.yaml

  # Smoke test trên CPU (32 mẫu, 2 steps — kiểm tra pipeline):
  python train.py --config configs/train_smoke.yaml --smoke

  # Resume từ checkpoint:
  python train.py --config configs/train_v1.yaml --resume outputs/checkpoints/checkpoint-1000

  # Multi-GPU với accelerate:
  accelerate launch train.py --config configs/train_v1.yaml

  # Override tham số bất kỳ:
  python train.py --config configs/train_v1.yaml \\
      --override training.learning_rate=1e-4 \\
      --override training.num_train_epochs=3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Fix Unicode output on Windows (cp1252 -> utf-8)
import io as _io
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf-8-sig'):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Config helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load YAML config và trả về dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config không tìm thấy: {config_path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config: %s", config_path)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Override giá trị config từ command-line (dạng key.subkey=value)."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override phải có dạng key.subkey=value, nhận: {item!r}")
        key_path, value_str = item.split("=", 1)
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        # Tự động parse kiểu dữ liệu
        try:
            parsed = json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            parsed = value_str
        node[keys[-1]] = parsed
        logger.info("Override: %s = %r", key_path, parsed)
    return cfg


def apply_smoke_overrides(cfg: dict) -> dict:
    """Áp dụng giới hạn smoke test từ cfg['smoke_limits'] (nếu có)."""
    limits = cfg.get("smoke_limits", {})
    # Giới hạn steps (cho phép người dùng override qua command line)
    cfg.setdefault("training", {})
    if "max_steps" not in cfg["training"]:
        cfg["training"]["max_steps"] = 2
    # Tắt fp16/bf16 khi không có CUDA
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if not has_cuda:
        cfg["training"]["fp16"] = False
        cfg["training"]["bf16"] = False
        logger.info("[smoke] CPU detected → fp16/bf16 disabled")
    # Giới hạn mẫu
    if "smoke_limits" not in cfg:
        cfg["smoke_limits"] = {"max_train_samples": 32, "max_eval_samples": 16}
    logger.info("[smoke] max_train=%d, max_eval=%d, max_steps=%d",
                cfg["smoke_limits"].get("max_train_samples", 32),
                cfg["smoke_limits"].get("max_eval_samples", 16),
                cfg["training"]["max_steps"])
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# 2. Environment logging
# ══════════════════════════════════════════════════════════════════════════════

def log_environment() -> dict:
    """Log và trả về thông tin môi trường."""
    import torch
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = "not installed"
    try:
        import datasets as hf_datasets
        env["datasets"] = hf_datasets.__version__
    except ImportError:
        env["datasets"] = "not installed"
    try:
        import accelerate
        env["accelerate"] = accelerate.__version__
    except ImportError:
        env["accelerate"] = "not installed"

    if torch.cuda.is_available():
        env["cuda_version"] = torch.version.cuda
        env["gpu_count"] = torch.cuda.device_count()
        env["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
            for i in range(torch.cuda.device_count())
        ]
    else:
        env["cuda_version"] = None
        env["gpu_count"] = 0
        env["gpus"] = []

    logger.info("-" * 60)
    logger.info("ENVIRONMENT")
    for k, v in env.items():
        if k != "gpus":
            logger.info("  %-20s %s", k, v)
    for gpu in env.get("gpus", []):
        logger.info("  GPU[%d]: %s  VRAM=%.1f GB  bf16=%s",
                    gpu["index"], gpu["name"], gpu["vram_gb"], gpu["bf16_supported"])
    logger.info("-" * 60)
    return env


# ══════════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════════

def verify_parquet_sha256(path: Path, expected: str | None) -> None:
    """Xác nhận SHA-256 của file parquet khớp với manifest."""
    if not expected:
        logger.warning("expected_sha256 không được cấu hình, bỏ qua kiểm tra toàn vẹn.")
        return
    logger.info("Đang tính SHA-256 của %s …", path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected.lower():
        raise ValueError(
            f"SHA-256 không khớp!\n  Expected: {expected}\n  Actual:   {actual}\n"
            "Dataset có thể bị hỏng hoặc bị thay thế."
        )
    logger.info("SHA-256 OK: %s", actual[:16] + "…")


def load_splits(cfg: dict, is_smoke: bool) -> tuple:
    """Load train/validation split từ parquet, áp dụng giới hạn smoke nếu cần."""
    import pandas as pd
    from datasets import Dataset

    data_cfg = cfg["data"]
    parquet_path = Path(data_cfg["parquet_path"])
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset không tìm thấy: {parquet_path}\n"
            "Đảm bảo Git LFS đã pull file: git lfs pull"
        )

    verify_parquet_sha256(parquet_path, data_cfg.get("expected_sha256"))

    logger.info("Đang load dataset từ %s …", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Tổng số dòng: %d, Columns: %s", len(df), df.columns.tolist())

    split_col = data_cfg["split_column"]
    src_col = data_cfg["source_column"]
    tgt_col = data_cfg["target_column"]

    # Kiểm tra columns tồn tại
    for col in [split_col, src_col, tgt_col]:
        if col not in df.columns:
            raise KeyError(
                f"Cột '{col}' không tìm thấy trong parquet. "
                f"Các cột hiện có: {df.columns.tolist()}"
            )

    # Lọc theo split_v2
    train_df = df[df[split_col] == "train"][[src_col, tgt_col]].copy()
    val_df   = df[df[split_col] == "validation"][[src_col, tgt_col]].copy()

    logger.info("Trước smoke limit — train: %d, val: %d", len(train_df), len(val_df))

    # Smoke limits
    if is_smoke:
        limits = cfg.get("smoke_limits", {})
        n_train = limits.get("max_train_samples", 32)
        n_val   = limits.get("max_eval_samples", 16)
        train_df = train_df.sample(n=min(n_train, len(train_df)), random_state=42)
        val_df   = val_df.sample(n=min(n_val, len(val_df)), random_state=42)
        logger.info("[smoke] Sau limit — train: %d, val: %d", len(train_df), len(val_df))

    # Đổi tên cột để tokenizer dễ xử lý
    train_df = train_df.rename(columns={src_col: "source", tgt_col: "target"}).reset_index(drop=True)
    val_df   = val_df.rename(columns={src_col: "source", tgt_col: "target"}).reset_index(drop=True)

    # Xoá dòng có source hoặc target rỗng
    train_df = train_df.dropna(subset=["source", "target"])
    train_df = train_df[train_df["source"].str.strip() != ""]
    train_df = train_df[train_df["target"].str.strip() != ""]
    val_df   = val_df.dropna(subset=["source", "target"])
    val_df   = val_df[val_df["source"].str.strip() != ""]
    val_df   = val_df[val_df["target"].str.strip() != ""]

    logger.info("Sau cleanup — train: %d, val: %d", len(train_df), len(val_df))

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset   = Dataset.from_pandas(val_df,   preserve_index=False)
    return train_dataset, val_dataset


# ══════════════════════════════════════════════════════════════════════════════
# 4. Tokenization
# ══════════════════════════════════════════════════════════════════════════════

def build_tokenize_fn(tokenizer, max_source: int, max_target: int):
    """Trả về hàm tokenize cho map()."""

    def tokenize(examples):
        # Tokenize source (article) và target (abstract) cùng lúc
        # Dùng text_target= — tương thích transformers 4.x & 5.x (as_target_tokenizer đã deprecated)
        # Tokenize source
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_source,
            truncation=True,
            padding=False,
        )
        
        # Tokenize target separately to use max_target_length
        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_target,
            truncation=True,
            padding=False,
        )
        
        # QUAN TRỌNG: thay pad_token_id bằng -100 để loss không tính padding
        label_ids = labels["input_ids"]
        label_ids = [
            [(-100 if token == tokenizer.pad_token_id else token) for token in seq]
            for seq in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize


def tokenize_datasets(train_ds, val_ds, tokenizer, cfg: dict, is_smoke: bool) -> tuple:
    """Tokenize train/val datasets."""
    data_cfg = cfg["data"]
    max_src = data_cfg["max_source_length"]
    max_tgt = data_cfg["max_target_length"]

    logger.info("Tokenizing… max_source=%d, max_target=%d", max_src, max_tgt)

    tokenize_fn = build_tokenize_fn(tokenizer, max_src, max_tgt)

    num_proc = 1 if is_smoke else min(4, os.cpu_count() or 1)

    train_tok = train_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_tok = val_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    logger.info("Tokenization xong — train: %d, val: %d", len(train_tok), len(val_tok))
    return train_tok, val_tok


# ══════════════════════════════════════════════════════════════════════════════
# 5. Metrics (ROUGE)
# ══════════════════════════════════════════════════════════════════════════════

def build_compute_metrics(tokenizer):
    """Trả về hàm compute_metrics cho Seq2SeqTrainer."""
    from rouge_score import rouge_scorer as rs_module

    scorer = rs_module.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,       # tiếng Việt không dùng stemmer
        tokenizer=None,          # dùng whitespace tokenizer mặc định
    )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # preds có thể là tuple (sequences, scores) từ generate
        if isinstance(preds, tuple):
            preds = preds[0]

        # Xử lý NaN/overflow từ generate
        import numpy as np
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Thay -100 bằng pad_token_id trước khi decode labels
        labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Trim whitespace
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        # Tính ROUGE
        r1_scores, r2_scores, rL_scores = [], [], []
        for pred, ref in zip(decoded_preds, decoded_labels):
            if not pred:
                pred = " "
            if not ref:
                ref = " "
            scores = scorer.score(ref, pred)
            r1_scores.append(scores["rouge1"].fmeasure)
            r2_scores.append(scores["rouge2"].fmeasure)
            rL_scores.append(scores["rougeL"].fmeasure)

        return {
            "rouge1": round(float(np.mean(r1_scores)) * 100, 4),
            "rouge2": round(float(np.mean(r2_scores)) * 100, 4),
            "rouge_l": round(float(np.mean(rL_scores)) * 100, 4),
        }

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 6. Training arguments
# ══════════════════════════════════════════════════════════════════════════════

def build_training_args(cfg: dict, output_dir: str | None = None):
    """Xây dựng Seq2SeqTrainingArguments từ config."""
    from transformers import Seq2SeqTrainingArguments

    t = cfg["training"]
    gen = cfg.get("generation", {})

    out_dir = output_dir or t.get("output_dir", "outputs/checkpoints")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Mapping tên eval_strategy cho transformers 4.x vs 5.x compatibility
    eval_strategy = t.get("eval_strategy", t.get("evaluation_strategy", "epoch"))

    kwargs: dict[str, Any] = dict(
        output_dir=out_dir,
        # Epochs / steps
        num_train_epochs=t.get("num_train_epochs", 5),
        max_steps=t.get("max_steps", -1),
        # Batch
        per_device_train_batch_size=t.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        # LR
        learning_rate=float(t.get("learning_rate", 5e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        warmup_ratio=float(t.get("warmup_ratio", 0.05)),
        warmup_steps=int(t.get("warmup_steps", 0)),
        # Regularization
        weight_decay=float(t.get("weight_decay", 0.01)),
        max_grad_norm=float(t.get("max_grad_norm", 1.0)),
        # Precision & Memory
        fp16=bool(t.get("fp16", False)),
        bf16=bool(t.get("bf16", False)),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", False)),
        # Seed
        seed=int(t.get("seed", 42)),
        data_seed=int(t.get("data_seed", 42)),
        # Checkpointing
        save_strategy=t.get("save_strategy", "epoch"),
        save_steps=int(t.get("save_steps", 500)),
        save_total_limit=int(t.get("save_total_limit", 3)),
        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=int(t.get("eval_steps", 500)),
        load_best_model_at_end=bool(t.get("load_best_model_at_end", True)),
        metric_for_best_model=t.get("metric_for_best_model", "rouge_l"),
        greater_is_better=bool(t.get("greater_is_better", True)),
        # Logging
        logging_steps=int(t.get("logging_steps", 50)),
        logging_first_step=True,
        report_to=t.get("report_to", "none"),
        # DataLoader
        dataloader_num_workers=int(t.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(t.get("dataloader_pin_memory", False)),
        # Seq2Seq generate
        predict_with_generate=bool(t.get("predict_with_generate", True)),
        generation_max_length=int(gen.get("max_new_tokens", 128)),
        generation_num_beams=int(gen.get("num_beams", 4)),
    )

    # Xử lý các tham số không hợp lệ hoặc deprecated trong transformers 5.x
    import transformers as _hf
    _hf_version = tuple(int(x) for x in _hf.__version__.split(".")[:2])

    if _hf_version >= (5, 0):
        # transformers 5.x loại bỏ hoàn toàn warmup_ratio
        if "warmup_ratio" in kwargs:
            if kwargs["warmup_ratio"] > 0 and kwargs.get("warmup_steps", 0) == 0:
                logger.warning("transformers >= 5.x không còn hỗ trợ `warmup_ratio`. Hãy sử dụng `warmup_steps` thay thế trong file config (VD: warmup_steps: 500). Tạm thời đặt warmup_steps = 0.")
            del kwargs["warmup_ratio"]
    else:
        # transformers 4.x: chỉ được truyền 1 trong 2
        if kwargs.get("warmup_steps", 0) > 0:
            if "warmup_ratio" in kwargs:
                del kwargs["warmup_ratio"]
        elif kwargs.get("warmup_ratio", 0) > 0:
            if "warmup_steps" in kwargs:
                del kwargs["warmup_steps"]
        else:
            if "warmup_ratio" in kwargs:
                del kwargs["warmup_ratio"]

    return Seq2SeqTrainingArguments(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_last_checkpoint(output_dir: str) -> str | None:
    """Tìm checkpoint mới nhất trong output_dir."""
    from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint
    last = hf_get_last_checkpoint(output_dir)
    if last:
        logger.info("Tìm thấy checkpoint để resume: %s", last)
    return last


def save_training_info(output_dir: str, cfg: dict, env: dict, metrics: dict | None) -> None:
    """Lưu training_info.json vào output_dir."""
    info = {
        "environment": env,
        "config": cfg,
        "final_metrics": metrics or {},
    }
    path = Path(output_dir) / "training_info.json"
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved training_info.json → %s", path)


def plot_metrics(log_history: list[dict], output_dir: str) -> None:
    """Vẽ biểu đồ loss và metrics từ log_history."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Không tìm thấy matplotlib. Bỏ qua vẽ biểu đồ. Hãy cài đặt: pip install matplotlib")
        return

    train_loss = []
    val_loss = []
    val_rouge1 = []
    val_rouge2 = []
    val_rouge_l = []
    train_steps = []
    val_steps = []

    for log in log_history:
        step = log.get("step", 0)
        if "loss" in log:
            train_loss.append(log["loss"])
            train_steps.append(step)
        if "eval_loss" in log:
            val_loss.append(log["eval_loss"])
            val_steps.append(step)
            if "eval_rouge1" in log:
                val_rouge1.append(log["eval_rouge1"])
            if "eval_rouge2" in log:
                val_rouge2.append(log["eval_rouge2"])
            if "eval_rouge_l" in log:
                val_rouge_l.append(log["eval_rouge_l"])

    if not train_loss and not val_loss:
        logger.warning("Không có dữ liệu loss để vẽ.")
        return

    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    if train_loss:
        plt.plot(train_steps, train_loss, label='Train Loss')
    if val_loss:
        plt.plot(val_steps, val_loss, label='Val Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)

    # Metrics plot (được coi như Accuracy cho bài toán text generation)
    plt.subplot(1, 2, 2)
    if val_rouge1 or val_rouge2 or val_rouge_l:
        if val_rouge1:
            plt.plot(val_steps, val_rouge1, label='Val ROUGE-1')
        if val_rouge2:
            plt.plot(val_steps, val_rouge2, label='Val ROUGE-2')
        if val_rouge_l:
            plt.plot(val_steps, val_rouge_l, label='Val ROUGE-L')
        plt.xlabel('Steps')
        plt.ylabel('Score')
        plt.title('Validation ROUGE Scores (Accuracy)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plot_path = Path(output_dir) / "training_metrics.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Đã lưu biểu đồ metrics tại: %s", plot_path)


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune VietAI/vit5-base cho tóm tắt văn bản tiếng Việt"
    )
    parser.add_argument(
        "--config", required=True,
        help="Đường dẫn file YAML config, ví dụ: configs/train_v1.yaml"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Chạy smoke test (subset nhỏ, 2 steps) để kiểm tra pipeline"
    )
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume từ checkpoint. Không truyền giá trị = auto-detect; "
             "truyền đường dẫn = resume từ checkpoint cụ thể"
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override config, ví dụ: --override training.learning_rate=1e-4"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load & patch config ──────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.smoke:
        cfg = apply_smoke_overrides(cfg)

    # ── Import heavy libs (sau config để fail sớm nếu config lỗi) ───────────
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            set_seed,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as e:
        logger.error("Thiếu dependency: %s", e)
        logger.error("Chạy: pip install -r requirements.txt")
        sys.exit(1)

    # ── Log environment ──────────────────────────────────────────────────────
    env = log_environment()

    # ── Cảnh báo nếu không có GPU ────────────────────────────────────────────
    if not torch.cuda.is_available() and not args.smoke:
        logger.warning(
            "⚠️  Không có CUDA/GPU. Training trên CPU sẽ rất chậm.\n"
            "   Dùng --smoke để chỉ chạy smoke test, hoặc chuyển sang máy cloud có GPU."
        )

    # ── Set seed ─────────────────────────────────────────────────────────────
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    logger.info("Random seed: %d", seed)

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = cfg["training"]["output_dir"]
    if args.smoke:
        out_dir = cfg["training"].get("output_dir", "outputs/smoke_test")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Resolve resume checkpoint ─────────────────────────────────────────────
    resume_ckpt = None
    if args.resume:
        if args.resume == "auto":
            resume_ckpt = get_last_checkpoint(out_dir) if Path(out_dir).exists() else None
            if resume_ckpt is None:
                logger.info("Không tìm thấy checkpoint để resume, bắt đầu train mới.")
        else:
            resume_ckpt = args.resume
            if not Path(resume_ckpt).exists():
                raise FileNotFoundError(f"Checkpoint không tìm thấy: {resume_ckpt}")
            logger.info("Resume từ: %s", resume_ckpt)
    # Override từ config
    cfg_resume = cfg["training"].get("resume_from_checkpoint")
    if cfg_resume and not resume_ckpt:
        resume_ckpt = cfg_resume

    # Patch trainer_state.json to fix transformers version mismatch error
    if resume_ckpt:
        trainer_state_path = Path(resume_ckpt) / "trainer_state.json"
        if trainer_state_path.exists():
            import json
            try:
                with open(trainer_state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                if "best_global_step" in state_data:
                    del state_data["best_global_step"]
                    with open(trainer_state_path, "w", encoding="utf-8") as f:
                        json.dump(state_data, f, indent=2)
                    logger.info(f"Đã patch {trainer_state_path} (xóa best_global_step)")
            except Exception as e:
                logger.warning("Không thể patch trainer_state.json: %s", e)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    model_name = cfg["model"]["name_or_path"]
    model_revision = cfg["model"].get("revision", "main")
    logger.info("Loading tokenizer: %s (revision=%s)", model_name, model_revision)
    # use_fast=True: dùng tokenizers lib (Rust) thay vì sentencepiece slow tokenizer
    # tránh TypeError với sentencepiece >= 0.2.0 và transformers >= 5.x
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, use_fast=True)

    # ── Load & tokenize dataset ───────────────────────────────────────────────
    train_ds, val_ds = load_splits(cfg, is_smoke=args.smoke)
    train_tok, val_tok = tokenize_datasets(train_ds, val_ds, tokenizer, cfg, is_smoke=args.smoke)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model: %s …", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=model_revision)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s (%.0fM)", f"{n_params:,}", n_params / 1e6)

    # ── Data collator ─────────────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,     # không tính loss trên padding
        pad_to_multiple_of=8 if cfg["training"].get("fp16") else None,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = build_training_args(cfg, output_dir=out_dir)
    logger.info("Output dir: %s", training_args.output_dir)
    logger.info("Effective batch size: %d",
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * max(1, training_args.world_size))

    # ── Compute metrics ───────────────────────────────────────────────────────
    compute_metrics = build_compute_metrics(tokenizer)

    # ── Trainer ───────────────────────────────────────────────────────────────
    # transformers 5.x: 'tokenizer' argument renamed to 'processing_class'
    import transformers as _hf
    _hf_version = tuple(int(x) for x in _hf.__version__.split(".")[:2])
    _trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if _hf_version >= (5, 0):
        _trainer_kwargs["processing_class"] = tokenizer
    else:
        _trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**_trainer_kwargs)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    if args.smoke:
        logger.info("[SMOKE TEST] kiem tra pipeline end-to-end")
    else:
        logger.info("[START] BAT DAU TRAINING")
    logger.info("=" * 60)

    t0 = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    elapsed = time.time() - t0
    logger.info("Training xong sau %.1f phút", elapsed / 60)

    # ── Save model & tokenizer ────────────────────────────────────────────────
    logger.info("Lưu model vào: %s", out_dir)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # ── Final evaluation ──────────────────────────────────────────────────────
    logger.info("Evaluation trên validation set…")
    eval_metrics = trainer.evaluate(
        metric_key_prefix="val",
        max_length=cfg["generation"].get("max_new_tokens", 128),
        num_beams=cfg["generation"].get("num_beams", 4),
    )
    logger.info("Validation metrics: %s", eval_metrics)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # ── Log train stats ───────────────────────────────────────────────────────
    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(train_tok)
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # ── Save training_info.json ───────────────────────────────────────────────
    all_metrics = {**train_metrics, **eval_metrics}
    save_training_info(out_dir, cfg, env, all_metrics)

    # ── Plot metrics ──────────────────────────────────────────────────────────
    plot_metrics(trainer.state.log_history, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HOAN TAT%s", " (SMOKE TEST)" if args.smoke else "")
    logger.info("  Output dir : %s", out_dir)
    logger.info("  ROUGE-1    : %.2f", eval_metrics.get("val_rouge1", 0))
    logger.info("  ROUGE-2    : %.2f", eval_metrics.get("val_rouge2", 0))
    logger.info("  ROUGE-L    : %.2f", eval_metrics.get("val_rouge_l", 0))
    logger.info("  Time (min) : %.1f", elapsed / 60)
    if not args.smoke:
        logger.info("")
        logger.info("Buoc tiep theo -- dong goi model tot nhat:")
        logger.info("  python package_model.py --checkpoint %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
