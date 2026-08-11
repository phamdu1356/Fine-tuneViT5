#!/usr/bin/env python3
"""
package_model.py — Đóng gói model tốt nhất để bàn giao cho Khải (G5).

Script này:
  1. Copy model + tokenizer + generation config từ best checkpoint
  2. Ghi training_info.json với đầy đủ metadata (seed, dataset hash, config, metrics)
  3. Chạy inference test để xác nhận model hoạt động trong process mới
  4. Tạo HANDOFF.md — hướng dẫn sử dụng cho Khải

Cách dùng
─────────
  python package_model.py --checkpoint outputs/checkpoints/checkpoint-best
  python package_model.py --checkpoint outputs/checkpoints --out outputs/checkpoints/best
  python package_model.py --checkpoint outputs/checkpoints --config configs/train_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_best_checkpoint(output_dir: str) -> str:
    """Tìm best checkpoint trong output_dir.
    Ưu tiên: trainer_state.json → best_model_checkpoint; fallback last checkpoint.
    """
    from transformers.trainer_utils import get_last_checkpoint

    # Thử đọc trainer_state.json
    state_file = Path(output_dir) / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        best = state.get("best_model_checkpoint")
        if best and Path(best).exists():
            logger.info("Best checkpoint (từ trainer_state): %s", best)
            return best

    # Fallback: last checkpoint hoặc output_dir chính
    last = get_last_checkpoint(output_dir)
    if last:
        logger.info("Last checkpoint: %s", last)
        return last

    # Nếu output_dir chứa config.json trực tiếp
    if (Path(output_dir) / "config.json").exists():
        return output_dir

    raise FileNotFoundError(
        f"Không tìm thấy checkpoint trong: {output_dir}\n"
        "Truyền đường dẫn cụ thể qua --checkpoint"
    )


def load_training_info(checkpoint_dir: str) -> dict:
    """Load training_info.json nếu có."""
    info_path = Path(checkpoint_dir).parent / "training_info.json"
    if not info_path.exists():
        info_path = Path(checkpoint_dir) / "training_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text(encoding="utf-8"))
    return {}


def load_eval_metrics(checkpoint_dir: str) -> dict:
    """Load eval_results.json hoặc all_results.json nếu có."""
    for fname in ["eval_results.json", "all_results.json"]:
        p = Path(checkpoint_dir).parent / fname
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        p = Path(checkpoint_dir) / fname
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def run_inference_test(model_dir: str, test_text: str) -> str:
    """Chạy inference test trong function mới (độc lập với training context)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    logger.info("Inference test với text: %s…", test_text[:60])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    inputs = tokenizer(
        test_text,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            num_beams=2,
        )

    result = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    logger.info("Inference output: %s", result[:120])
    return result


def create_handoff_doc(out_dir: Path, meta: dict) -> None:
    """Tạo HANDOFF.md hướng dẫn sử dụng model cho G5."""
    content = f"""\
# HANDOFF — Fine-tuned VietAI/vit5-base

**Ngày đóng gói:** {meta.get('packaged_at', 'N/A')}
**Checkpoint gốc:** `{meta.get('source_checkpoint', 'N/A')}`

## Nội dung thư mục này

| File / Folder | Mô tả |
|---|---|
| `config.json` | Model architecture config |
| `model.safetensors` / `pytorch_model.bin` | Model weights |
| `tokenizer_config.json`, `spiece.model` | Tokenizer |
| `generation_config.json` | Generation parameters |
| `training_info.json` | Metadata đầy đủ (seed, dataset hash, config) |

## Cách load và inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "{out_dir}"  # hoặc đường dẫn trên cloud

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

article = \"\"\"
Chính phủ vừa ban hành quyết định mới về phát triển kinh tế số
giai đoạn 2025-2030...
\"\"\"

inputs = tokenizer(article, max_length=512, truncation=True, return_tensors="pt")
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
)
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(summary)
```

## Chạy evaluation (G5 — Khải)

```bash
# Xác nhận dataset checksum
python evaluate.py --model {out_dir} \\
    --config configs/eval_baseline.yaml \\
    --split test \\
    --output_dir outputs/final_eval

# Kết quả sẽ được lưu tại:
#   outputs/final_eval/predictions.jsonl
#   outputs/final_eval/metrics.json
```

> ⚠️ **Quan trọng:** Dùng ĐÚNG config `configs/eval_baseline.yaml` mà Hải Anh đã khóa.
> Không thay đổi preprocessing, max_length, num_beams, hoặc normalization.

## Thông tin kỹ thuật

- **Model gốc:** `VietAI/vit5-base` (T5 encoder-decoder)
- **Dataset:** VietNews v2 — SHA256: `8064243b0e8fc6dbb841fe459d7440f87a12b7b8515f45f932c47112a8451681`
- **Task:** article_norm → abstract_norm (Vietnamese abstractive summarisation)
- **Seed:** 42
- **Metrics training:** `training_info.json`
"""
    (out_dir / "HANDOFF.md").write_text(content, encoding="utf-8")
    logger.info("Created HANDOFF.md")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Đóng gói best checkpoint để bàn giao")
    p.add_argument("--checkpoint", required=True,
                   help="Thư mục chứa checkpoint (hoặc training output_dir)")
    p.add_argument("--out", default=None,
                   help="Thư mục đích (mặc định: outputs/checkpoints/best)")
    p.add_argument("--config", default=None,
                   help="Training config YAML để ghi vào training_info.json")
    p.add_argument("--skip_inference_test", action="store_true",
                   help="Bỏ qua inference test (dùng khi không có model weights)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as e:
        logger.error("Thiếu dependency: %s", e)
        sys.exit(1)

    # ── Tìm checkpoint ────────────────────────────────────────────────────────
    checkpoint_dir = args.checkpoint
    if not Path(checkpoint_dir).exists():
        logger.error("Không tìm thấy: %s", checkpoint_dir)
        sys.exit(1)

    # Nếu checkpoint_dir không chứa config.json trực tiếp, tìm best checkpoint
    if not (Path(checkpoint_dir) / "config.json").exists():
        checkpoint_dir = find_best_checkpoint(checkpoint_dir)

    logger.info("Sử dụng checkpoint: %s", checkpoint_dir)

    # ── Thư mục đích ─────────────────────────────────────────────────────────
    out_dir = Path(args.out or "outputs/checkpoints/best")
    if out_dir.exists():
        logger.warning("Thư mục đích đã tồn tại: %s — sẽ ghi đè.", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Copy model files ──────────────────────────────────────────────────────
    logger.info("Copy model files → %s …", out_dir)
    keep_extensions = {".json", ".model", ".txt", ".safetensors", ".bin", ".pt", ".vocab"}
    keep_names      = {"special_tokens_map.json", "tokenizer_config.json",
                       "tokenizer.json", "config.json", "generation_config.json",
                       "spiece.model", "sentencepiece.bpe.model"}

    copied = 0
    for f in Path(checkpoint_dir).iterdir():
        if f.is_file() and (f.suffix in keep_extensions or f.name in keep_names):
            shutil.copy2(f, out_dir / f.name)
            logger.info("  Copied: %s", f.name)
            copied += 1

    if copied == 0:
        logger.error("Không có file nào được copy. Kiểm tra lại checkpoint_dir.")
        sys.exit(1)

    # ── Load training info & metrics ──────────────────────────────────────────
    training_info = load_training_info(checkpoint_dir)
    eval_metrics  = load_eval_metrics(checkpoint_dir)

    # ── Load config ───────────────────────────────────────────────────────────
    train_cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
    elif training_info.get("config"):
        train_cfg = training_info["config"]

    # ── Tạo generation_config.json nếu chưa có ────────────────────────────────
    gen_cfg_path = out_dir / "generation_config.json"
    if not gen_cfg_path.exists():
        gen_params = train_cfg.get("generation", {})
        gen_config = {
            "max_new_tokens": gen_params.get("max_new_tokens", 128),
            "num_beams": gen_params.get("num_beams", 4),
            "early_stopping": gen_params.get("early_stopping", True),
            "no_repeat_ngram_size": gen_params.get("no_repeat_ngram_size", 3),
            "length_penalty": gen_params.get("length_penalty", 1.0),
        }
        gen_cfg_path.write_text(json.dumps(gen_config, indent=2), encoding="utf-8")
        logger.info("Created generation_config.json")

    # ── Inference test ────────────────────────────────────────────────────────
    inference_result = None
    test_text = train_cfg.get("packaging", {}).get(
        "test_inference_text",
        "Chính phủ ban hành quyết định mới về phát triển kinh tế."
    )
    if not args.skip_inference_test:
        try:
            inference_result = run_inference_test(str(out_dir), test_text)
            logger.info("✅ Inference test thành công")
        except Exception as e:
            logger.error("❌ Inference test thất bại: %s", e)
            logger.error("Model có thể chưa đủ weights. Kiểm tra lại checkpoint.")
            sys.exit(1)

    # ── Ghi training_info.json ────────────────────────────────────────────────
    meta = {
        "packaged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_checkpoint": str(Path(checkpoint_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "dataset_sha256": "8064243b0e8fc6dbb841fe459d7440f87a12b7b8515f45f932c47112a8451681",
        "seed": train_cfg.get("training", {}).get("seed", 42),
        "config": train_cfg,
        "eval_metrics": eval_metrics,
        "training_info": training_info,
        "inference_test": {
            "input": test_text,
            "output": inference_result,
        },
    }
    (out_dir / "training_info.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved training_info.json")

    # ── Tạo HANDOFF.md ────────────────────────────────────────────────────────
    create_handoff_doc(out_dir, meta)

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("✅ ĐÓNG GÓI HOÀN TẤT")
    logger.info("  Output : %s", out_dir)
    logger.info("  Files  : %d file(s)", copied + 2)  # +gen_config +training_info
    if inference_result:
        logger.info("  Test   : OK (\"%s…\")", inference_result[:50])
    logger.info("")
    logger.info("Bàn giao cho Khải (G5):")
    logger.info("  python evaluate.py --model %s --config configs/eval_baseline.yaml "
                "--split test --output_dir outputs/final_eval", out_dir)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
