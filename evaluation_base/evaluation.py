#!/usr/bin/env python3
"""Evaluate VietAI/vit5-base on test split v2 without a LoRA adapter."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "eval_base_v2.yaml"
SEED = 42


def resolve_path(value: str | Path) -> Path:
    """Resolve repository-relative paths from the config."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the base-model evaluation config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    parquet_path = resolve_path(data_cfg["parquet_path"])
    if not parquet_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {parquet_path}")

    expected_sha256 = data_cfg.get("expected_sha256")
    actual_sha256 = sha256_file(parquet_path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Dataset checksum không khớp.\n"
            f"Expected: {expected_sha256}\nActual:   {actual_sha256}"
        )

    df = pd.read_parquet(parquet_path)
    required_columns = {
        data_cfg["id_column"],
        data_cfg["split_column"],
        data_cfg["source_column"],
        data_cfg["reference_column"],
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột: {sorted(missing)}")

    split_column = data_cfg["split_column"]
    split_name = data_cfg["split_name"]
    id_column = data_cfg["id_column"]
    df_test = df[df[split_column] == split_name].copy()
    df_test[id_column] = df_test[id_column].astype(str)

    if df_test[id_column].duplicated().any():
        raise ValueError("Test split có ID trùng.")

    limit = cfg["evaluation"].get("limit")
    if limit is not None:
        df_test = df_test.head(int(limit))
    elif data_cfg.get("expected_test_samples") is not None:
        expected_count = int(data_cfg["expected_test_samples"])
        if len(df_test) != expected_count:
            raise ValueError(
                f"Số mẫu test sai: expected {expected_count}, got {len(df_test)}"
            )

    if df_test.empty:
        raise ValueError("Không có mẫu test để đánh giá.")
    df_test = df_test.reset_index(drop=True)

    use_fast = bool(model_cfg.get("use_fast_tokenizer", True))
    model_name = model_cfg["name_or_path"]
    tokenizer_path = model_cfg.get("tokenizer_path", model_name)
    revision = model_cfg.get("revision")
    load_kwargs = {"revision": revision} if revision else {}

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=use_fast,
        **load_kwargs,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    model.eval()

    gen_cfg = cfg["generation"]
    generation_kwargs = {
        "num_beams": int(gen_cfg["num_beams"]),
        "max_new_tokens": int(gen_cfg["max_new_tokens"]),
        "do_sample": bool(gen_cfg["do_sample"]),
        "no_repeat_ngram_size": int(gen_cfg["no_repeat_ngram_size"]),
        "early_stopping": bool(gen_cfg.get("early_stopping", False)),
        "length_penalty": float(gen_cfg.get("length_penalty", 1.0)),
    }

    output_cfg = cfg["output"]
    output_root = resolve_path(output_cfg["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / output_cfg["predictions_filename"]
    source_column = data_cfg["source_column"]
    reference_column = data_cfg["reference_column"]
    batch_size = int(cfg["evaluation"]["batch_size"])
    max_source_length = int(data_cfg["max_source_length"])

    all_predictions: list[str] = []
    all_references: list[str] = []
    all_ids: list[str] = []
    start_time = time.time()

    with gzip.open(prediction_path, "wt", encoding="utf-8") as output_file:
        for start in tqdm(
            range(0, len(df_test), batch_size),
            desc="Inference - ViT5-base",
        ):
            batch_df = df_test.iloc[start : start + batch_size]
            sources = batch_df[source_column].fillna("").astype(str).tolist()
            references = batch_df[reference_column].fillna("").astype(str).tolist()
            ids = batch_df[id_column].tolist()

            inputs = tokenizer(
                sources,
                max_length=max_source_length,
                truncation=bool(data_cfg.get("truncation", True)),
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_kwargs)

            predictions = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            predictions = [text.strip() for text in predictions]

            for sample_id, prediction, reference in zip(ids, predictions, references):
                output_file.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "prediction": prediction,
                            "reference": reference,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            all_ids.extend(ids)
            all_predictions.extend(predictions)
            all_references.extend(references)

    elapsed_seconds = time.time() - start_time
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    rouge_scores = rouge.compute(
        predictions=all_predictions,
        references=all_references,
        use_stemmer=False,
    )
    meteor_scores = meteor.compute(
        predictions=all_predictions,
        references=all_references,
    )
    metrics = {
        "rouge1_f1": float(rouge_scores["rouge1"]),
        "rouge2_f1": float(rouge_scores["rouge2"]),
        "rougeL_f1": float(rouge_scores["rougeL"]),
        "meteor": float(meteor_scores["meteor"]),
    }

    metrics_path = output_root / output_cfg["metrics_filename"]
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_info = {
        "model_name_or_path": model_name,
        "tokenizer_path": tokenizer_path,
        "dataset_path": str(parquet_path),
        "dataset_sha256": actual_sha256,
        "split": split_name,
        "num_samples": len(df_test),
        "device": str(device),
        "batch_size": batch_size,
        "max_source_length": max_source_length,
        "generation": generation_kwargs,
        "prediction_file": str(prediction_path),
        "elapsed_seconds": elapsed_seconds,
        "seed": SEED,
    }
    run_info_path = output_root / output_cfg["run_info_filename"]
    run_info_path.write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_path = output_root / "predictions_preview.txt"
    with gzip.open(prediction_path, "rt", encoding="utf-8") as input_file:
        preview_lines = []
        for index, line in enumerate(input_file):
            if index >= 10:
                break
            record = json.loads(line)
            preview_lines.extend(
                [
                    "=" * 80,
                    f"ID: {record['id']}",
                    f"REFERENCE: {record['reference']}",
                    f"PREDICTION: {record['prediction']}",
                ]
            )
    preview_path.write_text("\n".join(preview_lines) + "\n", encoding="utf-8")

    print("Model:", model_name)
    print("Dataset SHA-256:", actual_sha256)
    print("Test samples:", len(df_test))
    print("Metrics:", metrics)
    print("Output:", output_root)


if __name__ == "__main__":
    main()
