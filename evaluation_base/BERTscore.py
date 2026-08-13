#!/usr/bin/env python3
"""Compute BERTScore for ViT5-base evaluator predictions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "eval_base_v2.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    from bert_score import score

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config

    with config_path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    output_cfg = cfg["output"]
    output_root = resolve_path(output_cfg["root_dir"])
    prediction_path = output_root / output_cfg["predictions_filename"]
    metrics_path = output_root / output_cfg["metrics_filename"]
    bertscore_path = output_root / output_cfg["bertscore_filename"]
    final_metrics_path = output_root / output_cfg["final_metrics_filename"]

    with gzip.open(prediction_path, "rt", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    if not records:
        raise ValueError(f"Không có prediction trong {prediction_path}")

    predictions = [record["prediction"] for record in records]
    references = [record["reference"] for record in records]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluation_cfg = cfg["evaluation"]

    precision, recall, f1 = score(
        predictions,
        references,
        model_type=evaluation_cfg["bertscore_model_type"],
        lang=evaluation_cfg["bertscore_language"],
        device=device,
        batch_size=int(evaluation_cfg["bertscore_batch_size"]),
        verbose=True,
    )
    bertscore_metrics = {
        "bertscore_precision": float(precision.mean()),
        "bertscore_recall": float(recall.mean()),
        "bertscore_f1": float(f1.mean()),
    }
    bertscore_path.write_text(
        json.dumps(bertscore_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final_metrics = {**metrics, **bertscore_metrics}
    final_metrics_path.write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Final metrics:", final_metrics)


if __name__ == "__main__":
    main()
