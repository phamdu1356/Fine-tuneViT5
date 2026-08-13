import json
import gzip
from pathlib import Path

import yaml
import torch
from bert_score import score


CONFIG_PATH = "configs/eval_finetune_v2.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

output_root = Path(cfg["output"]["root_dir"])

prediction_path = (
    output_root / cfg["output"]["predictions_filename"]
)

metrics_path = (
    output_root / cfg["output"]["metrics_filename"]
)

bertscore_path = (
    output_root / cfg["output"]["bertscore_filename"]
)

final_metrics_path = (
    output_root / cfg["output"]["final_metrics_filename"]
)

with gzip.open(prediction_path, "rt", encoding="utf-8") as f:
    records = [
        json.loads(line)
        for line in f
    ]

predictions = [
    record["prediction"]
    for record in records
]

references = [
    record["reference"]
    for record in records
]

device = "cuda" if torch.cuda.is_available() else "cpu"

P, R, F1 = score(
    predictions,
    references,
    model_type=cfg["evaluation"]["bertscore_model_type"],
    lang=cfg["evaluation"]["bertscore_language"],
    device=device,
    batch_size=cfg["evaluation"]["bertscore_batch_size"],
    verbose=True,
)

bertscore_metrics = {
    "bertscore_precision": float(P.mean()),
    "bertscore_recall": float(R.mean()),
    "bertscore_f1": float(F1.mean()),
}

with open(bertscore_path, "w", encoding="utf-8") as f:
    json.dump(
        bertscore_metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

final_metrics = {
    **metrics,
    **bertscore_metrics,
}

with open(final_metrics_path, "w", encoding="utf-8") as f:
    json.dump(
        final_metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(final_metrics)