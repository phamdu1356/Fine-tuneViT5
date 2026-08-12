# ============================================================
# Cell 0 — Imports
# ============================================================
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import hashlib

# ============================================================
# Cell 1 — Đọc cấu hình
# ============================================================
CONFIG_PATH = "configs/eval_smoke.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("Loaded config:", CONFIG_PATH)


# ============================================================
# Cell 2 — Chọn model
# ============================================================
MODEL_PATH = cfg["model"]["name_or_path"]
MODEL_REVISION = cfg["model"].get("revision", "main")

# Chỉ dùng cho progress bar và run_info; không tạo thêm thư mục output.
RUN_NAME = Path(cfg["output"]["root_dir"]).name


# ============================================================
# Cell 3 — Kiểm tra device
# ============================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )


# ============================================================
# Cell 4 — Kiểm tra dataset
# ============================================================
data_cfg = cfg["data"]
parquet_path = Path(data_cfg["parquet_path"])

assert parquet_path.exists(), f"Không tìm thấy: {parquet_path}"

expected_sha256 = data_cfg.get("expected_sha256")

if expected_sha256:
    sha256 = hashlib.sha256()

    with parquet_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)

    actual_sha256 = sha256.hexdigest()

    print("Expected SHA-256:", expected_sha256)
    print("Actual SHA-256:  ", actual_sha256)

    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Dataset checksum không khớp. "
            "Có thể đang dùng sai file vietnews_v2.parquet."
        )

    print("Dataset checksum: OK")

df = pd.read_parquet(parquet_path)

required_columns = {
    data_cfg["id_column"],
    data_cfg["split_column"],
    data_cfg["source_column"],
    data_cfg["reference_column"],
}
missing = required_columns - set(df.columns)
assert not missing, f"Thiếu cột: {missing}"

print("Dataset rows:", len(df))
print("Columns:", list(df.columns))
print(df[data_cfg["split_column"]].value_counts())


# ============================================================
# Cell 5 — Nạp dữ liệu test
# ============================================================
split_column = data_cfg["split_column"]
split_name = data_cfg["split_name"]
id_column = data_cfg["id_column"]

# Chỉ lấy các mẫu thuộc split test.
df_test = df[df[split_column] == split_name].copy()
df_test[id_column] = df_test[id_column].astype(str)

# eval_pack_300.csv có thể chứa ID của nhiều split.
# Chỉ lấy phần giao giữa eval pack và split test.
eval_pack_path = data_cfg.get("eval_pack_path")

if eval_pack_path:
    eval_pack = pd.read_csv(eval_pack_path)
    pack_ids = eval_pack[id_column].astype(str).tolist()

    df_test = df_test[df_test[id_column].isin(pack_ids)].copy()

    # Giữ đúng thứ tự các ID hợp lệ trong eval pack.
    test_id_set = set(df_test[id_column])
    available_ids = [sample_id for sample_id in pack_ids if sample_id in test_id_set]

    df_test = (
        df_test.set_index(id_column).loc[available_ids].reset_index()
    )
else:
    limit = cfg["evaluation"].get("limit")
    if limit is not None:
        df_test = df_test.head(int(limit))

df_test = df_test.reset_index(drop=True)

assert len(df_test) > 0, "Khong co mau test de chay"
print("Test samples:", len(df_test))
print(df_test[split_column].value_counts())


# ============================================================
# Cell 6 — Nạp tokenizer và model
# ============================================================
use_fast = cfg["model"].get("use_fast_tokenizer", True)
tokenizer_kwargs = {"use_fast": use_fast}

if MODEL_REVISION:
    tokenizer_kwargs["revision"] = MODEL_REVISION

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **tokenizer_kwargs)

model_kwargs = {}
if MODEL_REVISION:
    model_kwargs["revision"] = MODEL_REVISION

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, **model_kwargs)
model.to(device)
model.eval()

print("Model loaded from:", MODEL_PATH)
print("Tokenizer fast:", use_fast)
print("Vocab size:", len(tokenizer))


# ============================================================
# Cell 7 — Cấu hình generation
# ============================================================
gen_cfg = cfg["generation"]
max_source_length = int(data_cfg["max_source_length"])
batch_size = int(cfg["evaluation"]["batch_size"])

generation_kwargs = {
    "num_beams": int(gen_cfg["num_beams"]),
    "max_new_tokens": int(gen_cfg["max_new_tokens"]),
    "do_sample": bool(gen_cfg["do_sample"]),
    "no_repeat_ngram_size": int(gen_cfg["no_repeat_ngram_size"]),
    "early_stopping": bool(gen_cfg.get("early_stopping", True)),
    "length_penalty": float(gen_cfg.get("length_penalty", 1.0)),
}
print("Generation config:", generation_kwargs)


# ============================================================
# Cell 8 — Inference và lưu predictions
# ============================================================
source_column = data_cfg["source_column"]
reference_column = data_cfg["reference_column"]

output_root = Path(cfg["output"]["root_dir"])
output_root.mkdir(parents=True, exist_ok=True)
prediction_path = output_root / cfg["output"]["predictions_filename"]

all_predictions = []
all_references = []
all_ids = []
start_time = time.time()

with gzip.open(prediction_path, "wt", encoding="utf-8") as fout:
    for start in tqdm(
        range(0, len(df_test), batch_size),
        desc=f"Inference - {RUN_NAME}",
    ):
        batch_df = df_test.iloc[start : start + batch_size]
        sources = batch_df[source_column].fillna("").astype(str).tolist()
        references = batch_df[reference_column].fillna("").astype(str).tolist()
        ids = batch_df[id_column].tolist()

        inputs = tokenizer(
            sources,
            max_length=max_source_length,
            truncation=True,
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
            record = {
                "id": sample_id,
                "prediction": prediction,
                "reference": reference,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

        all_ids.extend(ids)
        all_predictions.extend(predictions)
        all_references.extend(references)

elapsed = time.time() - start_time
print("Đã lưu:", prediction_path)
print("Số prediction:", len(all_predictions))
print("Thời gian:", round(elapsed / 60, 2), "phút")


# ============================================================
# Cell 9 — Kiểm tra một số summary output
# ============================================================
for i in range(min(5, len(all_predictions))):
    print("=" * 80)
    print("ID:", all_ids[i])
    print("REFERENCE:", all_references[i])
    print("PREDICTION:", all_predictions[i])


# ============================================================
# Cell 10 — Tính ROUGE và METEOR
# ============================================================
import evaluate

rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

rouge_scores = rouge_metric.compute(
    predictions=all_predictions,
    references=all_references,
    use_stemmer=False,
)
meteor_scores = meteor_metric.compute(
    predictions=all_predictions,
    references=all_references,
)

metrics = {
    "rouge1_f1": float(rouge_scores["rouge1"]),
    "rouge2_f1": float(rouge_scores["rouge2"]),
    "rougeL_f1": float(rouge_scores["rougeL"]),
    "meteor": float(meteor_scores["meteor"]),
}
print("Metrics:", metrics)


# ============================================================
# Cell 11 — Lưu metrics và run info
# ============================================================
metrics_path = output_root / cfg["output"]["metrics_filename"]
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

run_info = {
    "run_name": RUN_NAME,
    "model_path": str(MODEL_PATH),
    "dataset_path": str(parquet_path),
    "split": split_name,
    "num_samples": len(df_test),
    "device": str(device),
    "batch_size": batch_size,
    "max_source_length": max_source_length,
    "generation": generation_kwargs,
    "prediction_file": str(prediction_path),
    "elapsed_seconds": elapsed,
}

run_info_path = output_root / cfg["output"]["run_info_filename"]
with open(run_info_path, "w", encoding="utf-8") as f:
    json.dump(run_info, f, ensure_ascii=False, indent=2)

print("Đã lưu:", metrics_path)
print("Đã lưu:", run_info_path)


# ============================================================
# Cell 12 — Đọc predictions.jsonl.gz và lưu preview .txt
# ============================================================
preview_path = output_root / "predictions_preview.txt"
preview_count = 10

with gzip.open(prediction_path, "rt", encoding="utf-8") as fin, open(
    preview_path, "w", encoding="utf-8"
) as fout:
    for index, line in enumerate(fin):
        if index >= preview_count:
            break

        record = json.loads(line)
        fout.write("=" * 80 + "\n")
        fout.write(f"ID: {record['id']}\n")
        fout.write(f"REFERENCE: {record['reference']}\n")
        fout.write(f"PREDICTION: {record['prediction']}\n")

print("Preview saved:", preview_path)
