# PROJECT_CONTEXT.md — Fine-tuneViT5

> **Last updated:** 2026-08-10 (AI session — G4 training pipeline)
> **Maintained by:** All contributors + AI assistants
> **Rule:** Any AI assistant working on this repo MUST read this file before
> beginning significant work, and MUST update the relevant sections whenever
> architecture, dataset versions, or project state changes.

---

## 1. Project Purpose

Fine-tune **`VietAI/vit5-base`** (a Vietnamese T5 model) for **abstractive text
summarisation** on Vietnamese news articles. The pipeline goes from raw data
ingestion through cleaning, dataset freeze, baseline evaluation, fine-tuning,
and final comparative evaluation.

- **Task:** Sequence-to-sequence summarisation (`article` → `abstract`)
- **Language:** Vietnamese
- **Base model:** `VietAI/vit5-base` (encoder-decoder Transformer)
- **Dataset source:** `nam194/vietnews` on HuggingFace

---

## 2. Architecture Overview

```
Fine-tuneViT5/
├── Xu_Ly_DataV2.ipynb          ← Main data-processing pipeline (v2)
├── data/
│   ├── raw/vietnews/           ← Raw parquet files (git-ignored, NOT committed)
│   │   ├── train-00000-of-00001-84acb79f6c6547a5.parquet
│   │   ├── validation-00000-of-00001-210cc51bf3cdb90f.parquet
│   │   └── test-00000-of-00001-123f98d55067eb7b.parquet
│   └── processed/
│       └── v2/                 ← LOCKED dataset (current canonical version)
│           ├── vietnews_v2.parquet   (tracked via Git LFS, ~460 MB)
│           ├── manifest.json         (full provenance record)
│           ├── data_card.md          (human-readable dataset card)
│           ├── LOCK.txt              (immutability declaration)
│           └── eval_pack_300.csv     (300 fixed eval sample IDs)
├── docs/
│   └── TEAM_TASKS.md           ← Team roles, checklists, and handoff gates
├── train.py                    ← Script training chính (G4)
├── evaluate.py                 ← Script evaluation dùng chung G3 & G5
├── package_model.py            ← Đóng gói model tốt nhất (G4)
├── requirements.txt            ← Python dependencies (pinned)
├── setup_env.bat               ← Windows venv setup
├── setup_env.sh                ← Linux/cloud venv setup (auto CUDA detect)
├── configs/
│   ├── train_v1.yaml           ← Config training đầy đủ (cloud GPU)
│   ├── train_smoke.yaml        ← Config smoke test (CPU)
│   └── eval_baseline.yaml      ← (planned) Locked eval config (Hải Anh)
├── reports/
│   └── hardware_and_training.md ← Template báo cáo (điền sau training)
├── outputs/                    ← Git-ignored (predictions, checkpoints)
├── models/
│   └── vit5-base/              ← (planned) local tokenizer cache
├── venv/                       ← Virtual environment (git-ignored)
├── .gitattributes              ← Git LFS rule for vietnews_v2.parquet
├── .gitignore                  ← Excludes raw data, checkpoints, venv, secrets
├── LICENSE                     ← Project license
└── README.md                   ← Setup & usage guide
```

**Pipeline stages (per team task document):**

```
[G1] Data Audit (Duy Anh)
  → [G2] Dataset Freeze (Minh Anh)    ← COMPLETE at v2
    → [G3] Baseline Evaluation (Hai Anh)  ← NOT YET STARTED
      → [G4] Fine-tuning (Giang)          ← NOT YET STARTED
        → [G5] Final Evaluation (Khai)    ← NOT YET STARTED
```

---

## 3. Technologies & Dependencies

### Core ML Stack (declared in notebook / task doc)

| Library | Role |
|---|---|
| `transformers` (HuggingFace) | Model loading, tokeniser, seq2seq training |
| `datasets` (HuggingFace) | Dataset loading and management |
| `torch` (PyTorch) | Deep learning backend |
| `pandas` | DataFrame manipulation in data pipeline |
| `pyarrow` / `pyarrow.parquet` | Parquet I/O |
| `numpy` | Numerical utilities |
| `hashlib` (stdlib) | SHA-256 checksums for data integrity |
| `re`, `unicodedata` (stdlib) | Text normalisation |

### Evaluation Metrics

| Metric | Notes |
|---|---|
| ROUGE-1, ROUGE-2, ROUGE-L | Primary evaluation; library + version must be locked per eval config |

### Infrastructure

| Tool | Role |
|---|---|
| Git LFS | Tracks `data/processed/v2/vietnews_v2.parquet` |
| Jupyter Notebook | Data-processing pipeline (`Xu_Ly_DataV2.ipynb`) |
| Python (version not pinned yet) | Runtime |
| CUDA / GPU | Required for training; precision (fp32/fp16/bf16) TBD |

### Model

| Item | Value |
|---|---|
| Base model ID | `VietAI/vit5-base` |
| Architecture | T5 (encoder-decoder) |
| Tokeniser | Bundled with model; revision must be pinned |
| Task | Conditional text generation (summarisation) |

---

## 4. Dataset Details — v2 (LOCKED)

| Property | Value |
|---|---|
| Version | `v2` |
| Total rows | 143,811 |
| Parquet SHA-256 | `8064243b0e8fc6dbb841fe459d7440f87a12b7b8515f45f932c47112a8451681` |
| Dataset hash | `25732a293d436fd3216d7ba6325e8c27b878d3a458b562782f7d38a7c4d56f93` |
| Split ratio | 65 / 15 / 20 (train / validation / test) |
| Split counts | train: 93,471 · validation: 21,578 · test: 28,762 |
| Seed | 42 |
| PII masked | 1,124 cells (policy: mask email / URL / phone) |
| Locked at | 2026-08-09T02:55:59+00:00 |

**Schema columns:**
`id`, `original_split`, `split_v2`, `guid`, `title`, `title_norm`,
`abstract`, `abstract_norm`, `article`, `article_norm`,
`n_tokens_article`, `n_tokens_abstract`

**Token statistics (VietAI/vit5-base tokeniser):**

| Split | Mean article tokens | Samples > 512 tok | Samples > 1024 tok |
|---|---|---|---|
| train | ~822 | 74.25 % | 26.37 % |
| validation | ~829 | 74.61 % | 27.07 % |
| test | ~830 | 74.75 % | 27.11 % |

**Key thresholds applied during processing:**
- `min_abstract_chars`: 20
- `max_abstract_to_article`: 0.75
- `near_dup_jaccard`: 0.90
- `max_article_chars`: 20,000
- `near_dup_cap_group`: 50

**Eval pack:** 300 fixed IDs (`eval_pack_300.csv`) — 100 per token bucket
(short / mid / long)

---

## 5. Configuration (Planned / In-Progress)

The following config files are expected but **not yet created**:
- `configs/data_cleaning.yaml` — cleaning rules (Duy Anh)
- `configs/eval_baseline.yaml` — locked evaluation config (Hai Anh)
- `configs/train_v1.yaml` — training hyperparameters (Giang)

**Key training parameters (to be determined):**
- `max_source_length`: likely 512 or 1024 (truncation at tokenise time only)
- `max_target_length`: TBD from abstract token stats (mean ~58 tokens)
- `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `warmup_steps` / `warmup_ratio`
- `seed`: 42 (team-wide convention)
- `label_pad_token_id`: `-100` (HuggingFace seq2seq standard)

---

## 6. Build / Run Commands

### Data Pipeline

```bash
# Smoke test (subset of 120 rows per split)
XULY_V2_SMOKE=1 jupyter nbconvert --to notebook --execute Xu_Ly_DataV2.ipynb

# Full pipeline run (requires data/raw/vietnews/*.parquet to exist)
jupyter nbconvert --to notebook --execute Xu_Ly_DataV2.ipynb
```

> NOTE: Running the full pipeline produces a new versioned directory.
> Change `VERSION = "v3"` in cell 1 before re-running to avoid overwriting v2.

### Training

```bash
# Smoke test (CPU, local)
venv\Scripts\activate
python train.py --config configs/train_smoke.yaml --smoke

# Full training (cloud GPU)
source venv/bin/activate
python train.py --config configs/train_v1.yaml

# Resume
python train.py --config configs/train_v1.yaml --resume auto

# Multi-GPU
accelerate launch train.py --config configs/train_v1.yaml
```

### Evaluation

```bash
# Baseline (G3)
python evaluate.py --model VietAI/vit5-base --config configs/eval_baseline.yaml --split validation

# Fine-tuned (G5)
python evaluate.py --model outputs/checkpoints/best --config configs/eval_baseline.yaml \
    --split test --output_dir outputs/final_eval
```

### Packaging

```bash
python package_model.py --checkpoint outputs/checkpoints --config configs/train_v1.yaml
```

---

## 7. APIs

No external APIs are currently used in committed code. The pipeline reads from
local parquet files. Future training will pull `VietAI/vit5-base` from
HuggingFace Hub (internet required unless cached at `models/vit5-base/`).

---

## 8. Database / Storage

| Storage | Location | Notes |
|---|---|---|
| Raw dataset | `data/raw/vietnews/` | Git-ignored; download separately |
| Processed dataset v2 | `data/processed/v2/vietnews_v2.parquet` | Git LFS; ~460 MB |
| Review queues | `data/processed/v2/review_queue/` | Git-ignored; CSV flag files |
| Model checkpoints | `outputs/checkpoints/` | Git-ignored |
| Predictions | `outputs/baseline/`, `outputs/final_eval/` | Git-ignored |

---

## 9. Current Implementation State

| Gate | Owner | Status |
|---|---|---|
| G1 — Data Audit | Duy Anh | COMPLETE (embedded in `Xu_Ly_DataV2.ipynb`) |
| G2 — Dataset Freeze | Minh Anh | COMPLETE — `data/processed/v2/` locked |
| G3 — Baseline Evaluation | Hai Anh | NOT STARTED |
| G4 — Fine-tuning | Giang | IN PROGRESS — scripts created, smoke test pending |
| G5 — Final Evaluation | Khai | NOT STARTED |

**What exists in the repo now:**
- Locked, processed dataset (v2 parquet + manifest + lock + data card + eval pack)
- Complete data-processing notebook (`Xu_Ly_DataV2.ipynb`) with executed outputs
- Team task specification (`docs/TEAM_TASKS.md`)
- `train.py` — full training script (Seq2SeqTrainer, ROUGE metrics, smoke/resume/override)
- `evaluate.py` — standalone eval script (shared G3 + G5, saves predictions JSONL)
- `package_model.py` — packaging + HANDOFF.md generation
- `requirements.txt` — pinned dependencies
- `setup_env.bat` / `setup_env.sh` — venv setup for Windows/Linux
- `configs/train_v1.yaml` — full cloud training config
- `configs/train_smoke.yaml` — CPU smoke test config
- `reports/hardware_and_training.md` — template (Giang to fill)
- `README.md` — full setup + usage guide
- `venv/` — local virtual environment (git-ignored)
- No model checkpoint yet (training pending on cloud)

---

## 10. Known Issues

| ID | Description | Status |
|---|---|---|
| I-001 | `fact_flags.csv` has 76,098 rows — kept with note in data card | Accepted |
| I-002 | 108 near-duplicate cross-split pairs; 0 exact leaks; screening is approximate | Accepted risk |
| I-003 | `data/raw/` git-ignored; no download script exists for new contributors | Open |
| I-004 | `models/vit5-base/` local tokeniser path referenced in notebook but does not exist | Open |
| I-005 | `README.md` was a one-line stub | RESOLVED — full README created |
| I-006 | PyTorch installed as CPU build on local machine — needs CUDA rebuild for real training | Open (use cloud) |

---

## 11. TODO

- [ ] Create `configs/eval_baseline.yaml` and lock evaluation config (Hai Anh — G3)
- [x] Create training script `train.py` + `configs/train_v1.yaml` (Giang — G4)
- [x] Create `evaluate.py` for shared evaluation (G3 + G5)
- [x] Create `package_model.py` for model packaging (G4)
- [x] Create `requirements.txt` with pinned dependencies
- [x] Create `setup_env.bat` / `setup_env.sh`
- [x] Expand `README.md` with full setup instructions
- [ ] Run smoke test on local machine (pending venv install)
- [ ] Run full training on cloud GPU (pending G3 baseline)
- [ ] Fill in `reports/hardware_and_training.md` after training
- [ ] Write a data download script for `data/raw/vietnews/`
- [ ] Write `reports/baseline_evaluation.md` after baseline run (Hai Anh)
- [ ] Write `reports/final_comparison.md` (Khai)

---

## 12. Important Technical Decisions

| Decision | Rationale |
|---|---|
| Article NOT truncated in parquet | Truncation happens at tokenise time; preserves full text for future experiments |
| Split ratio 65/15/20 | Larger test set than original 80/10/10 for more reliable evaluation |
| Seed 42 everywhere | Team-wide reproducibility convention |
| PII policy: mask (not remove) | Keeps dataset size intact; masked text signals removed info |
| Flagging over deletion | Suspicious rows go to `review_queue/`; humans decide |
| group_id / near-dup grouping before split | Prevents data leakage between train/val/test |
| Labels use `-100` padding | HuggingFace standard for excluding padding from loss |
| Test set frozen after G2 | Only one final evaluation run allowed |

---

## 13. Constraints — MUST NOT Change Accidentally

> CRITICAL: Read this section before modifying anything.

1. **`data/processed/v2/vietnews_v2.parquet`** — LOCKED. Do not modify, delete,
   or overwrite. SHA-256: `8064243b0e8fc6dbb841fe459d7440f87a12b7b8515f45f932c47112a8451681`

2. **`data/processed/v2/LOCK.txt`** — Do not modify or delete.

3. **`data/processed/v2/manifest.json`** — Do not modify; it is the provenance record.

4. **Test split rows** — The 28,762 test samples must never be used to tune
   hyperparameters, checkpoints, prompts, or decoding strategies.

5. **Seed = 42** — Must stay consistent across all pipeline stages.

6. **Evaluation config (once locked at G3)** — Preprocessing, prompt,
   normalization, beam size, max length must not change after Hai Anh locks
   `configs/eval_baseline.yaml`.

7. **Raw data integrity** — `data/raw/vietnews/` files must never be modified;
   checksums recorded in `manifest.json`.

8. **Versioning discipline** — New dataset runs must use a new version number
   (v3, v4, …). Never overwrite an existing version directory.

9. **Git hygiene** — Do not commit raw data, checkpoints, `.env` files, or model
   weights (`*.pt`, `*.safetensors`, `*.bin`, `*.ckpt`).

---

## 14. Team Members

| Member | Primary Role |
|---|---|
| Duy Anh | Data audit & initial cleaning |
| Minh Anh | Dataset processing & freeze |
| Hai Anh | Baseline evaluation |
| Giang | Fine-tuning & model packaging |
| Khai | Post-training evaluation & final report |
