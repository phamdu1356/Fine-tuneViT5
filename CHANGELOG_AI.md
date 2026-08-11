# CHANGELOG_AI.md — AI Session Change Log

This file records significant changes, decisions, and discoveries made during
AI-assisted sessions. It is maintained alongside `PROJECT_CONTEXT.md` and is
updated by the AI whenever meaningful work is done on the repository.

**Format per entry:**
```
## [YYYY-MM-DD] — Session summary title
- **By:** AI assistant (model name if known)
- **Scope:** Files / components touched
- **Changes:** What was done
- **Rationale:** Why it was done
- **Open items:** Anything left unresolved
```

---

## [2026-08-09] — Initial AI memory system bootstrap

- **By:** Antigravity AI (Claude Sonnet 4.6 Thinking)
- **Scope:** Repository-level metadata files only (no source code modified)
- **Changes:**
  - Created `PROJECT_CONTEXT.md` — comprehensive project snapshot documenting
    purpose, architecture, dataset details (v2), technology stack, pipeline
    gates, implementation state, known issues, constraints, and team conventions.
  - Created `.agents/rules/project-context.md` — standing rule instructing all
    AI assistants to read `PROJECT_CONTEXT.md` before significant work and to
    keep it updated.
  - Created `.agents/skills/project-memory/SKILL.md` — skill definition for
    how AI maintains and updates project memory across sessions.
  - Created `.agents/workflows/save-project-state.md` — workflow for
    checkpointing current project state (git diff review, context update,
    changelog update, issue recording).
  - Created `CHANGELOG_AI.md` (this file) — session log.
- **Rationale:** Project has reached G2 milestone (dataset v2 frozen). Before
  further development begins (G3–G5), establishing AI memory infrastructure
  ensures future AI sessions start with full context and consistent conventions.
- **Discoveries during analysis:**
  - Repository is at a clean state (`git status` shows nothing to commit).
  - Four commits in history: initial, team tasks docs, gitignore/AGENTS.md
    removal, dataset freeze.
  - Dataset v2 parquet is stored via Git LFS (~460 MB).
  - No Python environment or `requirements.txt` is present; dependencies are
    inferred from notebook imports only.
  - `models/vit5-base/` local tokeniser directory referenced in notebook but
    does not exist in the repo.
  - `reports/`, `configs/`, `outputs/` directories expected by team task doc
    do not yet exist.
- **Open items:**
  - No baseline evaluation script exists yet.
  - No training script exists yet.
  - Python environment not pinned.
  - `README.md` is a one-line stub.

---

## [2026-08-10] — G4: Xây dựng training pipeline (Giang)

- **By:** Antigravity AI (Claude Sonnet 4.6 Thinking)
- **Scope:** New scripts, configs, setup scripts, reports template — no source code modified
- **Changes:**
  - Created `train.py` — full Seq2SeqTrainer pipeline với ROUGE metrics, --smoke flag, --resume, --override CLI; xác nhận SHA-256 dataset; log environment; lưu training_info.json
  - Created `evaluate.py` — standalone evaluation script dùng chung G3 (baseline) và G5 (fine-tuned); lưu predictions.jsonl theo id; tính ROUGE-1/2/L
  - Created `package_model.py` — đóng gói best checkpoint, tạo generation_config.json, inference test, tạo HANDOFF.md cho Khải
  - Created `requirements.txt` — pinned Python dependencies
  - Created `setup_env.bat` — Windows venv + PyTorch install (CPU/CUDA)
  - Created `setup_env.sh` — Linux/cloud venv + auto CUDA detection
  - Created `configs/train_v1.yaml` — full cloud training config với tất cả hyperparameters có comment
  - Created `configs/train_smoke.yaml` — CPU smoke test config (32 mẫu, 2 steps, greedy decode)
  - Created `reports/hardware_and_training.md` — template báo cáo cho Giang điền
  - Updated `README.md` — full setup & usage guide (từ 1 dòng thành đầy đủ)
  - Updated `.gitignore` — thêm `venv/` entry
  - Created `venv/` — local Python venv cho smoke test
  - Updated `PROJECT_CONTEXT.md` — architecture, implementation state, TODO, known issues
- **Discoveries:**
  - PyTorch trên máy local là CPU build (2.13.0+cpu), không có CUDA — cần cloud để train thực sự
  - GPU RTX 4050 Laptop (6 GB VRAM) có CUDA driver nhưng PyTorch chưa được cài CUDA build
  - `datasets` chưa được cài trong system Python
  - G3 (baseline) chưa bắt đầu — scripts đã sẵn sàng nhưng full training nên chờ G3
- **Open items:**
  - Smoke test chưa chạy (chờ venv install xong)
  - Full training chưa chạy (cần cloud GPU + G3 baseline)
  - `configs/eval_baseline.yaml` chưa tạo (Hải Anh chịu trách nhiệm G3)
  - `reports/hardware_and_training.md` chờ Giang điền sau khi train
