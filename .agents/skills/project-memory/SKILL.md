---
name: project-memory
description: >
  Skill for maintaining, reading, and updating the persistent AI project memory
  system in the Fine-tuneViT5 repository. Use this skill whenever you need to
  orient yourself in the project, update project state, or record a session log.
---

# Project Memory Skill — Fine-tuneViT5

## Purpose

This skill defines how an AI assistant reads and updates the project memory
files so that context is preserved across sessions, team members, and different
AI tools.

---

## Memory Files

| File | Purpose | When to Read | When to Update |
|---|---|---|---|
| `PROJECT_CONTEXT.md` | Canonical project snapshot | Before every significant task | After architecture/state changes |
| `CHANGELOG_AI.md` | AI session history | At session start | At session end (if work was done) |
| `docs/TEAM_TASKS.md` | Team roles & handoff gates | When touching pipeline stages | Do not modify (owned by team) |
| `data/processed/v2/manifest.json` | Dataset provenance | Before any dataset work | Never (locked) |
| `data/processed/v2/data_card.md` | Dataset card | Before any dataset work | Never (locked) |

---

## Step 1 — Session Orientation (Do This First)

```
1. Read PROJECT_CONTEXT.md (root)
2. Read CHANGELOG_AI.md (last 2–3 entries)
3. Run: git status
4. Run: git log --oneline -5
5. Identify which pipeline gate is active (G1–G5)
6. Review the relevant section of docs/TEAM_TASKS.md
```

After orientation, summarise in one paragraph:
- Current gate / phase
- What has been completed
- What the user's current request relates to
- Any relevant constraints

---

## Step 2 — Working

Apply the hard constraints from `.agents/rules/project-context.md` at all times.

Key reminders:
- **Never truncate articles in the parquet** — truncation is for tokenise time only.
- **Always use seed 42** for any random operation.
- **Never touch `data/processed/v2/`** contents (LOCK.txt, manifest.json, parquet).
- **Do not use test split** for any tuning decision.
- **Create new version directories** (v3, v4…) for new dataset runs.

---

## Step 3 — Updating Project Memory

### When to update PROJECT_CONTEXT.md

Update the relevant section(s) when:
- A new script, config, or notebook is created → update Section 2 (directory tree)
- A pipeline gate is completed → update Section 9 (implementation state)
- A bug is found or fixed → update Section 10 (known issues)
- A TODO is completed → update Section 11
- A technical decision is made → update Section 12
- A new constraint is agreed upon → update Section 13

**Always update the "Last updated" timestamp at the top of PROJECT_CONTEXT.md.**

### How to append to CHANGELOG_AI.md

Add a new `## [YYYY-MM-DD]` section at the **bottom** of the "entries" block
(before any trailing notes). Include:
- What files were created / modified / deleted
- What was discovered
- What decisions were made
- Open items remaining

---

## Step 4 — Session Close Checklist

Before ending a session where files were modified:

```
[ ] Updated PROJECT_CONTEXT.md "Last updated" date
[ ] Updated relevant sections of PROJECT_CONTEXT.md
[ ] Appended entry to CHANGELOG_AI.md
[ ] Verified no locked files were modified (git diff data/processed/v2/)
[ ] Verified no secrets or large files are staged for commit
[ ] Left TODO items clearly noted in CHANGELOG_AI.md
```

---

## Patterns and Conventions

### Dataset versioning
- Current version: `v2`
- New runs: increment to `v3`, `v4`, etc.
- Version directory: `data/processed/{VERSION}/`
- Always write a new `manifest.json`, `LOCK.txt`, and `data_card.md`

### Naming conventions (inferred from notebook)
- Sample IDs: `{VERSION}-{original_split}-{guid:06d}` (e.g., `v2-train-001234`)
- Config files: `configs/{purpose}_{version}.yaml`
- Reports: `reports/{purpose}_{version}.md`
- Checkpoint selection: best validation ROUGE-L

### Smoke testing
```bash
XULY_V2_SMOKE=1 jupyter nbconvert --to notebook --execute Xu_Ly_DataV2.ipynb
```
This runs the data pipeline on 120 rows per split to validate the pipeline
before a full run.

---

## Quick Reference — Key Numbers

| Item | Value |
|---|---|
| Total dataset rows | 143,811 |
| Train / Val / Test | 93,471 / 21,578 / 28,762 |
| Parquet SHA-256 | `8064243b...` (see PROJECT_CONTEXT.md §4) |
| Random seed | 42 |
| Mean article tokens | ~822–830 |
| Mean abstract tokens | ~58 |
| Eval pack size | 300 (100 per length bucket) |
| PII masked cells | 1,124 |
