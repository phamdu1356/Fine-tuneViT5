# Workflow — Save Project State

## Purpose

This workflow checkpoints the current state of the repository into the AI
memory files without modifying any source code. Run it:
- At the end of a significant AI session
- After completing a pipeline gate (G1–G5)
- When the team wants a clean snapshot for the next session

---

## Prerequisites

- Git is available on the system
- The user is in the repository root: `Fine-tuneViT5/`

---

## Step 1 — Check Working Tree Status

```bash
git status
```

Record:
- Branch name
- Whether working tree is clean
- Any untracked files
- Any staged or unstaged changes

---

## Step 2 — Review Recent Commits

```bash
git log --oneline -10
git log --oneline --stat -3
```

Record:
- The last 3–5 meaningful commit messages
- What files were changed in the most recent commit
- Whether any locked files were touched (check `data/processed/v2/`)

---

## Step 3 — Check for Changes to Locked Files

```bash
git diff HEAD -- data/processed/v2/
git status data/processed/v2/
```

**If any locked files have been modified:** STOP. Alert the user immediately.
Do not proceed until the modification is explained and intentional.

Locked files that must never be modified:
- `data/processed/v2/vietnews_v2.parquet`
- `data/processed/v2/LOCK.txt`
- `data/processed/v2/manifest.json`

---

## Step 4 — Identify Completed and In-Progress Work

Review the current gate status from `docs/TEAM_TASKS.md` and recent git history:

| Gate | Owner | Check |
|---|---|---|
| G1 — Data Audit | Duy Anh | `reports/data_audit.md` exists? |
| G2 — Dataset Freeze | Minh Anh | `data/processed/v2/LOCK.txt` exists? |
| G3 — Baseline Eval | Hai Anh | `configs/eval_baseline.yaml` + `outputs/baseline/predictions.jsonl` exist? |
| G4 — Fine-tuning | Giang | `outputs/checkpoints/best/` exists? |
| G5 — Final Eval | Khai | `reports/final_comparison.md` exists? |

---

## Step 5 — Update PROJECT_CONTEXT.md

Edit `PROJECT_CONTEXT.md` to reflect current state:

1. Update the **"Last updated"** timestamp at the top.
2. Update **Section 2** (Architecture) if any new directories or files exist.
3. Update **Section 9** (Current Implementation State) with gate statuses.
4. Update **Section 10** (Known Issues) with any new or resolved issues.
5. Update **Section 11** (TODO) — check off completed items, add new ones.
6. Update **Section 12** (Technical Decisions) if any new decisions were made.

**Do NOT modify Sections 1, 3, 4, 12, or 13 unless the corresponding reality
has actually changed.**

---

## Step 6 — Update CHANGELOG_AI.md

Append a new entry at the bottom of CHANGELOG_AI.md using this template:

```markdown
## [YYYY-MM-DD] — Brief description of session

- **By:** [AI assistant name / model]
- **Scope:** [Files or components touched]
- **Changes:**
  - [Bullet list of what was created, modified, deleted]
- **Rationale:** [Why these changes were made]
- **Discoveries:** [Anything newly learned about the codebase]
- **Open items:**
  - [Unresolved issues or next steps]
```

---

## Step 7 — Record Unresolved Issues

For each unresolved issue:
1. Ensure it appears in `PROJECT_CONTEXT.md` Section 10 (Known Issues) with an
   ID (e.g., `I-006`).
2. Note it in the `CHANGELOG_AI.md` entry under "Open items".
3. If blocking a gate, note which gate is blocked and why.

---

## Step 8 — Final Verification

```bash
# Confirm no source code was modified
git diff HEAD -- "*.py" "*.ipynb" "*.yaml" "*.json" "*.md"

# Confirm only memory files changed
git diff HEAD -- PROJECT_CONTEXT.md CHANGELOG_AI.md .agents/
```

If source code was unintentionally modified, restore it:
```bash
git checkout HEAD -- <file>
```

---

## What This Workflow Does NOT Do

- Does NOT modify any source code, notebooks, or scripts.
- Does NOT commit or push changes (user decides whether to commit).
- Does NOT modify locked dataset files.
- Does NOT delete any project files.
- Does NOT invent or fabricate project state — only records what actually exists.

---

## Output

After running this workflow, the following should be true:
- `PROJECT_CONTEXT.md` accurately reflects the current repository state.
- `CHANGELOG_AI.md` has a new entry documenting the session.
- Any unresolved issues are documented.
- No source code was touched.
