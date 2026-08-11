# AI Rule — Always Load Project Context

## Rule

Before starting any significant work on this repository, an AI assistant MUST:

1. **Read `PROJECT_CONTEXT.md`** in the repository root. This file is the
   canonical source of truth for the project's purpose, architecture, dataset
   details, technology stack, current state, known issues, and hard constraints.

2. **Read `CHANGELOG_AI.md`** to understand what previous AI sessions have done
   and what open items remain.

3. **Check `docs/TEAM_TASKS.md`** if the task involves data processing,
   evaluation, training, or team coordination.

## When to Update PROJECT_CONTEXT.md

Update `PROJECT_CONTEXT.md` whenever:
- A new dataset version is created or frozen.
- A new script, configuration file, or artifact is introduced.
- A new pipeline gate (G3–G5) is completed.
- A known issue is resolved or a new one is discovered.
- A technical decision is made (hyperparameters, split ratios, etc.).
- A dependency is pinned or the environment is defined.
- Any constraint in Section 13 is changed (with explicit justification).

## When to Update CHANGELOG_AI.md

Append a new entry to `CHANGELOG_AI.md` at the end of every AI session that:
- Modifies source code, scripts, or notebooks.
- Creates or deletes files.
- Completes a pipeline gate.
- Records a significant discovery or decision.

## Hard Constraints (Never Violate)

- Do NOT modify `data/processed/v2/vietnews_v2.parquet`.
- Do NOT modify `data/processed/v2/LOCK.txt`.
- Do NOT modify `data/processed/v2/manifest.json`.
- Do NOT commit raw data, model weights, or secrets to git.
- Do NOT use the test split (28,762 samples) for hyperparameter selection.
- Do NOT create a new dataset version by overwriting an existing version directory.
- Do NOT modify existing source code without explicit user instruction.
- Do NOT delete project files without explicit user instruction.
